# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Train a GAN using the techniques described in the paper
"Alias-Free Generative Adversarial Networks"."""

import os
import click
import re
import json
import torch

import dnnlib
from training_ito import training_loop
from torch_utils import distributed as dist

#----------------------------------------------------------------------------

def init_dataset_kwargs(data):
    try:
        dataset_kwargs = dnnlib.EasyDict(class_name='training_ito.dataset.ImageFolderDataset', path=data, use_labels=True, max_size=None, xflip=False)
        dataset_obj = dnnlib.util.construct_class_by_name(**dataset_kwargs) # Subclass of training.dataset.Dataset.
        dataset_kwargs.resolution = dataset_obj.resolution # Be explicit about resolution.
        dataset_kwargs.use_labels = dataset_obj.has_labels # Be explicit about labels.
        dataset_kwargs.max_size = len(dataset_obj) # Be explicit about dataset size.
        return dataset_kwargs, dataset_obj.name
    except IOError as err:
        raise click.ClickException(f'--data: {err}')

#----------------------------------------------------------------------------

def parse_comma_separated_list(s):
    if isinstance(s, list):
        return s
    if s is None or s.lower() == 'none' or s == '':
        return []
    return s.split(',')

#----------------------------------------------------------------------------

@click.command()

# Required.
@click.option('--outdir',       help='Where to save the results', metavar='DIR',                required=True)
@click.option('--data',         help='Training data', metavar='[ZIP|DIR]',                      type=str, required=True)
@click.option('--batch',        help='Total batch size', metavar='INT',                         type=click.IntRange(min=1), required=True)
@click.option('--batch_edm',    help='Total batch size for V', metavar='INT',                   type=click.IntRange(min=1), required=True)
@click.option('--batch_disc',   help='Total batch size for V', metavar='INT',                   type=click.IntRange(min=1), required=True)

# Optional features.
@click.option('--cond',         help='Train conditional model', metavar='BOOL',                 type=bool, default=False, show_default=True)
@click.option('--mirror',       help='Enable dataset x-flips', metavar='BOOL',                  type=bool, default=False, show_default=True)
@click.option('--resume',       help='Resume from given network pickle', metavar='[PATH|URL]',  type=str)
@click.option('--freezed',      help='Freeze first layers of D', metavar='INT',                 type=click.IntRange(min=0), default=0, show_default=True)

# Misc hyperparameters.
@click.option('--batch-gpu',    help='Limit batch size per GPU', metavar='INT',                 type=click.IntRange(min=1))
@click.option('--cbase',        help='Capacity multiplier', metavar='INT',                      type=click.IntRange(min=1), default=32768, show_default=True)
@click.option('--cmax',         help='Max. feature maps', metavar='INT',                        type=click.IntRange(min=1), default=512, show_default=True)
@click.option('--glr',          help='G learning rate  [default: varies]', metavar='FLOAT',     type=click.FloatRange(min=0), default=1e-4, show_default=True)
@click.option('--dlr',          help='D learning rate', metavar='FLOAT',                        type=click.FloatRange(min=0), default=2e-3, show_default=True)
@click.option('--map-depth',    help='Mapping network depth  [default: varies]', metavar='INT', type=click.IntRange(min=1))

# Misc settings.
@click.option('--desc',         help='String to include in result dir name', metavar='STR',     type=str)
@click.option('--kimg',         help='Total training duration', metavar='KIMG',                 type=click.IntRange(min=1), default=960, show_default=True)
@click.option('--tick',         help='How often to print progress', metavar='KIMG',             type=click.IntRange(min=1), default=5, show_default=True)
@click.option('--snap',         help='How often to save snapshots', metavar='TICKS',            type=click.IntRange(min=1), default=1, show_default=True)
@click.option('--snap_img',     help='How often to save example snapshots', metavar='TICKS',    type=click.IntRange(min=1), default=1, show_default=True)
@click.option('--seed',         help='Random seed', metavar='INT',                              type=click.IntRange(min=0), default=0, show_default=True)
@click.option('--fp32',         help='Disable mixed-precision', metavar='BOOL',                 type=bool, default=False, show_default=True)
@click.option('--nobench',      help='Disable cuDNN benchmarking', metavar='BOOL',              type=bool, default=False, show_default=True)
@click.option('--workers',      help='DataLoader worker processes', metavar='INT',              type=click.IntRange(min=1), default=3, show_default=True)
@click.option('-n','--dry-run', help='Print training options and exit',                         is_flag=True)

# StyleGAN-XL additions
@click.option('--stem',         help='Train the stem.', is_flag=True)
@click.option('--syn_layers',   help='Number of layers in the stem', type=click.IntRange(min=1), default=14, show_default=True)
@click.option('--superres',     help='Train superresolution stage. You have to provide the path to a pretrained stem.', is_flag=True)
@click.option('--path_stem',    help='Path to pretrained stem',  type=str)
@click.option('--head_layers',  help='Layers of added superresolution head.', type=click.IntRange(min=1), default=7, show_default=True)
@click.option('--cls_weight',   help='class guidance weight', type=float, default=0.0, show_default=True)
@click.option('--up_factor',    help='Up sampling factor of superres head', type=click.IntRange(min=2), default=2, show_default=True)

# Generator hyperparameters
@click.option('--net_pkl',                  help='Path to pretrained model', metavar='DIR',                    type=str, required=True)
@click.option('--scale',                    help='Scale for MAC', metavar='FLOAT',                             type=float, required=True)
@click.option('--basis_num',                help='basis_num for MAC', metavar='INT',                           type=int, required=True)
@click.option('--multidimensionality',      help='multidimensionality for MAC', metavar='LIST',                type=list, required=True)
@click.option('--lpf',                      help='lpf for MAC', metavar='BOOL',                                type=bool, required=True)
@click.option('--gauss_kernel_sigma',       help='gauss_kernel_sigma for MAC', metavar='FLOAT',                type=click.FloatRange(min=0, min_open=True), required=True)
@click.option('--num_steps',                help='N for sampler', metavar='INT',                               type=int, required=True)
@click.option('--rho',                      help='rho for sampler', metavar='INT',                             type=int, required=True)
@click.option('--s_churn',                  help='For sampler', metavar='INT',                                 type=int, required=True)
@click.option('--s_min',                    help='For sampler', metavar='FLOAT',                               type=float, required=True)
@click.option('--s_max',                    help='For sampler', metavar='FLOAT',                               type=float, required=True)
@click.option('--s_noise',                  help='For sampler', metavar='FLOAT',                               type=float, required=True)
@click.option('--x_0_conditioning',         help='lambda for reg loss', metavar='BOOL',                        type=bool, required=True)
@click.option('--k',                        help='Path smoothness for path calculator', metavar='INT',         type=int, required=True)
@click.option('--train_v',                  help='Train V', metavar='BOOL',                                    type=bool, required=True)
@click.option('--train_c',                  help='Train C', metavar='BOOL',                                    type=bool, required=True)
@click.option('--coefficient_conditioning', help='coefficient_conditioning', metavar='BOOL',                   type=bool, required=True)

def main(**kwargs):
    '''Training code for Inference Trajectory Optimization '''
    
    dist.init()
    torch.multiprocessing.set_start_method('spawn')

    # Initialize config.
    opts = dnnlib.EasyDict(kwargs)  # Command line arguments
    c = dnnlib.EasyDict()  # Main config dict.
    c.V_opt_kwargs = dnnlib.EasyDict(class_name='torch.optim.Adam', lr=1e-5, betas=[0.0, 0.99], eps=1e-8)
    c.C_opt_kwargs = dnnlib.EasyDict(class_name='torch.optim.Adam', lr=opts.glr, betas=[0.0, 0.99], eps=1e-8)
    c.D_opt_kwargs = dnnlib.EasyDict(class_name='torch.optim.Adam', lr=opts.dlr, betas=[0.0, 0.99], eps=1e-8)
    c.data_loader_kwargs = dnnlib.EasyDict(pin_memory=True, prefetch_factor=2)

    # Training set.
    c.dataset_kwargs, dataset_name = init_dataset_kwargs(data=opts.data)
    if opts.cond and not c.dataset_kwargs.use_labels:
        raise click.ClickException('--cond=True requires labels specified in dataset.json')
    c.dataset_kwargs.use_labels = opts.cond
    c.dataset_kwargs.xflip = opts.mirror

    # Hyperparameters & settings.
    c.batch_size = opts.batch
    c.batch_size_edm = opts.batch_edm
    c.batch_size_disc = opts.batch_disc
    c.total_kimg = opts.kimg
    c.kimg_per_tick = opts.tick
    c.network_snapshot_ticks = opts.snap
    c.image_snapshot_ticks = opts.snap_img
    c.seed = c.dataset_kwargs.random_seed = opts.seed
    c.data_loader_kwargs.num_workers = opts.workers
    c.train_v = opts.train_v
    c.train_c = opts.train_c
    
    # Description string.
    desc = f'{dataset_name:s}-ito-gpus{dist.get_world_size():d}-batch{c.batch_size:d}'
    if opts.desc is not None:
        desc += f'-{opts.desc}'

    # Generator
    c.V_pkl = opts.net_pkl
    c.C_kwargs = dnnlib.EasyDict(
        class_name='mac.coefficient_generator.CoefficientGenerator',
        resolution=c.dataset_kwargs.resolution,
        multidimensionality=opts.multidimensionality,
        scale=opts.scale,
        M=opts.basis_num,
        lpf=opts.lpf,
        gauss_kernel_sigma=opts.gauss_kernel_sigma,
        x_0_conditioning=opts.x_0_conditioning
    )
    c.MC_kwargs = dnnlib.EasyDict(
        class_name='mac.mac_calculator.SinusoidalMacCalculator',
        M=opts.basis_num,
        rho=opts.rho,
        k=opts.k
    )
    c.G_kwargs = dnnlib.EasyDict(
        class_name='training_ito.generator.Generator',
        sigma_min=0.002, 
        sigma_max=80,
        num_steps=opts.num_steps,
        rho=opts.rho,
        S_churn=opts.s_churn, 
        S_min=opts.s_min, 
        S_max=opts.s_max, 
        S_noise=opts.s_noise,
        coefficient_conditioning=opts.coefficient_conditioning
    )

    # Discriminator
    c.D_kwargs = dnnlib.EasyDict(
        class_name='pg_modules.discriminator.ProjectedDiscriminator',
        backbones=['deit_base_distilled_patch16_224', 'tf_efficientnet_lite0'],
        diffaug=True,
        interp224=(c.dataset_kwargs.resolution < 224),
        backbone_kwargs=dnnlib.EasyDict(),
    )
    c.D_kwargs.backbone_kwargs.cout = 64
    c.D_kwargs.backbone_kwargs.expand = True
    c.D_kwargs.backbone_kwargs.proj_type = 2 if c.dataset_kwargs.resolution <= 16 else 2  # CCM only works better on very low resolutions
    c.D_kwargs.backbone_kwargs.num_discs = 4
    c.D_kwargs.backbone_kwargs.cond = opts.cond

    # Loss
    c.loss_kwargs = dnnlib.EasyDict(class_name='training_ito.loss.ProjectedGANLoss')

    # Pick output directory.
    if dist.get_rank() != 0:
        c.run_dir = None
    else:
        prev_run_dirs = []
        if os.path.isdir(opts.outdir):
            prev_run_dirs = [x for x in os.listdir(opts.outdir) if os.path.isdir(os.path.join(opts.outdir, x))]
        prev_run_ids = [re.match(r'^\d+', x) for x in prev_run_dirs]
        prev_run_ids = [int(x.group()) for x in prev_run_ids if x is not None]
        cur_run_id = max(prev_run_ids, default=-1) + 1
        c.run_dir = os.path.join(opts.outdir, f'{cur_run_id:05d}-{desc}')
        assert not os.path.exists(c.run_dir)
    
    # Print options.
    dist.print0()
    dist.print0('Training options:')
    dist.print0(json.dumps(c, indent=2))
    dist.print0()
    dist.print0(f'Output directory:    {c.run_dir}')
    dist.print0(f'Number of GPUs:      {dist.get_world_size()}')
    dist.print0(f'Batch size:          {c.batch_size} images')
    dist.print0(f'Training duration:   {c.total_kimg} kimg')
    dist.print0(f'Dataset path:        {c.dataset_kwargs.path}')
    dist.print0(f'Dataset size:        {c.dataset_kwargs.max_size} images')
    dist.print0(f'Dataset resolution:  {c.dataset_kwargs.resolution}')
    dist.print0(f'Dataset labels:      {c.dataset_kwargs.use_labels}')
    dist.print0(f'Dataset x-flips:     {c.dataset_kwargs.xflip}')
    dist.print0()

    # Create output directory.
    print('Creating output directory...')
    if dist.get_rank() == 0:
        os.makedirs(c.run_dir, exist_ok=True)
        os.makedirs(os.path.join(c.run_dir, 'image_snapshot', 'coefficients'), exist_ok=True)
        os.makedirs(os.path.join(c.run_dir, 'image_snapshot', 'paths'), exist_ok=True)
        with open(os.path.join(c.run_dir, 'training_options.json'), 'wt+') as f:
            json.dump(c, f, indent=2)

    # Execute training loop.
    training_loop.training_loop(**c)

#----------------------------------------------------------------------------

if __name__ == "__main__":
    os.environ["NCCL_TIMEOUT"] = "1800"
    main()  # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------
