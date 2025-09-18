# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Generate random images using the techniques described in the paper
"Elucidating the Design Space of Diffusion-Based Generative Models"."""

import os
import re
import click
import tqdm
import pickle
import torch
import PIL.Image
import dnnlib
from torch_utils import distributed as dist

#----------------------------------------------------------------------------
# Wrapper for torch.Generator that allows specifying a different random seed
# for each sample in a minibatch.

class StackedRandomGenerator:
    def __init__(self, device, seeds):
        super().__init__()
        self.generators = [torch.Generator(device).manual_seed(int(seed) % (1 << 32)) for seed in seeds]

    def randn(self, size, **kwargs):
        assert size[0] == len(self.generators)
        return torch.stack([torch.randn(size[1:], generator=gen, **kwargs) for gen in self.generators])

    def randn_like(self, input):
        return self.randn(input.shape, dtype=input.dtype, layout=input.layout, device=input.device)

    def randint(self, *args, size, **kwargs):
        assert size[0] == len(self.generators)
        return torch.stack([torch.randint(*args, size=size[1:], generator=gen, **kwargs) for gen in self.generators])

#----------------------------------------------------------------------------
# Parse a comma separated list of numbers or ranges and return a list of ints.
# Example: '1,2,5-10' returns [1, 2, 5, 6, 7, 8, 9, 10]

def parse_int_list(s):
    if isinstance(s, list): return s
    ranges = []
    range_re = re.compile(r'^(\d+)-(\d+)$')
    for p in s.split(','):
        m = range_re.match(p)
        if m:
            ranges.extend(range(int(m.group(1)), int(m.group(2))+1))
        else:
            ranges.append(int(p))
    return ranges

#----------------------------------------------------------------------------

@click.command()
@click.option('--network', 'network_pkl',  help='Network pickle filename', metavar='PATH|URL',                      type=str, required=True)
@click.option('--outdir',                  help='Where to save the output images', metavar='DIR',                   type=str, required=True)
@click.option('--seeds',                   help='Random seeds (e.g. 1,2,5-10)', metavar='LIST',                     type=parse_int_list, default='0-63', show_default=True)
@click.option('--subdirs',                 help='Create subdirectory for every 1000 seeds',                         is_flag=True)
@click.option('--class', 'class_idx',      help='Class label  [default: random]', metavar='INT',                    type=click.IntRange(min=0), default=None)
@click.option('--batch', 'max_batch_size', help='Maximum batch size', metavar='INT',                                type=click.IntRange(min=1), default=64, show_default=True)

def main(network_pkl, outdir, seeds, class_idx, subdirs=True, max_batch_size=64, device=torch.device('cuda')):
    """Generate random images using

    torchrun --standalone --nproc_per_node=1 train_ito.py --outdir /workspace/mac_trained --data /workspace/dataset/cifar10-32x32.zip --batch 1 --batch_edm 32 --batch_disc 32 --kimg 2000 --net_pkl /workspace/pretrained_model/edm-cifar10-32x32-cond-vp.pkl --cond True --coefficient_conditioning False --train_v True --train_c True --x_0_conditioning True --basis_num 5 --multidimensionality 111 --gauss_kernel_sigma 4.0 --num_steps 5 --rho 7 --s_churn 0 --s_min 0.0 --s_max 0.0 --s_noise 1.0 --glr 1e-4 --k 0 --scale 0.05 --lpf True


    Examples:

    # Generate 1024 images using 2 GPUs
    torchrun --standalone --nproc_per_node=1 generate.py --outdir=/workspace/mac_generated_samples/00001 --seeds=0-49999 \
    --network=/workspace/mac_trained/00036-cifar10-32x32-ito-gpus1-batch1/network-snapshot-000000.pkl

    torchrun --standalone --nproc_per_node=1 fid.py calc --images=/ssd3/leedohoon/mpo_generated_samples/00000-cifar10-32x32-mpo-gpus2-batch16/network-snapshot-000020 --ref=/ssd1/cifar10/fid-refs/cifar10-32x32.npz
    """
    dist.init()
    num_batches = ((len(seeds) - 1) // (max_batch_size) + 1)
    all_batches = torch.as_tensor(seeds).tensor_split(num_batches)
    rank_batches = all_batches

    # Rank 0 goes first.
    if dist.get_rank() != 0:
        torch.distributed.barrier()

    # Load network.
    print(f'Loading network from "{network_pkl}"...')
    with dnnlib.util.open_url(network_pkl) as f:
        net = pickle.load(f)['G_ema'].to(device)

    # Other ranks follow.
    if dist.get_rank() == 0:
        torch.distributed.barrier()

    # Loop over batches.
    print(f'Generating {len(seeds)} images to "{outdir}"...')
    for batch_seeds in tqdm.tqdm(rank_batches, unit='batch', disable=(dist.get_rank() != 0)):
        # torch.distributed.barrier()
        batch_size = len(batch_seeds)
        if batch_size == 0:
            continue

        # Pick latents and labels.
        rnd = StackedRandomGenerator(device, batch_seeds)
        latents = rnd.randn([batch_size, 3, net.V.img_resolution, net.V.img_resolution], device=device)
        class_labels = None
        if net.V.label_dim:
            class_labels = torch.eye(net.V.label_dim, device=device)[rnd.randint(net.V.label_dim, size=[batch_size], device=device)]
        if class_idx is not None:
            class_labels[:, :] = 0
            class_labels[:, class_idx] = 1

        # Generate images.
        images, _, _ = net(z=latents, c=class_labels, simulation=True)

        # Save images.
        images_np = (images * 127.5 + 128).clip(0, 255).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
        for seed, image_np in zip(batch_seeds, images_np):
            image_dir = os.path.join(outdir, f'{seed-seed%1000:06d}') if subdirs else outdir
            os.makedirs(image_dir, exist_ok=True)
            image_path = os.path.join(image_dir, f'{seed:06d}.png')
            if image_np.shape[2] == 1:
                PIL.Image.fromarray(image_np[:, :, 0], 'L').save(image_path)
            else:
                PIL.Image.fromarray(image_np, 'RGB').save(image_path)

    # Done.
    torch.distributed.barrier()
    dist.print0('Done.')

    # Disallocate the net from GPU and free memory
    del net
    torch.cuda.empty_cache()

#----------------------------------------------------------------------------

if __name__ == "__main__":
    main()