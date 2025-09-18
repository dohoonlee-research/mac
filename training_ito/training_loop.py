# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Main training loop."""

import os
import time
import copy
import psutil
import PIL.Image
import numpy as np
import torch
import dnnlib
import pickle
import random
from torch_utils import misc_stylegan as misc
from torch_utils import training_stats
from torch_utils.ops import conv2d_gradfix
from torch_utils.ops import grid_sample_gradfix
from torch_utils import distributed as dist

#----------------------------------------------------------------------------

def setup_snapshot_image_grid(dataset_obj, random_seed=0, gw=None, gh=None, max_images=36):
    rnd = np.random.RandomState(random_seed)
    
    # Calculate grid size with a maximum limit
    if gw is None or gh is None:
        # Estimate the grid dimensions if not provided
        if gw is None and gh is None:
            gw = int(np.sqrt(max_images))  # Grid width
            gh = int(np.ceil(max_images / gw))  # Grid height
        elif gw is None:
            gw = int(np.ceil(max_images / gh))
        elif gh is None:
            gh = int(np.ceil(max_images / gw))
    
    gw = min(gw, max_images)
    gh = min(gh, max_images // gw)

    # No labels => show random subset of training samples.
    if not dataset_obj.has_labels:
        all_indices = list(range(len(dataset_obj)))
        rnd.shuffle(all_indices)
        grid_indices = [all_indices[i % len(all_indices)] for i in range(gw * gh)]

    else:
        # Group training samples by label.
        label_groups = dict()  # label => [idx, ...]
        for idx in range(len(dataset_obj)):
            label = tuple(dataset_obj.get_details(idx).raw_label.flat[::-1])
            if label not in label_groups:
                label_groups[label] = []
            label_groups[label].append(idx)

        # Reorder.
        label_order = sorted(label_groups.keys())
        for label in label_order:
            rnd.shuffle(label_groups[label])

        # Organize into grid.
        grid_indices = []
        for y in range(gh):
            label = label_order[y % len(label_order)]
            indices = label_groups[label]
            grid_indices += [indices[x % len(indices)] for x in range(gw)]
            label_groups[label] = [indices[(i + gw) % len(indices)] for i in range(len(indices))]

    # Load data.
    images, labels = zip(*[dataset_obj[i] for i in grid_indices])
    return (gw, gh), np.stack(images), np.stack(labels)

#----------------------------------------------------------------------------

def save_image_grid(img, fname, drange, grid_size):
    lo, hi = drange
    img = np.asarray(img, dtype=np.float32)
    img = (img - lo) * (255 / (hi - lo))
    img = np.rint(img).clip(0, 255).astype(np.uint8)

    gw, gh = grid_size
    _N, C, H, W = img.shape
    img = img.reshape([gh, gw, C, H, W])
    img = img.transpose(0, 3, 1, 4, 2)
    img = img.reshape([gh * H, gw * W, C])

    assert C in [1, 3]
    if C == 1:
        PIL.Image.fromarray(img[:, :, 0], 'L').save(fname)
    if C == 3:
        PIL.Image.fromarray(img, 'RGB').save(fname)

#----------------------------------------------------------------------------

def training_loop(
    run_dir                 = '.',      # Output directory.
    dataset_kwargs          = {},       # Options for training set.
    data_loader_kwargs      = {},       # Options for torch.utils.data.DataLoader.
    V_pkl                   = '',       # Options for pretrained edm network
    C_kwargs                = {},       # Options for coefficient generator network.
    MC_kwargs               = {},       # Options for mac calculator.
    G_kwargs                = {},       # Options for generator
    D_kwargs                = {},       # Options for discriminator network.
    V_opt_kwargs            = {},       # Options for edm optimizer.
    C_opt_kwargs            = {},       # Options for generator optimizer.
    D_opt_kwargs            = {},       # Options for discriminator optimizer.
    loss_kwargs             = {},       # Options for loss function.
    seed                    = 0,        # Global random seed.
    batch_size              = 32,       # Total batch size for one training iteration. Can be larger than batch_gpu * num_gpus.
    batch_size_edm          = 256,
    batch_size_disc         = 256,
    ema_kimg                = 10,       # Half-life of the exponential moving average (EMA) of generator weights.
    ema_rampup              = 0.05,     # EMA ramp-up coefficient. None = no rampup.
    total_kimg              = 25000,    # Total length of the training, measured in thousands of real images.
    kimg_per_tick           = 4,        # Progress snapshot interval.
    image_snapshot_ticks    = 5,        # How often to save image snapshots? None = disable.
    network_snapshot_ticks  = 50,       # How often to save network snapshots? None = disable.
    cudnn_benchmark         = False,     # Enable torch.backends.cudnn.benchmark?
    device                  = torch.device('cuda'),
    train_v                 = True,
    train_c                 = True
):
    
    # Initialize.
    # seed_0 = (seed * dist.get_world_size() + dist.get_rank()) % (1 << 31)
    # seed_1 = np.random.randint(1 << 31)
    seed_0, seed_1 = 0, 0
    dist.print0('seed_0: ', seed_0)
    dist.print0('seed_1: ', seed_1)
    start_time = time.time()
    random.seed(seed_0)
    np.random.seed(seed_0)
    torch.manual_seed(seed_1)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = cudnn_benchmark    # Improves training speed.
    torch.backends.cudnn.allow_tf32 = False             # Improves numerical accuracy.
    torch.backends.cuda.matmul.allow_tf32 = False       # Improves numerical accuracy.
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    conv2d_gradfix.enabled = True                       # Improves training speed.
    grid_sample_gradfix.enabled = True                  # Avoids errors with the augmentation pipe.

    # Specify the path to fid file
    if dist.get_rank() == 0:
        fid_txt_file = os.path.join(run_dir, 'fid_values.txt')

    # Select batch size per GPU
    batch_gpu = batch_size // dist.get_world_size()
    assert batch_size == batch_gpu * dist.get_world_size()
    batch_edm_gpu = batch_size_edm // dist.get_world_size()
    batch_disc_gpu = batch_size_disc // dist.get_world_size()

    # Load training set.
    dist.print0('Loading dataset...')
    dataset_obj = dnnlib.util.construct_class_by_name(**dataset_kwargs) # subclass of training.dataset.Dataset
    dataset_sampler = misc.InfiniteSampler(dataset=dataset_obj, rank=dist.get_rank(), num_replicas=dist.get_world_size(), seed=seed)
    dataset_iterator = iter(torch.utils.data.DataLoader(dataset=dataset_obj, sampler=dataset_sampler, batch_size=batch_gpu, **data_loader_kwargs))
    dataset_iterator_edm = iter(torch.utils.data.DataLoader(dataset=dataset_obj, sampler=dataset_sampler, batch_size=batch_edm_gpu, **data_loader_kwargs))
    dataset_iterator_disc = iter(torch.utils.data.DataLoader(dataset=dataset_obj, sampler=dataset_sampler, batch_size=batch_disc_gpu, **data_loader_kwargs))

    # Construct networks.
    dist.print0('Constructing networks...')
    common_kwargs = dict(c_dim=dataset_obj.label_dim, img_resolution=dataset_obj.resolution, img_channels=dataset_obj.num_channels)
    # Load and distribute pretrained network weights
    with dnnlib.util.open_url(V_pkl) as f:
        V = pickle.load(f)['ema']
    V = V.eval().requires_grad_(True).to(device)  # Set the network to evaluation mode initially
    C = dnnlib.util.construct_class_by_name(**C_kwargs, label_dim=dataset_obj.label_dim).to(device).train()
    MC = dnnlib.util.construct_class_by_name(**MC_kwargs)
    G = dnnlib.util.construct_class_by_name(V=V, C=C, MC=MC, **G_kwargs).requires_grad_(True).to(device) # subclass of torch.nn.Module
    D = dnnlib.util.construct_class_by_name(**D_kwargs, **common_kwargs).train().requires_grad_(True).to(device) # subclass of torch.nn.Module
    G_ema = copy.deepcopy(G).eval().requires_grad_(False)

    # Print network summary tables.
    if dist.get_rank() == 0:
        with torch.no_grad():
            z = torch.empty([batch_gpu, 3, dataset_obj.resolution, dataset_obj.resolution], device=device)
            c = torch.empty([batch_gpu, dataset_obj.label_dim], device=device)
            img, _, _ = misc.print_module_summary(G, [z, c])
            misc.print_module_summary(D, [img, c])

    # Setup optimizer
    V_opt = dnnlib.util.construct_class_by_name(params=V.parameters(), **V_opt_kwargs)
    C_opt = dnnlib.util.construct_class_by_name(params=C.UNet.parameters(), **C_opt_kwargs) # Optimize only for coefficient generator
    D_opt = dnnlib.util.construct_class_by_name(params=D.parameters(), **D_opt_kwargs) # subclass of torch.optim.Optimizer

    # Distribute across GPUs using DDP
    G_ddp = torch.nn.parallel.DistributedDataParallel(G, device_ids=[device], find_unused_parameters=True)
    for param in misc.params_and_buffers(D):
        torch.distributed.broadcast(param, src=0)

    # Loss functions
    loss_fn_gan = dnnlib.util.construct_class_by_name(G=G_ddp, D=D, **loss_kwargs) # subclass of training.loss.Loss

    # Export sample images.
    grid_size = None
    grid_z = None
    grid_c = None
    if dist.get_rank() == 0:
        with torch.no_grad():
            print('Exporting sample images...')
            grid_size, images, labels = setup_snapshot_image_grid(dataset_obj=dataset_obj)
            save_image_grid(images, os.path.join(run_dir, 'image_snapshot/reals.png'), drange=[0,255], grid_size=grid_size)

            grid_z = torch.randn([labels.shape[0], 3, dataset_obj.resolution, dataset_obj.resolution], device=device).split(batch_gpu)
            grid_c = torch.from_numpy(labels).to(device).split(batch_gpu)
            images = torch.cat([G(z=z, c=c, simulation=True)[0].cpu() for z, c in zip(grid_z, grid_c)]).numpy()
            
            save_image_grid(images, os.path.join(run_dir, 'image_snapshot/fakes_init.png'), drange=[-1,1], grid_size=grid_size)

            images = torch.cat([G.generate_with_unidimensional_coefficient(z=z, c=c).cpu() for z, c in zip(grid_z, grid_c)]).numpy()
            save_image_grid(images, os.path.join(run_dir, 'image_snapshot/fakes_init_scalar.png'), drange=[-1,1], grid_size=grid_size)

            # Coefficients
            coefficients = G.C(grid_z[0][0].unsqueeze(0), grid_c[0][0].unsqueeze(0))
            coefficients = coefficients.squeeze(0).cpu()  # Remove batch dimension and move to CPU

            # Create directory for saving the images
            output_dir = os.path.join(run_dir, 'image_snapshot', 'coefficients', 'init')
            os.makedirs(output_dir, exist_ok=True)

            # Iterate over channels and save each one as a separate .png file
            for i in range(coefficients.shape[0]):  # Iterate over channels
                channel = coefficients[i].numpy()  # Convert to numpy array (shape: [H, W])
                
                # Normalize and scale the image to the range [0, 255] based on C_kwargs['scale']
                img = np.asarray(channel, dtype=np.float32)
                if C_kwargs['scale'] != 0:
                    img = (img + C_kwargs['scale']) * (255 / (2 * C_kwargs['scale']))  # Normalize [-scale, scale] -> [0, 255]
                img = np.rint(img).clip(0, 255).astype(np.uint8)  # Round, clip, and convert to uint8
                
                # Create PIL image from the processed array
                img_pil = PIL.Image.fromarray(img, mode='L')  # 'L' mode is for grayscale images
                
                # Save the image with the format "i.png" where i is the channel index
                img_pil.save(os.path.join(output_dir, f'{i}.png'))
            
            # Mac
            _, gamma_1 = MC.get_mac(G.t_steps.to(device), coefficients.unsqueeze(0).repeat(G.t_steps.shape[0], 1, 1, 1).to(device))
            
            # Create directory for saving the images
            output_dir = os.path.join(run_dir, 'image_snapshot', 'paths', 'init')
            os.makedirs(output_dir, exist_ok=True)

            # Iterate over each batch
            for b in range(gamma_1.shape[0]):
                # Convert the tensor to numpy array and move to CPU
                img = gamma_1[b].cpu().numpy()  # Shape [C, H, W]
                
                # Normalize and scale to [0, 255] (assuming C_kwargs['scale'] is defined)
                img = np.asarray(img, dtype=np.float32)
                img = img * (255 / 80)  # Normalize [-scale, scale] -> [0, 255]
                img = np.rint(img).clip(0, 255).astype(np.uint8)  # Round, clip, and convert to uint8
                
                # Transpose the image to [H, W, C] to be compatible with PIL (channels last)
                img = np.transpose(img, (1, 2, 0))  # Convert [C, H, W] -> [H, W, C]
                
                # Convert to PIL image
                img_pil = PIL.Image.fromarray(img)  # No need to specify 'L' mode, as this is RGB
                
                # Save the image as "batch_{b}.png"
                img_pil.save(os.path.join(output_dir, f'{b}.png'))

    # Train.
    dist.print0(f'Training for {total_kimg} kimg...')
    cur_nimg = 0
    cur_tick = 0
    tick_start_nimg = cur_nimg
    tick_start_time = time.time()
    maintenance_time = tick_start_time - start_time
    dist.update_progress(cur_nimg // 1000, total_kimg)
    torch.distributed.barrier()

    while True:
        # ------------------------------------------------------------------
        if train_v:
            # 1. Generator loss (for coefficient generator)
                # for V
            G.zero_grad()
            D.zero_grad()
            with misc.ddp_sync(G_ddp, True):
                real_img, real_c = next(dataset_iterator_edm)
                real_img = real_img.to(device).to(torch.float32) / 127.5 - 1
                real_c = real_c.to(device)
                gen_z = torch.randn_like(real_img)

                loss_G_V = loss_fn_gan.generator_loss_V(z=gen_z, c=real_c, img=real_img)
                loss = loss_G_V
                loss.backward()

                # Update weights.
            params = [param for param in G.parameters() if param.grad is not None]
            if len(params) > 0:
                flat = torch.cat([param.grad.flatten() for param in params])
                if dist.get_world_size() > 1:
                    torch.distributed.all_reduce(flat)
                    flat /= dist.get_world_size()
                misc.nan_to_num(flat, nan=0, posinf=1e5, neginf=-1e5, out=flat)
                grads = flat.split([param.numel() for param in params])
                for param, grad in zip(params, grads):
                    param.grad = grad.reshape(param.shape)
            V_opt.step()
        else:
            loss_G_V = torch.zeros(1)
        
        if train_c:
                # for C
            G.zero_grad()
            D.zero_grad()
            with misc.ddp_sync(G_ddp, True):
                real_img, real_c = next(dataset_iterator)
                real_img = real_img.to(device).to(torch.float32) / 127.5 - 1
                real_c = real_c.to(device)
                gen_z = torch.randn_like(real_img)
                gen_c = real_c

                loss_G_C = loss_fn_gan.generator_loss_C(gen_z, gen_c)
                loss = loss_G_C
                loss.backward()

                # Update weights.
            params = [param for param in G.parameters() if param.grad is not None]
            if len(params) > 0:
                flat = torch.cat([param.grad.flatten() for param in params])
                if dist.get_world_size() > 1:
                    torch.distributed.all_reduce(flat)
                    flat /= dist.get_world_size()
                misc.nan_to_num(flat, nan=0, posinf=1e5, neginf=-1e5, out=flat)
                grads = flat.split([param.numel() for param in params])
                for param, grad in zip(params, grads):
                    param.grad = grad.reshape(param.shape)
            C_opt.step()
        else:
            loss_G_C = torch.zeros(1)

        # ------------------------------------------------------------------
        
        # 2. Discriminator Loss
        G.zero_grad()
        D.zero_grad()
        D.feature_networks.requires_grad_(False)
        with misc.ddp_sync(G_ddp, True):
            real_img, real_c = next(dataset_iterator_disc)
            _, gen_c = next(dataset_iterator_disc)

            real_img = real_img.to(device).to(torch.float32) / 127.5 - 1
            real_c = real_c.to(device)
            gen_z = torch.randn_like(real_img).to(device)
            gen_c = gen_c.to(device)

            loss_Dgen, loss_Dreal, coefficient = loss_fn_gan.discriminator_loss(real_img, real_c, gen_z, gen_c)
            loss = loss_Dgen + loss_Dreal
            loss.backward()
        
        # Update weights.
        params = [param for param in D.parameters() if param.grad is not None]
        if len(params) > 0:
            flat = torch.cat([param.grad.flatten() for param in params])
            if dist.get_world_size() > 1:
                torch.distributed.all_reduce(flat)
                flat /= dist.get_world_size()
            misc.nan_to_num(flat, nan=0, posinf=1e5, neginf=-1e5, out=flat)
            grads = flat.split([param.numel() for param in params])
            for param, grad in zip(params, grads):
                param.grad = grad.reshape(param.shape)
        D_opt.step()
        D.feature_networks.requires_grad_(True)

        # ------------------------------------------------------------------

        # Update EMA
        with torch.no_grad():
            ema_nimg = ema_kimg * 1000
            if ema_rampup is not None:
                ema_nimg = min(ema_nimg, cur_nimg * ema_rampup)
            ema_beta = 0.5 ** (batch_size / max(ema_nimg, 1e-8))
            for p_ema, p in zip(G_ema.parameters(), G.parameters()):
                p_ema.copy_(p.lerp(p_ema, ema_beta))
            for b_ema, b in zip(G_ema.buffers(), G.buffers()):
                b_ema.copy_(b)

        # ------------------------------------------------------------------

        if dist.get_rank() == 0:
            # Logging for generator and discriminator phase
            total_norm_c = 0
            for param in C.parameters():
                total_norm_c += torch.norm(param)
            
            total_norm_c_ema = 0
            for param in G_ema.C.parameters():
                total_norm_c_ema += torch.norm(param)
            
            total_norm_d = 0
            for param in D.parameters():
                total_norm_d += torch.norm(param)

            total_norm_v = 0
            for param in V.parameters():
                total_norm_v += torch.norm(param)
            
            total_norm_v_ema = 0
            for param in G_ema.V.parameters():
                total_norm_v_ema += torch.norm(param)

        # ------------------------------------------------------------------

        # Update state.
        cur_nimg += batch_size

        # Perform maintenance tasks once per tick.
        done = (cur_nimg >= total_kimg * 1000)
        if (not done) and (cur_tick != 0) and (cur_nimg < tick_start_nimg + kimg_per_tick * 1000):
            continue

        # Print status line, accumulating the same information in training_stats.
        tick_end_time = time.time()
        fields = []
        fields += [f"tick {training_stats.report0('Progress/tick', cur_tick):<5d}"]
        fields += [f"kimg {training_stats.report0('Progress/kimg', cur_nimg / 1e3):<8.1f}"]
        fields += [f"time {dnnlib.util.format_time(training_stats.report0('Timing/total_sec', tick_end_time - start_time)):<12s}"]
        fields += [f"sec/tick {training_stats.report0('Timing/sec_per_tick', tick_end_time - tick_start_time):<7.1f}"]
        fields += [f"sec/kimg {training_stats.report0('Timing/sec_per_kimg', (tick_end_time - tick_start_time) / (cur_nimg - tick_start_nimg) * 1e3):<7.2f}"]
        fields += [f"maintenance {training_stats.report0('Timing/maintenance_sec', maintenance_time):<6.1f}"]
        fields += [f"cpumem {training_stats.report0('Resources/cpu_mem_gb', psutil.Process(os.getpid()).memory_info().rss / 2**30):<6.2f}"]
        fields += [f"gpumem {training_stats.report0('Resources/peak_gpu_mem_gb', torch.cuda.max_memory_allocated(device) / 2**30):<6.2f}"]
        fields += [f"reserved {training_stats.report0('Resources/peak_gpu_mem_reserved_gb', torch.cuda.max_memory_reserved(device) / 2**30):<6.2f}"]
        torch.cuda.reset_peak_memory_stats()
        training_stats.report0('Timing/total_hours', (tick_end_time - start_time) / (60 * 60))
        training_stats.report0('Timing/total_days', (tick_end_time - start_time) / (24 * 60 * 60))
        if dist.get_rank() == 0:
            print(' '.join(fields))

        # Save image snapshot.
        if (dist.get_rank() == 0) and (image_snapshot_ticks is not None) and (done or cur_tick % image_snapshot_ticks == 0):
            with torch.no_grad():
                # Non-ema
                images = torch.cat([G(z=z, c=c, simulation=True)[0].cpu() for z, c in zip(grid_z, grid_c)]).numpy()
                save_image_grid(images, os.path.join(run_dir, f'image_snapshot/fakes{cur_nimg//1000:06d}.png'), drange=[-1,1], grid_size=grid_size)

                # Ema
                images = torch.cat([G_ema(z=z, c=c, simulation=True)[0].cpu() for z, c in zip(grid_z, grid_c)]).numpy()
                save_image_grid(images, os.path.join(run_dir, f'image_snapshot/fakes{cur_nimg//1000:06d}_ema.png'), drange=[-1,1], grid_size=grid_size)

                # Coefficients
                coefficients = G_ema.C(grid_z[0][0].unsqueeze(0), grid_c[0][0].unsqueeze(0))
                coefficients = coefficients.squeeze(0).cpu()  # Remove batch dimension and move to CPU

                # Create directory for saving the images
                output_dir = os.path.join(run_dir, 'image_snapshot', 'coefficients', f'{cur_nimg // 1000:06d}')
                os.makedirs(output_dir, exist_ok=True)

                # Iterate over channels and save each one as a separate .png file
                for i in range(coefficients.shape[0]):  # Iterate over channels
                    channel = coefficients[i].numpy()  # Convert to numpy array (shape: [H, W])
                    
                    # Normalize and scale the image to the range [0, 255] based on C_kwargs['scale']
                    img = np.asarray(channel, dtype=np.float32)
                    if C_kwargs['scale'] != 0:
                        img = (img + C_kwargs['scale']) * (255 / (2 * C_kwargs['scale']))  # Normalize [-scale, scale] -> [0, 255]            
                    img = np.rint(img).clip(0, 255).astype(np.uint8)  # Round, clip, and convert to uint8
                    
                    # Create PIL image from the processed array
                    img_pil = PIL.Image.fromarray(img, mode='L')  # 'L' mode is for grayscale images
                    
                    # Save the image with the format "i.png" where i is the channel index
                    img_pil.save(os.path.join(output_dir, f'{i}.png'))

                # Mac
                _, gamma_1 = MC.get_mac(G.t_steps.to(device), coefficients.unsqueeze(0).repeat(G.t_steps.shape[0], 1, 1, 1).to(device))

                # Create directory for saving the images
                output_dir = os.path.join(run_dir, 'image_snapshot', 'paths', f'{cur_nimg // 1000:06d}')
                os.makedirs(output_dir, exist_ok=True)

                # Iterate over each batch
                for b in range(gamma_1.shape[0]):
                    # Convert the tensor to numpy array and move to CPU
                    img = gamma_1[b].cpu().numpy()  # Shape [C, H, W]
                    
                    # Normalize and scale to [0, 255] (assuming C_kwargs['scale'] is defined)
                    img = np.asarray(img, dtype=np.float32)
                    img = img * (255 / 80)  # Normalize [-scale, scale] -> [0, 255]
                    img = np.rint(img).clip(0, 255).astype(np.uint8)  # Round, clip, and convert to uint8
                    
                    # Transpose the image to [H, W, C] to be compatible with PIL (channels last)
                    img = np.transpose(img, (1, 2, 0))  # Convert [C, H, W] -> [H, W, C]
                    
                    # Convert to PIL image
                    img_pil = PIL.Image.fromarray(img)  # No need to specify 'L' mode, as this is RGB
                    
                    # Save the image as "batch_{b}.png"
                    img_pil.save(os.path.join(output_dir, f'{b}.png'))

        # Save network snapshot.
        if cur_tick and (network_snapshot_ticks is not None) and (done or cur_tick % network_snapshot_ticks == 0):
            with torch.no_grad():
                data = dict(G_ema=G_ema)
                for key, value in data.items():
                    if isinstance(value, torch.nn.Module):
                        value = copy.deepcopy(value).eval().requires_grad_(False)
                        misc.check_ddp_consistency(value)
                        data[key] = value.cpu()
                    del value # conserve memory
                if dist.get_rank() == 0:
                    with open(os.path.join(run_dir, f'network-snapshot-{cur_nimg//1000:06d}.pkl'), 'wb') as f:
                        pickle.dump(data, f)
                del data # conserve memory

        dist.update_progress(cur_nimg // 1000, total_kimg)

        # Update state.
        cur_tick += 1
        tick_start_nimg = cur_nimg
        tick_start_time = time.time()
        maintenance_time = tick_start_time - tick_end_time
        if done:
            break

    # Done.
    if dist.get_rank() == 0:
        print()
        print('Exiting...')

#----------------------------------------------------------------------------