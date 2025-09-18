import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from training.networks import SongUNetModified

#----------------------------------------------------------------------------
# Coefficient generator

class RandomCoefficientSampler(torch.nn.Module):
    def __init__(self, 
                 resolution, # whether 32X32 or 64X64
                 M,
                 scale,
                 multidimensionality, # control multidimensionality by dim
                 lpf, # low-pass filtering
                 gauss_kernel_sigma=None) : # low-pass filter parameters
        super().__init__()
        # initialize
        self.M = M
        self.scale = scale
        multidimensionality = [item == '1' for item in multidimensionality]
        self.lpf = lpf
        self.gauss_kernel_size = 20 * int(resolution / 32) - 1
        if lpf:
            gauss_kernel_sigma = gauss_kernel_sigma * int(resolution / 32)

        # calculate initial height and width
        if lpf:
            self.max_dim = [3, resolution + self.gauss_kernel_size + 1, resolution + self.gauss_kernel_size + 1]
        else:
            self.max_dim = [3, resolution, resolution]
        self.initial_resolution = [self.max_dim[i] if multidimensionality[i] else 1 for i in range(len(multidimensionality))]
        
        # Create a Gaussian kernel
        if lpf:
            gauss_kernel = torch.Tensor(cv2.getGaussianKernel(self.gauss_kernel_size, gauss_kernel_sigma))
            gauss_kernel = gauss_kernel * gauss_kernel.transpose(0, 1)
            gauss_kernel = gauss_kernel.expand(1, 6*self.M, gauss_kernel.shape[0], gauss_kernel.shape[1])
            self.gauss_kernel = gauss_kernel / torch.sum(gauss_kernel)
            self.padding = int((self.gauss_kernel_size - 1)/2)
            self.crop_value = int((self.gauss_kernel_size + 1)/2)
        
    def dimension_expansion(self, c):
        if self.initial_resolution[0] == 1:
            c_f, c_g = c[:, :self.M, :, :], c[:, self.M:, :, :]
            c_f, c_g = c_f.repeat(1, self.max_dim[0], 1, 1), c_f.repeat(1, self.max_dim[0], 1, 1)
            c = torch.cat((c_f, c_g), dim=1)
        if self.initial_resolution[1] == 1:
            c = c.repeat(1, 1, self.max_dim[1], 1)
        if self.initial_resolution[2] == 1:
            c = c.repeat(1, 1, 1, self.max_dim[2])
        return c

    def sample(self, img):
        device = img.device

        # Randomly sample
        c = torch.rand(img.shape[0], 2*self.initial_resolution[0]*self.M, self.initial_resolution[1], self.initial_resolution[2], device=device)
        c = 2 * c - 1  # Scale to [c_min, c_max] \in [-1, 1]

        # Dimension expansion
        c = self.dimension_expansion(c)

        # Low-pass filtering
        if self.lpf:
            # Save original max and min for each batch item
            c_max_orig = torch.amax(c, dim=(1,2,3)).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            c_min_orig = torch.amin(c, dim=(1,2,3)).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

            # Perform low-pass filtering
            c = F.conv2d(c, self.gauss_kernel.to(device), padding=self.padding, groups=1)

            # Get min & max values
            c_max = torch.amax(c, dim=(1,2,3)).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            c_min = torch.amin(c, dim=(1,2,3)).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

            # Normalize each batch item to [0, 1]
            c = (c - c_min) / (c_max - c_min + 1e-8)

            # Rescale to original [c_min, c_max] for each batch item
            c = c * (c_max_orig - c_min_orig) + c_min_orig

            # Edge cropping
            c = c[:, :, self.crop_value:-self.crop_value, self.crop_value:-self.crop_value]
        
        # Scale to [-s, s]
        c = c * self.scale

        return c

#----------------------------------------------------------------------------
# Coefficient generator

class CoefficientGenerator(nn.Module):
    def __init__(self, resolution, multidimensionality, scale, M, lpf, gauss_kernel_sigma, label_dim, x_0_conditioning):
        super(CoefficientGenerator, self).__init__()
        # initialize
        self.resolution = resolution
        self.multidimensionality = [item == '1' for item in multidimensionality]
        self.scale = scale
        self.M = M
        self.lpf = lpf
        gauss_kernel_size = 20 * int(resolution / 32) - 1
        gauss_kernel_sigma = gauss_kernel_sigma * int(resolution / 32)
        if self.lpf:
            self.pad = (
                (gauss_kernel_size + 1)//2, (gauss_kernel_size + 1)//2, 
                (gauss_kernel_size + 1)//2, (gauss_kernel_size + 1)//2)
        self.label_dim = label_dim
        self.x_0_conditioning = x_0_conditioning
        
        # UNet
        self.UNet = SongUNetModified(
            img_resolution=resolution,
            in_channels=3, 
            out_channels=M * 6,
            label_dim=label_dim,
            num_blocks=4
            )

        # Create a Gaussian kernel
        gauss_kernel = torch.Tensor(cv2.getGaussianKernel(gauss_kernel_size, gauss_kernel_sigma))
        gauss_kernel = gauss_kernel * gauss_kernel.transpose(0, 1)
        gauss_kernel = gauss_kernel.expand(M * 6, 1, gauss_kernel.shape[0], gauss_kernel.shape[1])
        self.gauss_kernel = gauss_kernel / torch.sum(gauss_kernel)
        self.padding = int((gauss_kernel_size - 1)/2)
        self.crop_value = int((gauss_kernel_size + 1)/2)

    def dimension_reduction_by_mean(self, coefficient):
        coefficient_f, coefficient_g = coefficient[:, :3*self.M, :, :], coefficient[:, 3*self.M:, :, :]

        if self.multidimensionality[0] == False:
            # Taking the mean value through channel
            coefficient_f = coefficient_f.view(coefficient_f.size(0), 3, self.M, coefficient_f.size(2), coefficient_f.size(3))
            coefficient_f = torch.mean(coefficient_f, dim=1)
            coefficient_g = coefficient_g.view(coefficient_g.size(0), 3, self.M, coefficient_g.size(2), coefficient_g.size(3))
            coefficient_g = torch.mean(coefficient_g, dim=1)

            # Dimension repeat
            coefficient_f = coefficient_f.repeat(1, 3, 1, 1)
            coefficient_g = coefficient_g.repeat(1, 3, 1, 1)
            coefficient = torch.cat((coefficient_f, coefficient_g), dim=1)

        if self.multidimensionality[1] == False:
            height = coefficient_f.size(2)

            # Taking the mean value through height
            coefficient_f, coefficient_g = coefficient_f.mean(dim=2, keepdim=True), coefficient_g.mean(dim=2, keepdim=True)

            # Dimension expansion
            coefficient_f = coefficient_f.expand(-1, -1, height, -1)
            coefficient_g = coefficient_g.expand(-1, -1, height, -1)
            coefficient = torch.cat((coefficient_f, coefficient_g), dim=1)

        if self.multidimensionality[2] == False:
            width = coefficient_f.size(2)

            # Taking the mean value through width
            coefficient_f, coefficient_g = coefficient_f.mean(dim=3, keepdim=True), coefficient_g.mean(dim=3, keepdim=True)

            # Dimension expansion
            coefficient_f = coefficient_f.expand(-1, -1, -1, width)
            coefficient_g = coefficient_g.expand(-1, -1, -1, width)
            coefficient = torch.cat((coefficient_f, coefficient_g), dim=1)
            
        return coefficient
    
    def forward(self, x_0, class_labels):
        # Pad x_0
        if self.lpf:
            x_0 = F.pad(x_0, self.pad, "constant", 0)
        
        # Generate coefficient by UNet
        if self.x_0_conditioning:
            input_unet = x_0
        else:
            input_unet = torch.ones_like(x_0)

        coefficient = self.UNet(input_unet, class_labels)

        # Bound the range to -1 ~ 1
        coefficient = torch.tanh(coefficient)

        # dimension reduction
        if False in self.multidimensionality:
            coefficient = self.dimension_reduction_by_mean(coefficient)
        
        # Low-pass filtering
        if self.lpf:
            # Save original max and min for each batch item
            c_max_orig = torch.amax(coefficient, dim=(1,2,3)).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            c_min_orig = torch.amin(coefficient, dim=(1,2,3)).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

            # Perform low-pass filtering
            coefficient = F.conv2d(coefficient, self.gauss_kernel.to(x_0.device), padding=self.padding, groups=self.M * 6)
            
            # Get min & max values
            c_max = torch.amax(coefficient, dim=(1,2,3)).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            c_min = torch.amin(coefficient, dim=(1,2,3)).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

            # Normalize each batch item to [0, 1]
            coefficient = (coefficient - c_min) / (c_max - c_min + 1e-8)

            # Rescale to original [c_min, c_max] for each batch item
            coefficient = coefficient * (c_max_orig - c_min_orig) + c_min_orig

            # Edge cropping
            coefficient = coefficient[:, :, self.crop_value:-self.crop_value, self.crop_value:-self.crop_value]

        coefficient = coefficient * self.scale

        return coefficient