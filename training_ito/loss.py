# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#
# modified by Axel Sauer for "StyleGAN-XL: Scaling StyleGAN to Large Diverse Datasets"
#

"""Loss functions."""

import numpy as np
import torch
import torch.nn.functional as F

class ProjectedGANLoss:
    def __init__(self, G, D):
        self.G = G
        self.D = D

    def generator_loss_V(self, z, c, img):
        # Maximize logits for generated images.
        prediction_x_0 = self.G(z=z, c=c, img=img, simulation=False)
        gen_logits = self.D(prediction_x_0, c)
        loss = sum([(-l).mean() for l in gen_logits])

        return loss
    
    def generator_loss_C(self, gen_z, gen_c):
        # Maximize logits for generated images.
        gen_img, _, _ = self.G(z=gen_z, c=gen_c, simulation=True)
        gen_logits = self.D(gen_img, gen_c)
        loss = sum([(-l).mean() for l in gen_logits])

        return loss
    
    def discriminator_loss(self, real_img, real_c, gen_z, gen_c):
        # Minimize logits for generated images.
        with torch.no_grad():
            gen_img, _, coefficient = self.G(z=gen_z, c=gen_c, simulation=True)
        gen_logits = self.D(gen_img, gen_c)
        loss_Dgen = sum([(F.relu(torch.ones_like(l) + l)).mean() for l in gen_logits])

        # Maximize logits for real images.
        real_logits = self.D(real_img, real_c)
        loss_Dreal = sum([(F.relu(torch.ones_like(l) - l)).mean() for l in real_logits])

        return loss_Dgen, loss_Dreal, coefficient