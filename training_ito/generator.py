import torch
import torch.nn as nn
import numpy as np

#----------------------------------------------------------------------------
# EDM sampler.

def edmmpo_sampler(
    net, coefficient, MC,
    latents, class_labels=None, randn_like=torch.randn_like, t_steps=None,
    num_steps=18, S_churn=0, S_min=0, S_max=float('inf'), S_noise=1,
    precision = torch.float64, coefficient_conditioning=False
):  
    # initialize
    t_steps = t_steps.to(latents.device)
    denoised_tensors = []

    # Get current mac from time steps
    _, gamma_1_cur = MC.get_mac(t_steps[0].unsqueeze(0), coefficient)

    # Main sampling loop.
    x_next = latents.to(precision) * gamma_1_cur

    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])): # 0, ..., N-1
        x_cur = x_next
        t_cur, t_next = t_cur.unsqueeze(0), t_next.unsqueeze(0)

        # Get next mac from time steps
        _, gamma_1_next = MC.get_mac(t_next, coefficient)

        # Increase noise temporarily.
        gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
        if gamma != 0:
            # Add noise
            t_hat = net.round_sigma(t_cur + gamma * t_cur)
            gamma_1_hat, _ = MC.get_mac(t_hat, coefficient)
            x_hat = x_cur + (gamma_1_hat ** 2 - gamma_1_cur ** 2).sqrt() * S_noise * randn_like(x_cur)
        else:
            t_hat = t_cur
            gamma_1_hat = gamma_1_cur
            x_hat = x_cur

        # Euler step.
        if coefficient_conditioning:
            denoised = net(x_hat, t_hat, gamma_1_hat, class_labels).to(precision)
        else:
            # use mean of gamma_1_hat as time for H_\theta
            t_hat_by_gamma_1_hat = gamma_1_hat.mean(dim=(1,2,3))
            denoised = net(x_hat, t_hat_by_gamma_1_hat, class_labels).to(precision)

        # Append to denoised list
        denoised_tensors.append(denoised)

        # Calculate x_next
        d_cur = (x_hat - denoised) / gamma_1_hat
        x_next = x_hat + (gamma_1_next - gamma_1_hat) * d_cur

        gamma_1_cur = gamma_1_next
    
    # Turn list to tensor
    denoised_tensors = torch.cat(denoised_tensors, dim=0)
    
    return x_next, denoised_tensors


#----------------------------------------------------------------------------
# Generator

class Generator(nn.Module):
    def __init__(self, V, C, MC, coefficient_conditioning, num_steps, rho, S_churn, S_min, S_max, S_noise, sigma_min, sigma_max):
        super().__init__()
        self.P_mean = -1.2
        self.P_std = 1.2

        # Pretrained edm network
        self.V = V

        # Coefficient generator
        self.C = C

        # MAC calculator
        self.MC = MC

        # coefficient conditioning
        self.coefficient_conditioning = coefficient_conditioning

        # sampler config
        self.sampler = edmmpo_sampler
        self.num_steps = num_steps
        self.rho = rho
        self.S_churn = S_churn
        self.S_min = S_min
        self.S_max = S_max
        self.S_noise = S_noise
        self.precision = torch.float32

        # Adjust noise levels based on what's supported by the network.
        self.sigma_min = max(sigma_min, self.V.sigma_min)
        self.sigma_max = min(sigma_max, self.V.sigma_max)

        # Time step discretization.
        step_indices = torch.arange(self.num_steps, dtype=self.precision)
        t_steps = (self.sigma_max ** (1 / self.rho) + step_indices / (self.num_steps - 1) * (self.sigma_min ** (1 / self.rho) - self.sigma_max ** (1 / self.rho))) ** self.rho
        self.t_steps = torch.cat([self.V.round_sigma(t_steps), torch.zeros_like(t_steps[:1])]) # t_N = 0
    
    @torch.no_grad()
    def generate_with_unidimensional_coefficient(self, z, c=None):
        # Get coefficient
        coefficient = torch.zeros(z.size(0), 6 * self.C.M, z.size(2), z.size(3)).to(z.device)

        # Generate images
        pred, _ = self.sampler(
            net=self.V, 
            coefficient=coefficient, 
            MC=self.MC, 
            latents=z, 
            class_labels=c,
            precision=self.precision,
            t_steps=self.t_steps,
            num_steps=self.num_steps,
            S_churn=self.S_churn, 
            S_min=self.S_min, 
            S_max=self.S_max, 
            S_noise=self.S_noise,
            coefficient_conditioning=self.coefficient_conditioning
            )

        return pred

    def forward(self, z, c=None, img=None, simulation=True):
        # Get coefficient
        if simulation:
            coefficient = self.C(z, c)

            # Generate images
            pred, denoised_tensors = self.sampler(
                net=self.V, 
                coefficient=coefficient, 
                MC=self.MC, 
                latents=z, 
                class_labels=c,
                precision=self.precision,
                t_steps=self.t_steps,
                num_steps=self.num_steps,
                S_churn=self.S_churn, 
                S_min=self.S_min, 
                S_max=self.S_max, 
                S_noise=self.S_noise,
                coefficient_conditioning=self.coefficient_conditioning
                )
            
            return pred, denoised_tensors, coefficient
        else:
            # Randomly sample time
            rnd_normal = torch.randn([img.shape[0],], device=img.device)
            t = (rnd_normal * self.P_std + self.P_mean).exp()

            #------ Quantize t to the closest value in self.t_steps ------#
            # Expand t and self.t_steps to compute pairwise differences
            t_expanded = t.unsqueeze(1)
            t_steps_expanded = self.t_steps[:-1].unsqueeze(0).to(img.device)

            # Compute pairwise differences and find the index of the minimum difference
            diff = torch.abs(t_expanded - t_steps_expanded)
            closest_indices = torch.argmin(diff, dim=1).cpu()

            # Replace t with the closest value in self.t_steps
            t = self.t_steps[closest_indices].to(img.device)
            #-------------------------------------------------------------#

            # Get coefficient and MAC
            with torch.no_grad():
                coefficient = self.C(torch.randn_like(img), c)
                _, gamma_1 = self.MC.get_mac(t, coefficient)

            # Predict x_0
            if self.coefficient_conditioning:
                prediction_x_0 = self.V(img + gamma_1 * z , t.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1), gamma_1, c)
            else:
                t_by_gamma_1 = gamma_1.mean(dim=(1,2,3))
                prediction_x_0 = self.V(img + gamma_1 * z , t_by_gamma_1.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1), c)
            
            return prediction_x_0