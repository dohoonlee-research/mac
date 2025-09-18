import torch


class SinusoidalMacCalculator:
    def __init__(self, M, t_max=80, rho=7, k=0): 
        self.M = M
        self.t_max = t_max
        self.rho = rho
        self.m_term_0 = torch.arange(1, self.M + 1).unsqueeze(0).repeat(1, 3)
        self.m_term_1 = (1/torch.cat((self.m_term_0**k, self.m_term_0**k), dim=1).unsqueeze(2).unsqueeze(3))
    
    def calculate_pi_term(self, t):
        """
        pi_term = [pi_term_f, pi_term_g] = [pi * m * (t/t_max)**(1/rho), pi * m * (t/t_max)**(1/rho)]
        """
        temp = torch.pi * self.m_term_0.to(t.device) * ((t.unsqueeze(-1) / self.t_max)**(1/self.rho))
        return torch.cat((temp, temp), dim=1)
         
    def calculate_sin_term(self, pi_term, coefficient):
        ''' 
        sin_term <- [c * (1/m**k) * sin(pi_term_f), c * (1/m**k) * sin(pi_term_g)]
        '''
        sin_term = coefficient * torch.sin(pi_term).unsqueeze(2).unsqueeze(3)
        sin_term = sin_term * self.m_term_1.to(pi_term.device)

        """
        sin_term <- Sigma_1^M { sin_term }
        """
        sin_term = sin_term.view(sin_term.size(0), 6, self.M, sin_term.size(2), sin_term.size(3))
        sin_term = torch.sum(sin_term, dim=2)
        
        return sin_term
    
    def calculate_f_g(self, t, sin_term):
        ''' 
        sin_term <- sin_term ** 2
        '''
        sin_term = sin_term ** 2
                
        ''' 
        f = 1 - t/t_max + sin_term_f
        g = t/t_max + sin_term_g
        '''

        f = 1 - t[:, None, None, None] / self.t_max + sin_term[:, :3, :, :]
        g = t[:, None, None, None] / self.t_max + sin_term[:, 3:, :, :]

        return f, g
    
    def calculate_mac(self, f, g):
        gamma_1 = (self.t_max * g) / (f + g)
        gamma_0 = torch.ones_like(gamma_1)
        return gamma_0, gamma_1
    
    def get_mac(self, t, coefficient):
        # Calculate pi term
        pi_term = self.calculate_pi_term(t)

        # Get sinusoidal term
        sin_term = self.calculate_sin_term(pi_term, coefficient)

        # Get f & g
        f, g = self.calculate_f_g(t, sin_term)

        # Get Mac
        gamma_0, gamma_1 = self.calculate_mac(f, g)

        return gamma_0, gamma_1