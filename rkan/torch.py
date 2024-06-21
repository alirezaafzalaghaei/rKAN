import torch
import torch.nn as nn
import torch.nn.functional as F

from .jacobi_polynomials import *

class PadeRKAN(nn.Module):
    def __init__(self, degree_p, degree_q):
        """
        Initialize the PadeRKAN module.

        Args:
            degree_p (int): Degree of the P polynomial.
            degree_q (int): Degree of the Q polynomial.
        """
        super(PadeRKAN, self).__init__()

        if not 0 < degree_p < 7 or not 0 < degree_q < 7:
            raise ValueError('Both degree_p and degree_q must be between one and six (inclusive).')

        self.degree_p = degree_p
        self.degree_q = degree_q

        # Define the trainable parameters for the P polynomial
        self.alpha_p = nn.Parameter(torch.ones(1))
        self.beta_p = nn.Parameter(torch.ones(1))
        self.zeta_p = nn.Parameter(torch.zeros(1))
        self.w_p = nn.Parameter(torch.ones(self.degree_p))

        # Define the trainable parameters for the Q polynomial
        self.alpha_q = nn.Parameter(torch.ones(1))
        self.beta_q = nn.Parameter(torch.ones(1))
        self.zeta_q = nn.Parameter(torch.zeros(1))
        self.w_q = nn.Parameter(torch.ones(self.degree_q))

    def forward(self, inputs):
        """
        Forward pass of the PadeRKAN module.

        Args:
            inputs (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor after applying the Pade rational function.
        """
        # Normalize parameters for the P polynomial
        normalized_alpha_p = F.elu(self.alpha_p, 1)
        normalized_beta_p = F.elu(self.beta_p, 1)
        normalized_zeta_p = torch.sigmoid(self.zeta_p)

        # Normalize parameters for the Q polynomial
        normalized_alpha_q = F.elu(self.alpha_q, 1)
        normalized_beta_q = F.elu(self.beta_q, 1)
        normalized_zeta_q = torch.sigmoid(self.zeta_q)

        # Normalize inputs
        normalized_inputs = torch.sigmoid(inputs)

        # Calculate the P polynomial
        p = self.w_p[0] + self.w_p[1] * normalized_inputs
        for deg in range(2, self.degree_p):
            p += self.w_p[deg] * shifted_jacobi_polynomial(
                normalized_inputs,
                deg,
                normalized_alpha_p,
                normalized_beta_p,
                normalized_zeta_p,
                0,
                1,
                backend=torch
            )

        # Calculate the Q polynomial
        q = self.w_q[0] + self.w_q[1] * normalized_inputs
        for deg in range(2, self.degree_q):
            q += self.w_q[deg] * shifted_jacobi_polynomial(
                normalized_inputs,
                deg,
                normalized_alpha_q,
                normalized_beta_q,
                normalized_zeta_q,
                0,
                1,
                backend=torch
            )

        # Return the Pade rational function
        return p / q


class JacobiRKAN(nn.Module):
    def __init__(self, degree):
        """
        Initialize the JacobiRKAN module.

        Args:
            degree (int): Degree of the Jacobi polynomial.
        """
        super(JacobiRKAN, self).__init__()

        if not 0 < degree < 7:
            raise ValueError('Parameter degree must be between one and six (inclusive).')

        self.degree = degree

        # Define the trainable parameters
        self.alpha = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.ones(1))
        self.iota = nn.Parameter(torch.ones(1))

    def forward(self, inputs):
        """
        Forward pass of the JacobiRKAN module.

        Args:
            inputs (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor after applying the Jacobi polynomial transformation.
        """
        # Normalize parameters
        normalized_alpha = F.elu(self.alpha, 1)
        normalized_beta = F.elu(self.beta, 1)
        normalized_iota = F.softplus(self.iota)

        # Return the result of the rational Jacobi polynomial
        return rational_jacobi_polynomial(
            inputs, self.degree, normalized_alpha, normalized_beta, 1, normalized_iota, backend=torch
        )
