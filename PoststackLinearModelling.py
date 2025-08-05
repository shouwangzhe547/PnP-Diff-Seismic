import copy
from typing import Optional

import numpy as np
import torch
import torch.fft as fft
from scipy.signal import filtfilt
from skimage.metrics import structural_similarity as ssim

import pylops.avo.poststack
from pylops import Identity, PoststackLinearModelling, ricker
from torch import Tensor

from deepinv.physics.forward import LinearPhysics
from deepinv.utils.metric import norm


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


def recover_x_torch(A, y):
    A = torch.tensor(A, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)
    A_pseudo_inv = torch.linalg.pinv(A)

    y_reshaped = y[0]  # assume shape (1, H, W) → (H, W)
    x_recovered = torch.zeros_like(y_reshaped)

    for i in range(y_reshaped.shape[0]):
        for j in range(y_reshaped.shape[1]):
            x_recovered[i, j] = torch.matmul(A_pseudo_inv, y_reshaped[i, j])

    x_recovered = x_recovered.unsqueeze(0)
    return x_recovered


def snr_db(reference: torch.Tensor,
           estimate: torch.Tensor,
           dim=None,
           eps: float = 1e-12):
    assert reference.shape == estimate.shape

    noise = reference - estimate
    signal_power = reference.pow(2).mean(dim=dim)
    noise_power = noise.pow(2).mean(dim=dim)

    snr = 10 * torch.log10((signal_power + eps) / (noise_power + eps))
    return snr


def calculate_snr_2d(reference: np.ndarray,
                     noisy: np.ndarray,
                     eps: float = 1e-12) -> float:
    assert reference.shape == noisy.shape

    reference = reference.astype(np.float64)
    noisy = noisy.astype(np.float64)

    signal_power = np.mean(reference ** 2)
    noise_power = np.mean((reference - noisy) ** 2)

    snr = 10 * np.log10((signal_power + eps) / (noise_power + eps))
    return snr


def calculate_snr(reference: torch.Tensor, noisy: torch.Tensor) -> float:
    assert reference.shape == noisy.shape, "The tensors must have the same shape."

    signal_power = torch.sum(reference ** 2)
    noise_power = torch.sum((reference - noisy) ** 2)

    if noise_power == 0:
        return float('inf')

    snr = 10 * torch.log10(signal_power / noise_power)
    return snr.item()


def calculate_rmse(true_model, inverted_model):
    return np.sqrt(np.mean((true_model - inverted_model) ** 2))


def compute_ssim(imageA, imageB):
    data_range = imageA.max() - imageA.min()
    ssim_score, diff = ssim(imageA, imageB, data_range=data_range, full=True)
    return ssim_score, diff


def normalize(data, min_val=None, max_val=None):
    if min_val is None:
        min_val = data.min()
    if max_val is None:
        max_val = data.max()

    normalized_data = (data - min_val) / (max_val - min_val)
    normalized_data = torch.from_numpy(normalized_data).unsqueeze(0).unsqueeze(0).float()
    normalized_data = torch.cat((normalized_data,) * 3, dim=1)

    return normalized_data, min_val, max_val


def denormalize(normalized_data, min_val, max_val):
    scaled = normalized_data * (max_val - min_val) + min_val
    result = scaled[0, 1, :, :].cpu().numpy()
    return result


def normalize1(data, min_val=None, max_val=None):
    if min_val is None:
        min_val = data.min()
    if max_val is None:
        max_val = data.max()

    normalized_data = (data - min_val) / (max_val - min_val)
    return normalized_data, min_val, max_val


def denormalize1(normalized_data, min_val, max_val):
    return normalized_data * (max_val - min_val) + min_val


class CustomForwardOperator(LinearPhysics):
    def __init__(self,
                 _A,
                 wav: Tensor,
                 nt0: int,
                 spatdims: Optional[tuple] = None,
                 explicit: bool = False,
                 sparse: bool = False,
                 kind: str = "centered"):
        super().__init__()

        self.wav = wav
        self.nt0 = nt0
        self.spatdims = spatdims
        self.explicit = explicit
        self.sparse = sparse
        self.kind = kind
        self._A = _A

    def A(self, x: Tensor) -> Tensor:
        y_np = self._A.float() @ x
        return torch.tensor(y_np, dtype=x.dtype)


    def A_adjoint(self, y: Tensor) -> Tensor:
        A_T = self._A.T.float().to(y.device)
        x_np = torch.einsum('ij,bcjk->bcik', A_T, y)
        return x_np.to(dtype=y.dtype)

    def norm(self) -> float:
        if hasattr(self.operator, "norm"):
            return self.operator.norm()

        x = torch.randn(self.operator.shape[1]).float()
        for _ in range(10):
            x = self.adjoint(self.forward(x))
            x = x / torch.norm(x)
        return torch.norm(self.forward(x))
