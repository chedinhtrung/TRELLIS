from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from easydict import EasyDict as edict

from ... import models
from .flow_matching import ImageConditionedFlowMatchingCFGTrainer


def predict_clean_latent(
    x_t: torch.Tensor,
    pred_velocity: torch.Tensor,
    t: torch.Tensor,
    sigma_min: float,
) -> torch.Tensor:
    """Recover the clean endpoint implied by a rectified-flow velocity."""
    t_grid = t.view(-1, *[1 for _ in range(x_t.ndim - 1)])
    noise_scale = sigma_min + (1 - sigma_min) * t_grid
    return (1 - sigma_min) * x_t - noise_scale * pred_velocity


class ImageConditionedDecoderAwareFlowMatchingCFGTrainer(ImageConditionedFlowMatchingCFGTrainer):
    """Train SS flow with direct supervision on decoded 64^3 coordinates."""

    def __init__(
        self,
        *args,
        pretrained_ss_decoder: str = (
            'microsoft/TRELLIS-image-large/ckpts/ss_dec_conv3d_16l8_fp16'
        ),
        coordinate_loss_weight: float = 0.1,
        empty_loss_weight: float = 0.4,
        exterior_loss_weight: float = 0.2,
        interior_loss_weight: float = 0.4,
        **kwargs,
    ):
        if coordinate_loss_weight <= 0:
            raise ValueError('coordinate_loss_weight must be positive')
        group_weights = [empty_loss_weight, exterior_loss_weight, interior_loss_weight]
        if any(weight < 0 for weight in group_weights) or sum(group_weights) <= 0:
            raise ValueError('Coordinate group weights must be non-negative and sum to more than zero')

        self.coordinate_loss_weight = float(coordinate_loss_weight)
        self.empty_loss_weight = float(empty_loss_weight)
        self.exterior_loss_weight = float(exterior_loss_weight)
        self.interior_loss_weight = float(interior_loss_weight)
        super().__init__(*args, **kwargs)

        self.ss_decoder = models.from_pretrained(pretrained_ss_decoder).to(self.device).eval()
        self.ss_decoder.requires_grad_(False)

    @staticmethod
    def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return (values * mask).sum() / mask.sum().clamp_min(1)

    def _decoder_input(self, latent: torch.Tensor) -> torch.Tensor:
        if getattr(self.dataset, 'normalization', None) is None:
            return latent
        mean = self.dataset.mean.to(device=latent.device, dtype=latent.dtype)
        std = self.dataset.std.to(device=latent.device, dtype=latent.dtype)
        return latent * std + mean

    def training_losses(
        self,
        x_0: torch.Tensor,
        occupancy: torch.Tensor,
        internal_occupancy: torch.Tensor,
        cond=None,
        **kwargs,
    ) -> Tuple[Dict, Dict]:
        noise = torch.randn_like(x_0)
        t = self.sample_t(x_0.shape[0]).to(x_0.device).float()
        x_t = self.diffuse(x_0, t, noise=noise)
        cond = self.get_cond(cond, **kwargs)

        pred, coordinate_residual = self.training_models['denoiser'](
            x_t,
            t * 1000,
            cond,
            return_coordinate_head=True,
            **kwargs,
        )
        if pred.shape != x_0.shape or coordinate_residual is None:
            raise RuntimeError('SS denoiser returned invalid velocity or coordinate-head output')

        target_velocity = self.get_v(x_0, noise, t)
        flow_mse = F.mse_loss(pred, target_velocity)

        x_0_hat = predict_clean_latent(x_t, pred, t, self.sigma_min)
        base_logits = self.ss_decoder(self._decoder_input(x_0_hat))
        logits = base_logits + coordinate_residual
        if logits.shape != occupancy.shape or internal_occupancy.shape != occupancy.shape:
            raise RuntimeError(
                'Coordinate target shape mismatch: '
                f'logits={tuple(logits.shape)}, occupancy={tuple(occupancy.shape)}, '
                f'internal={tuple(internal_occupancy.shape)}'
            )

        occupied = occupancy.bool()
        internal = internal_occupancy.bool()
        if torch.any(internal & ~occupied):
            raise RuntimeError('Internal occupancy must be a subset of occupancy')
        exterior = occupied & ~internal
        empty = ~occupied

        voxel_bce = F.binary_cross_entropy_with_logits(
            logits.float(), occupied.float(), reduction='none'
        )
        empty_losses = []
        exterior_losses = []
        interior_losses = []
        coordinate_losses = []
        for index in range(x_0.shape[0]):
            empty_loss = self._masked_mean(voxel_bce[index], empty[index])
            exterior_loss = self._masked_mean(voxel_bce[index], exterior[index])
            interior_loss = self._masked_mean(voxel_bce[index], internal[index])
            empty_losses.append(empty_loss)
            exterior_losses.append(exterior_loss)
            interior_losses.append(interior_loss)
            coordinate_losses.append(
                self.empty_loss_weight * empty_loss
                + self.exterior_loss_weight * exterior_loss
                + self.interior_loss_weight * interior_loss
            )

        empty_loss = torch.stack(empty_losses).mean()
        exterior_loss = torch.stack(exterior_losses).mean()
        interior_loss = torch.stack(interior_losses).mean()
        coordinate_per_sample = torch.stack(coordinate_losses)
        coordinate_loss = coordinate_per_sample.mean()
        weighted_coordinate_loss = ((1 - t) * coordinate_per_sample).mean()

        terms = edict()
        terms['mse'] = flow_mse
        terms['coordinate'] = coordinate_loss
        terms['coordinate_weighted'] = weighted_coordinate_loss
        terms['coordinate_empty'] = empty_loss
        terms['coordinate_exterior'] = exterior_loss
        terms['coordinate_internal'] = interior_loss
        terms['loss'] = flow_mse + self.coordinate_loss_weight * weighted_coordinate_loss

        mse_per_instance = np.array([
            F.mse_loss(pred[index], target_velocity[index]).item()
            for index in range(x_0.shape[0])
        ])
        time_bin = np.digitize(t.detach().cpu().numpy(), np.linspace(0, 1, 11)) - 1
        for index in range(10):
            if (time_bin == index).sum() != 0:
                terms[f'bin_{index}'] = {'mse': mse_per_instance[time_bin == index].mean()}

        with torch.no_grad():
            status = {
                'gt_voxels': occupied.flatten(1).sum(1).float().mean(),
                'pred_voxels': (logits > 0).flatten(1).sum(1).float().mean(),
                'coordinate_residual_abs': coordinate_residual.abs().mean(),
            }
        return terms, status
