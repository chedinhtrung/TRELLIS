from typing import Dict, Tuple

import torch
import torch.nn.functional as F
from easydict import EasyDict as edict

from ... import models
from .flow_matching import ImageConditionedFlowMatchingCFGTrainer


class ImageConditionedDecoderAwareFlowMatchingCFGTrainer(ImageConditionedFlowMatchingCFGTrainer):
    """Train only the coordinate head on generated Objective-1 endpoints."""

    def __init__(
        self,
        *args,
        pretrained_ss_decoder: str = (
            'microsoft/TRELLIS-image-large/ckpts/ss_dec_conv3d_16l8_fp16'
        ),
        empty_loss_weight: float = 1.0,
        exterior_loss_weight: float = 2.0,
        interior_loss_weight: float = 4.0,
        interior_aux_weight: float = 0.1,
        **kwargs,
    ):
        weights = [empty_loss_weight, exterior_loss_weight, interior_loss_weight]
        if any(weight <= 0 for weight in weights):
            raise ValueError('Coordinate voxel weights must be positive')
        if interior_aux_weight < 0:
            raise ValueError('interior_aux_weight must be non-negative')

        self.empty_loss_weight = float(empty_loss_weight)
        self.exterior_loss_weight = float(exterior_loss_weight)
        self.interior_loss_weight = float(interior_loss_weight)
        self.interior_aux_weight = float(interior_aux_weight)
        super().__init__(*args, **kwargs)

        if self.p_uncond != 0:
            raise ValueError('Coordinate-head endpoint training requires p_uncond=0')
        trainable = [
            name
            for name, parameter in self.models['denoiser'].named_parameters()
            if parameter.requires_grad
        ]
        if not trainable or any(not name.startswith('coordinate_head.') for name in trainable):
            raise RuntimeError(
                'Coordinate-head endpoint training requires the complete SS backbone and LoRA '
                f'to be frozen; trainable parameters are {trainable[:10]}'
            )

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
        # x_0 is not a GT/noisy training latent in this trainer. It is the cached
        # final z_s produced by the frozen Objective-1 sampler for this same image.
        cond = self.get_cond(cond, **kwargs)
        with torch.no_grad():
            base_logits = self.ss_decoder(self._decoder_input(x_0))

        zero_t = torch.zeros(x_0.shape[0], device=x_0.device, dtype=torch.float32)
        _, logits = self.training_models['denoiser'](
            x_0,
            zero_t,
            cond,
            return_coordinate_head=True,
            base_logits=base_logits,
            **kwargs,
        )
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
        voxel_weights = torch.full_like(voxel_bce, self.empty_loss_weight)
        voxel_weights[exterior] = self.exterior_loss_weight
        voxel_weights[internal] = self.interior_loss_weight
        coordinate_loss = (voxel_bce * voxel_weights).sum() / voxel_weights.sum()
        interior_loss = self._masked_mean(voxel_bce, internal)
        total_loss = coordinate_loss + self.interior_aux_weight * interior_loss

        terms = edict()
        terms['coordinate'] = coordinate_loss
        terms['coordinate_empty'] = self._masked_mean(voxel_bce, empty)
        terms['coordinate_exterior'] = self._masked_mean(voxel_bce, exterior)
        terms['coordinate_internal'] = interior_loss
        terms['interior_aux'] = self.interior_aux_weight * interior_loss
        terms['loss'] = total_loss

        with torch.no_grad():
            clipped_base = base_logits.clamp(
                -self.models['denoiser'].coordinate_head.base_logit_clip,
                self.models['denoiser'].coordinate_head.base_logit_clip,
            )
            coordinate_residual = logits - clipped_base
            status = {
                'gt_voxels': occupied.flatten(1).sum(1).float().mean(),
                'base_voxels': (base_logits > 0).flatten(1).sum(1).float().mean(),
                'pred_voxels': (logits > 0).flatten(1).sum(1).float().mean(),
                'coordinate_residual_abs': coordinate_residual.abs().mean(),
                'coordinate_residual_max': coordinate_residual.abs().max(),
            }
        return terms, status
