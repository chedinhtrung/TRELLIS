from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from easydict import EasyDict as edict

from ... import models
from .flow_matching import ImageConditionedFlowMatchingCFGTrainer


class ImageConditionedDecoderAwareSSFlowMatchingCFGTrainer(
    ImageConditionedFlowMatchingCFGTrainer
):
    """Fine-tune SS-flow LoRA with flow matching and decoded occupancy loss."""

    def __init__(
        self,
        *args,
        pretrained_ss_decoder: str = (
            'microsoft/TRELLIS-image-large/ckpts/ss_dec_conv3d_16l8_fp16'
        ),
        decoder_loss_weight: float = 0.1,
        **kwargs,
    ):
        if decoder_loss_weight <= 0:
            raise ValueError('decoder_loss_weight must be positive')
        self.decoder_loss_weight = float(decoder_loss_weight)
        super().__init__(*args, **kwargs)

        if not getattr(self.dataset, 'load_occupancy', False):
            raise ValueError('Decoder-aware SS-flow training requires load_occupancy=true')

        denoiser = self.models['denoiser']
        trainable = [
            name for name, parameter in denoiser.named_parameters()
            if parameter.requires_grad
        ]
        if not trainable or any(
            not (name.endswith('.lora_down') or name.endswith('.lora_up'))
            for name in trainable
        ):
            raise RuntimeError(
                'Decoder-aware SS-flow training must train only LoRA parameters; '
                f'trainable parameters are {trainable[:10]}'
            )
        if getattr(denoiser, 'coordinate_head', None) is not None:
            raise RuntimeError('Decoder-aware SS-flow training must not use a coordinate head')

        self.ss_decoder = models.from_pretrained(pretrained_ss_decoder).to(self.device).eval()
        self.ss_decoder.requires_grad_(False)

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
        internal_occupancy: torch.Tensor = None,
        cond=None,
        **kwargs,
    ) -> Tuple[Dict, Dict]:
        # Keep TRELLIS's original flow-matching objective.
        noise = torch.randn_like(x_0)
        t = self.sample_t(x_0.shape[0]).to(x_0.device).float()
        x_t = self.diffuse(x_0, t, noise=noise)
        cond = self.get_cond(cond, **kwargs)

        pred_v = self.training_models['denoiser'](x_t, t * 1000, cond, **kwargs)
        if pred_v.shape != x_0.shape:
            raise RuntimeError(
                f'SS-flow output shape {tuple(pred_v.shape)} does not match latent '
                f'shape {tuple(x_0.shape)}'
            )
        target_v = self.get_v(x_0, noise, t)
        flow_mse = F.mse_loss(pred_v, target_v)

        # This is exactly FlowEulerSampler._v_to_xstart_eps(). The decoder is
        # frozen, but this forward pass must retain its graph so occupancy loss
        # can update the SS-flow LoRA through pred_x_0.
        t_view = t.view(-1, *([1] * (x_0.ndim - 1)))
        noise_scale = self.sigma_min + (1 - self.sigma_min) * t_view
        pred_x_0 = (1 - self.sigma_min) * x_t - noise_scale * pred_v
        logits = self.ss_decoder(self._decoder_input(pred_x_0))

        if logits.shape != occupancy.shape:
            raise RuntimeError(
                f'Decoder logits shape {tuple(logits.shape)} does not match occupancy '
                f'shape {tuple(occupancy.shape)}'
            )
        if not logits.requires_grad:
            raise RuntimeError('Frozen SS decoder did not preserve gradients to the SS-flow LoRA')

        occupancy = occupancy.float()
        decoder_bce = F.binary_cross_entropy_with_logits(
            logits.float(), occupancy, reduction='mean'
        )

        terms = edict()
        terms['mse'] = flow_mse
        terms['decoder_bce'] = decoder_bce
        terms['decoder_weighted'] = self.decoder_loss_weight * decoder_bce
        terms['loss'] = flow_mse + terms['decoder_weighted']

        # Preserve the standard time-bin MSE logging from FlowMatchingTrainer.
        mse_per_instance = np.array([
            F.mse_loss(pred_v[i], target_v[i]).item()
            for i in range(x_0.shape[0])
        ])
        time_bin = np.digitize(t.detach().cpu().numpy(), np.linspace(0, 1, 11)) - 1
        for index in range(10):
            in_bin = time_bin == index
            if in_bin.any():
                terms[f'bin_{index}'] = {'mse': mse_per_instance[in_bin].mean()}

        with torch.no_grad():
            predicted = logits > 0
            occupied = occupancy.bool()
            intersection = (predicted & occupied).flatten(1).sum(1).float()
            union = (predicted | occupied).flatten(1).sum(1).float().clamp_min(1)
            status = {
                'gt_voxels': occupied.flatten(1).sum(1).float().mean(),
                'pred_voxels': predicted.flatten(1).sum(1).float().mean(),
                'decoded_iou': (intersection / union).mean(),
                'mean_t': t.mean(),
            }
        return terms, status


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
