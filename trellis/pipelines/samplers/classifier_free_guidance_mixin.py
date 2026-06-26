from typing import *


class ClassifierFreeGuidanceSamplerMixin:
    """
    A mixin class for samplers that apply classifier-free guidance.
    """

    def _inference_model(self, model, x_t, t, cond, neg_cond, cfg_strength, **kwargs):
        # Classifier-free guidance happens at the velocity-prediction level. The
        # same noisy state x_t and timestep are evaluated with positive and negative
        # context, then extrapolated toward the conditional prediction.
        pred = super()._inference_model(model, x_t, t, cond, **kwargs)
        neg_pred = super()._inference_model(model, x_t, t, neg_cond, **kwargs)
        return (1 + cfg_strength) * pred - cfg_strength * neg_pred
