import torch
import torch.nn as nn
import torch.nn.functional as F


class CategoryConditioningMixin:
    def enable_category_conditioning(self, categories):
        categories = tuple(categories)
        if not categories or len(categories) != len(set(categories)):
            raise ValueError("categories must be a non-empty list of unique names")
        self.category_names = categories
        self.category_to_id = {name: index for index, name in enumerate(categories)}
        self.category_embedding = nn.Embedding(len(categories), self.cond_channels).to(self.device)
        nn.init.normal_(self.category_embedding.weight, std=0.02)

    def append_category_token(self, cond, category):
        if not hasattr(self, "category_embedding"):
            return cond
        if category is None:
            raise ValueError("A category is required by this model")

        if isinstance(category, torch.Tensor):
            category_ids = category.to(device=cond.device, dtype=torch.long).reshape(-1)
        else:
            names = [category] if isinstance(category, str) else list(category)
            try:
                category_ids = torch.tensor(
                    [self.category_to_id[name] for name in names],
                    device=cond.device,
                    dtype=torch.long,
                )
            except KeyError as error:
                raise ValueError(f"Unknown category: {error.args[0]}") from error

        if category_ids.numel() == 1 and cond.shape[0] > 1:
            category_ids = category_ids.repeat(cond.shape[0])
        if category_ids.numel() != cond.shape[0]:
            raise ValueError(
                f"Category batch size {category_ids.numel()} does not match conditioning batch size {cond.shape[0]}"
            )

        token = self.category_embedding(category_ids)
        token = F.layer_norm(token, token.shape[-1:]).unsqueeze(1).to(cond.dtype)
        return torch.cat([cond, token], dim=1)
