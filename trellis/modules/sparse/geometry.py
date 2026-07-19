import torch


def axis_interior_mask(coords: torch.Tensor, margin: int = 1) -> torch.Tensor:
    """Mark sparse voxels behind both extrema of every axis-aligned line.

    ``coords`` must contain ``(batch, x, y, z)`` rows.  Including the batch
    column in each group keeps objects in the same batch independent.
    """
    if coords.ndim != 2 or coords.shape[1] != 4:
        raise ValueError(f"coords must have shape [N, 4], got {tuple(coords.shape)}")
    if margin < 1:
        raise ValueError("margin must be at least 1")
    if coords.shape[0] == 0:
        return torch.zeros(0, dtype=torch.bool, device=coords.device)

    coords = coords.long()
    batch = coords[:, 0]
    xyz = coords[:, 1:]
    interior = torch.ones(coords.shape[0], dtype=torch.bool, device=coords.device)

    for axis in range(3):
        other_axes = [index for index in range(3) if index != axis]
        groups = torch.stack(
            [batch, xyz[:, other_axes[0]], xyz[:, other_axes[1]]], dim=1
        )
        unique_groups, inverse = torch.unique(groups, dim=0, return_inverse=True)
        values = xyz[:, axis]
        minimum = torch.full(
            (unique_groups.shape[0],),
            torch.iinfo(values.dtype).max,
            dtype=values.dtype,
            device=values.device,
        )
        maximum = torch.full(
            (unique_groups.shape[0],),
            torch.iinfo(values.dtype).min,
            dtype=values.dtype,
            device=values.device,
        )
        minimum.scatter_reduce_(0, inverse, values, reduce="amin", include_self=True)
        maximum.scatter_reduce_(0, inverse, values, reduce="amax", include_self=True)
        interior &= (values - minimum[inverse] >= margin) & (
            maximum[inverse] - values >= margin
        )

    return interior
