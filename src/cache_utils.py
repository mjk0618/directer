# Scale the key vectors of the instruction tokens inside the KV cache.
from typing import List, Optional, Tuple, Union

import copy

import torch
from transformers import Cache


def _parse_token_indices(
    token_indices: List[Union[int, Tuple[int, int], range]],
    max_len: int,
) -> List[int]:
    # Flatten a mixed list of ints/tuples/ranges into sorted unique indices.
    if not token_indices:
        return []
    final_indices = set()
    for item in token_indices:
        if isinstance(item, int):
            final_indices.add(item)
        elif isinstance(item, tuple) and len(item) == 2:
            start, end = item
            final_indices.update(range(start, end))
        elif isinstance(item, range):
            final_indices.update(item)
    return sorted(i for i in final_indices if 0 <= i < max_len)


def scale_kv_cache(
    past_key_values: Optional[Cache],
    *,
    key_scale_factor: float,
    value_scale_factor: float,
    token_indices: List[Union[int, Tuple[int, int], range]],
    layer_indices: Optional[List[int]] = None,
    in_place: bool = False,
) -> Optional[Cache]:
    # Return a (copied) cache whose selected token/layer key-value vectors are
    # multiplied by the scaling factors.
    if past_key_values is None or (key_scale_factor == 1.0 and value_scale_factor == 1.0):
        return past_key_values

    cache = past_key_values if in_place else copy.deepcopy(past_key_values)

    seq_len = cache.get_seq_length()
    num_layers = len(cache.key_cache)
    layers = range(num_layers) if layer_indices is None else [
        idx for idx in layer_indices if 0 <= idx < num_layers
    ]

    selected_idx = _parse_token_indices(token_indices, seq_len)
    if not selected_idx or not layers:
        return cache

    device = cache.key_cache[0].device
    selected = torch.as_tensor(selected_idx, device=device, dtype=torch.long)

    for l in layers:
        if key_scale_factor != 1.0:
            cache.key_cache[l][:, :, selected, :] *= key_scale_factor
        if value_scale_factor != 1.0:
            cache.value_cache[l][:, :, selected, :] *= value_scale_factor

    return cache
