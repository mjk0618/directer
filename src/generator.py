from copy import deepcopy

import torch
import torch.nn.functional as F

from .cache_utils import scale_kv_cache


class DirecterGenerator:
    def __init__(
        self,
        model,
        tokenizer,
        *,
        key_scale_factor: float = 100.0,
        value_scale_factor: float = 1.0,
        plausibility: float = 0.5,
        max_new_tokens: int = 2048,
        max_trial: int = 10,
        fast_decoding: bool = False,
        decoding_attn_implementation: str = "sdpa",
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = model.device

        self.key_scale_factor = key_scale_factor
        self.value_scale_factor = value_scale_factor
        self.plausibility = plausibility
        self.max_new_tokens = max_new_tokens
        self.max_trial = max_trial

        # Layer ranking reads attention internals and needs eager attention.
        # The subsequent decoding loop only consumes logits, so it can run with a
        # fast attention implementation (e.g. sdpa / flash_attention_2). KV cache
        # scaling does not touch the attention kernel, so it stays compatible.
        self.fast_decoding = fast_decoding
        self.decoding_attn_implementation = decoding_attn_implementation

        self.num_layers = model.config.num_hidden_layers
        self.layer_indices = list(range(self.num_layers))
        self.token_indices = None  # instruction span; set in configure()

        # Forward hooks read each attention block's input/output hidden states.
        self.attention_states = []
        self.hooks = []

    @staticmethod
    def _fp32(x: torch.Tensor) -> torch.Tensor:
        return x.to(torch.float32)

    def configure(self, start_idx: int, end_idx: int):
        self.token_indices = [range(start_idx, end_idx + 1)]

    def _attach_attention_hooks(self):
        self.attention_states = []

        def post_hook_fn(module, args, kwargs, result):
            pre_hidden_states = kwargs.get("hidden_states", None)
            post_hidden_states = result[0]
            if pre_hidden_states is not None and post_hidden_states is not None:
                self.attention_states.append(
                    (pre_hidden_states.detach().clone(), post_hidden_states.detach().clone())
                )

        for name, module in self.model.named_modules():
            if "layers." in name and name.endswith(".self_attn"):
                self.hooks.append(module.register_forward_hook(post_hook_fn, with_kwargs=True))

    def _remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
        self.attention_states.clear()

    def _set_attn_implementation(self, impl):
        self.model.config._attn_implementation = impl

    def _prefill(self, input_ids):
        pseudo_next_token_id = input_ids[:, -1:]
        outputs = self.model(input_ids[:, :-1], return_dict=True)
        return outputs.past_key_values, pseudo_next_token_id

    def _can_skip_steering(self, raw_logits, ranked_layers, eps=1e-9):
        # Skip the steered pass when no steered distribution could change the
        # top-1 token and still pass the plausibility constraint.
        if ranked_layers is None:
            return False

        probs = torch.softmax(self._fp32(raw_logits), dim=-1)
        p = self.plausibility
        x, y = probs.topk(k=2, dim=-1).values.flatten()[:2]

        c1 = (x + y) <= (1.0 + eps)
        c2 = y <= (x + eps)
        c3 = y >= (p * x - eps)
        box = (x >= -eps) & (x <= 1.0 + eps) & (y >= -eps) & (y <= 1.0 + eps)
        scaling_possible = bool(c1 & c2 & c3 & box)
        return not scaling_possible

    def _is_plausible(self, raw_logits, scaled_logits):
        # Accept the steered token iff its new top-1 was plausible under the raw
        raw_flat = raw_logits.view(-1, raw_logits.size(-1))
        scaled_flat = scaled_logits.view(-1, scaled_logits.size(-1))

        top1_scaled_idx = torch.argmax(scaled_flat, dim=-1)
        top1_raw_idx = torch.argmax(raw_flat, dim=-1)

        raw_probs = F.softmax(raw_flat, dim=-1)
        prob_top1_scaled_in_raw = raw_probs.gather(-1, top1_scaled_idx.unsqueeze(-1)).squeeze(-1)
        prob_top1_raw_in_raw = raw_probs.gather(-1, top1_raw_idx.unsqueeze(-1)).squeeze(-1)

        margin = prob_top1_scaled_in_raw - self.plausibility * prob_top1_raw_in_raw
        return margin.item() > 0

    def _halve_layers(self, ranked_layers, trial):
        # Keep the top L / 2**trial most sensitive layers.
        ranked = list(ranked_layers.keys())
        keep_num = max(1, len(ranked) // (2 ** trial))
        return ranked[:keep_num]

    def _rank_layers(self, next_token_id, raw_cache_prev, raw_attention_states, device_type):
        # Steer each layer alone and measure the attention disturbance it
        # propagates across all layers; rank layers from most to least sensitive.
        target_layers = list(self.layer_indices)
        attn_rows = []  # one [L] disturbance vector per probed layer

        with torch.autocast(device_type=device_type, enabled=False):
            for layer_id in target_layers:
                scaled_cache = scale_kv_cache(
                    raw_cache_prev,
                    key_scale_factor=self.key_scale_factor,
                    value_scale_factor=self.value_scale_factor,
                    token_indices=self.token_indices,
                    layer_indices=[layer_id],
                )

                self.attention_states.clear()
                scaled_outputs = self.model(
                    input_ids=next_token_id,
                    past_key_values=scaled_cache,
                    use_cache=True,
                    output_attentions=True,  # forces eager attention -> hooks fire
                )
                scaled_attention_states = deepcopy(self.attention_states)

                # Cosine-distance disturbance at each layer j.
                local = []
                for j in range(self.num_layers):
                    h_pre = self._fp32(raw_attention_states[j][0])
                    h_post = self._fp32(raw_attention_states[j][1])
                    h_pre_s = self._fp32(scaled_attention_states[j][0])
                    h_post_s = self._fp32(scaled_attention_states[j][1])
                    s1 = 1.0 - F.cosine_similarity(h_pre, h_post_s, dim=-1)
                    s2 = 1.0 - F.cosine_similarity(h_post, h_pre_s, dim=-1)
                    diff = (s1 + s2).mean().clamp_min(0.0)
                    local.append(diff)
                attn_rows.append(torch.stack(local, dim=0))  # [L]

        attn_mat = self._fp32(torch.stack(attn_rows, dim=0))           # [P, L]
        per_layer = attn_mat.mean(dim=0)                               # [L]
        diffs = torch.stack([per_layer[l] for l in target_layers], dim=0)

        order = torch.argsort(diffs, descending=True)  # most sensitive first
        sorted_layers = [int(target_layers[i]) for i in order.tolist()]
        sorted_vals = [float(diffs[i].item()) for i in order.tolist()]
        return {layer: val for layer, val in zip(sorted_layers, sorted_vals)}

    @torch.inference_mode()
    def generate(self, prompt):
        input_ids = prompt.input_ids.to(self.device)
        generated_ids = []

        self._attach_attention_hooks()
        # Layer ranking runs under the model's (eager) attention; remember it so
        # we can restore it after we optionally switch to a fast kernel.
        original_attn_impl = self.model.config._attn_implementation

        try:
            raw_cache, next_token_id = self._prefill(input_ids)
            ranked_layers = None  # computed once, on the first decoding step

            device_type = "cuda" if next(self.model.parameters()).is_cuda else "cpu"

            for _ in range(self.max_new_tokens):
                raw_cache_prev = deepcopy(raw_cache)
                first_step = ranked_layers is None  # ranking happens on this step

                # Raw forward pass. Attentions (eager) are only needed for the
                # one-time layer ranking on the first step.
                self.attention_states.clear()
                raw_outputs = self.model(
                    input_ids=next_token_id,
                    past_key_values=raw_cache,
                    use_cache=True,
                    output_attentions=first_step,
                )
                raw_logits = raw_outputs.logits
                raw_attention_states = deepcopy(self.attention_states) if first_step else None

                is_decoding_valid = self._can_skip_steering(raw_logits, ranked_layers)
                if is_decoding_valid:
                    scaled_logits = raw_logits

                if ranked_layers is None:
                    ranked_layers = self._rank_layers(
                        next_token_id, raw_cache_prev, raw_attention_states, device_type
                    )
                    # Ranking is done; the remaining decoding only needs logits, so
                    # we can switch to a fast attention implementation.
                    if self.fast_decoding:
                        self._remove_hooks()
                        self._set_attn_implementation(self.decoding_attn_implementation)

                # Plausibility-guided decoding loop.
                current_layers = list(self.layer_indices)  # start by steering all layers
                trial = 0
                while not is_decoding_valid:
                    trial += 1
                    if len(current_layers) == 1 or trial > self.max_trial:
                        scaled_logits = raw_logits  # fall back to the raw distribution
                        break

                    scaled_cache = scale_kv_cache(
                        raw_cache_prev,
                        key_scale_factor=self.key_scale_factor,
                        value_scale_factor=self.value_scale_factor,
                        token_indices=self.token_indices,
                        layer_indices=current_layers,
                    )
                    scaled_outputs = self.model(
                        input_ids=next_token_id,
                        past_key_values=scaled_cache,
                        use_cache=True,
                        output_attentions=False,
                    )
                    scaled_logits = scaled_outputs.logits

                    is_decoding_valid = self._is_plausible(raw_logits, scaled_logits)
                    if is_decoding_valid:
                        break
                    # Weaken steering: drop the least sensitive half of the layers.
                    current_layers = self._halve_layers(ranked_layers, trial)
                    if not current_layers:
                        scaled_logits = raw_logits
                        break

                next_token_id = torch.argmax(scaled_logits[:, -1, :], dim=-1, keepdim=True)
                generated_ids.append(next_token_id.item())

                if next_token_id.item() == self.tokenizer.eos_token_id:
                    break
        finally:
            self._remove_hooks()
            self._set_attn_implementation(original_attn_impl)

        return self.tokenizer.decode(generated_ids, skip_special_tokens=True)
