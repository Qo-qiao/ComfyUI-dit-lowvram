"""Step Cache: skip-step caching for DiT transformer forward.

Adapted from comfyui-cache-dit. Skips transformer forward calls and reuses
cached results with small noise perturbation to avoid visual artifacts.

Supported models: MiniMax H3, MiniMax Music3, LTX, Wan.
"""

import torch
import time
import weakref


class StepCacheState:
    """Per-model cache state."""

    __slots__ = (
        "model_id", "enabled", "skip_interval", "warmup_steps", "noise_scale",
        "call_count", "skip_count", "compute_times", "last_result", "original_forward",
    )

    def __init__(self, model_id, skip_interval=2, warmup_steps=3, noise_scale=0.001):
        self.model_id = model_id
        self.enabled = True
        self.skip_interval = skip_interval
        self.warmup_steps = warmup_steps
        self.noise_scale = noise_scale
        self.call_count = 0
        self.skip_count = 0
        self.compute_times = []
        self.last_result = None
        self.original_forward = None

    def should_skip(self):
        if self.call_count <= self.warmup_steps:
            return False
        return self.call_count % self.skip_interval == 0

    def reset(self):
        self.call_count = 0
        self.skip_count = 0
        self.compute_times = []
        self.last_result = None


class StepCache:
    """Singleton step-cache manager for DiT models."""

    def __init__(self):
        self._states = {}
        self._refs = weakref.WeakKeyDictionary()

    def _state_key(self, model):
        return f"{type(model).__name__}_{id(model)}"

    def _get_state(self, model):
        key = self._state_key(model)
        return self._states.get(key)

    def enable(self, model, skip_interval=2, warmup_steps=3, noise_scale=0.001):
        """Enable step caching for a model. Returns (patched_model, applied)."""
        key = self._state_key(model)

        if key in self._states and self._states[key].original_forward is not None:
            self.disable(model)

        state = StepCacheState(key, skip_interval, warmup_steps, noise_scale)
        self._states[key] = state
        self._refs[model] = key

        transformer = self._find_transformer(model)
        if transformer is None:
            return model, False

        if hasattr(transformer, "_dit_cache_original_forward"):
            return model, False

        state.original_forward = transformer.forward
        transformer._dit_cache_original_forward = transformer.forward

        def cached_forward(*args, **kwargs):
            if not state.enabled:
                return state.original_forward(*args, **kwargs)

            state.call_count += 1

            if state.should_skip() and state.last_result is not None:
                state.skip_count += 1
                noise = torch.randn_like(state.last_result) * state.noise_scale
                return state.last_result + noise

            start = time.time()
            result = state.original_forward(*args, **kwargs)
            state.compute_times.append(time.time() - start)

            if isinstance(result, torch.Tensor):
                state.last_result = result.clone().detach()
            elif isinstance(result, (tuple, list)) and len(result) > 0 and isinstance(result[0], torch.Tensor):
                state.last_result = result[0].clone().detach()

            return result

        transformer.forward = cached_forward
        return model, True

    def disable(self, model):
        """Disable step caching and restore original forward."""
        key = self._state_key(model)
        state = self._states.get(key)
        if state is None:
            return False

        transformer = self._find_transformer(model)
        if transformer is not None and hasattr(transformer, "_dit_cache_original_forward"):
            transformer.forward = transformer._dit_cache_original_forward
            del transformer._dit_cache_original_forward

        state.original_forward = None
        state.last_result = None
        return True

    def get_stats(self, model=None):
        """Return cache stats dict."""
        if model is not None:
            state = self._get_state(model)
            if state is None:
                return {"enabled": False}
            avg = sum(state.compute_times) / max(len(state.compute_times), 1)
            return {
                "enabled": state.enabled,
                "calls": state.call_count,
                "hits": state.skip_count,
                "hit_rate": state.skip_count / max(state.call_count, 1) * 100,
                "avg_time": avg,
                "skip_interval": state.skip_interval,
                "warmup_steps": state.warmup_steps,
            }

        total_calls = sum(s.call_count for s in self._states.values())
        total_hits = sum(s.skip_count for s in self._states.values())
        return {
            "enabled": any(s.enabled for s in self._states.values()),
            "calls": total_calls,
            "hits": total_hits,
            "hit_rate": total_hits / max(total_calls, 1) * 100,
            "models": len(self._states),
        }

    def _find_transformer(self, model):
        """Locate the transformer/diffusion model inside ComfyUI model hierarchy."""
        try:
            inner = model.get_model_object("model") if hasattr(model, "get_model_object") else None
        except Exception:
            inner = None

        candidates = []
        if inner is not None:
            for attr in ("diffusion_transformer", "diffusion_model", "transformer", "model"):
                obj = getattr(inner, attr, None)
                if obj is not None:
                    candidates.append(obj)

        if hasattr(model, "model"):
            for attr in ("diffusion_model", "diffusion_transformer", "transformer"):
                obj = getattr(model.model, attr, None)
                if obj is not None:
                    candidates.append(obj)

        for obj in candidates:
            if hasattr(obj, "forward") and callable(obj.forward):
                if hasattr(obj, "blocks") or hasattr(obj, "transformer_blocks") or hasattr(obj, "layers"):
                    return obj

        for obj in candidates:
            if hasattr(obj, "forward") and callable(obj.forward):
                return obj

        return None

    def reset_stats(self):
        for state in self._states.values():
            state.reset()


global_step_cache = StepCache()
