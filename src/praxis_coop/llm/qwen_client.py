"""Local Qwen text-generation client.

The import of transformers is intentionally lazy so the package can still be
imported in lightweight tests. Actual Qwen-backed experiments fail loudly if
the model runtime or local checkpoint is missing.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from praxis_coop.configs import PHASE1_MODELS


_MODEL_CACHE: dict[tuple[str, str, str, str, str], tuple[Any, Any]] = {}


@dataclass
class QwenTextClient:
    """Small wrapper around a local Qwen instruction model."""

    model_name: str = "qwen3-8b-instruct"
    device_map: str = "auto"
    torch_dtype: str = "auto"
    max_new_tokens: int = 1024
    temperature: float = 0.2
    top_p: float = 0.9
    max_gpu_memory: str | None = None
    max_cpu_memory: str | None = None

    @property
    def model_dir(self) -> Path:
        if self.model_name not in PHASE1_MODELS:
            known = ", ".join(sorted(PHASE1_MODELS))
            raise ValueError(f"Unknown Qwen model {self.model_name!r}. Known models: {known}")
        return PHASE1_MODELS[self.model_name].local_dir

    def generate(self, *, system: str, user: str) -> str:
        tokenizer, model = self._load()
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        device = self._input_device(model)
        inputs = tokenizer([prompt], return_tensors="pt").to(device)
        generation_kwargs = {
            "max_new_tokens": int(self.max_new_tokens),
            "do_sample": self.temperature > 0.0,
        }
        if self.temperature > 0.0:
            generation_kwargs["temperature"] = float(self.temperature)
            generation_kwargs["top_p"] = float(self.top_p)
        generated = model.generate(**inputs, **generation_kwargs)
        new_tokens = generated[0][inputs.input_ids.shape[-1] :]
        return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    def preload(self) -> None:
        """Load tokenizer/model once so later generations reuse the cache."""

        self._load()

    def _load(self):
        model_dir = self.model_dir
        cache_key = (
            str(model_dir.resolve()),
            str(self.device_map),
            str(self.torch_dtype),
            str(self.max_gpu_memory or ""),
            str(self.max_cpu_memory or ""),
        )
        if cache_key in _MODEL_CACHE:
            return _MODEL_CACHE[cache_key]
        if not model_dir.exists():
            raise FileNotFoundError(
                f"Qwen model directory does not exist: {model_dir}. "
                f"Download it first with: python scripts/download_phase1_models.py --model {self.model_name}"
            )
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "Qwen critic requires transformers. Install the Isaac/Praxis Python dependencies "
                "or run: pip install transformers accelerate"
            ) from exc

        tokenizer = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)
        max_memory = self._max_memory()
        load_kwargs = {
            "device_map": self.device_map,
            "trust_remote_code": True,
        }
        if max_memory is not None:
            load_kwargs["max_memory"] = max_memory
        try:
            model = AutoModelForCausalLM.from_pretrained(
                str(model_dir),
                dtype=self.torch_dtype,
                **load_kwargs,
            )
        except TypeError:
            model = AutoModelForCausalLM.from_pretrained(
                str(model_dir),
                torch_dtype=self.torch_dtype,
                **load_kwargs,
            )
        model.eval()
        _MODEL_CACHE[cache_key] = (tokenizer, model)
        return tokenizer, model

    @staticmethod
    def _input_device(model: Any):
        try:
            device = model.get_input_embeddings().weight.device
            if str(device) != "meta":
                return device
        except Exception:
            pass
        for param in model.parameters():
            if str(param.device) != "meta":
                return param.device
        raise RuntimeError("Could not find a concrete Qwen input device")

    def _max_memory(self) -> dict[Any, str] | None:
        if self.device_map != "auto":
            return None
        max_memory: dict[Any, str] = {}
        if self.max_gpu_memory:
            max_memory[0] = self.max_gpu_memory
        if self.max_cpu_memory:
            max_memory["cpu"] = self.max_cpu_memory
        return max_memory or None
