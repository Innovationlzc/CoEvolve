"""Local Qwen text-generation client.

The import of transformers is intentionally lazy so the package can still be
imported in lightweight tests. Actual Qwen-backed experiments fail loudly if
the model runtime or local checkpoint is missing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import time
from typing import Any

from praxis_coop.configs import DEFAULT_MULTIMODAL_MODEL, PHASE1_MODELS


_MODEL_CACHE: dict[tuple[str, ...], tuple[Any, Any]] = {}
_VISION_MODEL_CACHE: dict[tuple[str, ...], tuple[Any, Any]] = {}


def clear_text_model_cache() -> None:
    """Release cached text models before loading a large vision model."""

    _MODEL_CACHE.clear()
    _empty_cuda_cache()


def clear_vision_model_cache() -> None:
    """Release cached vision models before returning to text generation."""

    _VISION_MODEL_CACHE.clear()
    _empty_cuda_cache()


def _empty_cuda_cache() -> None:
    try:
        import gc
        import torch
    except ImportError:
        return
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


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
    call_records: list[dict[str, Any]] = field(default_factory=list, init=False)

    @property
    def model_dir(self) -> Path:
        if self.model_name not in PHASE1_MODELS:
            known = ", ".join(sorted(PHASE1_MODELS))
            raise ValueError(f"Unknown Qwen model {self.model_name!r}. Known models: {known}")
        return PHASE1_MODELS[self.model_name].local_dir

    def generate(self, *, system: str, user: str) -> str:
        started = time.time()
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
        else:
            generation_kwargs.update(temperature=None, top_p=None, top_k=None)
        generated = model.generate(**inputs, **generation_kwargs)
        new_tokens = generated[0][inputs.input_ids.shape[-1] :]
        result = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        self.call_records.append(
            {
                "model_name": self.model_name,
                "kind": "text_generation",
                "started_at_unix": started,
                "latency_s": time.time() - started,
                "input_tokens": int(inputs.input_ids.numel()),
                "output_tokens": int(new_tokens.numel()),
            }
        )
        return result

    def usage_summary(self) -> dict[str, Any]:
        return _usage_summary(self.call_records)

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


@dataclass
class QwenVisionClient:
    """Shared local Qwen-VL client for text planning and visual evaluation."""

    model_name: str = DEFAULT_MULTIMODAL_MODEL
    device_map: str = "auto"
    torch_dtype: str = "auto"
    max_new_tokens: int = 768
    temperature: float = 0.0
    top_p: float = 0.9
    text_max_new_tokens: int | None = None
    text_temperature: float | None = None
    text_top_p: float | None = None
    max_gpu_memory: str | None = None
    max_cpu_memory: str | None = None
    quantization: str = "none"
    call_records: list[dict[str, Any]] = field(default_factory=list, init=False)

    @property
    def model_dir(self) -> Path:
        if self.model_name not in PHASE1_MODELS:
            known = ", ".join(sorted(PHASE1_MODELS))
            raise ValueError(f"Unknown Qwen-VL model {self.model_name!r}. Known models: {known}")
        return PHASE1_MODELS[self.model_name].local_dir

    def generate(self, *, system: str, user: str) -> str:
        """Run a text-only orchestration/evolution turn on the same VL model."""

        messages = [
            {"role": "system", "content": [{"type": "text", "text": system}]},
            {"role": "user", "content": [{"type": "text", "text": user}]},
        ]
        return self._generate_messages(
            messages=messages,
            kind="vision_text_generation",
            max_new_tokens=(
                self.max_new_tokens
                if self.text_max_new_tokens is None
                else self.text_max_new_tokens
            ),
            temperature=(
                self.temperature
                if self.text_temperature is None
                else self.text_temperature
            ),
            top_p=self.top_p if self.text_top_p is None else self.text_top_p,
        )

    def analyze_image(self, *, system: str, user: str, image_path: str | Path) -> str:
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"visual frame does not exist: {image_path}")
        messages = [
            {"role": "system", "content": [{"type": "text", "text": system}]},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": str(image_path.resolve())},
                    {"type": "text", "text": user},
                ],
            },
        ]
        return self._generate_messages(
            messages=messages,
            kind="vision_generation",
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            image_path=image_path,
        )

    def _generate_messages(
        self,
        *,
        messages: list[dict[str, Any]],
        kind: str,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        image_path: Path | None = None,
    ) -> str:
        started = time.time()
        processor, model = self._load()
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = inputs.to(QwenTextClient._input_device(model))
        generation_kwargs = {
            "max_new_tokens": int(max_new_tokens),
            "do_sample": temperature > 0.0,
        }
        if temperature > 0.0:
            generation_kwargs["temperature"] = float(temperature)
            generation_kwargs["top_p"] = float(top_p)
        else:
            generation_kwargs.update(temperature=None, top_p=None, top_k=None)
        generated = model.generate(**inputs, **generation_kwargs)
        trimmed = [output[len(input_ids) :] for input_ids, output in zip(inputs.input_ids, generated)]
        result = processor.batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()
        self.call_records.append(
            {
                "model_name": self.model_name,
                "kind": kind,
                "started_at_unix": started,
                "latency_s": time.time() - started,
                "input_tokens": int(inputs.input_ids.numel()),
                "output_tokens": int(sum(output.numel() for output in trimmed)),
                "image_path": "" if image_path is None else str(image_path),
            }
        )
        return result

    def usage_summary(self) -> dict[str, Any]:
        return _usage_summary(self.call_records)

    def preload(self) -> None:
        """Load processor/model once so later visual checks reuse the cache."""

        self._load()

    def _load(self):
        model_dir = self.model_dir
        cache_key = (
            str(model_dir.resolve()),
            str(self.device_map),
            str(self.torch_dtype),
            str(self.max_gpu_memory or ""),
            str(self.max_cpu_memory or ""),
            str(self.quantization),
        )
        if cache_key in _VISION_MODEL_CACHE:
            return _VISION_MODEL_CACHE[cache_key]
        if not model_dir.exists():
            raise FileNotFoundError(
                f"Qwen-VL model directory does not exist: {model_dir}. "
                f"Download it first with: python scripts/download_phase1_models.py --model {self.model_name}"
            )
        try:
            from transformers import AutoModelForImageTextToText, AutoProcessor
        except ImportError as exc:
            raise ImportError(
                "Qwen-VL verifier requires transformers. Install the Isaac/Praxis Python dependencies "
                "or run: pip install transformers accelerate pillow"
            ) from exc

        processor = AutoProcessor.from_pretrained(str(model_dir), trust_remote_code=True)
        max_memory = self._max_memory()
        load_kwargs = {
            "device_map": self.device_map,
            "trust_remote_code": True,
        }
        quantization_config = self._quantization_config()
        if quantization_config is not None:
            load_kwargs["quantization_config"] = quantization_config
        if max_memory is not None:
            load_kwargs["max_memory"] = max_memory
        try:
            model = AutoModelForImageTextToText.from_pretrained(
                str(model_dir),
                dtype=self.torch_dtype,
                **load_kwargs,
            )
        except TypeError:
            model = AutoModelForImageTextToText.from_pretrained(
                str(model_dir),
                torch_dtype=self.torch_dtype,
                **load_kwargs,
            )
        model.eval()
        _VISION_MODEL_CACHE[cache_key] = (processor, model)
        return processor, model

    def _quantization_config(self):
        mode = str(self.quantization).strip().lower()
        if mode in {"", "none"}:
            return None
        if mode not in {"8bit", "4bit"}:
            raise ValueError(f"unsupported Qwen-VL quantization mode: {self.quantization!r}")
        try:
            import torch
            from transformers import BitsAndBytesConfig
        except ImportError as exc:
            raise ImportError(
                "Quantized Qwen-VL requires bitsandbytes. Install it with: "
                "python -m pip install 'bitsandbytes>=0.46,<0.50'"
            ) from exc
        if mode == "8bit":
            return BitsAndBytesConfig(load_in_8bit=True)
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    def _max_memory(self) -> dict[Any, str] | None:
        if self.device_map != "auto":
            return None
        max_memory: dict[Any, str] = {}
        if self.max_gpu_memory:
            max_memory[0] = self.max_gpu_memory
        if self.max_cpu_memory:
            max_memory["cpu"] = self.max_cpu_memory
        return max_memory or None


def _usage_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "call_count": len(records),
        "input_tokens": sum(int(item.get("input_tokens", 0)) for item in records),
        "output_tokens": sum(int(item.get("output_tokens", 0)) for item in records),
        "latency_s": sum(float(item.get("latency_s", 0.0)) for item in records),
        "calls": list(records),
    }
