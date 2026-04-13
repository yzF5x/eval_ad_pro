from typing import Type
import inspect

from .base import BaseModelHandler
from .glm import GlmHandler
from .internvl import InternVLHandler
from .qwen import QwenHandler


class HandlerFactory:
    _registry: dict[str, Type[BaseModelHandler]] = {
        "qwen": QwenHandler,
        "internvl": InternVLHandler,
        "glm": GlmHandler,
    }

    @classmethod
    def infer_model_type(cls, model_path: str, model_type: str = "auto") -> str:
        if model_type and model_type.lower() != "auto":
            return model_type.lower()
        lower = model_path.lower()
        if "intern" in lower:
            return "internvl"
        if "glm" in lower:
            return "glm"
        return "qwen"

    @classmethod
    def create(cls, model_type: str, model_path: str, **kwargs) -> BaseModelHandler:
        key = cls.infer_model_type(model_path, model_type)
        if key not in cls._registry:
            raise ValueError(f"Unknown model_type: {model_type}. Supported: {list(cls._registry.keys())}")
        if key != "qwen" and "use_monkey_patch" in kwargs:
            kwargs = {k: v for k, v in kwargs.items() if k != "use_monkey_patch"}
        handler_cls = cls._registry[key]
        sig = inspect.signature(handler_cls.__init__)
        accepts_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
        if accepts_var_kw:
            filtered_kwargs = kwargs
        else:
            allowed = {name for name in sig.parameters if name not in {"self", "model_path"}}
            filtered_kwargs = {k: v for k, v in kwargs.items() if k in allowed}
        return handler_cls(model_path=model_path, **filtered_kwargs)

