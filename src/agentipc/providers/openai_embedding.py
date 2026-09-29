import importlib
import math

import numpy as np


def _load_openai_class() -> type:
    try:
        module = importlib.import_module("openai")
    except ModuleNotFoundError as exc:
        if exc.name != "openai":
            raise
        raise RuntimeError(
            "OpenAI-compatible embedding provider requires optional dependency; "
            "install agentipc[openai]"
        ) from exc
    return module.OpenAI


class OpenAICompatibleEmbeddingProvider:
    def __init__(
        self,
        *,
        model: str,
        dim: int,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout_sec: float = 60.0,
    ) -> None:
        if not isinstance(model, str):
            raise TypeError("model must be a str")
        if model == "":
            raise ValueError("model must be a non-empty str")

        if type(dim) is not int:
            raise TypeError("dim must be an int")
        if dim <= 0:
            raise ValueError("dim must be > 0")

        if api_key is not None and not isinstance(api_key, str):
            raise TypeError("api_key must be a str or None")
        if api_key == "":
            raise ValueError("api_key must be a non-empty str or None")

        if base_url is not None and not isinstance(base_url, str):
            raise TypeError("base_url must be a str or None")
        if base_url == "":
            raise ValueError("base_url must be a non-empty str or None")

        if isinstance(timeout_sec, bool) or not isinstance(timeout_sec, (int, float)):
            raise TypeError("timeout_sec must be an int or float")
        if not math.isfinite(timeout_sec):
            raise ValueError("timeout_sec must be finite")
        if timeout_sec <= 0:
            raise ValueError("timeout_sec must be > 0")

        self._model = model
        self._dim = dim

        openai_class = _load_openai_class()
        client_kwargs: dict[str, object] = {
            "timeout": float(timeout_sec),
            "max_retries": 0,
        }
        if api_key is not None:
            client_kwargs["api_key"] = api_key
        if base_url is not None:
            client_kwargs["base_url"] = base_url
        self._client = openai_class(**client_kwargs)

    @property
    def dim(self) -> int:
        return self._dim

    def embed(
        self,
        texts: list[str],
    ) -> np.ndarray:
        if not isinstance(texts, list):
            raise TypeError("texts must be a list[str]")
        if not all(isinstance(text, str) for text in texts):
            raise TypeError("each text must be a str")

        if not texts:
            return np.ascontiguousarray(
                np.empty((0, self._dim), dtype=np.float32),
                dtype=np.float32,
            )

        response = self._client.embeddings.create(
            model=self._model,
            input=texts,
        )

        data = response.data
        if len(data) != len(texts):
            raise ValueError(
                f"embedding response row count {len(data)} does not match "
                f"input text count {len(texts)}"
            )

        embeddings = np.zeros((len(texts), self._dim), dtype=np.float32)

        for idx, item in enumerate(data):
            embedding = item.embedding
            if not isinstance(embedding, list):
                raise TypeError(f"embedding at index {idx} must be a list")

            embedding_array = np.asarray(embedding, dtype=np.float32)

            if embedding_array.ndim != 1:
                raise ValueError(f"embedding at index {idx} must be 1-dimensional")
            if len(embedding_array) != self._dim:
                raise ValueError(
                    f"embedding at index {idx} has dimension {len(embedding_array)}, "
                    f"expected {self._dim}"
                )
            if not np.all(np.isfinite(embedding_array)):
                raise ValueError(
                    f"embedding at index {idx} contains non-finite values"
                )

            embeddings[idx] = embedding_array

        return np.ascontiguousarray(embeddings, dtype=np.float32)
