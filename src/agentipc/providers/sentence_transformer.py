import importlib
from typing import Any

import numpy as np


def _load_sentence_transformer_class() -> type[Any]:
    try:
        module = importlib.import_module("sentence_transformers")
    except ModuleNotFoundError as exc:
        if exc.name != "sentence_transformers":
            raise
        raise RuntimeError(
            "SentenceTransformer embedding provider requires optional dependency; "
            "install agentipc[sentence-transformers]"
        ) from exc
    return module.SentenceTransformer


class SentenceTransformerEmbeddingProvider:
    def __init__(
        self,
        model_name: str,
        *,
        device: str | None = None,
        local_files_only: bool = True,
    ) -> None:
        if not isinstance(model_name, str):
            raise TypeError("model_name must be a str")
        if model_name == "":
            raise ValueError("model_name must be a non-empty str")

        if device is not None and not isinstance(device, str):
            raise TypeError("device must be a str or None")
        if device == "":
            raise ValueError("device must be a non-empty str or None")

        if type(local_files_only) is not bool:
            raise TypeError("local_files_only must be a bool")

        self._model_name = model_name
        self._device = device
        self._local_files_only = local_files_only
        self._model: Any | None = None
        self._dim: int | None = None

    @property
    def dim(self) -> int:
        if self._dim is None:
            dimension = self._get_model().get_embedding_dimension()
            if type(dimension) is not int or dimension <= 0:
                raise ValueError("model embedding dimension must be a positive int")
            self._dim = dimension
        return self._dim

    def embed(self, texts: list[str]) -> np.ndarray:
        if not isinstance(texts, list):
            raise TypeError("texts must be a list[str]")
        if not all(isinstance(text, str) for text in texts):
            raise TypeError("each text must be a str")

        if not texts:
            return np.empty((0, self.dim), dtype=np.float32)

        model = self._get_model()
        result = model.encode(
            texts,
            convert_to_numpy=True,
            convert_to_tensor=False,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        array = np.asarray(result, dtype=np.float32)

        if array.ndim != 2:
            raise ValueError("embedding output must be 2-dimensional")
        if array.shape[0] != len(texts):
            raise ValueError("embedding output row count must match input text count")
        if array.shape[1] != self.dim:
            raise ValueError("embedding output dimension must match provider dim")
        if not np.all(np.isfinite(array)):
            raise ValueError("embedding output must contain only finite values")

        return np.ascontiguousarray(array, dtype=np.float32)

    def _get_model(self) -> Any:
        if self._model is None:
            sentence_transformer_class = _load_sentence_transformer_class()
            model_kwargs: dict[str, Any] = {
                "local_files_only": self._local_files_only,
            }
            if self._device is not None:
                model_kwargs["device"] = self._device
            self._model = sentence_transformer_class(
                self._model_name,
                **model_kwargs,
            )
        return self._model
