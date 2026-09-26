import hashlib
import re
import unicodedata

import numpy as np


DEFAULT_HASH_EMBEDDING_DIM = 128


class HashEmbeddingProvider:
    def __init__(
        self,
        dim: int = DEFAULT_HASH_EMBEDDING_DIM,
    ) -> None:
        if type(dim) is not int:
            raise TypeError("dim must be an int")
        if dim <= 0:
            raise ValueError("dim must be > 0")
        self._dim = dim

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

        embeddings = np.zeros((len(texts), self.dim), dtype=np.float32)

        for row_index, text in enumerate(texts):
            normalized_text = unicodedata.normalize("NFKC", text).casefold()
            tokens = re.findall(r"\w+", normalized_text, flags=re.UNICODE)
            vector = embeddings[row_index]

            for token in tokens:
                digest = hashlib.sha256(token.encode("utf-8")).digest()
                index = int.from_bytes(
                    digest[:8],
                    byteorder="big",
                    signed=False,
                ) % self.dim
                vector[index] += np.float32(1.0)

            norm = float(np.linalg.norm(vector))
            if norm > 0.0:
                vector /= np.float32(norm)

        return np.ascontiguousarray(embeddings, dtype=np.float32)
