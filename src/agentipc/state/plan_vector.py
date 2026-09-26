from collections.abc import Mapping
import hashlib
import json
import math

import numpy as np


PLAN_VECTOR_DIM = 64
PLAN_VECTOR_KIND = "plan_vector"


def encode_plan_vector(
    plan: Mapping[str, object],
) -> np.ndarray:
    """Encode a structured plan as deterministic feature-hashed compact state."""
    if not isinstance(plan, Mapping):
        raise TypeError("plan must be a Mapping[str, object]")
    if not plan:
        raise ValueError("plan must not be empty")

    vector = np.zeros(PLAN_VECTOR_DIM, dtype=np.float32)

    for token in _iter_feature_tokens(plan, path=[]):
        canonical = json.dumps(
            token,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        )
        digest = hashlib.sha256(canonical.encode("utf-8")).digest()
        index = int.from_bytes(
            digest[:8],
            byteorder="big",
            signed=False,
        ) % PLAN_VECTOR_DIM
        vector[index] += np.float32(1.0)

    norm = float(np.linalg.norm(vector))
    if norm == 0.0:
        raise RuntimeError("plan vector norm is unexpectedly zero")

    vector /= np.float32(norm)
    return np.ascontiguousarray(vector, dtype=np.float32)


def _iter_feature_tokens(
    value: object,
    *,
    path: list[object],
):
    if isinstance(value, Mapping):
        keys = list(value.keys())
        for key in keys:
            if not isinstance(key, str):
                raise TypeError("mapping keys must be str")
            if not key:
                raise ValueError("mapping keys must be non-empty str")

        yield ["mapping", path, len(keys)]
        for key in sorted(keys):
            child_path = [*path, ["key", key]]
            yield from _iter_feature_tokens(value[key], path=child_path)
        return

    if isinstance(value, (list, tuple)):
        yield ["sequence", path, len(value)]
        for index, item in enumerate(value):
            child_path = [*path, ["index", index]]
            yield from _iter_feature_tokens(item, path=child_path)
        return

    value_type = type(value)
    if value is None:
        type_tag = "none"
        canonical_value = None
    elif value_type is str:
        type_tag = "str"
        canonical_value = value
    elif value_type is bool:
        type_tag = "bool"
        canonical_value = value
    elif value_type is int:
        type_tag = "int"
        canonical_value = value
    elif value_type is float:
        if not math.isfinite(value):
            raise ValueError("float values must be finite")
        type_tag = "float"
        canonical_value = value
    else:
        raise TypeError(
            f"unsupported plan value type: {value_type.__name__}"
        )

    yield ["leaf", path, type_tag, canonical_value]