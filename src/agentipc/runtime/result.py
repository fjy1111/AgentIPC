from __future__ import annotations

from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field, StrictBool


NonEmptyStrictStr = Annotated[
    str,
    Field(
        min_length=1,
        strict=True,
    ),
]


class RunResult(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
    )

    task_id: NonEmptyStrictStr
    success: StrictBool
    answer: Annotated[
        str,
        Field(strict=True),
    ]

    error: dict[str, Any] | None = None
    metrics: dict[str, Any]

    trace_path: (
        Annotated[
            str,
            Field(strict=True),
        ]
        | None
    ) = None
