from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field


NonNegativeDimension = Annotated[int, Field(ge=0)]


class StateRef(BaseModel):
    model_config = ConfigDict(extra="forbid")

    uri: str = Field(min_length=1)
    kind: str = Field(min_length=1)
    shape: list[NonNegativeDimension]
    dtype: str = Field(min_length=1)
    nbytes: int = Field(ge=0)
    checksum: str = Field(min_length=1)
    transport: str = Field(min_length=1)
    summary: str


class ArtifactRef(BaseModel):
    model_config = ConfigDict(extra="forbid")

    uri: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-fA-F]{64}$")
    media_type: str = Field(min_length=1)
    size_bytes: int = Field(ge=0)
    summary: str


class MemoryRef(BaseModel):
    model_config = ConfigDict(extra="forbid")

    memory_id: str = Field(min_length=1)
    score: float = Field(allow_inf_nan=False)
    match_type: str = Field(min_length=1)
    summary: str