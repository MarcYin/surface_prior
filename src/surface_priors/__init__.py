"""Build native-grid surface prior products and persist them as STAC/GeoTIFF assets."""

from surface_priors._version import __version__
from surface_priors.chunks import ChunkLayout, ChunkWindow
from surface_priors.composite import ChunkedCompositor, MonthlyCompositor, PriorCompositor
from surface_priors.encoding import (
    EncodingConfig,
    decode_prior,
    decode_relative_uncertainty,
    encode_prior,
    encode_relative_uncertainty,
)
from surface_priors.provider import Provider, ProviderConfig
from surface_priors.selection import (
    SceneChunkStats,
    SelectionPlan,
    SelectionPolicy,
    select,
)
from surface_priors.types import GridSpec, Observation, PriorComposite, PriorProduct

__all__ = [
    "__version__",
    "ChunkLayout",
    "ChunkWindow",
    "ChunkedCompositor",
    "EncodingConfig",
    "GridSpec",
    "MonthlyCompositor",
    "Observation",
    "PriorComposite",
    "PriorCompositor",
    "PriorProduct",
    "Provider",
    "ProviderConfig",
    "SceneChunkStats",
    "SelectionPlan",
    "SelectionPolicy",
    "decode_prior",
    "decode_relative_uncertainty",
    "encode_prior",
    "encode_relative_uncertainty",
    "select",
]
