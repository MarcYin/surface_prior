"""Observation sources used by surface prior providers."""

from surface_priors.sources.base import ChunkedObservationSource, ObservationSource
from surface_priors.sources.gee import (
    EdownGeeSource,
    EdownSource,
    GeeEdownSource,
    GeeProductPreset,
    gee_product_preset,
)
from surface_priors.sources.local import InMemorySource, LocalNpzSource
from surface_priors.sources.s2_gee import S2L2AGeeSource
from surface_priors.sources.stac_api import (
    CDSE_S2_ALIASES,
    EARTH_SEARCH_S2_ALIASES,
    PLANETARY_COMPUTER_S2_ALIASES,
    AssetUrlSigner,
    CdseTokenSigner,
    NoOpSigner,
    PlanetaryComputerSigner,
    StacApiSource,
    StacAssetAliases,
)

__all__ = [
    "AssetUrlSigner",
    "CDSE_S2_ALIASES",
    "CdseTokenSigner",
    "ChunkedObservationSource",
    "EARTH_SEARCH_S2_ALIASES",
    "EdownGeeSource",
    "EdownSource",
    "GeeEdownSource",
    "GeeProductPreset",
    "InMemorySource",
    "LocalNpzSource",
    "NoOpSigner",
    "ObservationSource",
    "PLANETARY_COMPUTER_S2_ALIASES",
    "PlanetaryComputerSigner",
    "S2L2AGeeSource",
    "StacApiSource",
    "StacAssetAliases",
    "gee_product_preset",
]
