//! Production Rust port of the surface_priors monthly composite pipeline.
//!
//! Modules in dependency order:
//!   - [`error`]: typed errors via thiserror, shared `Result` alias.
//!   - [`cog`]: HTTP-backed COG reader (TIFF/IFD, overviews, range tiles, DEFLATE + predictor=2).
//!   - [`projx`]: PROJ-backed coordinate transforms and resampling primitives.
//!   - [`grid`]: AOI grid, chunk layout, COG-to-grid window math.
//!   - [`tile_classification`]: geometry-based exclusive-coverage classifier.
//!   - [`stac`]: async STAC search client.
//!   - [`disk_cache`]: persistent JSON cache for search / scout / partition.
//!   - [`pipeline`]: scout, select_top_k, fetch_band, fetch_quality, compose.
//!   - [`writer`]: tiled DEFLATE GeoTIFF output.

pub mod cog;
pub mod disk_cache;
pub mod endpoint;
pub mod error;
pub mod grid;
pub mod pipeline;
pub mod projx;
pub mod signer;
pub mod stac;
pub mod tile_classification;
pub mod writer;

#[cfg(feature = "python")]
pub mod py;

pub use error::{Error, Result};
