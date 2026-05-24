//! Centralised error type. Crate-level callers see one concrete error
//! enum; internal helpers use the `Result<T>` alias to avoid `anyhow`
//! creeping into the public API.

use std::path::PathBuf;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("HTTP request failed: {0}")]
    Http(#[from] reqwest::Error),

    #[error("STAC search returned status {status}: {body}")]
    StacResponse { status: u16, body: String },

    #[error("TIFF parse error: {0}")]
    Tiff(String),

    #[error("COG missing required asset {asset:?} on item {item_id:?}")]
    MissingAsset { item_id: String, asset: String },

    #[error("decompression failed: {0}")]
    Decompress(#[from] std::io::Error),

    #[error("PROJ transform failed: {0}")]
    Proj(String),

    #[error("output write failed at {path:?}: {source}")]
    Write {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    #[error("invalid configuration: {0}")]
    Config(String),

    #[error("cache I/O error at {path:?}: {source}")]
    Cache {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    #[error("JSON (de)serialisation error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("task join failed: {0}")]
    Join(#[from] tokio::task::JoinError),

    #[error("{0}")]
    Other(String),
}

impl Error {
    pub fn tiff(msg: impl Into<String>) -> Self {
        Self::Tiff(msg.into())
    }
    pub fn proj(msg: impl Into<String>) -> Self {
        Self::Proj(msg.into())
    }
    pub fn config(msg: impl Into<String>) -> Self {
        Self::Config(msg.into())
    }
    pub fn other(msg: impl Into<String>) -> Self {
        Self::Other(msg.into())
    }
}

pub type Result<T, E = Error> = std::result::Result<T, E>;
