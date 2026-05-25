//! Persistent disk cache for STAC search results, tile partition, and
//! per-scene scout outputs. Mirrors `surface_priors.sources.stac_cache`
//! in Python so a cache directory can be shared between the two
//! implementations during migration.

use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};

const SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Clone)]
pub struct DiskCache {
    pub root: PathBuf,
}

#[derive(Serialize, Deserialize)]
struct Envelope<T> {
    schema: u32,
    data: T,
}

impl DiskCache {
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    pub fn search_key(
        &self,
        stac_url: &str,
        collection: &str,
        bbox: [f64; 4],
        datetime: &str,
        max_cloud_cover: Option<f64>,
    ) -> String {
        use sha1::{Digest, Sha1};
        let mut h = Sha1::new();
        h.update(b"schema:1\0");
        h.update(stac_url.as_bytes());
        h.update(b"\0");
        h.update(collection.as_bytes());
        h.update(b"\0");
        for v in bbox {
            h.update(v.to_le_bytes());
        }
        h.update(datetime.as_bytes());
        h.update(b"\0");
        if let Some(c) = max_cloud_cover {
            h.update(c.to_le_bytes());
        }
        hex::encode(h.finalize())
    }

    pub fn partition_key(&self, scenes_signature: &str, grid_signature: &str, layout_chunk_size: u32) -> String {
        use sha1::{Digest, Sha1};
        let mut h = Sha1::new();
        h.update(b"schema:1\0");
        h.update(scenes_signature.as_bytes());
        h.update(b"\0");
        h.update(grid_signature.as_bytes());
        h.update(b"\0");
        h.update(layout_chunk_size.to_le_bytes());
        hex::encode(h.finalize())
    }

    pub fn scout_key(
        &self,
        stac_url: &str,
        collection: &str,
        item_id: &str,
        grid_signature: &str,
        layout_chunk_size: u32,
        scout_factor: u32,
    ) -> String {
        use sha1::{Digest, Sha1};
        let mut h = Sha1::new();
        h.update(b"schema:1\0scout\0");
        h.update(stac_url.as_bytes());
        h.update(b"\0");
        h.update(collection.as_bytes());
        h.update(b"\0");
        h.update(item_id.as_bytes());
        h.update(b"\0");
        h.update(grid_signature.as_bytes());
        h.update(b"\0");
        h.update(layout_chunk_size.to_le_bytes());
        h.update(scout_factor.to_le_bytes());
        hex::encode(h.finalize())
    }

    pub fn load_search(&self, key: &str) -> Result<Option<Vec<serde_json::Value>>> {
        self.load_at(self.path("search", key))
    }

    pub fn store_search(&self, key: &str, items: &[serde_json::Value]) -> Result<()> {
        self.store_slice(self.path("search", key), items)
    }

    pub fn load_partition(
        &self,
        key: &str,
    ) -> Result<Option<crate::tile_classification::TilePartition>> {
        self.load_at(self.path("partition", key))
    }

    pub fn store_partition(
        &self,
        key: &str,
        partition: &crate::tile_classification::TilePartition,
    ) -> Result<()> {
        self.store_at(self.path("partition", key), partition)
    }

    pub fn load_scout(&self, key: &str) -> Result<Option<Vec<crate::pipeline::SceneChunkStat>>> {
        self.load_at(self.path("scout", key))
    }

    pub fn store_scout(
        &self,
        key: &str,
        stats: &[crate::pipeline::SceneChunkStat],
    ) -> Result<()> {
        self.store_slice(self.path("scout", key), stats)
    }

    /// Stable key for caching a COG's parsed header. Hashes only the
    /// URL path (scheme+host+path), so SAS-token rotation on PC
    /// hrefs doesn't bust the cache.
    pub fn cog_profile_key(&self, url: &str) -> String {
        use sha1::{Digest, Sha1};
        let path_only = url.split_once('?').map(|(p, _)| p).unwrap_or(url);
        let mut h = Sha1::new();
        h.update(b"schema:1\0cog\0");
        h.update(path_only.as_bytes());
        hex::encode(h.finalize())
    }

    pub fn load_cog_profile(&self, key: &str) -> Result<Option<crate::cog::CogProfile>> {
        self.load_at(self.path("cog", key))
    }

    pub fn store_cog_profile(
        &self,
        key: &str,
        profile: &crate::cog::CogProfile,
    ) -> Result<()> {
        self.store_at(self.path("cog", key), profile)
    }

    fn store_slice<T: Serialize + Clone>(&self, path: PathBuf, slice: &[T]) -> Result<()> {
        let owned: Vec<T> = slice.to_vec();
        self.store_at(path, &owned)
    }

    fn path(&self, kind: &str, key: &str) -> PathBuf {
        self.root.join(kind).join(format!("{key}.json"))
    }

    fn load_at<T: for<'de> Deserialize<'de>>(&self, path: PathBuf) -> Result<Option<T>> {
        if !path.exists() {
            return Ok(None);
        }
        let text = match fs::read_to_string(&path) {
            Ok(s) => s,
            Err(_) => return Ok(None),
        };
        let env: Envelope<T> = match serde_json::from_str(&text) {
            Ok(e) => e,
            Err(_) => return Ok(None),
        };
        if env.schema != SCHEMA_VERSION {
            return Ok(None);
        }
        Ok(Some(env.data))
    }

    fn store_at<T: Serialize>(&self, path: PathBuf, data: &T) -> Result<()> {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).map_err(|e| Error::Cache {
                path: parent.to_path_buf(),
                source: e,
            })?;
        }
        let envelope = Envelope {
            schema: SCHEMA_VERSION,
            data,
        };
        let json = serde_json::to_string(&envelope)?;
        // Atomic write via temp file + rename.
        let tmp = path.with_extension("json.tmp");
        let mut f = fs::File::create(&tmp).map_err(|e| Error::Cache {
            path: tmp.clone(),
            source: e,
        })?;
        f.write_all(json.as_bytes()).map_err(|e| Error::Cache {
            path: tmp.clone(),
            source: e,
        })?;
        f.sync_all().ok();
        drop(f);
        fs::rename(&tmp, &path).map_err(|e| Error::Cache {
            path: path.clone(),
            source: e,
        })?;
        Ok(())
    }
}

/// Convenience: convert "string-or-Path-or-None" into an `Option<DiskCache>`.
pub fn resolve_disk_cache(arg: Option<&str>) -> Option<DiskCache> {
    arg.map(|s| DiskCache::new(s))
}

/// Stable grid signature for cache keys.
pub fn grid_signature(bounds: [f64; 4], epsg: u32, resolution: f64, width: u32, height: u32) -> String {
    use sha1::{Digest, Sha1};
    let mut h = Sha1::new();
    for v in bounds {
        h.update(v.to_le_bytes());
    }
    h.update(epsg.to_le_bytes());
    h.update(resolution.to_le_bytes());
    h.update(width.to_le_bytes());
    h.update(height.to_le_bytes());
    hex::encode(h.finalize())[..16].to_string()
}

#[allow(dead_code)]
fn _unused_path() -> &'static Path {
    Path::new("/")
}
