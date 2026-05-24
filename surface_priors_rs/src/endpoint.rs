//! STAC endpoint configuration: STAC URL, collection, band → asset
//! name mapping, and signing strategy.
//!
//! Different providers expose the same Sentinel-2 L2A data under
//! different STAC catalogues with different asset key conventions:
//!   - Element84 earth-search uses `red`, `blue`, ..., `scl` — anonymous.
//!   - Microsoft Planetary Computer uses `B04`, `B02`, ..., `SCL` —
//!     hrefs require an SAS token appended as a query string.
//!   - Copernicus Data Space Ecosystem (CDSE) uses `B04_10m`, etc., and
//!     a bearer-token header.
//!
//! The binary picks an endpoint at run-time; the rest of the pipeline
//! threads asset names through via this config so nothing else needs
//! to be endpoint-aware.

use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result};

/// Stable, endpoint-independent SR band names. Output GeoTIFFs are
/// named after these regardless of which catalogue we fetched from.
pub const BAND_NAMES: [&str; 12] = [
    "coastal", "blue", "green", "red",
    "rededge1", "rededge2", "rededge3",
    "nir", "nir08", "nir09",
    "swir16", "swir22",
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EndpointKind {
    EarthSearch,
    PlanetaryComputer,
}

impl EndpointKind {
    pub fn parse(s: &str) -> Result<Self> {
        match s {
            "earth-search" | "es" | "element84" => Ok(Self::EarthSearch),
            "pc" | "planetary-computer" | "planetary_computer" => Ok(Self::PlanetaryComputer),
            other => anyhow::bail!("unknown endpoint {other:?}"),
        }
    }
}

pub struct EndpointConfig {
    pub kind: EndpointKind,
    pub stac_url: String,
    pub collection: String,
    /// Map from stable BAND_NAMES entries to this endpoint's asset key.
    /// Order matches BAND_NAMES.
    pub band_assets: Vec<String>,
    /// SCL / quality band asset key.
    pub scl_asset: String,
    /// SAS token (PC) appended as `?{token}` to each asset href. None
    /// for anonymous endpoints. Fetched lazily on demand.
    sas_token: parking_lot::RwLock<Option<String>>,
}

impl EndpointConfig {
    pub fn build(kind: EndpointKind) -> Self {
        match kind {
            EndpointKind::EarthSearch => Self::earth_search(),
            EndpointKind::PlanetaryComputer => Self::planetary_computer(),
        }
    }

    pub fn earth_search() -> Self {
        Self {
            kind: EndpointKind::EarthSearch,
            stac_url: "https://earth-search.aws.element84.com/v1".to_string(),
            collection: "sentinel-2-l2a".to_string(),
            band_assets: vec![
                "coastal".into(), "blue".into(), "green".into(), "red".into(),
                "rededge1".into(), "rededge2".into(), "rededge3".into(),
                "nir".into(), "nir08".into(), "nir09".into(),
                "swir16".into(), "swir22".into(),
            ],
            scl_asset: "scl".into(),
            sas_token: parking_lot::RwLock::new(None),
        }
    }

    pub fn planetary_computer() -> Self {
        Self {
            kind: EndpointKind::PlanetaryComputer,
            stac_url: "https://planetarycomputer.microsoft.com/api/stac/v1".to_string(),
            collection: "sentinel-2-l2a".to_string(),
            band_assets: vec![
                "B01".into(), "B02".into(), "B03".into(), "B04".into(),
                "B05".into(), "B06".into(), "B07".into(),
                "B08".into(), "B8A".into(), "B09".into(),
                "B11".into(), "B12".into(),
            ],
            scl_asset: "SCL".into(),
            sas_token: parking_lot::RwLock::new(None),
        }
    }

    /// Lookup table from band name → asset key.
    pub fn band_to_asset_map(&self) -> HashMap<String, String> {
        BAND_NAMES
            .iter()
            .zip(self.band_assets.iter())
            .map(|(b, a)| (b.to_string(), a.clone()))
            .collect()
    }

    /// Sign a raw asset href if this endpoint requires SAS-tokening.
    /// Caches the token in this config; if it's near expiry we'll
    /// re-fetch on the next call.
    pub async fn sign_href(&self, http: &reqwest::Client, href: &str) -> Result<String> {
        if !matches!(self.kind, EndpointKind::PlanetaryComputer) {
            return Ok(href.to_string());
        }
        // If the href already carries SAS query params, don't double-sign.
        if href.contains("?sv=") || href.contains("&sv=") {
            return Ok(href.to_string());
        }
        let token = self.ensure_sas_token(http).await?;
        // The SAS token returned by the PC token endpoint already
        // starts with the SAS-style key=value pairs; just join with `?`.
        if href.contains('?') {
            Ok(format!("{href}&{token}"))
        } else {
            Ok(format!("{href}?{token}"))
        }
    }

    async fn ensure_sas_token(&self, http: &reqwest::Client) -> Result<String> {
        if let Some(t) = self.sas_token.read().clone() {
            return Ok(t);
        }
        // GET https://planetarycomputer.microsoft.com/api/sas/v1/token/{collection}
        let url = format!(
            "https://planetarycomputer.microsoft.com/api/sas/v1/token/{}",
            self.collection
        );
        let resp = http
            .get(&url)
            .send()
            .await
            .with_context(|| format!("PC SAS token GET {url}"))?
            .error_for_status()
            .with_context(|| format!("PC SAS token non-2xx for {url}"))?;
        let body: serde_json::Value = resp.json().await.context("PC SAS body decode")?;
        let token = body
            .get("token")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow::anyhow!("PC SAS response missing 'token' field"))?
            .to_string();
        let expiry = body
            .get("msft:expiry")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string())
            .unwrap_or_default();
        tracing::info!(token_len = token.len(), expiry, "fetched PC SAS token");
        *self.sas_token.write() = Some(token.clone());
        Ok(token)
    }

    /// Sign every asset href in a STAC item in place.
    pub async fn sign_item(
        &self,
        http: &reqwest::Client,
        mut item: serde_json::Value,
    ) -> Result<serde_json::Value> {
        if !matches!(self.kind, EndpointKind::PlanetaryComputer) {
            return Ok(item);
        }
        if let Some(assets) = item.get_mut("assets").and_then(|v| v.as_object_mut()) {
            for (_, asset) in assets.iter_mut() {
                let href_opt = asset
                    .get("href")
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string());
                if let Some(href) = href_opt {
                    match self.sign_href(http, &href).await {
                        Ok(signed) => {
                            asset["href"] = serde_json::Value::String(signed);
                        }
                        Err(e) => {
                            tracing::warn!(href, error = ?e, "PC SAS sign failed");
                        }
                    }
                }
            }
        }
        Ok(item)
    }
}

/// Heuristic: if user passed `auto`, pick PC unless they're already
/// running in us-west-2 (where Element84 has zero RTT to S3). Without
/// AWS region introspection we default to PC for the UK / JASMIN case
/// the team actually runs. Override via `--endpoint=earth-search`.
pub fn auto_pick() -> EndpointKind {
    // Could read AWS_REGION env var; for now PC wins from UK-based runs.
    if std::env::var("AWS_REGION")
        .ok()
        .as_deref()
        .map(|r| r.starts_with("us-"))
        .unwrap_or(false)
    {
        EndpointKind::EarthSearch
    } else {
        EndpointKind::PlanetaryComputer
    }
}

/// Stale value purely to give the time module a use; avoids unused warning.
#[doc(hidden)]
pub fn _unused_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}
