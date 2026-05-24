//! Asset URL signing for STAC endpoints with finite-TTL access tokens.
//!
//! Element84's `sentinel-cogs` is anonymously readable, so [`NoOpSigner`]
//! is the default. Planetary Computer and CDSE require their own
//! signing flows; they're left as thin async traits to keep the
//! production crate provider-agnostic.

use async_trait::async_trait;
use std::collections::HashMap;

use crate::error::{Error, Result};

#[async_trait]
pub trait AssetUrlSigner: Send + Sync {
    /// Sign every asset href in a raw STAC item. Returns the item dict
    /// with rewritten hrefs (token / SAS appended, or whatever the
    /// endpoint requires). Implementations may mutate the input
    /// in-place; the trait returns a fresh `serde_json::Value` so
    /// callers can keep the unsigned original for disk caching.
    async fn sign_item(&self, item: serde_json::Value) -> Result<serde_json::Value>;
}

/// Anonymous endpoints (Element84 sentinel-cogs).
pub struct NoOpSigner;

#[async_trait]
impl AssetUrlSigner for NoOpSigner {
    async fn sign_item(&self, item: serde_json::Value) -> Result<serde_json::Value> {
        Ok(item)
    }
}

/// Planetary Computer SAS-token signer. Calls the Planetary Computer
/// signing endpoint for each asset href, returning the SAS-signed URL.
///
/// SAS tokens expire (~1h); signing must happen per build, never
/// cached to disk. The `subscription_key` is optional but raises rate
/// limits.
pub struct PlanetaryComputerSigner {
    pub subscription_key: Option<String>,
    http: reqwest::Client,
}

impl PlanetaryComputerSigner {
    pub fn new(subscription_key: Option<String>, http: reqwest::Client) -> Self {
        Self {
            subscription_key,
            http,
        }
    }

    async fn sign_href(&self, href: &str) -> Result<String> {
        let url = format!(
            "https://planetarycomputer.microsoft.com/api/sas/v1/sign?href={}",
            urlencoding::encode(href)
        );
        let mut req = self.http.get(&url);
        if let Some(key) = &self.subscription_key {
            req = req.header("Ocp-Apim-Subscription-Key", key);
        }
        let resp = req.send().await?.error_for_status()?;
        let signed: serde_json::Value = resp.json().await?;
        let signed_href = signed
            .get("href")
            .and_then(|v| v.as_str())
            .ok_or_else(|| Error::other("PC sign endpoint returned no href"))?
            .to_string();
        Ok(signed_href)
    }
}

#[async_trait]
impl AssetUrlSigner for PlanetaryComputerSigner {
    async fn sign_item(&self, mut item: serde_json::Value) -> Result<serde_json::Value> {
        // Replace each asset's href with the SAS-signed URL.
        if let Some(assets) = item.get_mut("assets").and_then(|v| v.as_object_mut()) {
            // Sign each asset; small N (typically 12-15) so we do
            // sequential calls — the SAS endpoint enforces a per-key
            // rate limit and burst parallelism doesn't help.
            for (_, asset) in assets.iter_mut() {
                let href = asset
                    .get("href")
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string());
                if let Some(href) = href {
                    match self.sign_href(&href).await {
                        Ok(signed) => {
                            asset["href"] = serde_json::Value::String(signed);
                        }
                        Err(e) => {
                            tracing::warn!(href, error = ?e, "PC sign failed; using unsigned href");
                        }
                    }
                }
            }
        }
        Ok(item)
    }
}

/// CDSE bearer-token signer. Attaches the token via a header file
/// referenced by `GDAL_HTTP_HEADER_FILE`; since this Rust port uses
/// `reqwest` (not GDAL) we inject the `Authorization` header into the
/// HTTP client itself. The token is set globally on the client outside
/// this signer.
pub struct CdseTokenSigner {
    pub token: String,
}

#[async_trait]
impl AssetUrlSigner for CdseTokenSigner {
    async fn sign_item(&self, item: serde_json::Value) -> Result<serde_json::Value> {
        // Hrefs aren't rewritten; the auth header is set on the HTTP
        // client during construction (see bin spx_build.rs).
        // This signer exists so the protocol matches the others and
        // a future CDSE caller can pass the token through.
        let _ = &self.token;
        Ok(item)
    }
}

/// Sign each item in a batch; preserves order.
pub async fn sign_batch(
    signer: &dyn AssetUrlSigner,
    items: Vec<serde_json::Value>,
) -> Result<Vec<serde_json::Value>> {
    let mut out = Vec::with_capacity(items.len());
    for item in items {
        out.push(signer.sign_item(item).await?);
    }
    Ok(out)
}

/// Construct a default signer based on a CRS short name.
/// `"earth-search" | "pc" | "cdse"`.
pub fn signer_for_endpoint(
    name: &str,
    http: reqwest::Client,
    pc_key: Option<String>,
    cdse_token: Option<String>,
) -> Result<Box<dyn AssetUrlSigner>> {
    match name {
        "earth-search" | "" => Ok(Box::new(NoOpSigner)),
        "pc" | "planetary-computer" => Ok(Box::new(PlanetaryComputerSigner::new(pc_key, http))),
        "cdse" => {
            let token = cdse_token.ok_or_else(|| Error::config("CDSE signer requires --cdse-token"))?;
            Ok(Box::new(CdseTokenSigner { token }))
        }
        other => Err(Error::config(format!("unknown endpoint signer: {other}"))),
    }
}

/// Hash a `HashMap<String, String>` for testing.
#[doc(hidden)]
pub fn debug_map_eq(a: &HashMap<String, String>, b: &HashMap<String, String>) -> bool {
    a.len() == b.len() && a.iter().all(|(k, v)| b.get(k) == Some(v))
}
