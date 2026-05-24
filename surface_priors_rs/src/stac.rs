//! Async STAC search client for the Element84 earth-search endpoint.
//!
//! We only consume the fields needed downstream — item id, datetime,
//! geometry, mgrs tile, asset hrefs — and skip the rest. The endpoint
//! paginates via a `next` link in `links`; we follow it until exhausted.
//! `query` supports `eo:cloud_cover` server-side filtering so most-cloudy
//! scenes never hit the wire.

use anyhow::{anyhow, Context, Result};
use serde::Deserialize;
use serde_json::json;
use std::collections::HashMap;

pub const EARTH_SEARCH_URL: &str = "https://earth-search.aws.element84.com/v1";

#[derive(Debug, Clone)]
pub struct StacItem {
    pub id: String,
    pub datetime: String,
    pub mgrs_tile: String,
    pub geometry: serde_json::Value,
    pub assets: HashMap<String, String>,
    pub properties: serde_json::Value,
}

#[derive(Deserialize)]
struct Link {
    rel: String,
    href: String,
    #[serde(default)]
    method: Option<String>,
    #[serde(default)]
    body: Option<serde_json::Value>,
}

#[derive(Deserialize)]
struct SearchResponse {
    features: Vec<serde_json::Value>,
    #[serde(default)]
    links: Vec<Link>,
}

pub struct StacClient {
    pub base_url: String,
    pub collection: String,
    pub bbox: [f64; 4],
    pub datetime: String,
    pub max_cloud_cover: Option<f64>,
    http: reqwest::Client,
}

impl StacClient {
    pub fn new(
        base_url: impl Into<String>,
        collection: impl Into<String>,
        bbox: [f64; 4],
        datetime: impl Into<String>,
        max_cloud_cover: Option<f64>,
    ) -> Result<Self> {
        let http = reqwest::Client::builder()
            .gzip(true)
            .http2_adaptive_window(true)
            .pool_max_idle_per_host(64)
            .tcp_keepalive(std::time::Duration::from_secs(60))
            .build()
            .context("build reqwest client")?;
        Ok(Self {
            base_url: base_url.into(),
            collection: collection.into(),
            bbox,
            datetime: datetime.into(),
            max_cloud_cover,
            http,
        })
    }

    /// Issue the initial `/search` POST and follow `next` links.
    /// Returns the raw STAC item dicts so callers can persist them
    /// unchanged and apply signing on each load.
    pub async fn search_raw(&self) -> Result<Vec<serde_json::Value>> {
        let mut url = format!("{}/search", self.base_url.trim_end_matches('/'));
        let datetime = normalise_datetime(&self.datetime);
        let mut body = json!({
            "collections": [self.collection],
            "bbox": self.bbox,
            "datetime": datetime,
            "limit": 100,
        });
        if let Some(cc) = self.max_cloud_cover {
            if cc < 100.0 {
                body["query"] = json!({"eo:cloud_cover": {"lt": cc}});
            }
        }

        let mut features: Vec<serde_json::Value> = Vec::new();
        let mut method = "POST".to_string();
        let mut request_body = Some(body);
        loop {
            let req = if method == "POST" {
                self.http.post(&url).json(request_body.as_ref().unwrap())
            } else {
                self.http.get(&url)
            };
            let resp = req
                .send()
                .await
                .with_context(|| format!("STAC search POST {url}"))?
                .error_for_status()
                .context("STAC search non-2xx")?;
            let parsed: SearchResponse = resp.json().await.context("decode STAC response")?;
            features.extend(parsed.features);
            let Some(next) = parsed.links.iter().find(|l| l.rel == "next") else { break };
            url = next.href.clone();
            method = next.method.clone().unwrap_or_else(|| "GET".to_string());
            request_body = next.body.clone();
        }
        Ok(features)
    }

    /// Convert raw STAC items to `StacItem` (parses ids, datetime,
    /// asset hrefs, geometry, MGRS code). Stable datetime+id ordering.
    pub fn items_from_raw(features: Vec<serde_json::Value>) -> Vec<StacItem> {
        let mut out: Vec<StacItem> = Vec::with_capacity(features.len());
        for feat in features {
            let id = feat.get("id").and_then(|v| v.as_str()).unwrap_or("").to_string();
            let geometry = feat.get("geometry").cloned().unwrap_or(serde_json::Value::Null);
            let props = feat.get("properties").cloned().unwrap_or(serde_json::Value::Null);
            let datetime = props.get("datetime").and_then(|v| v.as_str()).unwrap_or("").to_string();
            let mgrs_tile = mgrs_from_props_or_id(&props, &id);
            let mut assets: std::collections::HashMap<String, String> = std::collections::HashMap::new();
            if let Some(map) = feat.get("assets").and_then(|v| v.as_object()) {
                for (key, val) in map {
                    if let Some(href) = val.get("href").and_then(|v| v.as_str()) {
                        assets.insert(key.clone(), href.to_string());
                    }
                }
            }
            out.push(StacItem {
                id,
                datetime,
                mgrs_tile,
                geometry,
                assets,
                properties: props,
            });
        }
        out.sort_by(|a, b| a.datetime.cmp(&b.datetime).then(a.id.cmp(&b.id)));
        out
    }

    /// Issue the initial `/search` POST and follow `next` links.
    pub async fn search(&self) -> Result<Vec<StacItem>> {
        let mut url = format!("{}/search", self.base_url.trim_end_matches('/'));
        let datetime = normalise_datetime(&self.datetime);
        let mut body = json!({
            "collections": [self.collection],
            "bbox": self.bbox,
            "datetime": datetime,
            "limit": 100,
        });
        if let Some(cc) = self.max_cloud_cover {
            if cc < 100.0 {
                body["query"] = json!({"eo:cloud_cover": {"lt": cc}});
            }
        }

        let mut all_features: Vec<serde_json::Value> = Vec::new();
        let mut method = "POST".to_string();
        let mut request_body = Some(body);

        loop {
            let req = if method == "POST" {
                self.http.post(&url).json(request_body.as_ref().unwrap())
            } else {
                self.http.get(&url)
            };
            let resp = req
                .send()
                .await
                .with_context(|| format!("STAC search POST {url}"))?
                .error_for_status()
                .context("STAC search non-2xx")?;
            let parsed: SearchResponse = resp.json().await.context("decode STAC response")?;
            let features = parsed.features;
            all_features.extend(features);

            let next = parsed.links.iter().find(|l| l.rel == "next");
            let Some(next) = next else { break };
            url = next.href.clone();
            method = next.method.clone().unwrap_or_else(|| "GET".to_string());
            request_body = next.body.clone();
        }

        let mut out = Vec::with_capacity(all_features.len());
        for feat in all_features {
            let id = feat
                .get("id")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            let geometry = feat
                .get("geometry")
                .cloned()
                .unwrap_or(serde_json::Value::Null);
            let props = feat
                .get("properties")
                .cloned()
                .unwrap_or(serde_json::Value::Null);
            let datetime = props
                .get("datetime")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            let mgrs_tile = mgrs_from_props_or_id(&props, &id);
            let mut assets: HashMap<String, String> = HashMap::new();
            if let Some(map) = feat.get("assets").and_then(|v| v.as_object()) {
                for (key, val) in map {
                    if let Some(href) = val.get("href").and_then(|v| v.as_str()) {
                        assets.insert(key.clone(), href.to_string());
                    }
                }
            }
            out.push(StacItem {
                id,
                datetime,
                mgrs_tile,
                geometry,
                assets,
                properties: props,
            });
        }
        // Stable ordering matches the Python implementation.
        out.sort_by(|a, b| a.datetime.cmp(&b.datetime).then(a.id.cmp(&b.id)));
        Ok(out)
    }
}

fn mgrs_from_props_or_id(props: &serde_json::Value, id: &str) -> String {
    for key in ["s2:mgrs_tile", "mgrs:tile"] {
        if let Some(v) = props.get(key).and_then(|v| v.as_str()) {
            return v.to_string();
        }
    }
    if let Some(grid) = props.get("grid:code").and_then(|v| v.as_str()) {
        if grid.to_ascii_uppercase().starts_with("MGRS-") {
            return grid[5..].to_string();
        }
    }
    // Parse from item id like "S2B_36RTU_20240705_0_L2A".
    let parts: Vec<&str> = id.split('_').collect();
    if parts.len() >= 2 && parts[0].starts_with("S2") {
        let tile = parts[1];
        if tile.len() == 5
            && tile[0..2].chars().all(|c| c.is_ascii_digit())
            && tile[2..5].chars().all(|c| c.is_ascii_uppercase())
        {
            return tile.to_string();
        }
    }
    String::new()
}

/// Element84 demands strict RFC3339 timestamps in `/search`. Promote
/// bare YYYY-MM-DD endpoints to start-of-day / end-of-day timestamps.
fn normalise_datetime(raw: &str) -> String {
    if !raw.contains('/') {
        return raw.to_string();
    }
    let parts: Vec<&str> = raw.splitn(2, '/').collect();
    let start = parts[0].trim();
    let end = parts.get(1).map(|s| s.trim()).unwrap_or("");
    let start = expand_to_rfc3339(start, true);
    let end = expand_to_rfc3339(end, false);
    format!("{start}/{end}")
}

fn expand_to_rfc3339(token: &str, start: bool) -> String {
    if token.is_empty() || token == ".." {
        return "..".to_string();
    }
    if token.contains('T') {
        return token.to_string();
    }
    if start {
        format!("{token}T00:00:00Z")
    } else {
        format!("{token}T23:59:59Z")
    }
}

fn _unused_marker() -> anyhow::Error {
    anyhow!("never")
}
