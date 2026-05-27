//! STAC endpoint configuration: STAC URL, one-or-more collections,
//! per-collection band → asset name mapping, per-collection quality
//! asset (SCL for S2 L2A, Fmask for HLS), and signing strategy.
//!
//! Different providers expose Sentinel-2 / HLS under different STAC
//! catalogues with different asset key conventions:
//!   - Element84 earth-search S2 L2A uses `red`, `blue`, ..., `scl` —
//!     anonymous, single collection.
//!   - Microsoft Planetary Computer S2 L2A uses `B04`, `B02`, ..., `SCL`,
//!     hrefs require an SAS token appended as a query string.
//!   - Microsoft Planetary Computer HLS v2.0 uses `B02`, `B03`, ...,
//!     `B11`/`B12`/`B8A`/`Fmask`, spread across two collections
//!     (`hls2-l30` for Landsat OLI, `hls2-s30` for Sentinel-2 MSI). The
//!     band → asset mapping differs between L30 and S30 (e.g. "nir"
//!     is `B05` on Landsat but `B8A` on Sentinel-2 in the harmonized
//!     spec). Hrefs require the same SAS-token signing as PC S2.
//!
//! The rest of the pipeline threads through stable band NAMES (the
//! coastal/blue/green/... canonical set) and per-scene asset lookup
//! resolves to the right key based on the scene's STAC collection.

use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result};

/// Stable, endpoint-independent surface-reflectance band names. Output
/// dicts / GeoTIFFs are keyed by these regardless of which catalogue
/// the scene was fetched from.
///
/// HLS supports only the first 7 of these (the harmonized common bands);
/// callers that pass `bands` constrained to the 7-common set work with
/// any endpoint, callers that pass the full 12 are S2 L2A only.
pub const BAND_NAMES: [&str; 12] = [
    "coastal", "blue", "green", "red",
    "rededge1", "rededge2", "rededge3",
    "nir", "nir08", "nir09",
    "swir16", "swir22",
];

/// Subset HLS supports across both Landsat (L30) and Sentinel-2 (S30).
/// These are the bands the Roy et al. c-factor NBAR normalisation
/// covers.
pub const BAND_NAMES_HLS_HARMONIZED: [&str; 7] = [
    "coastal", "blue", "green", "red", "nir", "swir16", "swir22",
];

/// MCD43A4 ships 7 native MODIS reflectance bands; we expose the 6
/// that overlap the canonical Sentinel-2 / HLS naming (MODIS Band 5
/// at 1240 nm has no S2/HLS counterpart and is omitted).
pub const BAND_NAMES_MCD43A4: [&str; 6] = [
    "blue", "green", "red", "nir", "swir16", "swir22",
];

/// What kind of cloud / quality raster the endpoint exposes. Drives
/// scout's "clear" classification and the per-pixel quality score.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QualityKind {
    /// Sentinel-2 L2A SCL: categorical labels 0..11.
    Scl,
    /// HLS Fmask: bit-packed flags + aerosol level.
    Fmask,
    /// MODIS MCD43A4 `BRDF_Albedo_Band_Mandatory_Quality_Band{N}`:
    /// 0 = full BRDF inversion, 1 = magnitude inversion, 255 = nodata.
    ModisMandatory,
}

impl QualityKind {
    pub fn is_nodata(&self, v: u8) -> bool {
        match self {
            Self::Scl => v == 0,
            Self::Fmask => v == 255,
            Self::ModisMandatory => v == 255,
        }
    }

    /// A byte that `is_nodata` treats as nodata. Used to pad quality
    /// buffers outside a windowed (Level 2) fetch so compose skips those
    /// pixels.
    pub fn nodata_fill(&self) -> u8 {
        match self {
            Self::Scl => 0,
            Self::Fmask | Self::ModisMandatory => 255,
        }
    }

    pub fn is_clear(&self, v: u8) -> bool {
        match self {
            // SCL: dark vegetation (4), vegetation (5), bare (6), snow (11).
            Self::Scl => matches!(v, 4 | 5 | 6 | 11),
            // Fmask: bits 0-3 clear (no cirrus / cloud / adj-cloud /
            // cloud-shadow); aerosol level <= low.
            Self::Fmask => {
                if v == 255 {
                    return false;
                }
                let cloudy = v & 0b0000_1111;
                let aerosol = (v >> 6) & 0b11;
                cloudy == 0 && aerosol <= 1
            }
            // MCD43A4 mandatory quality: 0 = full BRDF inversion. The
            // BRDF process already filtered clouds upstream, so "0" is
            // the only thing we count as "clear" here.
            Self::ModisMandatory => v == 0,
        }
    }

    /// Per-pixel quality score (0 = best, 65535 = nodata). Used by
    /// best-pixel compose to rank observations.
    pub fn score(&self, v: u8) -> u16 {
        const NODATA: u16 = 65535;
        match self {
            Self::Scl => match v {
                4 | 5 | 6 | 11 => 0,
                7 => 1,
                2 | 3 => 2,
                _ => NODATA,
            },
            Self::Fmask => {
                if v == 255 {
                    return NODATA;
                }
                let cirrus = (v & 0b0000_0001) != 0;
                let cloud = (v & 0b0000_0010) != 0;
                let adj = (v & 0b0000_0100) != 0;
                let shadow = (v & 0b0000_1000) != 0;
                let snow = (v & 0b0001_0000) != 0;
                let aerosol = (v >> 6) & 0b11;
                if cloud {
                    return NODATA;
                }
                if cirrus || shadow {
                    return 2;
                }
                if adj || aerosol >= 2 {
                    return 1;
                }
                if snow {
                    // Snow is real surface signal; treat as marginal so
                    // a non-snow observation will win when available.
                    return 1;
                }
                0
            }
            Self::ModisMandatory => match v {
                0 => 0,         // full BRDF inversion (best)
                1 => 1,         // magnitude inversion (acceptable)
                255 => NODATA,  // fill / no retrieval
                _ => NODATA,
            },
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EndpointKind {
    EarthSearch,
    PlanetaryComputer,
    /// Microsoft PC HLS v2.0: `hls2-l30` + `hls2-s30` combined into one
    /// harmonized composite pool.
    Hls,
    /// Microsoft PC MODIS MCD43A4 v6.1: daily 500 m Nadir BRDF-Adjusted
    /// Reflectance. Native CRS is MODIS Sinusoidal, output grid is the
    /// same CRS (no on-the-fly reprojection).
    Modis43A4,
}

impl EndpointKind {
    pub fn parse(s: &str) -> Result<Self> {
        match s {
            "earth-search" | "es" | "element84" => Ok(Self::EarthSearch),
            "pc" | "planetary-computer" | "planetary_computer" => Ok(Self::PlanetaryComputer),
            "hls" | "hls2" | "hls-l30-s30" => Ok(Self::Hls),
            "mcd43a4" | "modis-43a4" | "modis-43A4-061" | "modis43a4" => {
                Ok(Self::Modis43A4)
            }
            other => anyhow::bail!(
                "unknown endpoint {other:?}; supported: pc, earth-search, hls, mcd43a4"
            ),
        }
    }
}

/// Per-collection routing: which asset name carries each stable band
/// and which asset is the quality mask.
#[derive(Debug, Clone)]
pub struct CollectionAssets {
    /// band-name → asset key (only bands the collection exposes).
    pub bands: HashMap<String, String>,
    /// Quality raster asset name (`SCL` or `Fmask`).
    pub quality: String,
}

pub struct EndpointConfig {
    pub kind: EndpointKind,
    pub stac_url: String,
    /// STAC collections to search. Multi-collection endpoints (HLS)
    /// search them in one request via `collections: [..]`.
    pub collections: Vec<String>,
    /// Per-collection asset routing. Keyed by STAC `collection` id.
    pub per_collection: HashMap<String, CollectionAssets>,
    /// What dialect of quality mask this endpoint serves. All
    /// collections under one endpoint must use the same dialect.
    pub quality_kind: QualityKind,
    /// Stable order of bands this endpoint can produce. For S2 L2A
    /// this is all 12 BAND_NAMES; for HLS it's the 7-band harmonized
    /// subset.
    pub band_names_supported: &'static [&'static str],
    /// SAS tokens cached per Azure (storage_account, container) pair.
    /// PC's `/api/sas/v1/token/{collection}` endpoint only signs the
    /// default storage account for each collection; MCD43A4 (and other
    /// multi-account collections) need the per-href `/api/sas/v1/sign`
    /// endpoint to land on the right account. We populate this cache
    /// lazily as new container URLs appear.
    sas_tokens_per_container: parking_lot::RwLock<HashMap<(String, String), String>>,
}

impl EndpointConfig {
    pub fn build(kind: EndpointKind) -> Self {
        match kind {
            EndpointKind::EarthSearch => Self::earth_search(),
            EndpointKind::PlanetaryComputer => Self::planetary_computer(),
            EndpointKind::Hls => Self::hls_pc(),
            EndpointKind::Modis43A4 => Self::modis_43a4_pc(),
        }
    }

    /// `eo:cloud_cover` is only meaningful for S2-style scenes; HLS
    /// has the same property but MCD43A4 (already BRDF-cleaned)
    /// doesn't, so we skip the STAC `query` filter there.
    pub fn supports_cloud_cover_filter(&self) -> bool {
        !matches!(self.kind, EndpointKind::Modis43A4)
    }

    /// True if the endpoint's COGs are natively in MODIS sinusoidal —
    /// callers building the output grid use the matching constructor.
    pub fn uses_modis_sinusoidal(&self) -> bool {
        matches!(self.kind, EndpointKind::Modis43A4)
    }

    /// PROJ-readable CRS string for this endpoint's source COGs. For
    /// MCD43A4 this is MODIS Sinusoidal; for S2/HLS the source CRS
    /// varies per-scene UTM zone, so we return None — the existing
    /// same-CRS resampler handles those because the pipeline picks
    /// the target UTM zone to match the AOI.
    pub fn source_proj(&self) -> Option<&'static str> {
        match self.kind {
            EndpointKind::Modis43A4 => Some(crate::grid::MODIS_SINU_PROJ4),
            _ => None,
        }
    }

    pub fn earth_search() -> Self {
        let band_to_asset = [
            ("coastal", "coastal"),
            ("blue", "blue"),
            ("green", "green"),
            ("red", "red"),
            ("rededge1", "rededge1"),
            ("rededge2", "rededge2"),
            ("rededge3", "rededge3"),
            ("nir", "nir"),
            ("nir08", "nir08"),
            ("nir09", "nir09"),
            ("swir16", "swir16"),
            ("swir22", "swir22"),
        ];
        let assets = CollectionAssets {
            bands: band_to_asset
                .iter()
                .map(|(b, a)| (b.to_string(), a.to_string()))
                .collect(),
            quality: "scl".into(),
        };
        let mut per_collection = HashMap::new();
        per_collection.insert("sentinel-2-l2a".to_string(), assets);
        Self {
            kind: EndpointKind::EarthSearch,
            stac_url: "https://earth-search.aws.element84.com/v1".to_string(),
            collections: vec!["sentinel-2-l2a".to_string()],
            per_collection,
            quality_kind: QualityKind::Scl,
            band_names_supported: &BAND_NAMES,
            sas_tokens_per_container: parking_lot::RwLock::new(HashMap::new()),
        }
    }

    pub fn planetary_computer() -> Self {
        let band_to_asset = [
            ("coastal", "B01"),
            ("blue", "B02"),
            ("green", "B03"),
            ("red", "B04"),
            ("rededge1", "B05"),
            ("rededge2", "B06"),
            ("rededge3", "B07"),
            ("nir", "B08"),
            ("nir08", "B8A"),
            ("nir09", "B09"),
            ("swir16", "B11"),
            ("swir22", "B12"),
        ];
        let assets = CollectionAssets {
            bands: band_to_asset
                .iter()
                .map(|(b, a)| (b.to_string(), a.to_string()))
                .collect(),
            quality: "SCL".into(),
        };
        let mut per_collection = HashMap::new();
        per_collection.insert("sentinel-2-l2a".to_string(), assets);
        Self {
            kind: EndpointKind::PlanetaryComputer,
            stac_url: "https://planetarycomputer.microsoft.com/api/stac/v1".to_string(),
            collections: vec!["sentinel-2-l2a".to_string()],
            per_collection,
            quality_kind: QualityKind::Scl,
            band_names_supported: &BAND_NAMES,
            sas_tokens_per_container: parking_lot::RwLock::new(HashMap::new()),
        }
    }

    /// PC's MODIS MCD43A4 v6.1 (Nadir BRDF-Adjusted Reflectance, 500 m,
    /// daily). COGs are in MODIS Sinusoidal — the pipeline emits the
    /// composite in the same CRS (no on-the-fly reprojection).
    ///
    /// Band → asset mapping (MODIS band centres → S2-aligned names):
    ///   - blue   = Band 3 (459-479 nm)
    ///   - green  = Band 4 (545-565 nm)
    ///   - red    = Band 1 (620-670 nm)
    ///   - nir    = Band 2 (841-876 nm)   -- broad NIR
    ///   - swir16 = Band 6 (1628-1652 nm)
    ///   - swir22 = Band 7 (2105-2155 nm)
    /// MODIS Band 5 (1240 nm) is omitted: no S2 counterpart.
    ///
    /// Quality asset: `BRDF_Albedo_Band_Mandatory_Quality_Band1` is
    /// used as the scene-wide quality proxy — for a per-band quality
    /// breakdown each band's own _Quality_BandN exists but the
    /// mandatory flag tracks the BRDF inversion state across bands.
    pub fn modis_43a4_pc() -> Self {
        let assets = CollectionAssets {
            bands: [
                ("blue", "Nadir_Reflectance_Band3"),
                ("green", "Nadir_Reflectance_Band4"),
                ("red", "Nadir_Reflectance_Band1"),
                ("nir", "Nadir_Reflectance_Band2"),
                ("swir16", "Nadir_Reflectance_Band6"),
                ("swir22", "Nadir_Reflectance_Band7"),
            ]
            .iter()
            .map(|(b, a)| (b.to_string(), a.to_string()))
            .collect(),
            quality: "BRDF_Albedo_Band_Mandatory_Quality_Band1".into(),
        };
        let mut per_collection = HashMap::new();
        per_collection.insert("modis-43A4-061".to_string(), assets);
        Self {
            kind: EndpointKind::Modis43A4,
            stac_url: "https://planetarycomputer.microsoft.com/api/stac/v1".to_string(),
            collections: vec!["modis-43A4-061".to_string()],
            per_collection,
            quality_kind: QualityKind::ModisMandatory,
            band_names_supported: &BAND_NAMES_MCD43A4,
            sas_tokens_per_container: parking_lot::RwLock::new(HashMap::new()),
        }
    }

    /// PC's Harmonized Landsat-Sentinel-2 product (HLS v2.0). Two
    /// STAC collections (`hls2-l30` + `hls2-s30`) combined into one
    /// harmonized pool covering the 7 common-band NBAR bands.
    ///
    /// Asset mapping per the HLS v2.0 User Guide / Roy et al. 2021:
    ///   - L30 (Landsat OLI):  coastal=B01, blue=B02, green=B03,
    ///       red=B04, nir=B05, swir16=B06, swir22=B07
    ///   - S30 (Sentinel-2 MSI): coastal=B01, blue=B02, green=B03,
    ///       red=B04, nir=B8A (narrow NIR — the harmonized choice),
    ///       swir16=B11, swir22=B12
    pub fn hls_pc() -> Self {
        let l30 = CollectionAssets {
            bands: [
                ("coastal", "B01"),
                ("blue", "B02"),
                ("green", "B03"),
                ("red", "B04"),
                ("nir", "B05"),
                ("swir16", "B06"),
                ("swir22", "B07"),
            ]
            .iter()
            .map(|(b, a)| (b.to_string(), a.to_string()))
            .collect(),
            quality: "Fmask".into(),
        };
        let s30 = CollectionAssets {
            bands: [
                ("coastal", "B01"),
                ("blue", "B02"),
                ("green", "B03"),
                ("red", "B04"),
                ("nir", "B8A"),
                ("swir16", "B11"),
                ("swir22", "B12"),
            ]
            .iter()
            .map(|(b, a)| (b.to_string(), a.to_string()))
            .collect(),
            quality: "Fmask".into(),
        };
        let mut per_collection = HashMap::new();
        per_collection.insert("hls2-l30".to_string(), l30);
        per_collection.insert("hls2-s30".to_string(), s30);
        Self {
            kind: EndpointKind::Hls,
            stac_url: "https://planetarycomputer.microsoft.com/api/stac/v1".to_string(),
            collections: vec!["hls2-l30".to_string(), "hls2-s30".to_string()],
            per_collection,
            quality_kind: QualityKind::Fmask,
            band_names_supported: &BAND_NAMES_HLS_HARMONIZED,
            sas_tokens_per_container: parking_lot::RwLock::new(HashMap::new()),
        }
    }

    /// Stable identifier suitable for cache keys. For multi-collection
    /// endpoints we join with `+` so the key includes all participating
    /// collections.
    pub fn collections_key(&self) -> String {
        self.collections.join("+")
    }

    /// Primary collection — first in the list. Used by places that
    /// require a single identifier (legacy disk-cache scout key).
    pub fn primary_collection(&self) -> &str {
        self.collections
            .first()
            .map(|s| s.as_str())
            .unwrap_or("")
    }

    /// Resolve asset key for `band_name` on a scene from
    /// `scene_collection`. Returns `None` if the band doesn't exist
    /// on that collection (e.g. asking for `rededge1` on HLS-L30).
    pub fn asset_for(&self, scene_collection: &str, band_name: &str) -> Option<&str> {
        self.per_collection
            .get(scene_collection)
            .and_then(|a| a.bands.get(band_name).map(|s| s.as_str()))
    }

    /// Quality raster asset name for a scene from `scene_collection`.
    pub fn quality_asset_for(&self, scene_collection: &str) -> Option<&str> {
        self.per_collection
            .get(scene_collection)
            .map(|a| a.quality.as_str())
    }

    /// Sign a raw asset href via PC's `/api/sas/v1/sign?href=...`
    /// endpoint. The returned token is cached per Azure
    /// (storage_account, container) pair so subsequent hrefs in the
    /// same container reuse it without another network round-trip.
    /// The `_collection` argument is kept in the signature for API
    /// stability but no longer used — signing is purely href-driven.
    pub async fn sign_href(
        &self,
        http: &reqwest::Client,
        _collection: &str,
        href: &str,
    ) -> Result<String> {
        if !self.requires_sas() {
            return Ok(href.to_string());
        }
        if href.contains("?sv=") || href.contains("&sv=") {
            return Ok(href.to_string());
        }
        let Some((account, container)) = parse_account_container(href) else {
            return Ok(href.to_string());
        };
        // Cached?
        if let Some(t) = self
            .sas_tokens_per_container
            .read()
            .get(&(account.clone(), container.clone()))
            .cloned()
        {
            return Ok(append_token(href, &t));
        }
        // Cache miss — round-trip to PC's per-href sign endpoint.
        let token = self.fetch_sign_token(http, href).await?;
        self.sas_tokens_per_container
            .write()
            .insert((account, container), token.clone());
        Ok(append_token(href, &token))
    }

    fn requires_sas(&self) -> bool {
        matches!(
            self.kind,
            EndpointKind::PlanetaryComputer | EndpointKind::Hls | EndpointKind::Modis43A4
        )
    }

    /// POST `/api/sas/v1/sign?href=...` and return the bare SAS query
    /// string (without the leading `?`).
    async fn fetch_sign_token(
        &self,
        http: &reqwest::Client,
        href: &str,
    ) -> Result<String> {
        let url = format!(
            "https://planetarycomputer.microsoft.com/api/sas/v1/sign?href={}",
            urlencoding::encode(href)
        );
        let resp = http
            .get(&url)
            .send()
            .await
            .with_context(|| format!("PC SAS sign GET {url}"))?
            .error_for_status()
            .with_context(|| format!("PC SAS sign non-2xx for {url}"))?;
        let body: serde_json::Value = resp.json().await.context("PC SAS body decode")?;
        let signed_href = body
            .get("href")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow::anyhow!("PC sign response missing 'href' field"))?;
        // Extract the query part as the bare token.
        let token = signed_href
            .split_once('?')
            .map(|(_, q)| q.to_string())
            .ok_or_else(|| anyhow::anyhow!("signed href has no query string"))?;
        tracing::debug!(token_len = token.len(), "fetched PC SAS token via /sign");
        Ok(token)
    }

    /// Sign every asset href in a STAC item in place. Picks the right
    /// per-collection SAS token based on the item's `collection` field.
    pub async fn sign_item(
        &self,
        http: &reqwest::Client,
        mut item: serde_json::Value,
    ) -> Result<serde_json::Value> {
        if !self.requires_sas() {
            return Ok(item);
        }
        let collection = item
            .get("collection")
            .and_then(|v| v.as_str())
            .unwrap_or(self.primary_collection())
            .to_string();
        if let Some(assets) = item.get_mut("assets").and_then(|v| v.as_object_mut()) {
            for (_, asset) in assets.iter_mut() {
                let href_opt = asset
                    .get("href")
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string());
                if let Some(href) = href_opt {
                    match self.sign_href(http, &collection, &href).await {
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
/// AWS region introspection we default to PC for the UK / JASMIN case.
pub fn auto_pick() -> EndpointKind {
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

#[doc(hidden)]
pub fn _unused_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// Parse `(storage_account, container)` from an Azure blob URL like
/// `https://{account}.blob.core.windows.net/{container}/{blob...}`.
/// Returns `None` if the href doesn't match this layout.
fn parse_account_container(href: &str) -> Option<(String, String)> {
    let without_scheme = href.strip_prefix("https://").or_else(|| href.strip_prefix("http://"))?;
    let (host, rest) = without_scheme.split_once('/')?;
    let account = host.split_once('.')?.0.to_string();
    let container = rest.split('/').next()?.to_string();
    if account.is_empty() || container.is_empty() {
        return None;
    }
    Some((account, container))
}

/// Append a SAS token (query string body) to an href, joining with
/// `?` or `&` as appropriate.
fn append_token(href: &str, token: &str) -> String {
    if href.contains('?') {
        format!("{href}&{token}")
    } else {
        format!("{href}?{token}")
    }
}
