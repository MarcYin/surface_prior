//! Atmospheric correction of Sentinel-2 **L1C TOA** reflectance using
//! precomputed 6S coefficients (`xap`, `xbp`, `xcp`) carried in a per-scene
//! sidecar produced by the Python pre-step (GEE for MAIAC AOD / WVP /
//! geometry, SIAC native 6S for the coefficients).
//!
//! The MAIAC-select-then-correct pipeline keeps the radiative transfer in
//! Python/SIAC (MAIAC and 6S are not reachable from this crate) and does the
//! speed-critical per-pixel correction + compositing here. The 6S surface
//! reflectance relation is:
//!
//! ```text
//!   y       = xap * rho_toa - xbp
//!   rho_boa = y / (1 + xcp * y)
//! ```
//!
//! Coefficients depend on (AOD, water vapour, geometry, band). AOD and
//! geometry are fixed per scene; water vapour varies per pixel, so the sidecar
//! ships the coefficients over a small TCWV LUT and we interpolate by the
//! pixel's (or scene-mean) water vapour.

use std::collections::HashMap;

use serde::Deserialize;

/// DN convention: surface/TOA reflectance = DN / 10000. 65535 = nodata.
const SCALE: f32 = 10000.0;
const NODATA: u16 = 65535;

/// Per-scene 6S coefficients over a TCWV LUT, in the sidecar's band order.
/// `xap`/`xbp`/`xcp` are indexed `[tcwv_node][band]`.
#[derive(Debug, Clone, Deserialize)]
pub struct SceneAtmosphere {
    /// MAIAC AOD550 over the AOI — drives selection and is the AOD the
    /// coefficients were computed at.
    pub maiac_aod: f32,
    /// Scene-mean total column water vapour (cm), used when no per-pixel WVP
    /// raster is supplied.
    pub wvp: f32,
    /// TCWV grid (cm) the coefficient LUT is sampled on (ascending).
    pub tcwv_nodes: Vec<f32>,
    pub xap: Vec<Vec<f32>>,
    pub xbp: Vec<Vec<f32>>,
    pub xcp: Vec<Vec<f32>>,
}

impl SceneAtmosphere {
    /// Linearly interpolate `(xap, xbp, xcp)` for one band at water vapour
    /// `wvp` (clamped to the LUT range). Falls back to the single node when
    /// the LUT has one entry.
    pub fn coeffs(&self, band: usize, wvp: f32) -> (f32, f32, f32) {
        let n = self.tcwv_nodes.len();
        if n == 0 {
            return (1.0, 0.0, 0.0);
        }
        if n == 1 {
            return (self.xap[0][band], self.xbp[0][band], self.xcp[0][band]);
        }
        let w = wvp.clamp(self.tcwv_nodes[0], self.tcwv_nodes[n - 1]);
        // find upper node
        let hi = self
            .tcwv_nodes
            .iter()
            .position(|&t| t >= w)
            .unwrap_or(n - 1)
            .max(1);
        let lo = hi - 1;
        let (t0, t1) = (self.tcwv_nodes[lo], self.tcwv_nodes[hi]);
        let f = if (t1 - t0).abs() < 1e-6 { 0.0 } else { (w - t0) / (t1 - t0) };
        let lerp = |a: &[Vec<f32>]| a[lo][band] + (a[hi][band] - a[lo][band]) * f;
        (lerp(&self.xap), lerp(&self.xbp), lerp(&self.xcp))
    }
}

/// Sidecar: per-scene atmosphere keyed by STAC item id, plus the band order
/// the coefficients are in (must match the fetched band order).
#[derive(Debug, Clone, Deserialize)]
pub struct AtmoSidecar {
    pub bands: Vec<String>,
    pub scenes: HashMap<String, SceneAtmosphere>,
}

impl AtmoSidecar {
    pub fn load(path: &str) -> crate::Result<Self> {
        let txt = std::fs::read_to_string(path)
            .map_err(|e| crate::Error::Config(format!("read sidecar {path}: {e}")))?;
        Ok(serde_json::from_str(&txt)?)
    }

    /// Scene ids ranked by MAIAC AOD (ascending), keeping only the lowest
    /// `frac` — the "select clean days" step. `frac` is clamped to (0, 1].
    pub fn select_low_aod(&self, frac: f32) -> Vec<String> {
        let mut v: Vec<(&String, f32)> =
            self.scenes.iter().map(|(k, a)| (k, a.maiac_aod)).collect();
        v.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        let k = ((frac.clamp(1e-3, 1.0) * v.len() as f32).round() as usize).max(1).min(v.len());
        v.into_iter().take(k).map(|(id, _)| id.clone()).collect()
    }
}

/// Correct a single TOA reflectance value (already in 0..1) with 6S coeffs.
#[inline]
pub fn correct_refl(rho_toa: f32, xap: f32, xbp: f32, xcp: f32) -> f32 {
    let y = xap * rho_toa - xbp;
    y / (1.0 + xcp * y)
}

/// Correct one TOA band buffer (u16 DN, nodata=65535) to surface reflectance
/// DN using a single coefficient set (scene-mean WVP). Negative corrected
/// values clamp to 0; nodata passes through.
pub fn correct_band_scene(toa: &[u16], xap: f32, xbp: f32, xcp: f32) -> Vec<u16> {
    toa.iter()
        .map(|&t| {
            if t == NODATA {
                return NODATA;
            }
            let s = correct_refl(t as f32 / SCALE, xap, xbp, xcp);
            if s <= 0.0 {
                0
            } else {
                ((s * SCALE).round()).min(65534.0) as u16
            }
        })
        .collect()
}

/// Correct one TOA band using per-pixel water vapour (`wvp_cm`, same length as
/// `toa`); coefficients are interpolated from the scene LUT per pixel.
pub fn correct_band_perpixel(
    toa: &[u16],
    atm: &SceneAtmosphere,
    band: usize,
    wvp_cm: &[f32],
) -> Vec<u16> {
    toa.iter()
        .zip(wvp_cm.iter())
        .map(|(&t, &w)| {
            if t == NODATA {
                return NODATA;
            }
            let (xap, xbp, xcp) = atm.coeffs(band, if w.is_finite() { w } else { atm.wvp });
            let s = correct_refl(t as f32 / SCALE, xap, xbp, xcp);
            if s <= 0.0 { 0 } else { ((s * SCALE).round()).min(65534.0) as u16 }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn forward(rho_boa: f32, xap: f32, xbp: f32, xcp: f32) -> f32 {
        // inverse of correct_refl: TOA that yields this surface
        let y = rho_boa / (1.0 - xcp * rho_boa);
        (y + xbp) / xap
    }

    #[test]
    fn correction_inverts_forward() {
        let (xap, xbp, xcp) = (1.27f32, 0.080, 0.145); // ~blue at AOD 0.13
        for &boa in &[0.02f32, 0.08, 0.20, 0.4] {
            let toa = forward(boa, xap, xbp, xcp);
            let back = correct_refl(toa, xap, xbp, xcp);
            assert!((back - boa).abs() < 1e-5, "boa {boa} -> {back}");
        }
    }

    #[test]
    fn band_buffer_nodata_and_clamp() {
        let toa = [65535u16, 0u16, 1500u16];
        let out = correct_band_scene(&toa, 1.27, 0.080, 0.145);
        assert_eq!(out[0], 65535); // nodata passes through
        assert_eq!(out[1], 0); // 0 TOA -> negative -> clamp 0
        assert!(out[2] > 0 && out[2] < 1500); // path-removed -> darker than TOA
    }

    #[test]
    fn wvp_lut_interpolation() {
        let atm = SceneAtmosphere {
            maiac_aod: 0.1,
            wvp: 2.0,
            tcwv_nodes: vec![1.0, 3.0],
            xap: vec![vec![1.0], vec![2.0]],
            xbp: vec![vec![0.0], vec![0.0]],
            xcp: vec![vec![0.0], vec![0.0]],
        };
        let (xap, _, _) = atm.coeffs(0, 2.0); // midpoint
        assert!((xap - 1.5).abs() < 1e-6);
        let (xap_lo, _, _) = atm.coeffs(0, 0.0); // clamp low
        assert!((xap_lo - 1.0).abs() < 1e-6);
    }

    #[test]
    fn select_low_aod_keeps_cleanest() {
        let mk = |aod: f32| SceneAtmosphere {
            maiac_aod: aod, wvp: 2.0, tcwv_nodes: vec![2.0],
            xap: vec![vec![1.0]], xbp: vec![vec![0.0]], xcp: vec![vec![0.0]],
        };
        let mut scenes = HashMap::new();
        for (i, a) in [0.5, 0.1, 0.3, 0.2, 0.4].iter().enumerate() {
            scenes.insert(format!("s{i}"), mk(*a));
        }
        let sc = AtmoSidecar { bands: vec![], scenes };
        let sel = sc.select_low_aod(0.6); // lowest 3 of 5
        assert_eq!(sel.len(), 3);
        assert!(sel.contains(&"s1".to_string()) && sel.contains(&"s3".to_string()));
        assert!(!sel.contains(&"s0".to_string())); // 0.5 excluded
    }
}
