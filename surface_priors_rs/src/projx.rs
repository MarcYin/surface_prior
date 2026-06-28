//! PROJ-accurate coordinate transforms.
//!
//! Wraps the `proj` crate (system libproj 9.x). Transformers are
//! cached per CRS pair in a thread-local map so a single batch reuses
//! initialised PROJ contexts and avoids the ~5 ms per-transform setup.

use std::cell::RefCell;
use std::collections::HashMap;

use proj::Proj;

use crate::error::{Error, Result};

thread_local! {
    static TRANSFORMER_CACHE: RefCell<HashMap<(String, String), Proj>> =
        RefCell::new(HashMap::new());
}

/// `from`/`to` are anything libproj understands ("EPSG:4326", "+proj=…", WKT, etc).
pub fn transform_point(from: &str, to: &str, x: f64, y: f64) -> Result<(f64, f64)> {
    with_transformer(from, to, |t| {
        t.convert((x, y))
            .map_err(|e| Error::proj(format!("convert ({x}, {y}): {e}")))
    })?
}

/// Transform a sequence of points in-place. Faster than per-call when
/// there are many points because the PROJ pipeline runs once.
pub fn transform_points(from: &str, to: &str, points: &mut [(f64, f64)]) -> Result<()> {
    with_transformer(from, to, |t| {
        t.convert_array(points)
            .map_err(|e| Error::proj(format!("convert_array: {e}")))?;
        Ok::<(), Error>(())
    })?
}

/// Compute the densified bounds of a polygon in CRS `from` when
/// projected into CRS `to`. Densification points along edges keep the
/// resulting axis-aligned bbox tight enough for our grid snapping.
pub fn transform_bounds(
    from: &str,
    to: &str,
    bounds: [f64; 4],
    densify: usize,
) -> Result<[f64; 4]> {
    let (xmin, ymin, xmax, ymax) = (bounds[0], bounds[1], bounds[2], bounds[3]);
    let n = densify.max(2);
    let mut points: Vec<(f64, f64)> = Vec::with_capacity(4 * n);
    // Sample along each of the 4 edges.
    for i in 0..n {
        let t = i as f64 / (n - 1) as f64;
        points.push((xmin + t * (xmax - xmin), ymin)); // bottom edge
        points.push((xmin + t * (xmax - xmin), ymax)); // top edge
        points.push((xmin, ymin + t * (ymax - ymin))); // left edge
        points.push((xmax, ymin + t * (ymax - ymin))); // right edge
    }
    transform_points(from, to, &mut points)?;
    let mut out_xmin = f64::INFINITY;
    let mut out_ymin = f64::INFINITY;
    let mut out_xmax = f64::NEG_INFINITY;
    let mut out_ymax = f64::NEG_INFINITY;
    for (x, y) in &points {
        if !x.is_finite() || !y.is_finite() {
            continue;
        }
        out_xmin = out_xmin.min(*x);
        out_ymin = out_ymin.min(*y);
        out_xmax = out_xmax.max(*x);
        out_ymax = out_ymax.max(*y);
    }
    if !out_xmin.is_finite() {
        return Err(Error::proj("transform_bounds yielded no finite points"));
    }
    Ok([out_xmin, out_ymin, out_xmax, out_ymax])
}

/// Choose the UTM EPSG that minimises distortion for the AOI centre.
/// Mirrors Python's helper exactly.
pub fn utm_epsg_for_wgs84_bounds(bounds: [f64; 4]) -> u32 {
    let centre_lon = (bounds[0] + bounds[2]) / 2.0;
    let centre_lat = (bounds[1] + bounds[3]) / 2.0;
    let zone = (((centre_lon + 180.0) / 6.0).floor() as i32 + 1).clamp(1, 60);
    if centre_lat >= 0.0 { 32600 + zone as u32 } else { 32700 + zone as u32 }
}

/// UTM EPSG forcing the NORTHERN zone (false_northing = 0) regardless of
/// hemisphere. HLS georeferences every MGRS tile in the northern UTM zone with
/// continuous (signed) northing across the equator — a southern HLS tile is
/// stored as e.g. "UTM zone 48N" with negative northings, NOT zone 48S with the
/// 10,000,000 m false northing. Grids that must align pixel-for-pixel with HLS
/// COGs therefore have to use this convention; using 327xx for a southern AOI
/// leaves the grid northings ~10,000,000 m off the scene origin, which empties
/// the read window and drops every southern scene.
pub fn utm_epsg_for_wgs84_bounds_north(bounds: [f64; 4]) -> u32 {
    let centre_lon = (bounds[0] + bounds[2]) / 2.0;
    let zone = (((centre_lon + 180.0) / 6.0).floor() as i32 + 1).clamp(1, 60);
    32600 + zone as u32
}

/// UTM EPSG of an MGRS tile's own COG, from the tile code (`"16TCN"` →
/// zone 16, band `T` → northern → 32616). This is the *scene's* native
/// CRS, which is NOT necessarily the AOI grid's zone: an AOI straddling a
/// 6° UTM seam pulls tiles from the neighbouring zone too, and those COGs
/// live in that zone. Deriving the source CRS per scene (rather than
/// assuming every scene shares the grid zone) lets the seam tiles be
/// reprojected instead of read in the wrong zone and scored 0-usable.
///
/// `force_north` mirrors [`utm_epsg_for_wgs84_bounds_north`]: HLS stores every
/// tile in the northern zone (continuous signed northing), so its COGs are
/// always 326xx regardless of hemisphere; S2/PC use the true hemisphere from
/// the MGRS latitude band (`C`–`M` south → 327xx, `N`–`X` north → 326xx).
pub fn epsg_from_mgrs_tile(tile: &str, force_north: bool) -> Option<u32> {
    let b = tile.as_bytes();
    if b.len() < 3 {
        return None;
    }
    let zone: u32 = std::str::from_utf8(&b[0..2]).ok()?.parse().ok()?;
    if !(1..=60).contains(&zone) {
        return None;
    }
    let band = b[2].to_ascii_uppercase();
    if !band.is_ascii_alphabetic() {
        return None;
    }
    let north = force_north || band >= b'N'; // MGRS bands C..M south, N..X north
    Some(if north { 32600 + zone } else { 32700 + zone })
}

fn with_transformer<F, R>(from: &str, to: &str, f: F) -> Result<R>
where
    F: FnOnce(&Proj) -> R,
{
    TRANSFORMER_CACHE.with(|cell| {
        let mut map = cell.borrow_mut();
        if !map.contains_key(&(from.to_string(), to.to_string())) {
            let t = Proj::new_known_crs(from, to, None)
                .map_err(|e| Error::proj(format!("init {from} → {to}: {e}")))?;
            map.insert((from.to_string(), to.to_string()), t);
        }
        let t = map.get(&(from.to_string(), to.to_string())).unwrap();
        Ok(f(t))
    })
}

/// Convenience: WGS84 bbox → metric UTM bbox in the chosen EPSG.
pub fn wgs84_to_utm_bounds(wgs84: [f64; 4], utm_epsg: u32) -> Result<[f64; 4]> {
    transform_bounds("EPSG:4326", &format!("EPSG:{utm_epsg}"), wgs84, 21)
}

/// Snap bounds to whole multiples of `resolution`. Matches Python's
/// `_snap_bounds_to_resolution`.
pub fn snap_bounds(bounds: [f64; 4], resolution: f64) -> [f64; 4] {
    let r = resolution;
    [
        (bounds[0] / r).floor() * r,
        (bounds[1] / r).floor() * r,
        (bounds[2] / r).ceil() * r,
        (bounds[3] / r).ceil() * r,
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mgrs_tile_to_utm_epsg() {
        // Northern-hemisphere tiles (band >= N): 326xx. These are the three
        // AERONET seam sites whose neighbouring-zone tiles were being dropped.
        assert_eq!(epsg_from_mgrs_tile("16TCN", false), Some(32616)); // Wisconsin
        assert_eq!(epsg_from_mgrs_tile("15TYH", false), Some(32615)); // its seam neighbour
        assert_eq!(epsg_from_mgrs_tile("17TKE", false), Some(32617)); // Dayton
        assert_eq!(epsg_from_mgrs_tile("35SKC", false), Some(32635)); // Athens
        assert_eq!(epsg_from_mgrs_tile("33TVF", false), Some(32633)); // Napoli control
        // Southern-hemisphere band (C..M): 327xx unless force_north.
        assert_eq!(epsg_from_mgrs_tile("48MYT", false), Some(32748));
        assert_eq!(epsg_from_mgrs_tile("48MYT", true), Some(32648)); // HLS north convention
        // Garbage / short input → None (falls back to legacy same-CRS path).
        assert_eq!(epsg_from_mgrs_tile("", false), None);
        assert_eq!(epsg_from_mgrs_tile("XX", false), None);
        assert_eq!(epsg_from_mgrs_tile("99TCN", false), None); // zone out of range
    }
}
