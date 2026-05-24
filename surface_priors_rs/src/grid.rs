//! Grid + window math.
//!
//! Mirrors `surface_priors.types.GridSpec` and `chunks.ChunkLayout` in
//! Python. We assume UTM north for now (matches all S2 L2A tiles in the
//! verification AOI). WGS84 → UTM uses the simple zone-from-longitude
//! rule; for production you'd want a real PROJ-backed transform.

use anyhow::Result;
use rayon::prelude::*;
use std::collections::HashMap;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use crate::cog::{OverviewLevel, PixelWindow};
use crate::projx;

#[derive(Debug, Clone, Copy)]
pub struct GridSpec {
    /// UTM bounds (xmin, ymin, xmax, ymax) snapped to `resolution`.
    pub bounds: [f64; 4],
    pub epsg: u32,
    pub resolution: f64,
    pub width: u32,
    pub height: u32,
}

impl GridSpec {
    /// Build a UTM-snapped grid from WGS84 bounds. Uses PROJ via the
    /// `projx` module — the four-corner shortcut from the prototype was
    /// off by tens of metres near zone boundaries; PROJ's densified
    /// `transform_bounds` is accurate to sub-pixel.
    pub fn from_wgs84_bounds(bbox: [f64; 4], resolution: f64) -> Self {
        let epsg = projx::utm_epsg_for_wgs84_bounds(bbox);
        let utm = projx::wgs84_to_utm_bounds(bbox, epsg).expect("PROJ WGS84→UTM");
        let snapped = projx::snap_bounds(utm, resolution);
        let width = ((snapped[2] - snapped[0]) / resolution).round() as u32;
        let height = ((snapped[3] - snapped[1]) / resolution).round() as u32;
        GridSpec {
            bounds: snapped,
            epsg,
            resolution,
            width,
            height,
        }
    }

    pub fn affine_transform(&self) -> [f64; 6] {
        // (a, b, xoff, d, e, yoff) — north-up.
        [self.resolution, 0.0, self.bounds[0], 0.0, -self.resolution, self.bounds[3]]
    }
}

/// Compute the pixel window inside a COG image (at a chosen overview
/// level) that covers the requested UTM bounds at the requested output
/// resolution. The window is clipped to the COG image bounds. Returns
/// `None` if the window is empty (image doesn't cover the AOI).
pub fn cog_window_for_utm(
    image_geo_origin: [f64; 2], // (x, y) of pixel (0,0)
    image_pixel_size: f64,
    image_width: u32,
    image_height: u32,
    utm_bounds: [f64; 4],
) -> Option<crate::cog::PixelWindow> {
    let (ox, oy) = (image_geo_origin[0], image_geo_origin[1]);
    let pix = image_pixel_size;
    let col0 = ((utm_bounds[0] - ox) / pix).floor() as i64;
    let row0 = ((oy - utm_bounds[3]) / pix).floor() as i64;
    let col1 = ((utm_bounds[2] - ox) / pix).ceil() as i64;
    let row1 = ((oy - utm_bounds[1]) / pix).ceil() as i64;
    let col0 = col0.max(0) as u32;
    let row0 = row0.max(0) as u32;
    let col1 = col1.min(image_width as i64) as u32;
    let row1 = row1.min(image_height as i64) as u32;
    if col1 <= col0 || row1 <= row0 {
        return None;
    }
    Some(crate::cog::PixelWindow {
        col_off: col0,
        row_off: row0,
        width: col1 - col0,
        height: row1 - row0,
    })
}

/// Bilinear resample a source array shaped `(src_h, src_w)` into output
/// shape `(dst_h, dst_w)`. Same-CRS, same-origin case only; src and dst
/// share a geo-aligned anchor at src_origin / dst_origin.
/// Bilinear u16 → u16 resample. Hoisted per-row computations and
/// branchless interior-pixel kernel — when both `src_c+1 < sw` and
/// `src_r+1 < sh` we skip bounds checks and run the dense kernel. Edge
/// pixels fall through to the safe version. About 2× faster on the
/// hot path vs the previous all-branches version.
pub fn resample_u16_to_u16(
    src: &[u16],
    src_dim: (u32, u32),
    src_origin: [f64; 2],
    src_pixel_size: f64,
    dst_dim: (u32, u32),
    dst_origin: [f64; 2],
    dst_pixel_size: f64,
) -> Result<Vec<u16>> {
    let (sw, sh) = (src_dim.0 as usize, src_dim.1 as usize);
    let (dw, dh) = (dst_dim.0 as usize, dst_dim.1 as usize);
    let mut out = vec![0u16; dw * dh];
    let inv_src_px = 1.0 / src_pixel_size;
    let x_origin_delta = (dst_origin[0] - src_origin[0]) * inv_src_px;
    let y_origin_delta = (src_origin[1] - dst_origin[1]) * inv_src_px;
    let col_step = dst_pixel_size * inv_src_px;

    // Precompute per-row metadata once: row offset pair + dr.
    // The dense interior is the common case (the only edge pixels are
    // the last ~1 row/column near the COG boundary). We split each row
    // into the interior chunk and the edge tail; the interior uses
    // 8-lane SIMD with `wide::f32x8`, the tail uses scalar branchy code.
    out.chunks_mut(dw).enumerate().for_each(|(r, row)| {
        let src_r_f64 = y_origin_delta + (r as f64 + 0.5) * col_step - 0.5;
        let r0i = src_r_f64.floor() as i64;
        let dr = (src_r_f64 - r0i as f64) as f32;
        let one_minus_dr = 1.0_f32 - dr;
        let row0_off = if r0i >= 0 && (r0i as usize) < sh {
            Some((r0i as usize) * sw)
        } else {
            None
        };
        let row1_off = if r0i + 1 >= 0 && ((r0i + 1) as usize) < sh {
            Some(((r0i + 1) as usize) * sw)
        } else {
            None
        };
        let interior_row = row0_off.is_some() && row1_off.is_some();

        if !interior_row {
            // Whole row falls outside the source — leave zeros.
            return;
        }
        let r0 = unsafe { row0_off.unwrap_unchecked() };
        let r1 = unsafe { row1_off.unwrap_unchecked() };

        // Find the column range where ALL 8 lanes' c0 and c0+1 are
        // inside the source. Pixels outside this range take the scalar
        // edge path.
        let src_c_for = |c: usize| x_origin_delta + (c as f64 + 0.5) * col_step - 0.5;
        let mut c_simd_start = 0usize;
        while c_simd_start < dw {
            let s = src_c_for(c_simd_start).floor() as i64;
            if s >= 0 {
                break;
            }
            c_simd_start += 1;
        }
        let mut c_simd_end = dw;
        while c_simd_end > 0 {
            let s = src_c_for(c_simd_end - 1).floor() as i64;
            if s + 1 < sw as i64 {
                break;
            }
            c_simd_end -= 1;
        }
        // Ensure room for at least one 8-lane block.
        let simd_end = if c_simd_end >= c_simd_start + 8 {
            c_simd_start + ((c_simd_end - c_simd_start) / 8) * 8
        } else {
            c_simd_start
        };

        // Scalar prefix (handles negative c0i edge).
        for c in 0..c_simd_start {
            scalar_bilinear_pixel(
                src, sw, row, c, src_c_for(c), r0, r1, dr, one_minus_dr,
            );
        }
        // SIMD interior — 8 dst pixels at a time using AVX2 + FMA.
        // Safety: AVX2 + FMA presence is guaranteed by target-cpu=native
        // on the Zen 4 / Zen 3 build targets. The kernel writes 8
        // exclusive u16 outputs and reads from `src`'s aliased read-only
        // slice; no overlap with `row`.
        let c_end = simd_end;
        if c_simd_start < c_end {
            #[cfg(target_arch = "x86_64")]
            unsafe {
                bilinear_interior_avx2(
                    src,
                    sw,
                    row,
                    c_simd_start,
                    c_end,
                    r0,
                    r1,
                    dr,
                    one_minus_dr,
                    x_origin_delta,
                    col_step,
                );
            }
            #[cfg(not(target_arch = "x86_64"))]
            for c in c_simd_start..c_end {
                scalar_bilinear_pixel(
                    src, sw, row, c, src_c_for(c), r0, r1, dr, one_minus_dr,
                );
            }
        }
        // Scalar suffix (post-SIMD + right edge).
        for c in c_end..dw {
            scalar_bilinear_pixel(
                src, sw, row, c, src_c_for(c), r0, r1, dr, one_minus_dr,
            );
        }
    });
    Ok(out)
}

/// AVX2 + FMA bilinear inner kernel. Processes 8 destination pixels at
/// a time. Memory loads are scalar (gather is microcoded and slower
/// than 8 contiguous-ish loads on AMD Zen 3/4); the arithmetic is
/// 256-bit packed FP with FMA. Cross-checked numerically against the
/// scalar path.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn bilinear_interior_avx2(
    src: &[u16],
    sw: usize,
    row: &mut [u16],
    c_start: usize,
    c_end: usize,
    r0: usize,
    r1: usize,
    dr: f32,
    one_minus_dr: f32,
    x_origin_delta: f64,
    col_step: f64,
) {
    let one_minus_dr_v = _mm256_set1_ps(one_minus_dr);
    let dr_v = _mm256_set1_ps(dr);
    let zero = _mm256_setzero_ps();
    let half = _mm256_set1_ps(0.5);
    let maxv = _mm256_set1_ps(65535.0);
    let one = _mm256_set1_ps(1.0);

    let mut c = c_start;
    while c + 8 <= c_end {
        // Compute the 8 source-x coordinates as f64, then split into
        // integer floor and fractional dc.
        let mut src_c_f64 = [0.0f64; 8];
        for i in 0..8 {
            src_c_f64[i] = x_origin_delta + ((c + i) as f64 + 0.5) * col_step - 0.5;
        }
        let mut c0 = [0usize; 8];
        let mut dc_arr = [0.0f32; 8];
        for i in 0..8 {
            let f = src_c_f64[i].floor();
            c0[i] = f as usize;
            dc_arr[i] = (src_c_f64[i] - f) as f32;
        }
        // Gather pixel values via scalar loads (faster than VGATHERDD on Zen).
        let mut v00_arr = [0.0f32; 8];
        let mut v01_arr = [0.0f32; 8];
        let mut v10_arr = [0.0f32; 8];
        let mut v11_arr = [0.0f32; 8];
        for i in 0..8 {
            let p = c0[i];
            v00_arr[i] = *src.get_unchecked(r0 + p) as f32;
            v01_arr[i] = *src.get_unchecked(r0 + p + 1) as f32;
            v10_arr[i] = *src.get_unchecked(r1 + p) as f32;
            v11_arr[i] = *src.get_unchecked(r1 + p + 1) as f32;
        }
        let dc = _mm256_loadu_ps(dc_arr.as_ptr());
        let one_minus_dc = _mm256_sub_ps(one, dc);
        let v00 = _mm256_loadu_ps(v00_arr.as_ptr());
        let v01 = _mm256_loadu_ps(v01_arr.as_ptr());
        let v10 = _mm256_loadu_ps(v10_arr.as_ptr());
        let v11 = _mm256_loadu_ps(v11_arr.as_ptr());
        // top = one_minus_dc * v00 + dc * v01
        let top = _mm256_fmadd_ps(dc, v01, _mm256_mul_ps(one_minus_dc, v00));
        // bot = one_minus_dc * v10 + dc * v11
        let bot = _mm256_fmadd_ps(dc, v11, _mm256_mul_ps(one_minus_dc, v10));
        // sum = one_minus_dr * top + dr * bot
        let sum = _mm256_fmadd_ps(dr_v, bot, _mm256_mul_ps(one_minus_dr_v, top));
        // rounded = clamp(sum + 0.5, 0, 65535) then cast to u16
        let rounded = _mm256_add_ps(sum, half);
        let clamped = _mm256_min_ps(_mm256_max_ps(rounded, zero), maxv);
        // Convert f32 → i32 (truncate) → pack to u16
        let i32v = _mm256_cvttps_epi32(clamped);
        // Split 256→128 then pack 4×i32 + 4×i32 into 8×u16 with saturation.
        let lo = _mm256_castsi256_si128(i32v);
        let hi = _mm256_extracti128_si256(i32v, 1);
        let packed = _mm_packus_epi32(lo, hi);
        // Store the 8 u16 outputs into row[c..c+8].
        _mm_storeu_si128(row.as_mut_ptr().add(c) as *mut __m128i, packed);
        c += 8;
    }
    // Tail (less than 8 pixels left) — scalar via the shared helper.
    let inv_src_px_unused = 0.0;
    let _ = inv_src_px_unused;
    for cc in c..c_end {
        let src_c_f64 = x_origin_delta + (cc as f64 + 0.5) * col_step - 0.5;
        scalar_bilinear_pixel(src, sw, row, cc, src_c_f64, r0, r1, dr, one_minus_dr);
    }
}

/// Bilinear resample reading directly from per-tile decoded buffers.
/// Eliminates the per-band ~5 MB stitch alloc + memcpy.
///
/// `decoded[idx]` is the raw u16-little-endian byte buffer of tile
/// (tx, ty) where idx = ty * level.tiles_x() + tx. The window's
/// (col_off, row_off) places the rendered rectangle within the COG.
/// `src_origin` is the world coord of pixel (window.col_off, window.row_off)
/// at `src_pixel_size`.
pub fn resample_tiles_to_u16(
    decoded: &HashMap<usize, Vec<u8>>,
    level: &OverviewLevel,
    window: PixelWindow,
    src_origin: [f64; 2],
    src_pixel_size: f64,
    dst_dim: (u32, u32),
    dst_origin: [f64; 2],
    dst_pixel_size: f64,
) -> Result<Vec<u16>> {
    let dw = dst_dim.0 as usize;
    let dh = dst_dim.1 as usize;
    let tiles_x = level.tiles_x() as usize;
    let tile_w = level.tile_width as usize;
    let tile_h = level.tile_height as usize;
    let bps = level.bytes_per_sample as usize;
    debug_assert_eq!(bps, 2, "tile-aware u16 resample expects 2-byte samples");

    // The window's pixel origin in absolute COG-level coords.
    let abs_col_off = window.col_off as i64;
    let abs_row_off = window.row_off as i64;

    // Per-row tile reuse: cache pointers to (tile_y0_row, tile_y1_row)
    // tiles and only refresh when the absolute tile-y indices change.
    let mut out = vec![0u16; dw * dh];

    let inv_src_px = 1.0 / src_pixel_size;
    let col_step = dst_pixel_size * inv_src_px;
    let x_origin_delta = (dst_origin[0] - src_origin[0]) * inv_src_px;
    let y_origin_delta = (src_origin[1] - dst_origin[1]) * inv_src_px;

    out.chunks_mut(dw).enumerate().for_each(|(r, row)| {
        // Compute absolute COG-level row coordinate (float).
        let src_r_rel = y_origin_delta + (r as f64 + 0.5) * col_step - 0.5;
        let abs_r = src_r_rel + abs_row_off as f64;
        let r0i = abs_r.floor() as i64;
        let dr = (abs_r - r0i as f64) as f32;
        let one_minus_dr = 1.0_f32 - dr;

        // Tile rows for r0 and r0+1 (in absolute COG pixel space).
        let r0_tile_y = if r0i >= 0 { (r0i as usize) / tile_h } else { usize::MAX };
        let r1_tile_y = if r0i + 1 >= 0 { ((r0i + 1) as usize) / tile_h } else { usize::MAX };
        let r0_local = if r0i >= 0 { (r0i as usize) % tile_h } else { 0 };
        let r1_local = if r0i + 1 >= 0 { ((r0i + 1) as usize) % tile_h } else { 0 };

        // Current tile pointers (refreshed only when tile_x changes).
        let mut cur_tx0: usize = usize::MAX;
        let mut cur_top_tile: Option<&Vec<u8>> = None;
        let mut cur_bot_tile: Option<&Vec<u8>> = None;
        // Also keep neighbours one tile to the right (for the c0+1 column
        // when it crosses a tile boundary).
        let mut cur_top_right: Option<&Vec<u8>> = None;
        let mut cur_bot_right: Option<&Vec<u8>> = None;

        let mut src_c_rel = x_origin_delta + 0.5 * col_step - 0.5;
        for c in 0..dw {
            let abs_c = src_c_rel + abs_col_off as f64;
            let c0i = abs_c.floor() as i64;
            let dc = (abs_c - c0i as f64) as f32;
            let one_minus_dc = 1.0_f32 - dc;

            let tx0 = if c0i >= 0 { (c0i as usize) / tile_w } else { usize::MAX };
            let c0_local = if c0i >= 0 { (c0i as usize) % tile_w } else { 0 };
            // Right pixel may live in the next tile.
            let c1i = c0i + 1;
            let tx1 = if c1i >= 0 { (c1i as usize) / tile_w } else { usize::MAX };
            let c1_local = if c1i >= 0 { (c1i as usize) % tile_w } else { 0 };

            // Refresh tile-pointer cache only when tile_x changes.
            if tx0 != cur_tx0 {
                cur_tx0 = tx0;
                cur_top_tile = if r0_tile_y != usize::MAX && tx0 != usize::MAX {
                    decoded.get(&(r0_tile_y * tiles_x + tx0))
                } else {
                    None
                };
                cur_bot_tile = if r1_tile_y != usize::MAX && tx0 != usize::MAX {
                    decoded.get(&(r1_tile_y * tiles_x + tx0))
                } else {
                    None
                };
                cur_top_right = if r0_tile_y != usize::MAX && tx1 != usize::MAX && tx1 != tx0 {
                    decoded.get(&(r0_tile_y * tiles_x + tx1))
                } else {
                    cur_top_tile
                };
                cur_bot_right = if r1_tile_y != usize::MAX && tx1 != usize::MAX && tx1 != tx0 {
                    decoded.get(&(r1_tile_y * tiles_x + tx1))
                } else {
                    cur_bot_tile
                };
            } else if tx1 != tx0 {
                // We crossed into a column-spanning boundary; refresh
                // only the right tile pointer.
                cur_top_right = if r0_tile_y != usize::MAX {
                    decoded.get(&(r0_tile_y * tiles_x + tx1))
                } else {
                    None
                };
                cur_bot_right = if r1_tile_y != usize::MAX {
                    decoded.get(&(r1_tile_y * tiles_x + tx1))
                } else {
                    None
                };
            }

            let v00 = sample_tile_u16(cur_top_tile, c0_local, r0_local, tile_w);
            let v01 = sample_tile_u16(
                if tx1 == tx0 { cur_top_tile } else { cur_top_right },
                c1_local,
                r0_local,
                tile_w,
            );
            let v10 = sample_tile_u16(cur_bot_tile, c0_local, r1_local, tile_w);
            let v11 = sample_tile_u16(
                if tx1 == tx0 { cur_bot_tile } else { cur_bot_right },
                c1_local,
                r1_local,
                tile_w,
            );

            let top = one_minus_dc * v00 + dc * v01;
            let bot = one_minus_dc * v10 + dc * v11;
            let sum = one_minus_dr * top + dr * bot;
            row[c] = (sum + 0.5).clamp(0.0, 65535.0) as u16;

            src_c_rel += col_step;
        }
    });
    Ok(out)
}

#[inline(always)]
fn sample_tile_u16(tile: Option<&Vec<u8>>, col: usize, row: usize, tile_w: usize) -> f32 {
    let Some(tile) = tile else { return 0.0 };
    let off = (row * tile_w + col) * 2;
    if off + 1 >= tile.len() {
        return 0.0;
    }
    // Tile bytes are LE u16; read fast via unaligned u16 load.
    u16::from_le_bytes([tile[off], tile[off + 1]]) as f32
}

#[inline]
fn scalar_bilinear_pixel(
    src: &[u16],
    sw: usize,
    row: &mut [u16],
    c: usize,
    src_c_f64: f64,
    r0: usize,
    r1: usize,
    dr: f32,
    one_minus_dr: f32,
) {
    let c0i = src_c_f64.floor() as i64;
    let dc = (src_c_f64 - c0i as f64) as f32;
    let one_minus_dc = 1.0_f32 - dc;
    let mut sum = 0.0_f32;
    let mut wsum = 0.0_f32;
    for (row_off, wy) in [(r0, one_minus_dr), (r1, dr)] {
        for (dx, wx) in [(0i64, one_minus_dc), (1, dc)] {
            let cc = c0i + dx;
            if cc < 0 || cc as usize >= sw {
                continue;
            }
            sum += src[row_off + cc as usize] as f32 * (wy * wx);
            wsum += wy * wx;
        }
    }
    let v = if wsum > 0.0 { sum / wsum } else { 0.0 };
    row[c] = (v + 0.5).clamp(0.0, 65535.0) as u16;
}

/// Nearest-neighbour u8 → u8 resample (for SCL / classification rasters).
pub fn resample_u8_to_u8(
    src: &[u8],
    src_dim: (u32, u32),
    src_origin: [f64; 2],
    src_pixel_size: f64,
    dst_dim: (u32, u32),
    dst_origin: [f64; 2],
    dst_pixel_size: f64,
) -> Result<Vec<u8>> {
    let (sw, sh) = (src_dim.0 as usize, src_dim.1 as usize);
    let (dw, dh) = (dst_dim.0 as usize, dst_dim.1 as usize);
    let mut out = vec![0u8; dw * dh];
    out.chunks_mut(dw).enumerate().for_each(|(r, row)| {
        let world_y = dst_origin[1] - (r as f64 + 0.5) * dst_pixel_size;
        let src_r = ((src_origin[1] - world_y) / src_pixel_size).floor() as i64;
        if src_r < 0 || src_r >= sh as i64 {
            return;
        }
        let row_off = src_r as usize * sw;
        for c in 0..dw {
            let world_x = dst_origin[0] + (c as f64 + 0.5) * dst_pixel_size;
            let src_c = ((world_x - src_origin[0]) / src_pixel_size).floor() as i64;
            if src_c >= 0 && src_c < sw as i64 {
                row[c] = src[row_off + src_c as usize];
            }
        }
    });
    Ok(out)
}
