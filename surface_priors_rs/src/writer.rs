//! Minimal GeoTIFF writer.
//!
//! Emits a single-band uint16 GeoTIFF with DEFLATE compression, tiled
//! at 256×256, with the GeoKey directory needed for downstream GIS
//! tools to recognise a UTM north EPSG. Just enough to be opened by
//! `rasterio.open` for verification against the Python output.

use anyhow::{Context, Result};
use byteorder::{LittleEndian, WriteBytesExt};
use flate2::write::ZlibEncoder;
use flate2::Compression;
use std::collections::BTreeMap;
use std::fs::File;
use std::io::{BufWriter, Seek, SeekFrom, Write};

const TIFF_HEADER: [u8; 4] = [b'I', b'I', 42, 0];

// IFD tags we emit.
const T_IMAGE_WIDTH: u16 = 256;
const T_IMAGE_LENGTH: u16 = 257;
const T_BITS_PER_SAMPLE: u16 = 258;
const T_COMPRESSION: u16 = 259;
const T_PHOTOMETRIC: u16 = 262;
const T_SAMPLES_PER_PIXEL: u16 = 277;
const T_PLANAR_CONFIG: u16 = 284;
const T_PREDICTOR: u16 = 317;
const T_TILE_WIDTH: u16 = 322;
const T_TILE_LENGTH: u16 = 323;
const T_TILE_OFFSETS: u16 = 324;
const T_TILE_BYTE_COUNTS: u16 = 325;
const T_SAMPLE_FORMAT: u16 = 339;
const T_NODATA: u16 = 42113;
const T_PIXEL_SCALE: u16 = 33550;
const T_TIE_POINT: u16 = 33922;
const T_GEO_KEY_DIR: u16 = 34735;

// TIFF data types.
const TIFF_BYTE: u16 = 1;
const TIFF_ASCII: u16 = 2;
const TIFF_SHORT: u16 = 3;
const TIFF_LONG: u16 = 4;
const TIFF_DOUBLE: u16 = 12;

pub struct GeoTiffParams {
    pub width: u32,
    pub height: u32,
    pub tile_size: u32,
    pub epsg: u32,
    pub origin: [f64; 2], // (xmin, ymax)
    pub pixel_size: f64,
    pub nodata: u16,
}

pub fn write_int16_geotiff(path: &std::path::Path, data: &[i16], params: GeoTiffParams) -> Result<()> {
    let as_u16: Vec<u16> = data.iter().map(|&v| v as u16).collect();
    write_geotiff_impl(path, &as_u16, params, /* signed: */ true)
}

pub fn write_uint16_geotiff(path: &std::path::Path, data: &[u16], params: GeoTiffParams) -> Result<()> {
    write_geotiff_impl(path, data, params, false)
}

fn write_geotiff_impl(
    path: &std::path::Path,
    data: &[u16],
    params: GeoTiffParams,
    signed: bool,
) -> Result<()> {
    let file = File::create(path).with_context(|| format!("create {}", path.display()))?;
    let mut w = BufWriter::new(file);
    w.write_all(&TIFF_HEADER)?;
    // Placeholder for IFD offset; rewrite after data + tiles are written.
    let ifd_ptr = 4u64;
    w.write_u32::<LittleEndian>(0)?;

    let tw = params.tile_size;
    let th = params.tile_size;
    let tiles_x = (params.width + tw - 1) / tw;
    let tiles_y = (params.height + th - 1) / th;
    let mut offsets = Vec::with_capacity((tiles_x * tiles_y) as usize);
    let mut byte_counts = Vec::with_capacity((tiles_x * tiles_y) as usize);

    let mut tile_buf = vec![0u16; (tw * th) as usize];
    for ty in 0..tiles_y {
        for tx in 0..tiles_x {
            // Copy region from data into tile buffer (zero/nodata-padded).
            for b in tile_buf.iter_mut() {
                *b = params.nodata;
            }
            for r in 0..th {
                let src_row = ty * th + r;
                if src_row >= params.height {
                    break;
                }
                for c in 0..tw {
                    let src_col = tx * tw + c;
                    if src_col >= params.width {
                        break;
                    }
                    tile_buf[(r * tw + c) as usize] =
                        data[(src_row * params.width + src_col) as usize];
                }
            }
            // Serialize to LE u16 bytes.
            let mut raw: Vec<u8> = Vec::with_capacity((tw * th * 2) as usize);
            for &v in &tile_buf {
                raw.write_u16::<LittleEndian>(v)?;
            }
            // DEFLATE compress.
            let mut enc = ZlibEncoder::new(Vec::new(), Compression::new(6));
            enc.write_all(&raw)?;
            let compressed = enc.finish()?;
            let offset = w.stream_position()?;
            w.write_all(&compressed)?;
            offsets.push(offset as u32);
            byte_counts.push(compressed.len() as u32);
        }
    }

    // Align to 2 bytes before writing IFD.
    let ifd_offset = align(&mut w, 2)?;

    // Allocate a sidecar arena for tag values that don't fit in 4 bytes.
    // 18 entries is comfortably above the 16 push_short + 6 offset entries we emit.
    const N_ENTRIES: u64 = 18;
    let mut entries: Vec<IfdEntry> = Vec::new();
    let mut sidecar: Vec<u8> = Vec::new();
    let sidecar_base = ifd_offset + 2 + 12 * N_ENTRIES + 4; // entries + count + next-IFD pointer

    push_short(&mut entries, T_IMAGE_WIDTH, params.width);
    push_short(&mut entries, T_IMAGE_LENGTH, params.height);
    push_short(&mut entries, T_BITS_PER_SAMPLE, 16);
    push_short(&mut entries, T_COMPRESSION, 8);
    push_short(&mut entries, T_PHOTOMETRIC, 1); // BlackIsZero
    push_short(&mut entries, T_SAMPLES_PER_PIXEL, 1);
    push_short(&mut entries, T_PLANAR_CONFIG, 1); // chunky
    push_short(&mut entries, T_PREDICTOR, 1); // none
    push_short(&mut entries, T_TILE_WIDTH, tw);
    push_short(&mut entries, T_TILE_LENGTH, th);

    let offsets_offset = sidecar_base + sidecar.len() as u64;
    for o in &offsets {
        sidecar.write_u32::<LittleEndian>(*o)?;
    }
    entries.push(IfdEntry {
        tag: T_TILE_OFFSETS,
        typ: TIFF_LONG,
        count: offsets.len() as u32,
        value: TagValue::Offset(offsets_offset),
    });

    let bytecounts_offset = sidecar_base + sidecar.len() as u64;
    for c in &byte_counts {
        sidecar.write_u32::<LittleEndian>(*c)?;
    }
    entries.push(IfdEntry {
        tag: T_TILE_BYTE_COUNTS,
        typ: TIFF_LONG,
        count: byte_counts.len() as u32,
        value: TagValue::Offset(bytecounts_offset),
    });

    push_short(&mut entries, T_SAMPLE_FORMAT, if signed { 2 } else { 1 }); // 1=unsigned, 2=signed

    // Nodata as ASCII string.
    let nodata_str = format!("{}\0", params.nodata);
    let nodata_offset = sidecar_base + sidecar.len() as u64;
    sidecar.extend_from_slice(nodata_str.as_bytes());
    entries.push(IfdEntry {
        tag: T_NODATA,
        typ: TIFF_ASCII,
        count: nodata_str.len() as u32,
        value: TagValue::Offset(nodata_offset),
    });

    // PixelScale (3 doubles)
    let pixel_scale_offset = sidecar_base + sidecar.len() as u64;
    sidecar.write_f64::<LittleEndian>(params.pixel_size)?;
    sidecar.write_f64::<LittleEndian>(params.pixel_size)?;
    sidecar.write_f64::<LittleEndian>(0.0)?;
    entries.push(IfdEntry {
        tag: T_PIXEL_SCALE,
        typ: TIFF_DOUBLE,
        count: 3,
        value: TagValue::Offset(pixel_scale_offset),
    });

    // TiePoint (6 doubles): pixel (0,0,0) → world (xmin, ymax, 0)
    let tie_point_offset = sidecar_base + sidecar.len() as u64;
    for v in [0.0, 0.0, 0.0, params.origin[0], params.origin[1], 0.0] {
        sidecar.write_f64::<LittleEndian>(v)?;
    }
    entries.push(IfdEntry {
        tag: T_TIE_POINT,
        typ: TIFF_DOUBLE,
        count: 6,
        value: TagValue::Offset(tie_point_offset),
    });

    // Minimal GeoKey directory: declare ProjectedCRSTypeGeoKey = epsg.
    // Header: (1, 1, 0, 1) followed by one key triplet.
    let geo_keys_offset = sidecar_base + sidecar.len() as u64;
    sidecar.write_u16::<LittleEndian>(1)?; // KeyDirectoryVersion
    sidecar.write_u16::<LittleEndian>(1)?; // KeyRevision
    sidecar.write_u16::<LittleEndian>(0)?; // MinorRevision
    sidecar.write_u16::<LittleEndian>(1)?; // NumberOfKeys
    // GTModelTypeGeoKey would also be standard; for projected we set
    // ProjectedCSTypeGeoKey (key id 3072) → EPSG.
    sidecar.write_u16::<LittleEndian>(3072)?; // KeyID
    sidecar.write_u16::<LittleEndian>(0)?; // TIFFTagLocation (0 = stored in this directory)
    sidecar.write_u16::<LittleEndian>(1)?; // Count
    sidecar.write_u16::<LittleEndian>(params.epsg as u16)?; // Value
    entries.push(IfdEntry {
        tag: T_GEO_KEY_DIR,
        typ: TIFF_SHORT,
        count: 8,
        value: TagValue::Offset(geo_keys_offset),
    });

    // Now write entries sorted by tag.
    entries.sort_by_key(|e| e.tag);
    let n_entries = entries.len();
    if n_entries as u64 > N_ENTRIES {
        anyhow::bail!("internal: writer assumes ≤{N_ENTRIES} IFD entries; got {n_entries}");
    }
    // Pad entry list out to N_ENTRIES to keep sidecar_base stable. The
    // 0xFFFE padding tag is reserved-private; well-behaved readers ignore it.
    while (entries.len() as u64) < N_ENTRIES {
        entries.push(IfdEntry {
            tag: 0xFFFE,
            typ: TIFF_BYTE,
            count: 0,
            value: TagValue::Inline([0u8; 4]),
        });
    }

    w.write_u16::<LittleEndian>(entries.len() as u16)?;
    for e in &entries {
        w.write_u16::<LittleEndian>(e.tag)?;
        w.write_u16::<LittleEndian>(e.typ)?;
        w.write_u32::<LittleEndian>(e.count)?;
        match e.value {
            TagValue::Inline(b) => w.write_all(&b)?,
            TagValue::Offset(o) => w.write_u32::<LittleEndian>(o as u32)?,
        }
    }
    w.write_u32::<LittleEndian>(0)?; // no next IFD

    // Now write the sidecar block.
    let sidecar_actual_offset = w.stream_position()?;
    if sidecar_actual_offset != sidecar_base {
        return Err(anyhow::anyhow!(
            "internal sidecar alignment off: expected {sidecar_base}, got {sidecar_actual_offset}"
        ));
    }
    w.write_all(&sidecar)?;

    // Patch IFD pointer at byte 4 of the file.
    w.seek(SeekFrom::Start(ifd_ptr))?;
    w.write_u32::<LittleEndian>(ifd_offset as u32)?;
    w.flush()?;
    Ok(())
}

fn align<W: Seek + Write>(w: &mut W, align: u64) -> Result<u64> {
    let pos = w.stream_position()?;
    let pad = (align - (pos % align)) % align;
    if pad > 0 {
        let zeros = vec![0u8; pad as usize];
        w.write_all(&zeros)?;
    }
    Ok(w.stream_position()?)
}

#[derive(Debug)]
struct IfdEntry {
    tag: u16,
    typ: u16,
    count: u32,
    value: TagValue,
}

#[derive(Debug, Clone, Copy)]
enum TagValue {
    Inline([u8; 4]),
    Offset(u64),
}

fn push_short(entries: &mut Vec<IfdEntry>, tag: u16, value: u32) {
    let mut buf = [0u8; 4];
    buf[0..2].copy_from_slice(&(value as u16).to_le_bytes());
    entries.push(IfdEntry {
        tag,
        typ: TIFF_SHORT,
        count: 1,
        value: TagValue::Inline(buf),
    });
}

#[allow(dead_code)]
fn _silence(_: BTreeMap<u16, u16>) {}
