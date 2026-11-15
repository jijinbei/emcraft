//! EMCraft's app icon and logo, drawn in code so it is reproducible: SolidCraft's family of
//! marks — blue blocks seen isometrically, lit from above, the faces parted by a clear gap — here
//! an E-plane bend of waveguide, its two open ends dark, a wave (orange) leaving the lower one.
//! The window takes [`rgba`]; the Windows executable and installer take the `.ico` file [`ico`]
//! writes (`cargo xtask icon`, checked in as `apps/emcraft/assets/emcraft.ico`); the README
//! shows [`png`]'s picture (`docs/media/emcraft-logo.png`).
//!
//! Each pixel's coverage comes from the signed distance of its centre to the shape (as in
//! [`crate::canvas`]), so one sample per pixel draws smooth edges at every size.
//!
//! ```
//! let px = emcraft_render::icon::rgba(32);
//! assert_eq!(px.len(), 32 * 32 * 4);
//! assert_eq!(emcraft_render::icon::rgba(0).len(), 4); // sides are clamped to 1..=1024
//! let ico = emcraft_render::icon::ico().unwrap();
//! assert_eq!(&ico[..4], &[0, 0, 1, 0]); // an icon directory
//! ```

use crate::canvas::Rgb;

/// The sizes in the `.ico` file: Windows' small, list, large and jumbo icons at 100–200 % scale.
pub const ICO_SIDES: [u32; 9] = [16, 20, 24, 32, 40, 48, 64, 128, 256];

/// The largest side [`rgba`] draws.
pub const MAX_SIDE: u32 = 1024;

// The mark's colours are the app's (SolidCraft's blues, its orange for what acts), the same in
// every theme, so they are written here rather than in a theme file. Each face runs from its
// first colour at its top to its second at its bottom.
const TOP: [Rgb; 2] = [Rgb(0x51, 0xad, 0xff), Rgb(0x2b, 0x83, 0xf1)];
const LEFT: [Rgb; 2] = [Rgb(0x25, 0x8d, 0xf5), Rgb(0x14, 0x61, 0xda)];
const RIGHT: [Rgb; 2] = [Rgb(0x1b, 0x66, 0xcf), Rgb(0x12, 0x45, 0x96)];
const OPENING: [Rgb; 2] = [Rgb(0x0b, 0x2f, 0x66), Rgb(0x15, 0x49, 0x9e)];
const WAVE: Rgb = Rgb(0xff, 0x6a, 0x3d);

/// The gap between faces and the wave's width, as a share of the side.
const GAP: f64 = 0.022;
const WAVE_WIDTH: f64 = 0.05;
/// The clear margin about the mark, as a share of the side.
const MARGIN: f64 = 0.04;

/// A point in the mark's model (x, y, z: the waveguide's units) seen isometrically: x to the
/// lower right, y to the lower left, z up (y down on the picture).
fn iso([x, y, z]: [f64; 3]) -> (f64, f64) {
    let (c, s) = (30f64.to_radians().cos(), 0.5);
    ((x - y) * c, (x + y) * s - z)
}

/// A face of the mark: its corners on the picture (convex) and its two colours.
struct Face {
    corners: Vec<(f64, f64)>,
    colours: [Rgb; 2],
    /// Parted from the faces about it by the gap (the openings lie on their face, unparted).
    parted: bool,
}

/// A block's three faces seen (top, right, left) from `lo` to `hi`.
fn block(lo: [f64; 3], hi: [f64; 3]) -> [Face; 3] {
    let [x0, y0, z0] = lo;
    let [x1, y1, z1] = hi;
    let face = |pts: [[f64; 3]; 4], colours| Face { corners: pts.map(iso).to_vec(), colours, parted: true };
    [
        face([[x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]], TOP),
        face([[x1, y0, z0], [x1, y1, z0], [x1, y1, z1], [x1, y0, z1]], RIGHT),
        face([[x0, y1, z0], [x1, y1, z0], [x1, y1, z1], [x0, y1, z1]], LEFT),
    ]
}

/// The bend's faces in drawing order (what is nearer drawn later), then the wave's points: the
/// upright arm's top open, the lower arm's end open, the wave leaving it along y.
fn mark() -> (Vec<Face>, Vec<(f64, f64)>) {
    const N: usize = 64;
    let opening = |pts: [[f64; 3]; 4]| Face { corners: pts.map(iso).to_vec(), colours: OPENING, parted: false };
    let mut faces: Vec<Face> = block([0.0, 0.0, 0.0], [1.2, 1.2, 2.6]).into_iter().chain(block([0.0, 1.2, 0.0], [1.2, 3.0, 1.2])).collect();
    faces.push(opening([[0.3, 3.0, 0.3], [0.9, 3.0, 0.3], [0.9, 3.0, 0.9], [0.3, 3.0, 0.9]]));
    faces.push(opening([[0.3, 0.3, 2.6], [0.9, 0.3, 2.6], [0.9, 0.9, 2.6], [0.3, 0.9, 2.6]]));
    let wave = (0..=N)
        .map(|i| {
            let t = i as f64 / N as f64;
            iso([0.6, 3.05 + 1.3 * t, 0.6 + 0.3 * (std::f64::consts::TAU * 1.5 * t).sin()])
        })
        .collect();
    (faces, wave)
}

/// Distance from `p` to the segment `a`–`b`.
fn to_segment(p: (f64, f64), a: (f64, f64), b: (f64, f64)) -> f64 {
    let (dx, dy) = (b.0 - a.0, b.1 - a.1);
    let len2 = dx * dx + dy * dy;
    let t = if len2 > 0.0 { (((p.0 - a.0) * dx + (p.1 - a.1) * dy) / len2).clamp(0.0, 1.0) } else { 0.0 };
    (p.0 - a.0 - t * dx).hypot(p.1 - a.1 - t * dy)
}

/// Signed distance from `p` to the convex polygon `corners` (negative inside).
fn to_polygon(p: (f64, f64), corners: &[(f64, f64)]) -> f64 {
    let n = corners.len();
    let edges = (0..n).map(|i| (corners[i], corners[(i + 1) % n]));
    let nearest = edges.clone().map(|(a, b)| to_segment(p, a, b)).fold(f64::INFINITY, f64::min);
    // Inside: on the same side of every edge.
    let sides: Vec<f64> = edges.map(|(a, b)| (b.0 - a.0) * (p.1 - a.1) - (b.1 - a.1) * (p.0 - a.0)).collect();
    let inside = sides.iter().all(|s| *s >= 0.0) || sides.iter().all(|s| *s <= 0.0);
    if inside { -nearest } else { nearest }
}

/// The icon at `side` × `side` pixels (clamped to 1..=[`MAX_SIDE`]): RGBA, rows top to bottom,
/// alpha not premultiplied, clear about the mark.
pub fn rgba(side: u32) -> Vec<u8> {
    let side = side.clamp(1, MAX_SIDE);
    let px = 1.0 / f64::from(side);
    let (faces, wave) = mark();
    // The mark fitted into the square inside the margin, centred.
    let all = faces.iter().flat_map(|f| f.corners.iter()).chain(&wave);
    let (lo, hi) = all.fold(((f64::INFINITY, f64::INFINITY), (f64::NEG_INFINITY, f64::NEG_INFINITY)), |(lo, hi), p| {
        ((lo.0.min(p.0), lo.1.min(p.1)), (hi.0.max(p.0), hi.1.max(p.1)))
    });
    let span = (hi.0 - lo.0).max(hi.1 - lo.1);
    let scale = (1.0 - 2.0 * MARGIN) / span;
    let offset = (0.5 - 0.5 * (lo.0 + hi.0) * scale, 0.5 - 0.5 * (lo.1 + hi.1) * scale);
    let fit = |p: &(f64, f64)| (p.0 * scale + offset.0, p.1 * scale + offset.1);
    let faces: Vec<Face> = faces.into_iter().map(|f| Face { corners: f.corners.iter().map(fit).collect(), ..f }).collect();
    let wave: Vec<(f64, f64)> = wave.iter().map(fit).collect();
    // Strokes and gaps no thinner than about a pixel stay visible at 16 px.
    let gap = GAP.max(0.8 * px);
    let wave_half = (0.5 * WAVE_WIDTH).max(0.85 * px);
    let coverage = |d: f64| (0.5 - d / px).clamp(0.0, 1.0);
    let mut out = Vec::with_capacity((side * side * 4) as usize);
    for y in 0..side {
        for x in 0..side {
            let p = ((f64::from(x) + 0.5) * px, (f64::from(y) + 0.5) * px);
            // Painted in order over a clear ground: a nearer face covers (and parts) what is behind.
            let (mut c, mut alpha) = (TOP[0], 0.0f64);
            for f in &faces {
                let d = to_polygon(p, &f.corners);
                let (top, bottom) = f.corners.iter().fold((f64::INFINITY, f64::NEG_INFINITY), |(t, b), q| (t.min(q.1), b.max(q.1)));
                let shade = f.colours[0].mix(f.colours[1], (p.1 - top) / (bottom - top).max(px));
                if f.parted {
                    // The gap about a face clears what is behind it there.
                    let cleared = coverage(d - gap);
                    alpha *= 1.0 - cleared;
                    let a = coverage(d);
                    c = if alpha > 0.0 { c.mix(shade, a) } else { shade };
                    alpha = alpha + a * (1.0 - alpha);
                } else {
                    c = c.mix(shade, coverage(d));
                }
            }
            let d = wave.windows(2).map(|w| to_segment(p, w[0], w[1])).fold(f64::INFINITY, f64::min) - wave_half;
            let a = coverage(d);
            c = if alpha > 0.0 { c.mix(WAVE, a) } else { WAVE };
            alpha = alpha + a * (1.0 - alpha);
            out.extend_from_slice(&[c.0, c.1, c.2, (alpha * 255.0).round() as u8]);
        }
    }
    out
}

/// The icon at `side` × `side` pixels as a PNG file (a desktop entry's icon); `None` when the
/// encoder fails.
///
/// ```
/// let png = emcraft_render::icon::png(256).unwrap();
/// assert_eq!(&png[1..4], b"PNG");
/// ```
pub fn png(side: u32) -> Option<Vec<u8>> {
    let side = side.clamp(1, MAX_SIDE);
    let mut buf = Vec::new();
    {
        let mut enc = png::Encoder::new(&mut buf, side, side);
        enc.set_color(png::ColorType::Rgba);
        enc.set_depth(png::BitDepth::Eight);
        let mut w = enc.write_header().ok()?;
        w.write_image_data(&rgba(side)).ok()?;
    }
    Some(buf)
}

/// A Windows `.ico` file of the icon at [`ICO_SIDES`], each a 32-bit bitmap: read by every
/// Windows tool, and the same bytes in every build (a PNG entry would depend on the encoder's
/// features). `None` only when a size overflows the format.
pub fn ico() -> Option<Vec<u8>> {
    let images: Vec<Vec<u8>> = ICO_SIDES.iter().map(|&side| bitmap(side)).collect();
    let count = u16::try_from(images.len()).ok()?;
    let mut out = Vec::new();
    out.extend_from_slice(&[0, 0, 1, 0]);
    out.extend_from_slice(&count.to_le_bytes());
    let mut offset = 6 + 16 * u32::from(count);
    for (&side, image) in ICO_SIDES.iter().zip(&images) {
        let dim = if side >= 256 { 0 } else { side as u8 };
        let len = u32::try_from(image.len()).ok()?;
        out.extend_from_slice(&[dim, dim, 0, 0]);
        out.extend_from_slice(&1u16.to_le_bytes());
        out.extend_from_slice(&32u16.to_le_bytes());
        out.extend_from_slice(&len.to_le_bytes());
        out.extend_from_slice(&offset.to_le_bytes());
        offset = offset.checked_add(len)?;
    }
    for image in images {
        out.extend(image);
    }
    Some(out)
}

/// An icon entry as a device-independent bitmap: the header, BGRA rows bottom to top, then the
/// 1-bit transparency mask (rows padded to 4 bytes).
fn bitmap(side: u32) -> Vec<u8> {
    let side = side.clamp(1, MAX_SIDE);
    let n = side as usize;
    let px = rgba(side);
    let mask_row = n.div_ceil(32) * 4;
    let mut out = Vec::with_capacity(40 + n * n * 4 + mask_row * n);
    let size_image = (n * n * 4 + mask_row * n) as u32;
    out.extend_from_slice(&40u32.to_le_bytes());
    out.extend_from_slice(&side.to_le_bytes());
    out.extend_from_slice(&(2 * side).to_le_bytes());
    out.extend_from_slice(&1u16.to_le_bytes());
    out.extend_from_slice(&32u16.to_le_bytes());
    out.extend_from_slice(&0u32.to_le_bytes());
    out.extend_from_slice(&size_image.to_le_bytes());
    out.extend_from_slice(&[0; 16]);
    for row in px.chunks_exact(n * 4).rev() {
        for p in row.as_chunks::<4>().0 {
            out.extend_from_slice(&[p[2], p[1], p[0], p[3]]);
        }
    }
    for row in px.chunks_exact(n * 4).rev() {
        let mut bits = vec![0u8; mask_row];
        for (x, p) in row.as_chunks::<4>().0.iter().enumerate() {
            if p[3] == 0 {
                bits[x / 8] |= 0x80 >> (x % 8);
            }
        }
        out.extend(bits);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Writes the icon at a few sides into `EMCRAFT_ICON_DIR` to look at (nothing without it).
    #[test]
    fn pictures_for_a_look() {
        let Some(dir) = std::env::var_os("EMCRAFT_ICON_DIR") else { return };
        for side in [16u32, 32, 64, 256, 512] {
            std::fs::write(std::path::Path::new(&dir).join(format!("icon-{side}.png")), png(side).unwrap()).unwrap();
        }
    }

    #[test]
    fn the_icon_is_drawn_the_same_every_time() {
        assert_eq!(rgba(48), rgba(48));
        assert_eq!(ico(), ico());
    }

    /// The corners are clear and the bend's side is opaque blue; the wave is orange where it
    /// leaves the lower opening.
    #[test]
    fn the_mark_is_drawn_on_a_clear_ground() {
        for side in [16u32, 32, 256] {
            let px = rgba(side);
            let n = side as usize;
            let at = |x: usize, y: usize| &px[(y * n + x) * 4..(y * n + x) * 4 + 4];
            assert_eq!(at(0, 0)[3], 0, "{side}");
            assert_eq!(at(n - 1, n - 1)[3], 0, "{side}");
            let blue =
                (0..n).flat_map(|y| (0..n).map(move |x| (x, y))).filter(|&(x, y)| at(x, y)[3] >= 200 && at(x, y)[2] > 150 && at(x, y)[0] < 120);
            assert!(blue.count() > n * n / 8, "{side}: little blue");
            let orange =
                (0..n).flat_map(|y| (0..n).map(move |x| (x, y))).filter(|&(x, y)| at(x, y)[3] > 200 && at(x, y)[0] > 200 && at(x, y)[2] < 120);
            assert!(orange.count() > 0, "{side}: no wave");
        }
    }

    #[test]
    fn the_ico_directory_points_at_each_image() {
        let ico = ico().unwrap();
        let count = u16::from_le_bytes([ico[4], ico[5]]) as usize;
        assert_eq!(count, ICO_SIDES.len());
        for (i, &side) in ICO_SIDES.iter().enumerate() {
            let e = &ico[6 + 16 * i..6 + 16 * (i + 1)];
            assert_eq!(u32::from(e[0]), side % 256);
            let len = u32::from_le_bytes([e[8], e[9], e[10], e[11]]) as usize;
            let off = u32::from_le_bytes([e[12], e[13], e[14], e[15]]) as usize;
            let image = &ico[off..off + len];
            let n = side as usize;
            assert_eq!(len, 40 + n * n * 4 + n.div_ceil(32) * 4 * n);
            assert_eq!(u32::from_le_bytes([image[4], image[5], image[6], image[7]]), side);
            assert_eq!(u32::from_le_bytes([image[8], image[9], image[10], image[11]]), 2 * side);
        }
        let last = &ico[6 + 16 * (count - 1)..6 + 16 * count];
        let end = u32::from_le_bytes([last[12], last[13], last[14], last[15]]) + u32::from_le_bytes([last[8], last[9], last[10], last[11]]);
        assert_eq!(end as usize, ico.len());
    }
}
