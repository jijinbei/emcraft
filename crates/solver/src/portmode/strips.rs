//! Printed lines' cross-sections: strips (sheets, or of a thickness) at their heights in a stack of
//! dielectric slabs and blocks (a solder mask over the strips, say), inside a conducting box whose floor and ceiling are the lines' reference
//! planes (or far from them, for an open side), solved quasi-statically for their impedances —
//! one strip a single line, two a coupled pair (its odd and even modes, the differential and
//! common impedances).
//!
//! **Quasi-TEM by capacitance** (the standard of planar lines: R. E. Collin, *Foundations for
//! Microwave Engineering*, 2nd ed., 1992, §3.11; Paul 2008, §§5.2–5.3). Each strip held at 1 V,
//! the others and the box at 0, the potential solves ∇·(εr ∇φ) = 0 by linear finite elements; the
//! capacitances per unit length C_jk = ε₀ ∫ εr ∇φ_j·∇φ_k follow, and C⁰ alike with every slab
//! vacuum. A line's Z = 1/(c √(C C⁰)) and ε_eff = C/C⁰. A pair's odd and even capacitances are
//! C_o = C̄ − C₁₂, C_e = C̄ + C₁₂ (C̄ the mean of C₁₁ and C₂₂: exact for like strips, the
//! symmetric pair's), Z_o and Z_e from them, Z_diff = 2 Z_o and Z_common = Z_e/2 (Cohn 1955).
//!
//! **The grid.** A tensor grid of rectangles, each cut into two triangles: its lines through the
//! box's sides, each strip's edges, floor and ceiling, each slab's and block's faces and each
//! strip's bottom and top,
//! spaced geometrically from a fine size at the strips' edges and heights (where the field is
//! singular) to a coarse one away from them. A strip is the grid's nodes in its rectangle (a sheet:
//! along its height between its two edges); a triangle's permittivity is that of the block, else
//! the slab, its middle lies in. The same lines give the same grid, and the same impedances, bit
//! for bit.
//!
//! ```
//! use emcraft_solver::portmode::strips::{Section, Slab, Strip};
//! // A 0.6 mm strip on 0.635 mm of alumina (εr 9.8), open above: about 50 Ω.
//! let h = 0.635e-3;
//! let section = Section {
//!     half_width: 20.0 * h,
//!     floor: 0.0,
//!     ceiling: 20.0 * h,
//!     slabs: vec![Slab { bottom: 0.0, top: h, eps: 9.8 }],
//!     blocks: vec![],
//!     strips: vec![Strip { left: -0.3e-3, right: 0.3e-3, height: h, thickness: 0.0 }],
//! };
//! let line = section.solve()?;
//! assert!((line.single().unwrap() - 50.0).abs() < 2.0);
//! // A strip outside the box is refused.
//! let outside = Section { strips: vec![Strip { left: 0.0, right: 1.0, height: h, thickness: 0.0 }], ..section };
//! assert!(outside.solve().is_err());
//! # Ok::<(), emcraft_solver::SolveError>(())
//! ```

use emcraft_model::Point3;
use emcraft_model::units::{C0, EPS0};
use num_complex::Complex64 as C;

use crate::driven::{Result, invalid};
use crate::ldlt::Ldlt;
use crate::sparse::Csr;

/// A dielectric slab between `bottom` and `top` (m), of relative permittivity `eps`; the rest of
/// the box is vacuum.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Slab {
    pub bottom: f64,
    pub top: f64,
    pub eps: f64,
}

/// A rectangle of dielectric from `left` to `right` across and `bottom` to `top` up (m), of
/// relative permittivity `eps`: over the slabs where it lies.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Block {
    pub left: f64,
    pub right: f64,
    pub bottom: f64,
    pub top: f64,
    pub eps: f64,
}

/// A strip from `left` to `right` across, from `height` up by `thickness` (m; 0, a sheet).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Strip {
    pub left: f64,
    pub right: f64,
    pub height: f64,
    pub thickness: f64,
}

impl Strip {
    /// Its top face's height (m).
    pub fn top(&self) -> f64 {
        self.height + self.thickness
    }
}

/// A cross-section: the box from −`half_width` to `half_width` across and from `floor` to
/// `ceiling` (m), its walls conducting, its slabs, its blocks and its one or two strips.
#[derive(Clone, Debug, PartialEq)]
pub struct Section {
    pub half_width: f64,
    pub floor: f64,
    pub ceiling: f64,
    pub slabs: Vec<Slab>,
    pub blocks: Vec<Block>,
    pub strips: Vec<Strip>,
}

/// A mode of the line: its effective permittivity, its impedance per conductor (Ω) and whether
/// its strips' voltages are opposite (a pair's odd mode).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Mode {
    pub eps_eff: f64,
    pub impedance: f64,
    pub odd: bool,
}

/// The line's quasi-TEM modes, one per strip (a pair's even mode first).
#[derive(Clone, Debug, PartialEq)]
pub struct Line {
    pub modes: Vec<Mode>,
}

impl Line {
    /// A single strip's characteristic impedance (Ω).
    pub fn single(&self) -> Option<f64> {
        match self.modes[..] {
            [m] => Some(m.impedance),
            _ => None,
        }
    }

    /// A pair's odd and even modes.
    pub fn pair(&self) -> Option<(Mode, Mode)> {
        match self.modes[..] {
            [a, b] if a.odd != b.odd => Some(if a.odd { (a, b) } else { (b, a) }),
            _ => None,
        }
    }
}

/// Geometric spacing's growth from one line of the grid to the next.
const GROWTH: f64 = 1.2;
/// The finest spacing, as a fraction of the narrowest strip (or its nearest neighbour across).
const FINE: f64 = 1.0 / 100.0;
/// The coarsest spacing up, as a fraction of the box's height; across, at most `ASPECT` times it.
const COARSE: f64 = 1.0 / 20.0;
const ASPECT: f64 = 4.0;
/// The most grid lines along either axis.
const MAX_LINES: usize = 1200;
/// The most dielectric blocks of a section.
const MAX_BLOCKS: usize = 16;

/// The points strictly between `a` and `b`: the space at x the smaller of the grades from either
/// end — `fine` there growing by `GROWTH` a space (its spacing at distance d about fine + (GROWTH −
/// 1) d) — and `coarse`; an end without a fine spacing is coarse there.
fn graded(a: f64, b: f64, fine_a: Option<f64>, fine_b: Option<f64>, coarse: f64) -> Vec<f64> {
    let space = |x: f64| {
        let from = |fine: Option<f64>, d: f64| fine.map_or(coarse, |f| f + (GROWTH - 1.0) * d);
        from(fine_a, x - a).min(from(fine_b, b - x)).min(coarse)
    };
    let mut out = Vec::new();
    let mut x = a;
    while out.len() < MAX_LINES {
        let h = space(x);
        if !(h > 0.0) || x + 1.5 * h >= b {
            break;
        }
        x += h;
        out.push(x);
    }
    out
}

/// The grid lines along one axis through `keys` (sorted, distinct), each key's fine spacing if it
/// has one.
fn axis(keys: &[(f64, Option<f64>)], coarse: f64) -> Vec<f64> {
    let mut out = Vec::new();
    for w in keys.windows(2) {
        let [(a, fa), (b, fb)] = [w[0], w[1]];
        out.push(a);
        out.extend(graded(a, b, fa, fb, coarse));
    }
    if let Some((last, _)) = keys.last() {
        out.push(*last);
    }
    out
}

/// Sorted keys without near duplicates (within a billionth of `scale`), each keeping the finest
/// spacing asked at it.
fn keys(mut ks: Vec<(f64, Option<f64>)>, scale: f64) -> Vec<(f64, Option<f64>)> {
    ks.sort_by(|a, b| a.0.total_cmp(&b.0));
    let mut out: Vec<(f64, Option<f64>)> = Vec::new();
    for (x, f) in ks {
        match out.last_mut() {
            Some(last) if (x - last.0).abs() <= 1e-9 * scale => {
                last.1 = match (last.1, f) {
                    (Some(p), Some(q)) => Some(p.min(q)),
                    (p, q) => p.or(q),
                };
            }
            _ => out.push((x, f)),
        }
    }
    out
}

impl Section {
    fn check(&self) -> Result<()> {
        let ok = |x: f64| x.is_finite();
        let size = self.half_width.min(self.ceiling - self.floor);
        if !(ok(self.half_width) && ok(self.floor) && ok(self.ceiling) && self.half_width > 0.0 && self.ceiling > self.floor && size > 0.0) {
            return Err(invalid("line cross-section", "needs a box of finite, positive size"));
        }
        if self.strips.is_empty() || self.strips.len() > 2 {
            return Err(invalid("line cross-section", "needs one strip or a pair"));
        }
        for s in &self.strips {
            if !(ok(s.left)
                && ok(s.right)
                && ok(s.height)
                && ok(s.thickness)
                && s.thickness >= 0.0
                && s.left < s.right
                && -self.half_width < s.left
                && s.right < self.half_width
                && self.floor < s.height
                && s.top() < self.ceiling)
            {
                return Err(invalid(
                    "line cross-section",
                    "a strip must lie inside the box, off its walls, its left edge left of its right, its thickness ≥ 0",
                ));
            }
        }
        if let [a, b] = self.strips[..]
            && a.height <= b.top()
            && b.height <= a.top()
            && a.left < b.right
            && b.left < a.right
        {
            return Err(invalid("line cross-section", "the two strips overlap"));
        }
        if !self.slabs.iter().all(|s| ok(s.bottom) && ok(s.top) && s.bottom < s.top && s.eps.is_finite() && s.eps >= 1.0) {
            return Err(invalid("line cross-section", "a slab needs a finite bottom below its top and a permittivity of at least 1"));
        }
        if self.blocks.len() > MAX_BLOCKS
            || !self.blocks.iter().all(|b| {
                ok(b.left) && ok(b.right) && ok(b.bottom) && ok(b.top) && b.left < b.right && b.bottom < b.top && b.eps.is_finite() && b.eps >= 1.0
            })
        {
            return Err(invalid(
                "line cross-section",
                format!("at most {MAX_BLOCKS} blocks, each with finite sides in order and a permittivity of at least 1"),
            ));
        }
        Ok(())
    }

    /// The grid: its lines across (u) and up (v).
    fn grid(&self) -> Result<(Vec<f64>, Vec<f64>)> {
        let width = self.strips.iter().map(|s| s.right - s.left).fold(f64::INFINITY, f64::min);
        // The gap between a pair's strips, and each strip's height above the nearest plane or
        // slab face, also set the finest spacing.
        let gap = match self.strips[..] {
            [a, b] => (b.left - a.right).abs().min((a.left - b.right).abs()).max((a.height - b.height).abs()),
            _ => f64::INFINITY,
        };
        let fine = FINE * width.min(gap);
        let coarse_up = COARSE * (self.ceiling - self.floor);
        let coarse_across = (COARSE * 2.0 * self.half_width).min(ASPECT * coarse_up);
        let scale = 2.0 * self.half_width + (self.ceiling - self.floor);
        let mut across = vec![(-self.half_width, None), (self.half_width, None)];
        for s in &self.strips {
            across.extend([(s.left, Some(fine)), (s.right, Some(fine))]);
        }
        let mut up = vec![(self.floor, None), (self.ceiling, None)];
        for s in &self.strips {
            up.extend([(s.height, Some(fine)), (s.top(), Some(fine))]);
        }
        // A block's faces are fine too (they lie by the strips, so the grading from the strips
        // goes on past them).
        let inside = |v: f64, lo: f64, hi: f64| lo < v && v < hi;
        let faces = self.slabs.iter().flat_map(|s| [(s.bottom, None), (s.top, None)]);
        for (v, f) in faces.chain(self.blocks.iter().flat_map(|b| [(b.bottom, Some(fine)), (b.top, Some(fine))])) {
            if inside(v, self.floor, self.ceiling) {
                up.push((v, f));
            }
        }
        for u in self.blocks.iter().flat_map(|b| [b.left, b.right]) {
            if inside(u, -self.half_width, self.half_width) {
                across.push((u, Some(fine)));
            }
        }
        let (u, v) = (axis(&keys(across, scale), coarse_across), axis(&keys(up, scale), coarse_up));
        if u.len() > MAX_LINES || v.len() > MAX_LINES {
            return Err(invalid("line cross-section", "its box is too large for its strips: the grid would have too many lines"));
        }
        Ok((u, v))
    }

    /// The capacitances per unit length (F/m) between the strips, C_jk, with the slabs' permittivities
    /// (`vacuum` false) or without: the potential of each strip at 1 V by linear elements.
    fn capacitances(&self, u: &[f64], v: &[f64], vacuum: bool) -> Result<Vec<Vec<f64>>> {
        let (nu, nv) = (u.len(), v.len());
        let node = |i: usize, j: usize| i + nu * j;
        let eps_at = |p: [f64; 2]| {
            if vacuum {
                return 1.0;
            }
            let block = self.blocks.iter().find(|b| b.left <= p[0] && p[0] < b.right && b.bottom <= p[1] && p[1] < b.top);
            block.map(|b| b.eps).or_else(|| self.slabs.iter().find(|s| s.bottom <= p[1] && p[1] < s.top).map(|s| s.eps)).unwrap_or(1.0)
        };
        let find = |xs: &[f64], x: f64| xs.iter().position(|y| (y - x).abs() <= 1e-12 * (1.0 + x.abs()));
        // Each node's strip (k + 1), or 0 free, or the box (u32::MAX).
        let mut fixed = vec![0u32; nu * nv];
        for j in 0..nv {
            for i in 0..nu {
                if i == 0 || j == 0 || i + 1 == nu || j + 1 == nv {
                    fixed[node(i, j)] = u32::MAX;
                }
            }
        }
        for (k, s) in self.strips.iter().enumerate() {
            let (Some(i0), Some(i1), Some(j0), Some(j1)) = (find(u, s.left), find(u, s.right), find(v, s.height), find(v, s.top())) else {
                return Err(invalid("line cross-section", "a strip's edges are not on the grid"));
            };
            for j in j0..=j1 {
                for i in i0..=i1 {
                    fixed[node(i, j)] = k as u32 + 1;
                }
            }
        }
        // The free nodes' unknowns.
        let mut unknown = vec![usize::MAX; nu * nv];
        let mut points = Vec::new();
        for (n, f) in fixed.iter().enumerate() {
            if *f == 0 {
                unknown[n] = points.len();
                points.push(Point3::new(u[n % nu], v[n / nu], 0.0));
            }
        }
        // The triangles: corners, εr, and the element matrix εr ∇λ_a·∇λ_b area.
        let mut tris: Vec<([usize; 3], [[f64; 3]; 3])> = Vec::with_capacity(2 * (nu - 1) * (nv - 1));
        for j in 0..nv - 1 {
            for i in 0..nu - 1 {
                let [a, b, c, d] = [node(i, j), node(i + 1, j), node(i + 1, j + 1), node(i, j + 1)];
                for t in [[a, b, c], [a, c, d]] {
                    let p = t.map(|n| [u[n % nu], v[n / nu]]);
                    let e = eps_at([(p[0][0] + p[1][0] + p[2][0]) / 3.0, (p[0][1] + p[1][1] + p[2][1]) / 3.0]);
                    let area2 = (p[1][0] - p[0][0]) * (p[2][1] - p[0][1]) - (p[2][0] - p[0][0]) * (p[1][1] - p[0][1]);
                    // ∇λ_k = (y_{k+1} − y_{k+2}, x_{k+2} − x_{k+1}) / 2A.
                    let g: [[f64; 2]; 3] = std::array::from_fn(|k| {
                        let (q, r) = (p[(k + 1) % 3], p[(k + 2) % 3]);
                        [(q[1] - r[1]) / area2, (r[0] - q[0]) / area2]
                    });
                    let m = std::array::from_fn(|a| std::array::from_fn(|b| e * (g[a][0] * g[b][0] + g[a][1] * g[b][1]) * 0.5 * area2));
                    tris.push((t, m));
                }
            }
        }
        let n = points.len();
        let mut a = Csr::from_groups(n, tris.iter().map(|(t, _)| t.iter().map(|&x| unknown[x]).filter(|&x| x != usize::MAX).collect::<Vec<_>>()))?;
        let count = self.strips.len();
        let mut rhs = vec![vec![C::new(0.0, 0.0); n]; count];
        for (t, m) in &tris {
            for (ra, &na) in t.iter().enumerate() {
                let Some(row) = unknown.get(na).copied().filter(|&x| x != usize::MAX) else { continue };
                for (cb, &nb) in t.iter().enumerate() {
                    match (unknown[nb], fixed[nb]) {
                        (col, _) if col != usize::MAX => {
                            a.add(row, col, C::new(m[ra][cb], 0.0));
                        }
                        (_, f) if f != 0 && f != u32::MAX => rhs[(f - 1) as usize][row] -= m[ra][cb],
                        _ => {}
                    }
                }
            }
        }
        let factor = Ldlt::factor(&a, &points)?;
        factor.solve_many(&mut rhs)?;
        // The potential of strip k at every node, and C_jk = ε₀ Σ φ_jᵀ m φ_k.
        let potential = |k: usize, n: usize| match (unknown[n], fixed[n]) {
            (x, _) if x != usize::MAX => rhs[k][x].re,
            (_, f) if f as usize == k + 1 => 1.0,
            _ => 0.0,
        };
        let mut out = vec![vec![0.0; count]; count];
        for (jj, row) in out.iter_mut().enumerate() {
            for (kk, c) in row.iter_mut().enumerate() {
                let sum: f64 = tris
                    .iter()
                    .map(|(t, m)| {
                        let (pj, pk) = (t.map(|x| potential(jj, x)), t.map(|x| potential(kk, x)));
                        (0..3).map(|a| (0..3).map(|b| pj[a] * m[a][b] * pk[b]).sum::<f64>()).sum::<f64>()
                    })
                    .sum();
                *c = EPS0 * sum;
            }
        }
        Ok(out)
    }

    /// The line's quasi-TEM modes (module docs): one strip's, or a pair's even and odd.
    pub fn solve(&self) -> Result<Line> {
        self.check()?;
        let (u, v) = self.grid()?;
        let (c, c0) = (self.capacitances(&u, &v, false)?, self.capacitances(&u, &v, true)?);
        let mode = |cap: f64, cap0: f64, odd: bool| -> Result<Mode> {
            if !(cap > 0.0 && cap0 > 0.0 && cap.is_finite() && cap0.is_finite()) {
                return Err(invalid("line cross-section", "a strip holds no charge: is it shorted to its box?"));
            }
            Ok(Mode { eps_eff: cap / cap0, impedance: 1.0 / (C0 * (cap * cap0).sqrt()), odd })
        };
        let modes = match (&c[..], &c0[..]) {
            ([c], [c0]) => vec![mode(c[0], c0[0], false)?],
            ([c1, c2], [d1, d2]) => {
                let (mean, mean0) = (0.5 * (c1[0] + c2[1]), 0.5 * (d1[0] + d2[1]));
                let (m, m0) = (0.5 * (c1[1] + c2[0]), 0.5 * (d1[1] + d2[0]));
                vec![mode(mean + m, mean0 + m0, false)?, mode(mean - m, mean0 - m0, true)?]
            }
            _ => return Err(invalid("line cross-section", "needs one strip or a pair")),
        };
        Ok(Line { modes })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use emcraft_theory::{coupled, microstrip, stripline};

    /// A strip between planes `b` apart in εr 2.2, centred, `w` wide; with a second one `s` away.
    fn stripline_section(w: f64, s: Option<f64>) -> Section {
        let b = 3e-3;
        let strips = match s {
            None => vec![Strip { left: -0.5 * w, right: 0.5 * w, height: 0.5 * b, thickness: 0.0 }],
            Some(s) => {
                vec![
                    Strip { left: -0.5 * s - w, right: -0.5 * s, height: 0.5 * b, thickness: 0.0 },
                    Strip { left: 0.5 * s, right: 0.5 * s + w, height: 0.5 * b, thickness: 0.0 },
                ]
            }
        };
        Section { half_width: 10.0 * b, floor: 0.0, ceiling: b, slabs: vec![Slab { bottom: 0.0, top: b, eps: 2.2 }], blocks: vec![], strips }
    }

    #[test]
    fn a_microstrip_and_a_stripline_match_their_closed_forms() {
        // Hammerstad and Jensen's fit (within 0.2–1 %) for a strip on alumina under a lid 20 h up;
        // Cohn's exact stripline.
        let h = 0.635e-3;
        let ms = Section {
            half_width: 20.0 * h,
            floor: 0.0,
            ceiling: 20.0 * h,
            slabs: vec![Slab { bottom: 0.0, top: h, eps: 9.8 }],
            blocks: vec![],
            strips: vec![Strip { left: -0.3e-3, right: 0.3e-3, height: h, thickness: 0.0 }],
        };
        let line = ms.solve().unwrap();
        let (z, eps) = (line.single().unwrap(), line.modes[0].eps_eff);
        assert!((z / microstrip::z0(0.6 / 0.635, 9.8) - 1.0).abs() < 0.01, "{z} Ω");
        assert!((eps / microstrip::eps_eff(0.6 / 0.635, 9.8) - 1.0).abs() < 0.005, "{eps}");
        let z = stripline_section(2.4e-3, None).solve().unwrap().single().unwrap();
        assert!((z / stripline::z0(2.4e-3, 3e-3, 2.2) - 1.0).abs() < 0.01, "{z} Ω");
    }

    #[test]
    fn a_coupled_stripline_matches_cohns_even_and_odd_impedances() {
        let line = stripline_section(2.4e-3, Some(0.6e-3)).solve().unwrap();
        let (odd, even) = line.pair().unwrap();
        let (z_even, z_odd) = coupled::z0_even_odd(2.4e-3, 0.6e-3, 3e-3, 2.2).unwrap();
        assert!((even.impedance / z_even - 1.0).abs() < 0.01 && (odd.impedance / z_odd - 1.0).abs() < 0.01, "{line:?}");
        // A homogeneous line: both modes at εr.
        assert!((odd.eps_eff / 2.2 - 1.0).abs() < 1e-9 && (even.eps_eff / 2.2 - 1.0).abs() < 1e-9);
    }

    /// The 0.6 mm strip on alumina of `a_microstrip_and_a_stripline_match_their_closed_forms`,
    /// `t` thick, under a block of mask `mask` = (thickness, εr) over it and the substrate.
    fn coated(t: f64, mask: Option<(f64, f64)>) -> Section {
        let h = 0.635e-3;
        let strip = Strip { left: -0.3e-3, right: 0.3e-3, height: h, thickness: t };
        let blocks = mask.map_or(vec![], |(m, eps)| {
            vec![
                Block { left: -20.0 * h, right: 20.0 * h, bottom: h, top: h + m, eps },
                Block { left: strip.left - m, right: strip.right + m, bottom: h, top: strip.top() + m, eps },
            ]
        });
        Section {
            half_width: 20.0 * h,
            floor: 0.0,
            ceiling: 20.0 * h,
            slabs: vec![Slab { bottom: 0.0, top: h, eps: 9.8 }],
            blocks,
            strips: vec![strip],
        }
    }

    #[test]
    fn a_thick_strip_lowers_its_impedance_as_hammerstad_and_jensen_give() {
        // The change from the sheet, against the fit's own change: the box's and the fit's
        // offsets cancel in the ratio.
        let h = 0.635e-3;
        let solve = |t: f64| {
            let line = coated(t, None).solve().unwrap();
            (line.single().unwrap(), line.modes[0].eps_eff)
        };
        let (z_sheet, eps_sheet) = solve(0.0);
        for t in [0.017e-3, 0.035e-3, 0.07e-3] {
            let (z, eps) = solve(t);
            let (zf, epsf) = microstrip::thick(0.6e-3 / h, t / h, 9.8);
            let (z0, eps0) = (microstrip::z0(0.6e-3 / h, 9.8), microstrip::eps_eff(0.6e-3 / h, 9.8));
            assert!(((z / z_sheet) / (zf / z0) - 1.0).abs() < 0.005, "t {t}: {z} / {z_sheet} against {zf} / {z0}");
            assert!(((eps / eps_sheet) / (epsf / eps0) - 1.0).abs() < 0.005, "t {t}: {eps} / {eps_sheet} against {epsf} / {eps0}");
        }
    }

    #[test]
    fn a_mask_slows_the_line_and_a_mask_like_the_substrate_fills_the_box() {
        let bare = coated(0.035e-3, None).solve().unwrap().modes[0];
        let masked = coated(0.035e-3, Some((0.025e-3, 3.5))).solve().unwrap().modes[0];
        assert!(masked.eps_eff > bare.eps_eff && masked.impedance < bare.impedance, "{bare:?} {masked:?}");
        // The mask εr 1: the bare line (only the grid's lines differ).
        let air = coated(0.035e-3, Some((0.025e-3, 1.0))).solve().unwrap().modes[0];
        assert!((air.impedance / bare.impedance - 1.0).abs() < 1e-3, "{bare:?} {air:?}");
        // A mask of the substrate's εr up to the lid: one dielectric, ε_eff = εr exactly.
        let h = 0.635e-3;
        let full = coated(0.035e-3, Some((19.0 * h, 9.8))).solve().unwrap().modes[0];
        assert!((full.eps_eff / 9.8 - 1.0).abs() < 1e-9, "{full:?}");
    }

    #[test]
    fn hostile_sections_are_refused() {
        let good = stripline_section(1e-3, None);
        for bad in [
            Section { half_width: f64::NAN, ..good.clone() },
            Section { ceiling: -1.0, ..good.clone() },
            Section { strips: vec![], ..good.clone() },
            Section { strips: vec![Strip { left: 1e-3, right: -1e-3, height: 1e-3, thickness: 0.0 }], ..good.clone() },
            Section { strips: vec![Strip { left: -1e-3, right: 1e-3, height: 0.0, thickness: 0.0 }], ..good.clone() },
            Section {
                strips: vec![
                    Strip { left: -1e-3, right: 1e-3, height: 1e-3, thickness: 0.0 },
                    Strip { left: 0.0, right: 2e-3, height: 1e-3, thickness: 0.0 },
                ],
                ..good.clone()
            },
            Section { slabs: vec![Slab { bottom: 0.0, top: 1.0, eps: 0.5 }], ..good.clone() },
            Section { strips: vec![Strip { left: -1e-3, right: 1e-3, height: 1e-3, thickness: -1e-4 }], ..good.clone() },
            Section { strips: vec![Strip { left: -1e-3, right: 1e-3, height: 1e-3, thickness: 1.0 }], ..good.clone() },
            Section { blocks: vec![Block { left: 0.0, right: 1e-3, bottom: 0.0, top: 1e-3, eps: 0.5 }], ..good.clone() },
            Section { blocks: vec![Block { left: 0.0, right: f64::NAN, bottom: 0.0, top: 1e-3, eps: 2.0 }], ..good.clone() },
        ] {
            assert!(bad.solve().is_err(), "{bad:?}");
        }
        // A box far wider than its strip has more grid lines than allowed.
        assert!(Section { half_width: 1e3, ..good }.solve().is_err());
    }
}
