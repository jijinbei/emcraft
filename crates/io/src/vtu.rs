//! VTK's unstructured grid in its XML form (`.vtu`): linear tetrahedra with arrays on their
//! points and on their cells, the numbers appended raw after the XML (Kitware, *VTK File
//! Formats*, "Unstructured Grid" and "Appended Data"; VTK 9 user's guide). ParaView, VisIt and
//! meshio open it.
//!
//! The writer streams: the XML names each array's place in the appended block, then the block
//! follows, each array as its byte count (UInt64) and its little-endian values. Nothing grows
//! with the grid but what the caller already holds.
//!
//! ```
//! use emcraft_io::vtu::{Array, Grid, write};
//!
//! // One tetrahedron (points in mm, say) with a scalar on its points and its material.
//! let points = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
//! let grid = Grid {
//!     points: &points,
//!     tets: &[[0, 1, 2, 3]],
//!     point_data: vec![Array { name: "|E| (V/m)", components: 1, values: &[1.0, 2.0, 3.0, 4.0] }],
//!     cell_data: vec![Array { name: "material", components: 1, values: &[0.0] }],
//! };
//! let mut bytes = Vec::new();
//! write(&grid, &mut bytes).unwrap();
//! assert!(bytes.starts_with(b"<?xml"));
//!
//! // A tetrahedron naming a point that is not there is refused.
//! let bad = Grid { tets: &[[0, 1, 2, 9]], ..grid };
//! assert!(write(&bad, &mut Vec::new()).is_err());
//! ```

use std::io::Write;

use crate::{Error, Result};

/// VTK's cell type of a linear tetrahedron.
const VTK_TETRA: u8 = 10;

/// An array on the points or on the cells: its name, its values per item and the values, item
/// after item.
#[derive(Clone, Copy, Debug)]
pub struct Array<'a> {
    pub name: &'a str,
    pub components: usize,
    pub values: &'a [f64],
}

/// A grid of linear tetrahedra: the points, the tetrahedra's corners (indices into `points`)
/// and arrays on each.
#[derive(Clone, Debug)]
pub struct Grid<'a> {
    pub points: &'a [[f64; 3]],
    pub tets: &'a [[u32; 4]],
    pub point_data: Vec<Array<'a>>,
    pub cell_data: Vec<Array<'a>>,
}

/// Writes `grid` as a `.vtu` file to `out`. An index past the points, or an array whose length
/// is not its items times its components, is an error before anything is written.
pub fn write(grid: &Grid, out: &mut impl Write) -> Result<()> {
    check(grid)?;
    let blocks = Blocks::of(grid);
    out.write_all(blocks.xml(grid).as_bytes()).map_err(failed)?;
    out.write_all(b"  <AppendedData encoding=\"raw\">\n   _").map_err(failed)?;
    let coordinates = grid.points.iter().flatten().copied();
    raw_f64(out, grid.points.len() * 3, coordinates)?;
    for a in &grid.point_data {
        raw_f64(out, a.values.len(), a.values.iter().copied())?;
    }
    for a in &grid.cell_data {
        raw_f64(out, a.values.len(), a.values.iter().copied())?;
    }
    let corners = grid.tets.iter().flatten().map(|&i| i64::from(i).to_le_bytes());
    raw(out, grid.tets.len() * 4 * 8, corners)?;
    let offsets = (1..=grid.tets.len()).map(|k| (k as i64 * 4).to_le_bytes());
    raw(out, grid.tets.len() * 8, offsets)?;
    raw(out, grid.tets.len(), grid.tets.iter().map(|_| [VTK_TETRA]))?;
    out.write_all(b"\n  </AppendedData>\n</VTKFile>\n").map_err(failed)
}

fn failed(e: std::io::Error) -> Error {
    Error::Invalid(format!("could not write the grid: {e}"))
}

/// Every corner names a point and every array holds its items' values.
fn check(grid: &Grid) -> Result<()> {
    let n = grid.points.len();
    if let Some(t) = grid.tets.iter().position(|t| t.iter().any(|&i| i as usize >= n)) {
        return Err(Error::Invalid(format!("tetrahedron {t} names a point past the {n} there are")));
    }
    let sized = |a: &Array, items: usize| {
        if a.components == 0 || a.values.len() != items * a.components {
            Err(Error::Invalid(format!("array `{}` holds {} values, not {items} × {}", a.name, a.values.len(), a.components)))
        } else {
            Ok(())
        }
    };
    grid.point_data.iter().try_for_each(|a| sized(a, n))?;
    grid.cell_data.iter().try_for_each(|a| sized(a, grid.tets.len()))
}

/// Each block's offset in the appended data, in the order they are written: the points, the
/// point arrays, the cell arrays, then connectivity, offsets and types.
struct Blocks {
    offsets: Vec<u64>,
}

impl Blocks {
    fn of(grid: &Grid) -> Blocks {
        let tets = grid.tets.len() as u64;
        let sizes = std::iter::once(grid.points.len() as u64 * 3 * 8)
            .chain(grid.point_data.iter().map(|a| a.values.len() as u64 * 8))
            .chain(grid.cell_data.iter().map(|a| a.values.len() as u64 * 8))
            .chain([tets * 4 * 8, tets * 8, tets]);
        let offsets = sizes
            .scan(0u64, |at, size| {
                let here = *at;
                *at += 8 + size;
                Some(here)
            })
            .collect();
        Blocks { offsets }
    }

    /// The XML before the appended data.
    fn xml(&self, grid: &Grid) -> String {
        let mut at = self.offsets.iter();
        let mut next = || at.next().copied().unwrap_or_default();
        let array = |kind: &str, name: &str, components: usize, offset: u64| {
            format!(
                "     <DataArray type=\"{kind}\" Name=\"{}\" NumberOfComponents=\"{components}\" format=\"appended\" offset=\"{offset}\"/>\n",
                escaped(name)
            )
        };
        let mut x = format!(
            "<?xml version=\"1.0\"?>\n<VTKFile type=\"UnstructuredGrid\" version=\"1.0\" byte_order=\"LittleEndian\" header_type=\"UInt64\">\n <UnstructuredGrid>\n  <Piece NumberOfPoints=\"{}\" NumberOfCells=\"{}\">\n",
            grid.points.len(),
            grid.tets.len()
        );
        x += "   <Points>\n";
        x += &array("Float64", "points", 3, next());
        x += "   </Points>\n   <PointData>\n";
        for a in &grid.point_data {
            x += &array("Float64", a.name, a.components, next());
        }
        x += "   </PointData>\n   <CellData>\n";
        for a in &grid.cell_data {
            x += &array("Float64", a.name, a.components, next());
        }
        x += "   </CellData>\n   <Cells>\n";
        x += &array("Int64", "connectivity", 1, next());
        x += &array("Int64", "offsets", 1, next());
        x += &array("UInt8", "types", 1, next());
        x += "   </Cells>\n  </Piece>\n </UnstructuredGrid>\n";
        x
    }
}

/// A name as XML attribute text.
fn escaped(name: &str) -> String {
    name.replace('&', "&amp;").replace('"', "&quot;").replace('<', "&lt;").replace('>', "&gt;")
}

fn raw_f64(out: &mut impl Write, count: usize, values: impl Iterator<Item = f64>) -> Result<()> {
    raw(out, count * 8, values.map(f64::to_le_bytes))
}

/// One appended block: its byte count, then the bytes.
fn raw<const N: usize>(out: &mut impl Write, bytes: usize, items: impl Iterator<Item = [u8; N]>) -> Result<()> {
    out.write_all(&(bytes as u64).to_le_bytes()).map_err(failed)?;
    items.into_iter().try_for_each(|b| out.write_all(&b)).map_err(failed)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tet() -> ([[f64; 3]; 4], [[u32; 4]; 1]) {
        ([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], [[0, 1, 2, 3]])
    }

    /// The appended block read back: each array's bytes at the offset the XML gives it.
    fn block(bytes: &[u8], offset: usize) -> &[u8] {
        let mark = b"<AppendedData encoding=\"raw\">\n   _";
        let start = bytes.windows(mark.len()).position(|w| w == mark).unwrap();
        let data = &bytes[start + mark.len()..];
        let len = u64::from_le_bytes(data[offset..offset + 8].try_into().unwrap()) as usize;
        &data[offset + 8..offset + 8 + len]
    }

    #[test]
    fn the_arrays_land_where_the_xml_says() {
        let (points, tets) = tet();
        let values = [10.0, 20.0, 30.0, 40.0];
        let grid = Grid { points: &points, tets: &tets, point_data: vec![Array { name: "v", components: 1, values: &values }], cell_data: vec![] };
        let mut bytes = Vec::new();
        write(&grid, &mut bytes).unwrap();
        let text = String::from_utf8_lossy(&bytes);
        // Points first (offset 0: 12 values), then `v` after them: 8 + 96 bytes on.
        assert!(text.contains("Name=\"v\" NumberOfComponents=\"1\" format=\"appended\" offset=\"104\""));
        let v: Vec<f64> = block(&bytes, 104).chunks(8).map(|c| f64::from_le_bytes(c.try_into().unwrap())).collect();
        assert_eq!(v, values);
        let p = block(&bytes, 0);
        assert_eq!(p.len(), 96);
        assert_eq!(f64::from_le_bytes(p[24..32].try_into().unwrap()), 1.0);
    }

    #[test]
    fn arrays_of_the_wrong_length_and_bad_names_are_refused_or_escaped() {
        let (points, tets) = tet();
        let short = Grid { points: &points, tets: &tets, point_data: vec![Array { name: "v", components: 3, values: &[1.0; 4] }], cell_data: vec![] };
        assert!(write(&short, &mut Vec::new()).is_err());
        let none = Grid { point_data: vec![Array { name: "v", components: 0, values: &[] }], ..short.clone() };
        assert!(write(&none, &mut Vec::new()).is_err());
        let named = Grid { point_data: vec![Array { name: "a\"<b>&", components: 1, values: &[1.0; 4] }], ..short };
        let mut bytes = Vec::new();
        write(&named, &mut bytes).unwrap();
        assert!(String::from_utf8_lossy(&bytes).contains("Name=\"a&quot;&lt;b&gt;&amp;\""));
    }
}
