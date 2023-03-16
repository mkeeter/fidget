use crate::eval::{types::Interval, Family, IntervalEval, Tape};

////////////////////////////////////////////////////////////////////////////////
// Welcome to the strongly-typed zone!

#[derive(Copy, Clone, Debug)]
struct Axis(u8);

impl Axis {
    fn index(self) -> usize {
        self.0.trailing_zeros() as usize
    }
}

const X: Axis = Axis(1);
const Y: Axis = Axis(2);
const Z: Axis = Axis(4);

impl std::ops::Mul<bool> for Axis {
    type Output = Axis;
    fn mul(self, rhs: bool) -> Axis {
        if rhs {
            self
        } else {
            Axis(0)
        }
    }
}

impl std::ops::BitAnd<Corner> for Axis {
    type Output = bool;
    fn bitand(self, rhs: Corner) -> bool {
        (self.0 & rhs.0) != 0
    }
}

impl std::ops::BitOr<Axis> for Axis {
    type Output = Corner;
    fn bitor(self, rhs: Axis) -> Corner {
        Corner(self.0 | rhs.0)
    }
}

impl std::ops::BitOr<Corner> for Axis {
    type Output = Corner;
    fn bitor(self, rhs: Corner) -> Corner {
        Corner(self.0 | rhs.0)
    }
}

impl From<Axis> for Corner {
    fn from(a: Axis) -> Self {
        Corner(a.0)
    }
}

/// Strongly-typed cell corner, in the 0-8 range
#[derive(Copy, Clone, Debug)]
struct Corner(u8);

impl Corner {
    fn iter() -> impl Iterator<Item = Corner> {
        (0..8).map(Corner)
    }
}

impl std::ops::BitAnd<Axis> for Corner {
    type Output = bool;
    fn bitand(self, rhs: Axis) -> bool {
        (self.0 & rhs.0) != 0
    }
}

impl std::ops::BitOr<Corner> for Corner {
    type Output = Corner;
    fn bitor(self, rhs: Corner) -> Corner {
        Corner(self.0 | rhs.0)
    }
}

impl std::ops::BitOr<Axis> for Corner {
    type Output = Corner;
    fn bitor(self, rhs: Axis) -> Corner {
        Corner(self.0 | rhs.0)
    }
}

/// Represents a directed edge within an octree cell
#[derive(Copy, Clone, Debug)]
struct DirectedEdge {
    start: Corner,
    end: Corner,
}

/// Represents an undirected edge within an octree cell
///
/// With `(t, u, v)` as a right-handed coordinate system and `t` being the
/// varying axis of the edge, this is packed as `4 * t + 2 * v + 1 * u`
#[derive(Copy, Clone, Debug)]
struct Edge(u8);

/// Represents the relative offset of a vertex within [`Octree::verts`]
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
struct Offset(u8);

/// Represents an edge that includes a sign change (and thus an intersection)
#[derive(Copy, Clone, Debug)]
struct Intersection {
    /// Data offset of the vertex located within the cell
    vert: Offset,
    /// Data offset of the vertex located on the edge
    edge: Offset,
}

/// Cell mask, as an 8-bit value representing set corners
#[derive(Copy, Clone, Debug)]
struct Mask(u8);

////////////////////////////////////////////////////////////////////////////////

/// Marker trait for a right-handed coordinate frame
trait Frame {
    type Next: Frame;
    fn frame() -> (Axis, Axis, Axis);
}

#[allow(clippy::upper_case_acronyms)]
struct XYZ;
#[allow(clippy::upper_case_acronyms)]
struct YZX;
#[allow(clippy::upper_case_acronyms)]
struct ZXY;

impl Frame for XYZ {
    type Next = YZX;
    fn frame() -> (Axis, Axis, Axis) {
        (X, Y, Z)
    }
}

impl Frame for YZX {
    type Next = ZXY;
    fn frame() -> (Axis, Axis, Axis) {
        (Y, Z, X)
    }
}
impl Frame for ZXY {
    type Next = XYZ;
    fn frame() -> (Axis, Axis, Axis) {
        (Z, X, Y)
    }
}

////////////////////////////////////////////////////////////////////////////////

/// Raw cell data
///
/// Unpack to a [`Cell`] to actually use it
#[derive(Copy, Clone)]
struct CellData(u64);

impl From<Cell> for CellData {
    fn from(c: Cell) -> Self {
        let i = match c {
            Cell::Empty => 0b00 << 62,
            Cell::Full => 0b01 << 62,
            Cell::Branch { index } => {
                debug_assert!(index < (1 << 62));
                0b10 << 62 | index as u64
            }
            Cell::Leaf(Leaf { mask, index }) => {
                debug_assert!(index < (1 << 54));
                (0b11 << 62) | ((mask as u64) << 54) | index as u64
            }
        };
        CellData(i)
    }
}

static_assertions::const_assert_eq!(
    std::mem::size_of::<usize>(),
    std::mem::size_of::<u64>()
);

/// Unpacked form of [`CellData`]
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
enum Cell {
    Empty,
    Full,
    Branch { index: usize },
    Leaf(Leaf),
}

impl From<CellData> for Cell {
    fn from(c: CellData) -> Self {
        let i = c.0 as usize;
        match (i >> 62) & 0b11 {
            0b00 => Cell::Empty,
            0b01 => Cell::Full,
            0b10 => Cell::Branch {
                index: i & ((1 << 62) - 1),
            },
            0b11 => Cell::Leaf(Leaf {
                mask: (i >> 54) as u8,
                index: i & ((1 << 54) - 1),
            }),
            _ => unreachable!(),
        }
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
struct Leaf {
    mask: u8,
    index: usize,
}

impl Leaf {
    fn edge(&self, e: Edge) -> Intersection {
        let out = CELL_TO_EDGE_TO_VERT[self.mask as usize][e.0 as usize];
        debug_assert_ne!(out.vert.0, u8::MAX);
        debug_assert_ne!(out.edge.0, u8::MAX);
        out
    }
}

////////////////////////////////////////////////////////////////////////////////

/// Cell index used during iteration
///
/// Instead of storing the cell bounds in the leaf itself, we build them when
/// descending the tree.
///
/// `index` points to where this cell is stored in [`Octree::cells`]
#[derive(Copy, Clone, Debug)]
struct CellIndex {
    index: usize,
    depth: usize,
    x: Interval,
    y: Interval,
    z: Interval,
}

impl CellIndex {
    /// Returns the position of the given corner (0-7)
    ///
    /// Vertices are numbered as follows:
    ///
    /// ```text
    ///         6 -------- 7
    ///        /          /       Z
    ///       / |        / |      ^  _ Y
    ///      4----------5  |      | /
    ///      |  |       |  |      |/
    ///      |  2-------|--3      ---> X
    ///      | /        | /
    ///      |/         |/
    ///      0----------1
    /// ```
    ///
    /// The 8 octree cells are numbered equivalently, based on their corner
    /// vertex.
    fn corner(&self, i: Corner) -> (f32, f32, f32) {
        let x = if i & X {
            self.x.upper()
        } else {
            self.x.lower()
        };
        let y = if i & Y {
            self.y.upper()
        } else {
            self.y.lower()
        };
        let z = if i & Z {
            self.z.upper()
        } else {
            self.z.lower()
        };
        (x, y, z)
    }

    /// Returns the interval of the given child (0-7)
    fn interval(&self, i: Corner) -> (Interval, Interval, Interval) {
        // TODO: make this a function in `Interval`?
        let x = if i & X {
            Interval::new(self.x.midpoint(), self.x.upper())
        } else {
            Interval::new(self.x.lower(), self.x.midpoint())
        };
        let y = if i & Y {
            Interval::new(self.y.midpoint(), self.y.upper())
        } else {
            Interval::new(self.y.lower(), self.y.midpoint())
        };
        let z = if i & Z {
            Interval::new(self.z.midpoint(), self.z.upper())
        } else {
            Interval::new(self.z.lower(), self.z.midpoint())
        };
        (x, y, z)
    }

    fn pos(&self, p: nalgebra::Vector3<u16>) -> nalgebra::Vector3<f32> {
        let x = self.x.lerp(p.x as f32 / u16::MAX as f32);
        let y = self.y.lerp(p.y as f32 / u16::MAX as f32);
        let z = self.z.lerp(p.z as f32 / u16::MAX as f32);
        nalgebra::Vector3::new(x, y, z)
    }
}

////////////////////////////////////////////////////////////////////////////////

/// An indexed 3D mesh
#[derive(Default, Debug)]
pub struct Mesh {
    triangles: Vec<nalgebra::Vector3<usize>>,
    vertices: Vec<nalgebra::Vector3<f32>>,
}

impl Mesh {
    /// Builds a new mesh
    pub fn new() -> Self {
        Self::default()
    }

    /// Writes a binary STL to the given output
    pub fn write_stl<F: std::io::Write>(
        &self,
        out: &mut F,
    ) -> Result<(), crate::Error> {
        const HEADER: &[u8] = b"This is a binary STL file exported by Fidget";
        static_assertions::const_assert!(HEADER.len() <= 80);
        out.write_all(HEADER)?;
        out.write_all(&[0u8; 80 - HEADER.len()])?;
        out.write_all(&(self.triangles.len() as u32).to_le_bytes())?;
        for t in &self.triangles {
            // Not the _best_ way to calculate a normal, but good enough
            let a = self.vertices[t.x];
            let b = self.vertices[t.y];
            let c = self.vertices[t.z];
            let ab = b - a;
            let ac = c - a;
            let normal = ab.cross(&ac);
            out.write_all(&normal.x.to_le_bytes())?;
            out.write_all(&normal.y.to_le_bytes())?;
            out.write_all(&normal.z.to_le_bytes())?;
            for v in t {
                let v = self.vertices[*v];
                out.write_all(&v.x.to_le_bytes())?;
                out.write_all(&v.y.to_le_bytes())?;
                out.write_all(&v.z.to_le_bytes())?;
            }
            out.write_all(&[0u8; std::mem::size_of::<u16>()])?; // attributes
        }
        Ok(())
    }
}

#[derive(Default)]
struct MeshBuilder {
    /// Map from indexes in [`Octree::verts`] to `out.vertices`
    ///
    /// `usize::MAX` is used a marker for an unmapped vertex
    map: Vec<usize>,
    out: Mesh,
}

impl MeshBuilder {
    /// Looks up the given vertex, localizing it within a cell
    ///
    /// `v` is an absolute offset into `verts`, which should be a reference to
    /// [`Octree::verts`].
    fn get(
        &mut self,
        v: usize,
        cell: CellIndex,
        verts: &[nalgebra::Vector3<u16>],
    ) -> usize {
        if v >= self.map.len() {
            self.map.resize(v + 1, usize::MAX);
        }
        match self.map[v] {
            usize::MAX => {
                let next_vert = self.out.vertices.len();
                self.out.vertices.push(cell.pos(verts[v]));
                self.map[v] = next_vert;

                next_vert
            }
            u => u,
        }
    }
}

////////////////////////////////////////////////////////////////////////////////

pub struct Octree {
    /// The top two bits determine cell types
    cells: Vec<CellData>,

    /// Cell vertices, given as positions within the cell
    ///
    /// This is indexed by cell leaf index; the exact shape depends heavily on
    /// the number of intersections and vertices within each leaf.
    verts: Vec<nalgebra::Vector3<u16>>,
}

impl Octree {
    pub fn build<I: Family>(tape: &Tape<I>, depth: usize) -> Self {
        let i_handle = tape.new_interval_evaluator();
        let x = Interval::new(-1.0, 1.0);
        let y = Interval::new(-1.0, 1.0);
        let z = Interval::new(-1.0, 1.0);

        let mut out = Self {
            cells: vec![CellData(0); 8],
            verts: vec![],
        };

        out.recurse(
            &i_handle,
            CellIndex {
                index: 0,
                x,
                y,
                z,
                depth,
            },
        );
        out
    }

    fn recurse<I: Family>(
        &mut self,
        i_handle: &IntervalEval<I>,
        cell: CellIndex,
    ) {
        let (i, r) = i_handle.eval(cell.x, cell.y, cell.z, &[]).unwrap();
        if i.upper() < 0.0 {
            self.cells[cell.index] = Cell::Full.into();
        } else if i.lower() > 0.0 {
            self.cells[cell.index] = Cell::Empty.into();
        } else {
            let sub_tape = r.map(|r| r.simplify().unwrap());
            if cell.depth == 0 {
                let tape = sub_tape.unwrap_or_else(|| i_handle.tape());
                self.leaf(tape, cell)
            } else {
                let sub_eval = sub_tape.map(|s| s.new_interval_evaluator());
                let child = self.cells.len();
                for _ in Corner::iter() {
                    self.cells.push(CellData(0));
                }
                self.cells[cell.index] = Cell::Branch { index: child }.into();
                for i in Corner::iter() {
                    let (x, y, z) = cell.interval(i);
                    self.recurse(
                        sub_eval.as_ref().unwrap_or(i_handle),
                        CellIndex {
                            index: child + i.0 as usize,
                            x,
                            y,
                            z,
                            depth: cell.depth - 1,
                        },
                    );
                }
            }
        }
    }

    fn leaf<I: Family>(&mut self, tape: Tape<I>, cell: CellIndex) {
        let eval = tape.new_float_slice_evaluator();

        let mut xs = [0.0; 8];
        let mut ys = [0.0; 8];
        let mut zs = [0.0; 8];
        for i in Corner::iter() {
            let (x, y, z) = cell.corner(i);
            xs[i.0 as usize] = x;
            ys[i.0 as usize] = y;
            zs[i.0 as usize] = z;
        }

        // TODO: reuse evaluators, etc
        let out = eval.eval(&xs, &ys, &zs, &[]).unwrap();
        debug_assert_eq!(out.len(), 8);

        // Build a mask of active corners, which determines cell
        // topology / vertex count / active edges / etc.
        let mask = out
            .iter()
            .enumerate()
            .filter(|(_i, &v)| v < 0.0)
            .fold(0, |acc, (i, _v)| acc | (1 << i));

        // Pick fake intersections and fake vertex positions for now
        //
        // We have up to 4 vertices, followed by up to 12 intersections
        let mut intersections: arrayvec::ArrayVec<nalgebra::Vector3<u16>, 12> =
            arrayvec::ArrayVec::new();
        let mut verts: arrayvec::ArrayVec<nalgebra::Vector3<u16>, 4> =
            arrayvec::ArrayVec::new();
        for vs in CELL_TO_VERT_TO_EDGES[mask as usize].iter() {
            let mut center: nalgebra::Vector3<u32> = nalgebra::Vector3::zeros();
            for e in *vs {
                // Find the axis that's being used by this edge
                let axis = e.start.0 ^ e.end.0;
                debug_assert_eq!(axis.count_ones(), 1);
                debug_assert!(axis < 8);

                // Pick a position closer to the filled side
                let pos = if e.end.0 & axis != 0 {
                    16384
                } else {
                    u16::MAX - 16384
                };

                // Convert the intersection to a 3D position
                let mut v = nalgebra::Vector3::zeros();
                v[axis.trailing_zeros() as usize] = pos;
                let i = (axis.trailing_zeros() + 1) % 3;
                let j = (axis.trailing_zeros() + 2) % 3;
                v[i as usize] =
                    if e.start & Axis(1 << i) { u16::MAX } else { 0 };
                v[j as usize] =
                    if e.start & Axis(1 << j) { u16::MAX } else { 0 };
                intersections.push(v);

                // Accumulate a fake vertex by taking the average of all
                // intersection positions (for now)
                center += v.map(|i| i as u32);
            }
            verts.push(
                center.map(|i| (i / vs.len() as u32).try_into().unwrap()),
            );
        }
        let index = self.verts.len();
        self.verts.extend(verts.into_iter());
        self.verts.extend(intersections.into_iter());
        self.cells[cell.index] = Cell::Leaf(Leaf { mask, index }).into();
    }

    pub fn walk_dual(&self) -> Mesh {
        let x = Interval::new(-1.0, 1.0);
        let y = Interval::new(-1.0, 1.0);
        let z = Interval::new(-1.0, 1.0);
        let mut mesh = MeshBuilder::default();

        self.dc_cell(
            CellIndex {
                index: 0,
                x,
                y,
                z,
                depth: 0,
            },
            &mut mesh,
        );
        mesh.out
    }
}

#[allow(clippy::modulo_one, clippy::identity_op, unused_parens)]
impl Octree {
    fn dc_cell(&self, cell: CellIndex, out: &mut MeshBuilder) {
        if let Cell::Branch { index } = self.cells[cell.index].into() {
            debug_assert_eq!(index % 8, 0);
            for i in Corner::iter() {
                self.dc_cell(self.child(cell, i), out);
            }

            self.dc_faces::<XYZ>(cell, out);
            self.dc_faces::<YZX>(cell, out);
            self.dc_faces::<ZXY>(cell, out);

            for i in [false, true] {
                self.dc_edge::<XYZ>(
                    self.child(cell, (X * i)),
                    self.child(cell, (X * i) | Y),
                    self.child(cell, (X * i) | Y | Z),
                    self.child(cell, (X * i) | Z),
                    out,
                );
                self.dc_edge::<YZX>(
                    self.child(cell, (Y * i)),
                    self.child(cell, (Y * i) | Z),
                    self.child(cell, (Y * i) | X | Z),
                    self.child(cell, (Y * i) | X),
                    out,
                );
                self.dc_edge::<ZXY>(
                    self.child(cell, (Z * i)),
                    self.child(cell, (Z * i) | X),
                    self.child(cell, (Z * i) | X | Y),
                    self.child(cell, (Z * i) | Y),
                    out,
                );
            }
        }
    }

    /// Calls [`self.dc_face`] on all four face adjacencies in the given frame
    fn dc_faces<T: Frame>(&self, cell: CellIndex, out: &mut MeshBuilder) {
        let (t, u, v) = T::frame();
        for c in [Corner(0), u.into(), v.into(), u | v] {
            self.dc_face::<T>(
                self.child(cell, c),
                self.child(cell, c | t),
                out,
            );
        }
    }

    /// Looks up the given child of a cell.
    ///
    /// If the cell is a leaf node, returns that cell instead.
    fn child<C: Into<Corner>>(&self, cell: CellIndex, child: C) -> CellIndex {
        let child = child.into();
        debug_assert!(child.0 < 8);

        match self.cells[cell.index].into() {
            Cell::Leaf { .. } | Cell::Full | Cell::Empty => cell,
            Cell::Branch { index } => {
                let (x, y, z) = cell.interval(child);
                CellIndex {
                    index: index + child.0 as usize,
                    x,
                    y,
                    z,
                    depth: cell.depth + 1,
                }
            }
        }
    }

    fn is_leaf(&self, cell: CellIndex) -> bool {
        !matches!(self.cells[cell.index].into(), Cell::Branch { .. })
    }

    /// Handles two cells which share a common face
    ///
    /// `lo` is below `hi` on the `T` axis; the cells share a `UV` face where
    /// `T-U-V` is a right-handed coordinate system.
    fn dc_face<T: Frame>(
        &self,
        lo: CellIndex,
        hi: CellIndex,
        out: &mut MeshBuilder,
    ) {
        if self.is_leaf(lo) && self.is_leaf(hi) {
            return;
        }
        let (t, u, v) = T::frame();
        self.dc_face::<T>(self.child(lo, t), self.child(hi, Corner(0)), out);
        self.dc_face::<T>(self.child(lo, t | u), self.child(hi, u), out);
        self.dc_face::<T>(self.child(lo, t | Y), self.child(hi, v), out);
        self.dc_face::<T>(
            self.child(lo, t | u | v),
            self.child(hi, u | v),
            out,
        );
        for i in [false, true] {
            self.dc_edge::<T::Next>(
                self.child(lo, (u * i) | t),
                self.child(lo, (u * i) | v | t),
                self.child(hi, (u * i) | v),
                self.child(hi, (u * i)),
                out,
            );
            self.dc_edge::<<T::Next as Frame>::Next>(
                self.child(lo, (v * i) | t),
                self.child(hi, (v * i)),
                self.child(hi, (v * i) | u),
                self.child(lo, (v * i) | u | t),
                out,
            );
        }
    }

    /// Handles four cells that share a common edge aligned on axis `T`
    ///
    /// Cells positions are in the order `[0, U, U | V, U]`, i.e. a right-handed
    /// winding about `+T` (where `T, U, V` is a right-handed coordinate frame)
    ///
    /// - `dc_edge<X>` is `[0, Y, Y | Z, Z]`
    /// - `dc_edge<Y>` is `[0, Z, Z | X, X]`
    /// - `dc_edge<Z>` is `[0, X, X | Y, Y]`
    fn dc_edge<T: Frame>(
        &self,
        a: CellIndex,
        b: CellIndex,
        c: CellIndex,
        d: CellIndex,
        out: &mut MeshBuilder,
    ) {
        let cs = [a, b, c, d];
        if cs.iter().all(|v| self.is_leaf(*v)) {
            // If any of the leafs are Empty or Full, then this edge can't
            // include a sign change.  TODO: can we make this any -> all if we
            // collapse empty / filled leafs into Empty / Full cells?
            let leafs = cs.map(|cell| match self.cells[cell.index].into() {
                Cell::Leaf(leaf) => Some(leaf),
                Cell::Empty | Cell::Full => None,
                Cell::Branch { .. } => unreachable!(),
            });
            if leafs.iter().any(Option::is_none) {
                return;
            }
            let leafs = leafs.map(Option::unwrap);

            // TODO: check for a sign change on this edge
            let (t, u, v) = T::frame();
            let sign_change_count = leafs
                .iter()
                .zip([u | v, v.into(), Corner(0), u.into()])
                .filter(|(leaf, c)| {
                    (leaf.mask & (1 << c.0) == 0)
                        != (leaf.mask & (1 << (*c | t).0) == 0)
                })
                .count();
            if sign_change_count == 0 {
                return;
            }
            debug_assert_eq!(sign_change_count, 4);

            let verts = [
                leafs[0].edge(Edge((t.index() * 4 + 3) as u8)),
                leafs[1].edge(Edge((t.index() * 4 + 2) as u8)),
                leafs[2].edge(Edge((t.index() * 4 + 0) as u8)),
                leafs[3].edge(Edge((t.index() * 4 + 1) as u8)),
            ];

            // Pick the intersection vertex based on the deepest cell
            let deepest = (0..4).max_by_key(|i| cs[*i].depth).unwrap();
            let i = out.get(
                leafs[deepest].index + verts[deepest].edge.0 as usize,
                cs[deepest],
                &self.verts,
            );
            // Helper function to extract other vertices
            let mut vert = |i: usize| {
                out.get(
                    leafs[i].index + verts[i].vert.0 as usize,
                    cs[i],
                    &self.verts,
                )
            };
            let vs = [vert(0), vert(1), vert(2), vert(3)];

            // Pick a triangle winding depending on the edge direction
            let winding = if leafs[0].mask & (1 << (u | v).0) == 0 {
                3
            } else {
                1
            };
            for j in 0..4 {
                out.out.triangles.push(nalgebra::Vector3::new(
                    vs[j],
                    vs[(j + winding) % 4],
                    i,
                ))
            }
        } else {
            let (t, u, v) = T::frame();
            for i in [false, true] {
                self.dc_edge::<T>(
                    self.child(a, (t * i) | u | v),
                    self.child(b, (t * i) | v),
                    self.child(c, (t * i)),
                    self.child(d, (t * i) | u),
                    out,
                )
            }
        }
    }
}

include!(concat!(env!("OUT_DIR"), "/mdc_tables.rs"));

////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod test {
    use super::*;
    use crate::{context::Node, Context};
    use std::collections::BTreeMap;

    #[test]
    fn test_cell_encode_decode() {
        for c in [
            Cell::Empty,
            Cell::Full,
            Cell::Branch { index: 12345 },
            Cell::Branch {
                index: 0x1234000054322345,
            },
            Cell::Leaf(Leaf {
                index: 12345,
                mask: 0b101,
            }),
            Cell::Leaf(Leaf {
                index: 0x123400005432,
                mask: 0b11011010,
            }),
        ] {
            assert_eq!(c, Cell::from(CellData::from(c)));
        }
    }

    fn sphere(ctx: &mut Context, center: [f32; 3], radius: f32) -> Node {
        let x = ctx.x();
        let x = ctx.sub(x, center[0]).unwrap();
        let y = ctx.y();
        let y = ctx.sub(y, center[1]).unwrap();
        let z = ctx.z();
        let z = ctx.sub(z, center[2]).unwrap();
        let x2 = ctx.square(x).unwrap();
        let y2 = ctx.square(y).unwrap();
        let z2 = ctx.square(z).unwrap();
        let a = ctx.add(x2, y2).unwrap();
        let r = ctx.add(a, z2).unwrap();
        let r = ctx.sqrt(r).unwrap();
        ctx.sub(r, radius).unwrap()
    }

    #[test]
    fn test_mesh_basic() {
        let mut c = Context::new();
        let shape = sphere(&mut c, [0.0; 3], 0.2);

        let tape = c.get_tape::<crate::vm::Eval>(shape).unwrap();

        // If we only build a depth-0 octree, then it's a leaf without any
        // vertices (since all the corners are empty)
        let octree = Octree::build(&tape, 0);
        assert_eq!(octree.cells.len(), 8); // we always build at least 8 cells
        assert_eq!(
            Cell::Leaf(Leaf { mask: 0, index: 0 }),
            octree.cells[0].into(),
        );
        assert_eq!(octree.verts.len(), 0);
        // TODO: should we transform this into an Empty?

        let empty_mesh = octree.walk_dual();
        assert_eq!(empty_mesh.vertices.len(), 0);
        assert!(empty_mesh.triangles.is_empty());

        // Now, at depth-1, each cell should be a Leaf with one vertex
        let octree = Octree::build(&tape, 1);
        assert_eq!(octree.cells.len(), 16); // we always build at least 8 cells
        assert_eq!(Cell::Branch { index: 8 }, octree.cells[0].into());

        // Each of the 6 edges is counted 4 times and each cell has 1 vertex
        assert_eq!(octree.verts.len(), 6 * 4 + 8);

        // Each cell is a leaf with 4 vertices (3 edges, 1 center)
        for o in &octree.cells[8..] {
            let Cell::Leaf(Leaf { index, mask}) = (*o).into() else { panic!() };
            assert_eq!(mask.count_ones(), 1);
            assert_eq!(index % 4, 0);
        }

        let sphere_mesh = octree.walk_dual();
        assert!(sphere_mesh.vertices.len() > 1);
        assert!(!sphere_mesh.triangles.is_empty());
    }

    #[test]
    #[should_panic(expected = "not yet implemented")]
    fn test_sphere_verts() {
        let mut c = Context::new();
        let shape = sphere(&mut c, [0.0; 3], 0.2);

        let tape = c.get_tape::<crate::vm::Eval>(shape).unwrap();
        let octree = Octree::build(&tape, 1);
        let sphere_mesh = octree.walk_dual();

        for v in &sphere_mesh.vertices {
            // Edge vertices should be found via binary search and therefore
            // should be close to the true crossing point
            let x_edge = v.x.abs() > 1e-6;
            let y_edge = v.y.abs() > 1e-6;
            let z_edge = v.z.abs() > 1e-6;
            let edge_sum = x_edge as u8 + y_edge as u8 + z_edge as u8;
            assert!(edge_sum == 1 || edge_sum == 3);
            if edge_sum == 1 {
                assert!(
                    (v.norm() - 0.2) < 1e-6,
                    "edge vertex {v:?} is not at radius 0.2"
                );
            }
        }
    }

    #[test]
    fn test_mesh_manifold() {
        for i in 0..256 {
            let mut c = Context::new();
            let mut shape = vec![];
            for j in Corner::iter() {
                if i & (1 << j.0) != 0 {
                    shape.push(sphere(
                        &mut c,
                        [
                            if j & X { 0.5 } else { 0.0 },
                            if j & Y { 0.5 } else { 0.0 },
                            if j & Z { 0.5 } else { 0.0 },
                        ],
                        0.1,
                    ));
                }
            }
            let Some(start) = shape.pop() else { continue };
            let shape = shape
                .into_iter()
                .fold(start, |acc, s| c.min(acc, s).unwrap());

            // Now, we have our shape, which is 0-8 spheres placed at the
            // corners of the cell spanning [0, 0.25]
            let tape = c.get_tape::<crate::vm::Eval>(shape).unwrap();
            let octree = Octree::build(&tape, 2);

            let mesh = octree.walk_dual();
            if i != 0 && i != 255 {
                assert!(!mesh.vertices.is_empty());
                assert!(!mesh.triangles.is_empty());
            }
            check_for_vertex_dupes(i, &mesh);
            check_for_edge_matching(i, &mesh);
            mesh.write_stl(
                &mut std::fs::File::create(format!("out{i}.stl")).unwrap(),
            )
            .unwrap();
        }
    }

    fn check_for_vertex_dupes(mask: usize, mesh: &Mesh) {
        let mut verts = mesh.vertices.clone();
        verts.sort_by_key(|k| (k.x.to_bits(), k.y.to_bits(), k.z.to_bits()));
        for i in 1..verts.len() {
            assert_ne!(
                verts[i - 1],
                verts[i],
                "mask {mask:08b} has duplicate vertices"
            );
        }
    }

    fn check_for_edge_matching(mask: usize, mesh: &Mesh) {
        let mut edges: BTreeMap<_, usize> = BTreeMap::new();
        for t in &mesh.triangles {
            for edge in [(t.x, t.y), (t.y, t.z), (t.z, t.x)] {
                *edges.entry(edge).or_default() += 1;
            }
        }
        for (&(a, b), &i) in &edges {
            assert_eq!(i, 1, "mask {mask:08b} has duplicate edge ({a}, {b})");
            assert!(
                edges.contains_key(&(b, a)),
                "mask {mask:08b} has unpaired edges"
            );
        }
    }
}
