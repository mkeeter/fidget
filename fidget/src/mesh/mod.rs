use crate::eval::{types::Interval, Family, IntervalEval, Tape};

// TODO: make these strongly typed?
const X: usize = 1;
const Y: usize = 2;
const Z: usize = 4;

/// `(axis, next(axis), next(next(axis)))` is a right-handed coordinate system
const fn next(axis: usize) -> usize {
    match axis {
        X => Y,
        Y => Z,
        Z => X,
        _ => panic!(),
    }
}

/// `(axis, prev(prev((axis))), prev(axis))` is a right-handed coordinate system
const fn prev(axis: usize) -> usize {
    match axis {
        X => Y,
        Y => Z,
        Z => X,
        _ => panic!(),
    }
}

/// Cell mask, as an 8-bit value representing set corners
#[derive(Copy, Clone, Debug)]
struct Mask(u8);

/// Strongly-typed cell corner, in the 0-8 range
#[derive(Copy, Clone, Debug)]
struct Corner(u8);

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

impl Edge {
    /// Returns a right-handed coordinate system `(t, u, v)` for this edge
    fn axes(&self) -> (u8, u8, u8) {
        let t = self.0 / 4;
        let u = (t + 1) % 3;
        let v = (t + 2) % 3;

        (1 << t, 1 << u, 1 << v)
    }
    fn start(&self) -> Corner {
        let (t, u, v) = self.axes();
        Corner(((self.0 % 4 / 2) * v) | ((self.0 % 2) * u))
    }
    fn end(&self) -> Corner {
        let (t, _, _) = self.axes();
        Corner(self.start().0 | t)
    }
}

/// Represents the relative offset of a vertex within [`Octree::verts`]
#[derive(Copy, Clone, Debug)]
struct Offset(u8);

/// Represents an edge that includes a sign change (and thus an intersection)
#[derive(Copy, Clone, Debug)]
struct Intersection {
    /// Data offset of the vertex located within the cell
    vert: Offset,
    /// Data offset of the vertex located on the edge
    edge: Offset,
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
                assert!(index < (1 << 62));
                0b10 << 62 | index as u64
            }
            Cell::Leaf(Leaf { mask, index }) => {
                assert!(index < (1 << 54));
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
                index: i & ((1 << 54) - 1),
                mask: (i >> 54) as u8,
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
        CELL_TO_EDGE_TO_VERT[self.mask as usize][e.0 as usize]
    }
}

////////////////////////////////////////////////////////////////////////////////

/// Cell index used during iteration
///
/// Instead of storing the cell bounds in the leaf itself, we build them when
/// descending the tree.
///
/// `index` points to where this cell is stored in [`Octree::cells`]
#[derive(Copy, Clone)]
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
    fn corner(&self, i: usize) -> (f32, f32, f32) {
        let x = if i & X == 0 {
            self.x.lower()
        } else {
            self.x.upper()
        };
        let y = if i & Y == 0 {
            self.y.lower()
        } else {
            self.y.upper()
        };
        let z = if i & Z == 0 {
            self.z.lower()
        } else {
            self.z.upper()
        };
        (x, y, z)
    }

    /// Returns the interval of the given child (0-7)
    fn interval(&self, i: usize) -> (Interval, Interval, Interval) {
        // TODO: make this a function in `Interval`?
        let x = if i & X == 0 {
            Interval::new(self.x.lower(), self.x.midpoint())
        } else {
            Interval::new(self.x.midpoint(), self.x.upper())
        };
        let y = if i & Y == 0 {
            Interval::new(self.y.lower(), self.y.midpoint())
        } else {
            Interval::new(self.y.midpoint(), self.y.upper())
        };
        let z = if i & Z == 0 {
            Interval::new(self.z.lower(), self.z.midpoint())
        } else {
            Interval::new(self.z.midpoint(), self.z.upper())
        };
        (x, y, z)
    }
}

////////////////////////////////////////////////////////////////////////////////

#[derive(Default)]
pub struct Mesh {
    triangles: Vec<nalgebra::Vector3<usize>>,
    vertices: Vec<nalgebra::Vector3<f32>>,
}

impl Mesh {
    fn new() -> Self {
        Self::default()
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
    pub fn build<I: Family>(tape: Tape<I>, depth: usize) -> Self {
        let i_handle = tape.new_interval_evaluator();
        let x = Interval::new(-1.0, 1.0);
        let y = Interval::new(-1.0, 1.0);
        let z = Interval::new(-1.0, 1.0);

        let mut out = Self {
            cells: vec![CellData(0); 8],

            // Use the 0th index as an empty marker
            verts: vec![nalgebra::Vector3::zeros()],
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
                let eval = tape.new_float_slice_evaluator();

                let mut xs = [0.0; 8];
                let mut ys = [0.0; 8];
                let mut zs = [0.0; 8];
                for i in 0..8 {
                    let (x, y, z) = cell.corner(i);
                    xs[i] = x;
                    ys[i] = y;
                    zs[i] = z;
                }

                // TODO: reuse evaluators, etc
                let out = eval.eval(&xs, &ys, &zs, &[]).unwrap();
                assert_eq!(out.len(), 8);

                // Build a mask of active corners, which determines cell
                // topology / vertex count / active edges / etc.
                let mask = out
                    .iter()
                    .enumerate()
                    .filter(|(_i, &v)| v < 0.0)
                    .fold(0, |acc, (i, _v)| acc | (1 << i));

                // Pick fake intersections and fake vertex positions for now
                //
                // We have up to 12 intersections, followed by up to 4 vertices
                let mut intersections: arrayvec::ArrayVec<
                    nalgebra::Vector3<u16>,
                    12,
                > = arrayvec::ArrayVec::new();
                let mut verts: arrayvec::ArrayVec<nalgebra::Vector3<u16>, 4> =
                    arrayvec::ArrayVec::new();
                for vs in CELL_TO_VERT_TO_EDGES[mask as usize].iter() {
                    let mut center = nalgebra::Vector3::zeros();
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
                        intersections.push(v);

                        // Accumulate a fake vertex by taking the average of all
                        // intersection positions (for now)
                        center += v;
                    }
                    verts.push(center / vs.len() as u16);
                }
                let index = self.verts.len();
                self.verts.extend(verts.into_iter());
                self.verts.extend(intersections.into_iter());
                self.cells[cell.index] =
                    Cell::Leaf(Leaf { mask, index }).into();
            } else {
                let child = self.cells.len();
                for _ in 0..8 {
                    self.cells.push(CellData(0));
                }
                self.cells[cell.index] = Cell::Branch { index: child }.into();
                let (x_lo, x_hi) = cell.x.split();
                let (y_lo, y_hi) = cell.y.split();
                let (z_lo, z_hi) = cell.z.split();
                for i in 0..8 {
                    let x = if i & X == 0 { x_lo } else { x_hi };
                    let y = if i & Y == 0 { y_lo } else { y_hi };
                    let z = if i & Z == 0 { z_lo } else { z_hi };
                    self.recurse(
                        i_handle,
                        CellIndex {
                            index: child + i,
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

    pub fn walk_dual(&self) -> Mesh {
        let x = Interval::new(-1.0, 1.0);
        let y = Interval::new(-1.0, 1.0);
        let z = Interval::new(-1.0, 1.0);
        let mut mesh = Mesh::default();

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
        mesh
    }
}

#[allow(clippy::modulo_one, clippy::identity_op)]
impl Octree {
    fn dc_cell(&self, cell: CellIndex, out: &mut Mesh) {
        if let Cell::Branch { index } = self.cells[cell.index].into() {
            assert_eq!(index % 8, 0);
            for i in 0..8 {
                self.dc_cell(self.child(cell, i), out);
            }
            for i in 0..4 {
                self.dc_face_x(
                    self.child(cell, (2 * X * (i / X) + (i % X)) | 0),
                    self.child(cell, (2 * X * (i / X) + (i % X)) | X),
                    out,
                );
                self.dc_face_y(
                    self.child(cell, (2 * Y * (i / Y) + (i % Y)) | 0),
                    self.child(cell, (2 * Y * (i / Y) + (i % Y)) | Y),
                    out,
                );
                self.dc_face_z(
                    self.child(cell, (2 * Z * (i / Z) + (i % Z)) | 0),
                    self.child(cell, (2 * Z * (i / Z) + (i % Z)) | Z),
                    out,
                );
            }
            for i in 0..2 {
                self.dc_edge_x(
                    self.child(cell, (X * i) | 0),
                    self.child(cell, (X * i) | Y),
                    self.child(cell, (X * i) | Y | Z),
                    self.child(cell, (X * i) | Z),
                    out,
                );
                self.dc_edge_y(
                    self.child(cell, (Y * i) | 0),
                    self.child(cell, (Y * i) | Z),
                    self.child(cell, (Y * i) | X | Z),
                    self.child(cell, (Y * i) | X),
                    out,
                );
                self.dc_edge_z(
                    self.child(cell, (Z * i) | 0),
                    self.child(cell, (Z * i) | X),
                    self.child(cell, (Z * i) | X | Y),
                    self.child(cell, (Z * i) | Y),
                    out,
                );
            }
        }
    }

    /// Looks up the given child of a cell.
    ///
    /// If the cell is a leaf node, returns that cell instead.
    fn child(&self, cell: CellIndex, child: usize) -> CellIndex {
        assert!(child < 8);
        match self.cells[cell.index].into() {
            Cell::Leaf { .. } | Cell::Full | Cell::Empty => cell,
            Cell::Branch { index } => {
                let (x, y, z) = cell.interval(child);
                CellIndex {
                    index: index + child,
                    x,
                    y,
                    z,
                    depth: cell.depth + 1,
                }
            }
        }
    }

    fn is_leaf(&self, cell: CellIndex) -> bool {
        matches!(self.cells[cell.index].into(), Cell::Leaf { .. })
    }

    /// Handles two cells which share a common `YZ` face
    ///
    /// `lo` is below `hi` on the `X` axis
    fn dc_face_x(&self, lo: CellIndex, hi: CellIndex, out: &mut Mesh) {
        if self.is_leaf(lo) && self.is_leaf(hi) {
            return;
        }
        self.dc_face_x(self.child(lo, X), self.child(hi, 0), out);
        self.dc_face_x(self.child(lo, X | Y), self.child(hi, Y), out);
        self.dc_face_x(self.child(lo, X | Z), self.child(hi, Z), out);
        self.dc_face_x(self.child(lo, X | Y | Z), self.child(hi, Y | Z), out);
        for i in 0..2 {
            self.dc_edge_y(
                self.child(lo, (Y * i) | X),
                self.child(lo, (Y * i) | X | Z),
                self.child(hi, (Y * i) | Z),
                self.child(hi, (Y * i) | 0),
                out,
            );
            self.dc_edge_z(
                self.child(lo, (Z * i) | X),
                self.child(hi, (Z * i) | 0),
                self.child(hi, (Z * i) | Y),
                self.child(lo, (Z * i) | X | Y),
                out,
            );
        }
    }
    /// Handles two cells which share a common `XZ` face
    ///
    /// `lo` is below `hi` on the `Y` axis
    fn dc_face_y(&self, lo: CellIndex, hi: CellIndex, out: &mut Mesh) {
        if self.is_leaf(lo) && self.is_leaf(hi) {
            return;
        }
        self.dc_face_y(self.child(lo, Y), self.child(hi, 0), out);
        self.dc_face_y(self.child(lo, Y | X), self.child(hi, X), out);
        self.dc_face_y(self.child(lo, Y | Z), self.child(hi, Z), out);
        self.dc_face_y(self.child(lo, Y | X | Z), self.child(hi, X | Z), out);
        for i in 0..2 {
            self.dc_edge_x(
                self.child(lo, (X * i) | Y),
                self.child(lo, (X * i) | Y | Z),
                self.child(hi, (X * i) | Z),
                self.child(hi, (X * i) | 0),
                out,
            );
            self.dc_edge_z(
                self.child(lo, (Z * i) | Y),
                self.child(lo, (Z * i) | X | Y),
                self.child(hi, (Z * i) | X),
                self.child(hi, (Z * i) | 0),
                out,
            );
        }
    }
    /// Handles two cells which share a common `XY` face
    ///
    /// `lo` is below `hi` on the `Z` axis
    fn dc_face_z(&self, lo: CellIndex, hi: CellIndex, out: &mut Mesh) {
        if self.is_leaf(lo) && self.is_leaf(hi) {
            return;
        }
        self.dc_face_z(self.child(lo, Z), self.child(hi, 0), out);
        self.dc_face_z(self.child(lo, Z | X), self.child(hi, X), out);
        self.dc_face_z(self.child(lo, Z | Y), self.child(hi, Y), out);
        self.dc_face_z(self.child(lo, Z | X | Y), self.child(hi, X | Y), out);
        for i in 0..2 {
            self.dc_edge_x(
                self.child(lo, (X * i) | Z),
                self.child(lo, (X * i) | Y | Z),
                self.child(hi, (X * i) | Y),
                self.child(hi, (X * i) | 0),
                out,
            );
            self.dc_edge_y(
                self.child(lo, (Y * i) | Z),
                self.child(hi, (Y * i) | 0),
                self.child(hi, (Y * i) | X),
                self.child(lo, (Y * i) | X | Z),
                out,
            );
        }
    }
    /// Handles four cells that share a common `X` edge
    ///
    /// Cells positions are in the order `[0, Y, Y | Z, Z]`, i.e. a right-handed
    /// winding about `+X`.
    fn dc_edge_x(
        &self,
        a: CellIndex,
        b: CellIndex,
        c: CellIndex,
        d: CellIndex,
        out: &mut Mesh,
    ) {
        if [a, b, c, d].iter().all(|v| self.is_leaf(*v)) {
            /*
            let leaf_a = self.leafs[a.index & Self::CELL_INDEX_MASK];
            let leaf_b = self.leafs[b.index & Self::CELL_INDEX_MASK];
            let leaf_c = self.leafs[c.index & Self::CELL_INDEX_MASK];
            let leaf_d = self.leafs[d.index & Self::CELL_INDEX_MASK];

            let vert_a = get_edge_vert(leaf_a.mask, Z, 3);
            let vert_b = get_edge_vert(leaf_a.mask, Z, 2);
            let vert_c = get_edge_vert(leaf_a.mask, Z, 1);
            let vert_d = get_edge_vert(leaf_a.mask, Z, 0);

            let e =
                CELL_TO_VERT_TO_EDGES[leaf_a.mask as usize][vert_a as usize];
            */
            // terminate!
        }
        for i in 0..2 {
            self.dc_edge_x(
                self.child(a, (i * X) | Y | Z),
                self.child(b, (i * X) | Z),
                self.child(c, (i * X) | 0),
                self.child(d, (i * X) | Y),
                out,
            )
        }
    }
    /// Handles four cells that share a common `Y` edge
    ///
    /// Cells positions are in the order `[0, Z, X | Z, X]`, i.e. a right-handed
    /// winding about `+Y`.
    fn dc_edge_y(
        &self,
        a: CellIndex,
        b: CellIndex,
        c: CellIndex,
        d: CellIndex,
        out: &mut Mesh,
    ) {
        if [a, b, c, d].iter().all(|v| self.is_leaf(*v)) {
            /*
            let leaf_a = self.leafs[a.index & Self::CELL_INDEX_MASK];
            let leaf_b = self.leafs[b.index & Self::CELL_INDEX_MASK];
            let leaf_c = self.leafs[c.index & Self::CELL_INDEX_MASK];
            let leaf_d = self.leafs[d.index & Self::CELL_INDEX_MASK];
            */
            // terminate!
        }
        for i in 0..2 {
            self.dc_edge_y(
                self.child(a, (i * Y) | X | Z),
                self.child(b, (i * Y) | X),
                self.child(c, (i * Y) | 0),
                self.child(d, (i * Y) | Z),
                out,
            )
        }
    }
    /// Handles four cells that share a common `Z` edge
    ///
    /// Cells positions are in the order `[0, X, X | Y, Y]`, i.e. a right-handed
    /// winding about `+Z`.
    fn dc_edge_z(
        &self,
        a: CellIndex,
        b: CellIndex,
        c: CellIndex,
        d: CellIndex,
        out: &mut Mesh,
    ) {
        if [a, b, c, d].iter().all(|v| self.is_leaf(*v)) {
            /*
            let leaf_a = self.leafs[a.index & Self::CELL_INDEX_MASK];
            let leaf_b = self.leafs[b.index & Self::CELL_INDEX_MASK];
            let leaf_c = self.leafs[c.index & Self::CELL_INDEX_MASK];
            let leaf_d = self.leafs[d.index & Self::CELL_INDEX_MASK];
            */
            // terminate!
        }
        for i in 0..2 {
            self.dc_edge_z(
                self.child(a, (i * Z) | X | Y),
                self.child(b, (i * Z) | Y),
                self.child(c, (i * Z) | 0),
                self.child(d, (i * Z) | X),
                out,
            )
        }
    }
}

include!(concat!(env!("OUT_DIR"), "/mdc_tables.rs"));

#[cfg(test)]
mod test {
    use super::*;

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
}
