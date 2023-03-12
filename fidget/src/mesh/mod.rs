use std::num::NonZeroUsize;

use crate::eval::{types::Interval, Family, IntervalEval, Tape};

struct Axis(u8);

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

/// Cell mask, as an 8-bit value representing set corners
#[derive(Copy, Clone, Debug)]
struct Mask(u8);

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
    fn corner(&self, i: Corner) -> (f32, f32, f32) {
        let x = if i & X {
            self.x.lower()
        } else {
            self.x.upper()
        };
        let y = if i & Y {
            self.y.lower()
        } else {
            self.y.upper()
        };
        let z = if i & Z {
            self.z.lower()
        } else {
            self.z.upper()
        };
        (x, y, z)
    }

    /// Returns the interval of the given child (0-7)
    fn interval(&self, i: Corner) -> (Interval, Interval, Interval) {
        // TODO: make this a function in `Interval`?
        let x = if i & X {
            Interval::new(self.x.lower(), self.x.midpoint())
        } else {
            Interval::new(self.x.midpoint(), self.x.upper())
        };
        let y = if i & Y {
            Interval::new(self.y.lower(), self.y.midpoint())
        } else {
            Interval::new(self.y.midpoint(), self.y.upper())
        };
        let z = if i & Z {
            Interval::new(self.z.lower(), self.z.midpoint())
        } else {
            Interval::new(self.z.midpoint(), self.z.upper())
        };
        (x, y, z)
    }

    fn pos(&self, p: nalgebra::Vector3<u16>) -> nalgebra::Vector3<f32> {
        let x = self.x.lerp(p.x as f32 / u16::MAX as f32);
        let y = self.y.lerp(p.y as f32 / u16::MAX as f32);
        let z = self.z.lerp(p.y as f32 / u16::MAX as f32);
        nalgebra::Vector3::new(x, y, z)
    }
}

////////////////////////////////////////////////////////////////////////////////

/// An indexed 3D mesh
pub struct Mesh {
    triangles: Vec<nalgebra::Vector3<usize>>,
    vertices: Vec<nalgebra::Vector3<f32>>,
}

impl Default for Mesh {
    fn default() -> Self {
        Self {
            triangles: vec![],
            vertices: vec![nalgebra::Vector3::zeros()],
        }
    }
}

impl Mesh {
    /// Builds a new mesh
    ///
    /// Note that this performs memory allocation, because we reserve the 0th
    /// position in [`self.vertices`] as a marker.
    pub fn new() -> Self {
        Self::default()
    }
}

#[derive(Default)]
struct MeshBuilder {
    /// Map from indexes in [`Octree::verts`] to `out.vertices`
    map: Vec<Option<NonZeroUsize>>,
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
        if v > self.map.len() {
            self.map.resize(v, Option::None);
        }
        match self.map[v] {
            Some(u) => u.get(),
            None => {
                let next_vert = self.out.vertices.len();
                debug_assert!(next_vert >= 1);

                self.out.vertices.push(cell.pos(verts[v]));
                self.map[v] = NonZeroUsize::new(next_vert);

                next_vert
            }
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
                for _ in Corner::iter() {
                    self.cells.push(CellData(0));
                }
                self.cells[cell.index] = Cell::Branch { index: child }.into();
                let (x_lo, x_hi) = cell.x.split();
                let (y_lo, y_hi) = cell.y.split();
                let (z_lo, z_hi) = cell.z.split();
                for i in Corner::iter() {
                    let x = if i & X { x_lo } else { x_hi };
                    let y = if i & Y { y_lo } else { y_hi };
                    let z = if i & Z { z_lo } else { z_hi };
                    self.recurse(
                        i_handle,
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

            let mut dc_face_x = |a| {
                self.dc_face_x(
                    self.child(cell, a),
                    self.child(cell, a | X),
                    out,
                )
            };
            dc_face_x(Corner(0));
            dc_face_x(Y.into());
            dc_face_x(Z.into());
            dc_face_x(Y | Z);

            let mut dc_face_y = |a| {
                self.dc_face_y(
                    self.child(cell, a),
                    self.child(cell, a | Y),
                    out,
                )
            };
            dc_face_y(Corner(0));
            dc_face_y(X.into());
            dc_face_y(Z.into());
            dc_face_y(X | Z);

            let mut dc_face_z = |a| {
                self.dc_face_z(
                    self.child(cell, a),
                    self.child(cell, a | Z),
                    out,
                )
            };
            dc_face_z(Corner(0));
            dc_face_z(X.into());
            dc_face_z(Y.into());
            dc_face_z(X | Y);

            for i in [false, true] {
                self.dc_edge_x(
                    self.child(cell, (X * i)),
                    self.child(cell, (X * i) | Y),
                    self.child(cell, (X * i) | Y | Z),
                    self.child(cell, (X * i) | Z),
                    out,
                );
                self.dc_edge_y(
                    self.child(cell, (Y * i)),
                    self.child(cell, (Y * i) | Z),
                    self.child(cell, (Y * i) | X | Z),
                    self.child(cell, (Y * i) | X),
                    out,
                );
                self.dc_edge_z(
                    self.child(cell, (Z * i)),
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
        matches!(self.cells[cell.index].into(), Cell::Leaf { .. })
    }

    /// Handles two cells which share a common `YZ` face
    ///
    /// `lo` is below `hi` on the `X` axis
    fn dc_face_x(&self, lo: CellIndex, hi: CellIndex, out: &mut MeshBuilder) {
        if self.is_leaf(lo) && self.is_leaf(hi) {
            return;
        }
        self.dc_face_x(self.child(lo, X), self.child(hi, Corner(0)), out);
        self.dc_face_x(self.child(lo, X | Y), self.child(hi, Y), out);
        self.dc_face_x(self.child(lo, X | Z), self.child(hi, Z), out);
        self.dc_face_x(self.child(lo, X | Y | Z), self.child(hi, Y | Z), out);
        for i in [false, true] {
            self.dc_edge_y(
                self.child(lo, (Y * i) | X),
                self.child(lo, (Y * i) | X | Z),
                self.child(hi, (Y * i) | Z),
                self.child(hi, (Y * i)),
                out,
            );
            self.dc_edge_z(
                self.child(lo, (Z * i) | X),
                self.child(hi, (Z * i)),
                self.child(hi, (Z * i) | Y),
                self.child(lo, (Z * i) | X | Y),
                out,
            );
        }
    }
    /// Handles two cells which share a common `XZ` face
    ///
    /// `lo` is below `hi` on the `Y` axis
    fn dc_face_y(&self, lo: CellIndex, hi: CellIndex, out: &mut MeshBuilder) {
        if self.is_leaf(lo) && self.is_leaf(hi) {
            return;
        }
        self.dc_face_y(self.child(lo, Y), self.child(hi, Corner(0)), out);
        self.dc_face_y(self.child(lo, Y | X), self.child(hi, X), out);
        self.dc_face_y(self.child(lo, Y | Z), self.child(hi, Z), out);
        self.dc_face_y(self.child(lo, Y | X | Z), self.child(hi, X | Z), out);
        for i in [false, true] {
            self.dc_edge_x(
                self.child(lo, (X * i) | Y),
                self.child(lo, (X * i) | Y | Z),
                self.child(hi, (X * i) | Z),
                self.child(hi, (X * i)),
                out,
            );
            self.dc_edge_z(
                self.child(lo, (Z * i) | Y),
                self.child(lo, (Z * i) | X | Y),
                self.child(hi, (Z * i) | X),
                self.child(hi, (Z * i)),
                out,
            );
        }
    }
    /// Handles two cells which share a common `XY` face
    ///
    /// `lo` is below `hi` on the `Z` axis
    fn dc_face_z(&self, lo: CellIndex, hi: CellIndex, out: &mut MeshBuilder) {
        if self.is_leaf(lo) && self.is_leaf(hi) {
            return;
        }
        self.dc_face_z(self.child(lo, Z), self.child(hi, Corner(0)), out);
        self.dc_face_z(self.child(lo, Z | X), self.child(hi, X), out);
        self.dc_face_z(self.child(lo, Z | Y), self.child(hi, Y), out);
        self.dc_face_z(self.child(lo, Z | X | Y), self.child(hi, X | Y), out);
        for i in [false, true] {
            self.dc_edge_x(
                self.child(lo, (X * i) | Z),
                self.child(lo, (X * i) | Y | Z),
                self.child(hi, (X * i) | Y),
                self.child(hi, (X * i)),
                out,
            );
            self.dc_edge_y(
                self.child(lo, (Y * i) | Z),
                self.child(hi, (Y * i)),
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
        out: &mut MeshBuilder,
    ) {
        let cs = [a, b, c, d];
        if cs.iter().all(|v| self.is_leaf(*v)) {
            let leafs = cs.map(|cell| {
                let Cell::Leaf(leaf) = self.cells[cell.index].into() else
                    { unreachable!() };
                leaf
            });
            let verts = [
                leafs[0].edge(Edge(3)),
                leafs[1].edge(Edge(2)),
                leafs[2].edge(Edge(1)),
                leafs[3].edge(Edge(0)),
            ];

            // Pick the intersection vertex based on the deepest cell
            let deepest = (0..4).max_by_key(|i| cs[*i].depth).unwrap();
            let i = out.get(
                leafs[deepest].index + verts[deepest].edge.0 as usize,
                cs[deepest],
                &self.verts,
            );

            // Helper function to extract other vertices
            let mut v = |i: usize| {
                out.get(
                    leafs[i].index + verts[i].vert.0 as usize,
                    cs[i],
                    &self.verts,
                )
            };
            let vs = [v(0), v(1), v(2), v(3)];

            for j in 0..4 {
                out.out.triangles.push(nalgebra::Vector3::new(
                    vs[j],
                    vs[(j + 1) % 4],
                    i,
                ))
            }
        }
        for i in [false, true] {
            self.dc_edge_x(
                self.child(a, (X * i) | Y | Z),
                self.child(b, (X * i) | Z),
                self.child(c, (X * i)),
                self.child(d, (X * i) | Y),
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
        out: &mut MeshBuilder,
    ) {
        let cs = [a, b, c, d];
        if cs.iter().all(|v| self.is_leaf(*v)) {
            let leafs = cs.map(|cell| {
                let Cell::Leaf(leaf) = self.cells[cell.index].into() else
                    { unreachable!() };
                leaf
            });
            let verts = [
                leafs[0].edge(Edge(4 + 3)),
                leafs[1].edge(Edge(4 + 2)),
                leafs[2].edge(Edge(4 + 1)),
                leafs[3].edge(Edge(4 + 0)),
            ];

            // Pick the intersection vertex based on the deepest cell
            let deepest = (0..4).max_by_key(|i| cs[*i].depth).unwrap();
            let i = out.get(
                leafs[deepest].index + verts[deepest].edge.0 as usize,
                cs[deepest],
                &self.verts,
            );

            // Helper function to extract other vertices
            let mut v = |i: usize| {
                out.get(
                    leafs[i].index + verts[i].vert.0 as usize,
                    cs[i],
                    &self.verts,
                )
            };
            let vs = [v(0), v(1), v(2), v(3)];

            for j in 0..4 {
                out.out.triangles.push(nalgebra::Vector3::new(
                    vs[j],
                    vs[(j + 1) % 4],
                    i,
                ))
            }
        }
        for i in [false, true] {
            self.dc_edge_y(
                self.child(a, (Y * i) | X | Z),
                self.child(b, (Y * i) | X),
                self.child(c, (Y * i)),
                self.child(d, (Y * i) | Z),
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
        out: &mut MeshBuilder,
    ) {
        let cs = [a, b, c, d];
        if cs.iter().all(|v| self.is_leaf(*v)) {
            let leafs = cs.map(|cell| {
                let Cell::Leaf(leaf) = self.cells[cell.index].into() else
                    { unreachable!() };
                leaf
            });
            let verts = [
                leafs[0].edge(Edge(8 + 3)),
                leafs[1].edge(Edge(8 + 2)),
                leafs[2].edge(Edge(8 + 1)),
                leafs[3].edge(Edge(8 + 0)),
            ];

            // Pick the intersection vertex based on the deepest cell
            let deepest = (0..4).max_by_key(|i| cs[*i].depth).unwrap();
            let i = out.get(
                leafs[deepest].index + verts[deepest].edge.0 as usize,
                cs[deepest],
                &self.verts,
            );

            // Helper function to extract other vertices
            let mut v = |i: usize| {
                out.get(
                    leafs[i].index + verts[i].vert.0 as usize,
                    cs[i],
                    &self.verts,
                )
            };
            let vs = [v(0), v(1), v(2), v(3)];

            for j in 0..4 {
                out.out.triangles.push(nalgebra::Vector3::new(
                    vs[j],
                    vs[(j + 1) % 4],
                    i,
                ))
            }
        }
        for i in [false, true] {
            self.dc_edge_z(
                self.child(a, (Z * i) | X | Y),
                self.child(b, (Z * i) | Y),
                self.child(c, (Z * i)),
                self.child(d, (Z * i) | X),
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
