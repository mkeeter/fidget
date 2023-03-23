//! Octree construction and meshing
use crate::eval::{types::Interval, Family, IntervalEval, Tape};

mod frame;
pub mod types;

use frame::{Frame, XYZ, YZX, ZXY};
use types::{Axis, Corner, DirectedEdge, Edge, Intersection, Offset, X, Y, Z};

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

impl std::fmt::Debug for CellData {
    fn fmt(
        &self,
        f: &mut std::fmt::Formatter<'_>,
    ) -> Result<(), std::fmt::Error> {
        let c: Cell = (*self).into();
        c.fmt(f)
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
        let out = CELL_TO_EDGE_TO_VERT[self.mask as usize][e.index()];
        debug_assert_ne!(out.vert.0, u8::MAX);
        debug_assert_ne!(out.edge.0, u8::MAX);
        out
    }
}

#[derive(Copy, Clone, Debug)]
struct CellVertex {
    /// Position, as a relative offset within a cell's bounding box
    pos: nalgebra::Vector3<u16>,

    /// If a vertex is valid, its original position was within the bounding box
    _valid: bool,
}

impl From<CellVertex> for nalgebra::Vector3<u16> {
    fn from(v: CellVertex) -> Self {
        v.pos
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

impl Default for CellIndex {
    fn default() -> Self {
        Self::new()
    }
}

impl CellIndex {
    pub fn new() -> Self {
        let x = Interval::new(-1.0, 1.0);
        let y = Interval::new(-1.0, 1.0);
        let z = Interval::new(-1.0, 1.0);
        CellIndex {
            index: 0,
            x,
            y,
            z,
            depth: 0,
        }
    }

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

    /// Converts from a relative position in the cell to an absolute position
    fn pos<P: Into<nalgebra::Vector3<u16>>>(
        &self,
        p: P,
    ) -> nalgebra::Vector3<f32> {
        let p = p.into();
        let x = self.x.lerp(p.x as f32 / u16::MAX as f32);
        let y = self.y.lerp(p.y as f32 / u16::MAX as f32);
        let z = self.z.lerp(p.z as f32 / u16::MAX as f32);
        nalgebra::Vector3::new(x, y, z)
    }

    /// Converts from an absolute position to a relative position in the cell
    ///
    /// The `bool` indicates whether the vertex was clamped into the cell's
    /// bounding box.
    fn relative(&self, p: nalgebra::Vector3<f32>) -> CellVertex {
        let x = (p.x - self.x.lower()) / self.x.width() * u16::MAX as f32;
        let y = (p.y - self.y.lower()) / self.y.width() * u16::MAX as f32;
        let z = (p.z - self.z.lower()) / self.z.width() * u16::MAX as f32;

        let valid = x >= 0.0
            && x <= u16::MAX as f32
            && y >= 0.0
            && y <= u16::MAX as f32
            && z >= 0.0
            && z <= u16::MAX as f32;

        CellVertex {
            pos: nalgebra::Vector3::new(
                x.clamp(0.0, u16::MAX as f32) as u16,
                y.clamp(0.0, u16::MAX as f32) as u16,
                z.clamp(0.0, u16::MAX as f32) as u16,
            ),
            _valid: valid,
        }
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
            for p in &normal {
                out.write_all(&p.to_le_bytes())?;
            }
            for v in t {
                for p in &self.vertices[*v] {
                    out.write_all(&p.to_le_bytes())?;
                }
            }
            out.write_all(&[0u8; std::mem::size_of::<u16>()])?; // attributes
        }
        Ok(())
    }
}

/// Container used during construction of a [`Mesh`]
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
        verts: &[CellVertex],
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

/// Octree storing occupancy and vertex positions for Manifold Dual Contouring
#[derive(Debug)]
pub struct Octree {
    /// The top two bits determine cell types
    cells: Vec<CellData>,

    /// Cell vertices, given as positions within the cell
    ///
    /// The `bool` in the tuple indicates whether the vertex was clamped to fit
    /// into the cell's bounding box.
    ///
    /// This is indexed by cell leaf index; the exact shape depends heavily on
    /// the number of intersections and vertices within each leaf.
    verts: Vec<CellVertex>,
}

impl Octree {
    /// Builds an octree to the given depth
    ///
    /// The shape is evaluated on the region `[-1, 1]` on all axes
    pub fn build<I: Family>(tape: &Tape<I>, depth: usize) -> Self {
        let i_handle = tape.new_interval_evaluator();

        let mut out = Self {
            cells: vec![CellData(0); 8],
            verts: vec![],
        };

        out.recurse(
            &i_handle,
            CellIndex {
                depth,
                ..CellIndex::default()
            },
        );
        out
    }

    /// Recurse down the octree, rendering the given cell
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
                            index: child + i.index(),
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

    /// Evaluates the given leaf
    fn leaf<I: Family>(&mut self, tape: Tape<I>, cell: CellIndex) {
        let float_eval = tape.new_float_slice_evaluator();

        let mut xs = [0.0; 8];
        let mut ys = [0.0; 8];
        let mut zs = [0.0; 8];
        for i in Corner::iter() {
            let (x, y, z) = cell.corner(i);
            xs[i.index()] = x;
            ys[i.index()] = y;
            zs[i.index()] = z;
        }

        // TODO: reuse evaluators, etc
        let out = float_eval.eval(&xs, &ys, &zs, &[]).unwrap();
        debug_assert_eq!(out.len(), 8);

        // Build a mask of active corners, which determines cell
        // topology / vertex count / active edges / etc.
        let mask = out
            .iter()
            .enumerate()
            .filter(|(_i, &v)| v < 0.0)
            .fold(0, |acc, (i, _v)| acc | (1 << i));

        // Start and endpoints in 3D space for intersection searches
        let mut start = [nalgebra::Vector3::zeros(); 12];
        let mut end = [nalgebra::Vector3::zeros(); 12];
        let mut edge_count = 0;

        // Convert from the corner mask to start and end-points in 3D space,
        // relative to the cell bounds (i.e. as a `Matrix3<u16>`).
        for vs in CELL_TO_VERT_TO_EDGES[mask as usize].iter() {
            for e in *vs {
                // Find the axis that's being used by this edge
                let axis = e.start().index() ^ e.end().index();
                debug_assert_eq!(axis.count_ones(), 1);
                debug_assert!(axis < 8);

                // Pick a position closer to the filled side
                let (a, b) = if e.end().index() & axis != 0 {
                    (0, u16::MAX)
                } else {
                    (u16::MAX, 0)
                };

                // Convert the intersection to a 3D position
                let mut v = nalgebra::Vector3::zeros();
                let i = (axis.trailing_zeros() + 1) % 3;
                let j = (axis.trailing_zeros() + 2) % 3;
                v[i as usize] = if e.start() & Axis::new(1 << i) {
                    u16::MAX
                } else {
                    0
                };
                v[j as usize] = if e.start() & Axis::new(1 << j) {
                    u16::MAX
                } else {
                    0
                };

                v[axis.trailing_zeros() as usize] = a;
                start[edge_count] = v;
                v[axis.trailing_zeros() as usize] = b;
                end[edge_count] = v;
                edge_count += 1;
            }
        }
        // Slice off the unused sections of the arrays
        let start = &mut start[..edge_count]; // always inside
        let end = &mut end[..edge_count]; // always outside

        // Scratch arrays for edge search
        const EDGE_SEARCH_SIZE: usize = 16;
        const EDGE_SEARCH_DEPTH: usize = 4;
        let xs =
            &mut [0f32; 12 * EDGE_SEARCH_SIZE][..edge_count * EDGE_SEARCH_SIZE];
        let ys =
            &mut [0f32; 12 * EDGE_SEARCH_SIZE][..edge_count * EDGE_SEARCH_SIZE];
        let zs =
            &mut [0f32; 12 * EDGE_SEARCH_SIZE][..edge_count * EDGE_SEARCH_SIZE];
        let mut data = crate::eval::bulk::BulkEvalData::default();

        // This part looks hairy, but it's just doing an N-ary search along each
        // edge to find the intersection point.
        for _ in 0..EDGE_SEARCH_DEPTH {
            // Populate edge search arrays
            let mut i = 0;
            for (start, end) in start.iter().zip(end.iter()) {
                for j in 0..EDGE_SEARCH_SIZE {
                    let pos = ((start.map(|i| i as u32)
                        * (EDGE_SEARCH_SIZE - j - 1) as u32)
                        + (end.map(|i| i as u32) * j as u32))
                        / ((EDGE_SEARCH_SIZE - 1) as u32);
                    debug_assert!(pos.max() <= u16::MAX.into());

                    let pos = cell.pos(pos.map(|v| v as u16));
                    xs[i] = pos.x;
                    ys[i] = pos.y;
                    zs[i] = pos.z;
                    i += 1;
                }
            }
            debug_assert_eq!(i, EDGE_SEARCH_SIZE * edge_count);

            // Do the actual evaluation
            let out = float_eval.eval_with(xs, ys, zs, &[], &mut data).unwrap();

            // Update start and end positions based on evaluation
            for ((start, end), search) in start
                .iter_mut()
                .zip(end.iter_mut())
                .zip(out.chunks(EDGE_SEARCH_SIZE))
            {
                // The search must be inside-to-outside
                debug_assert!(search[0] < 0.0);
                debug_assert!(search[EDGE_SEARCH_SIZE - 1] >= 0.0);
                let frac = search
                    .iter()
                    .enumerate()
                    .find(|(_i, v)| **v >= 0.0)
                    .unwrap()
                    .0;
                debug_assert!(frac > 0);
                debug_assert!(frac < EDGE_SEARCH_SIZE);

                let f = |frac| {
                    ((start.map(|i| i as u32)
                        * (EDGE_SEARCH_SIZE - frac - 1) as u32)
                        + (end.map(|i| i as u32) * frac as u32))
                        / ((EDGE_SEARCH_SIZE - 1) as u32)
                };

                let a = f(frac - 1);
                let b = f(frac);

                debug_assert!(a.max() <= u16::MAX.into());
                debug_assert!(b.max() <= u16::MAX.into());
                *start = a.map(|v| v as u16);
                *end = b.map(|v| v as u16);
            }
        }

        // Populate intersections to the average of start and end
        let intersections: arrayvec::ArrayVec<nalgebra::Vector3<u16>, 12> =
            start
                .iter()
                .zip(end.iter())
                .map(|(a, b)| {
                    ((a.map(|v| v as u32) + b.map(|v| v as u32)) / 2)
                        .map(|v| v as u16)
                })
                .collect();

        for (i, xyz) in intersections.iter().enumerate() {
            let pos = cell.pos(*xyz);
            xs[i] = pos.x;
            ys[i] = pos.y;
            zs[i] = pos.z;
        }

        // TODO: special case for cells with multiple gradients ("features")
        let grad_eval = tape.new_grad_slice_evaluator();
        let grads = grad_eval.eval(xs, ys, zs, &[]).unwrap();

        // Now, we're going to solve a quadratic error function for every vertex
        // to position it at the right place.  This gets a little tricky; see
        // https://www.mattkeeter.com/projects/qef for a walkthrough of QEF
        // math and references to primary sources.
        let mut verts: arrayvec::ArrayVec<_, 4> = arrayvec::ArrayVec::new();
        let mut i = 0;
        for vs in CELL_TO_VERT_TO_EDGES[mask as usize].iter() {
            let mut ata = nalgebra::Matrix3::zeros();
            let mut atb = nalgebra::Vector3::zeros();
            // We won't track B^T B here, since it's only useful for calculating
            // the actual error, which we don't really care about.

            let mut mass_point = nalgebra::Vector3::zeros();
            for _ in 0..vs.len() {
                let pos = nalgebra::Vector3::new(xs[i], ys[i], zs[i]);
                mass_point += pos;

                let grad = grads[i];
                let norm = nalgebra::Vector3::new(grad.dx, grad.dy, grad.dz);

                // TODO: correct for non-zero distance value?

                ata += norm * norm.transpose();
                atb += norm * norm.dot(&pos);
                i += 1;
            }
            // Minimize towards mass point of intersections
            let center = mass_point / vs.len() as f32;
            atb -= ata * center;

            // Instead of getting tricky with SVD, let's just throw a LU solver
            // at the problem; this appears to be more robust than the
            // traditional SVD + eigenvalue clamping approach.
            let sol = nalgebra::linalg::LU::new(ata).solve(&atb);
            let pos = sol.map(|c| c + center).unwrap_or(center);

            // Convert back to a relative (within-cell) position and store it
            verts.push(cell.relative(pos));
        }

        let index = self.verts.len();
        self.verts.extend(verts.into_iter());
        // All intersections are valid (within the cell), by definition
        self.verts.extend(
            intersections
                .into_iter()
                .map(|pos| CellVertex { pos, _valid: true }),
        );
        self.cells[cell.index] = Cell::Leaf(Leaf { mask, index }).into();
    }

    /// Recursively walks the dual of the octree, building a mesh
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
        for c in [Corner::new(0), u.into(), v.into(), u | v] {
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

        match self.cells[cell.index].into() {
            Cell::Leaf { .. } | Cell::Full | Cell::Empty => cell,
            Cell::Branch { index } => {
                let (x, y, z) = cell.interval(child);
                CellIndex {
                    index: index + child.index(),
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
        self.dc_face::<T>(
            self.child(lo, t),
            self.child(hi, Corner::new(0)),
            out,
        );
        self.dc_face::<T>(self.child(lo, t | u), self.child(hi, u), out);
        self.dc_face::<T>(self.child(lo, t | v), self.child(hi, v), out);
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
                .zip([u | v, v.into(), Corner::new(0), u.into()])
                .filter(|(leaf, c)| {
                    (leaf.mask & (1 << c.index()) == 0)
                        != (leaf.mask & (1 << (*c | t).index()) == 0)
                })
                .count();
            if sign_change_count == 0 {
                return;
            }
            debug_assert_eq!(sign_change_count, 4);

            let verts = [
                leafs[0].edge(Edge::new((t.index() * 4 + 3) as u8)),
                leafs[1].edge(Edge::new((t.index() * 4 + 2) as u8)),
                leafs[2].edge(Edge::new((t.index() * 4 + 0) as u8)),
                leafs[3].edge(Edge::new((t.index() * 4 + 1) as u8)),
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
            let winding = if leafs[0].mask & (1 << (u | v).index()) == 0 {
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
    use crate::context::bound::{self, BoundContext, BoundNode};
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

    fn sphere(
        ctx: &BoundContext,
        center: [f32; 3],
        radius: f32,
    ) -> bound::BoundNode {
        let (x, y, z) = ctx.axes();
        ((x - center[0]).square()
            + (y - center[1]).square()
            + (z - center[2]).square())
        .sqrt()
            - radius
    }

    fn cube(
        ctx: &BoundContext,
        bx: [f32; 2],
        by: [f32; 2],
        bz: [f32; 2],
    ) -> BoundNode {
        let (x, y, z) = ctx.axes();
        let x_bounds = (bx[0] - x.clone()).max(x - bx[1]);
        let y_bounds = (by[0] - y.clone()).max(y - by[1]);
        let z_bounds = (bz[0] - z.clone()).max(z - bz[1]);
        x_bounds.max(y_bounds).max(z_bounds)
    }

    fn cone(
        ctx: &BoundContext,
        corner: nalgebra::Vector3<f32>,
        tip: nalgebra::Vector3<f32>,
        radius: f32,
    ) -> BoundNode {
        let dir = tip - corner;
        let length = dir.norm();
        let dir = dir.normalize();

        let corner = corner.map(|v| ctx.constant(v as f64));
        let dir = dir.map(|v| ctx.constant(v as f64));

        let (x, y, z) = ctx.axes();
        let point = nalgebra::Vector3::new(x, y, z);
        let offset = point.clone() - corner.clone();

        // a is the distance along the corner-tip direction
        let a = offset.x.clone() * dir.x.clone()
            + offset.y.clone() * dir.y.clone()
            + offset.z.clone() * dir.z.clone();

        // Position of the nearest point on the corner-tip axis
        let a_pos = corner + dir * a.clone();

        // b is the orthogonal distance
        let offset = point - a_pos;
        let b = (offset.x.clone().square()
            + offset.y.clone().square()
            + offset.z.clone().square())
        .sqrt();

        b - radius * (1.0 - a / length)
    }

    #[test]
    fn test_mesh_basic() {
        let ctx = BoundContext::new();
        let shape = sphere(&ctx, [0.0; 3], 0.2);
        let tape = shape.get_tape::<crate::vm::Eval>().unwrap();

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
        assert!(empty_mesh.vertices.is_empty());
        assert!(empty_mesh.triangles.is_empty());

        // Now, at depth-1, each cell should be a Leaf with one vertex
        let octree = Octree::build(&tape, 1);
        assert_eq!(octree.cells.len(), 16); // we always build at least 8 cells
        assert_eq!(Cell::Branch { index: 8 }, octree.cells[0].into());

        // Each of the 6 edges is counted 4 times and each cell has 1 vertex
        assert_eq!(octree.verts.len(), 6 * 4 + 8, "incorrect vertex count");

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
    fn test_sphere_verts() {
        let ctx = BoundContext::new();
        let shape = sphere(&ctx, [0.0; 3], 0.2);

        let tape = shape.get_tape::<crate::vm::Eval>().unwrap();
        let octree = Octree::build(&tape, 1);
        let sphere_mesh = octree.walk_dual();

        let mut edge_count = 0;
        for v in &sphere_mesh.vertices {
            // Edge vertices should be found via binary search and therefore
            // should be close to the true crossing point
            let x_edge = v.x != 0.0;
            let y_edge = v.y != 0.0;
            let z_edge = v.z != 0.0;
            let edge_sum = x_edge as u8 + y_edge as u8 + z_edge as u8;
            assert!(edge_sum == 1 || edge_sum == 3);
            if edge_sum == 1 {
                assert!(
                    (v.norm() - 0.2).abs() < 2.0 / u16::MAX as f32,
                    "edge vertex {v:?} is not at radius 0.2"
                );
                edge_count += 1;
            } else {
                // The sphere looks like a box at this sampling depth, since the
                // edge intersections are all axis-aligned; we'll check that
                // corner vertices are all at [±0.2, ±0.2, ±0.2]
                assert!(
                    (v.abs() - nalgebra::Vector3::new(0.2, 0.2, 0.2)).norm()
                        < 2.0 / 65535.0,
                    "cell vertex {v:?} is not at expected position"
                );
            }
        }
        assert_eq!(edge_count, 6);
    }

    #[test]
    fn test_sphere_manifold() {
        let ctx = BoundContext::new();
        let shape = sphere(&ctx, [0.0; 3], 0.85);

        let tape = shape.get_tape::<crate::vm::Eval>().unwrap();
        let octree = Octree::build(&tape, 3);
        let sphere_mesh = octree.walk_dual();

        if let Err(e) = check_for_vertex_dupes(&sphere_mesh) {
            panic!("{e}");
        }
        if let Err(e) = check_for_edge_matching(&sphere_mesh) {
            panic!("{e}");
        }
    }

    #[test]
    fn test_cube_verts() {
        let ctx = BoundContext::new();
        let shape = cube(&ctx, [-0.1, 0.6], [-0.2, 0.75], [-0.3, 0.4]);

        let tape = shape.get_tape::<crate::vm::Eval>().unwrap();
        let octree = Octree::build(&tape, 1);
        let mesh = octree.walk_dual();
        const EPSILON: f32 = 2.0 / u16::MAX as f32;
        assert!(!mesh.vertices.is_empty());
        for v in &mesh.vertices {
            // Edge vertices should be found via binary search and therefore
            // should be close to the true crossing point
            let x_edge = v.x != 0.0;
            let y_edge = v.y != 0.0;
            let z_edge = v.z != 0.0;
            let edge_sum = x_edge as u8 + y_edge as u8 + z_edge as u8;
            assert!(edge_sum == 1 || edge_sum == 3);

            if edge_sum == 1 {
                assert!(
                    (x_edge
                        && ((v.x - -0.1).abs() < EPSILON
                            || (v.x - 0.6).abs() < EPSILON))
                        || (y_edge
                            && ((v.y - -0.2).abs() < EPSILON
                                || (v.y - 0.75).abs() < EPSILON))
                        || (z_edge
                            && ((v.z - -0.3).abs() < EPSILON
                                || (v.z - 0.4).abs() < EPSILON)),
                    "bad edge position {v:?}"
                );
            } else {
                assert!(
                    ((v.x - -0.1).abs() < EPSILON
                        || (v.x - 0.6).abs() < EPSILON)
                        && ((v.y - -0.2).abs() < EPSILON
                            || (v.y - 0.75).abs() < EPSILON)
                        && ((v.z - -0.3).abs() < EPSILON
                            || (v.z - 0.4).abs() < EPSILON),
                    "bad vertex position {v:?}"
                );
            }
        }
    }

    #[test]
    fn test_plane_center() {
        const EPSILON: f32 = 1e-3;
        let ctx = BoundContext::new();
        for dx in [0.0, 0.25, -0.25, 2.0, -2.0] {
            for dy in [0.0, 0.25, -0.25, 2.0, -2.0] {
                for offset in [0.0, -0.2, 0.2] {
                    let (x, y, z) = ctx.axes();
                    let f = x * dx + y * dy + z + offset;
                    let tape = f.get_tape::<crate::vm::Eval>().unwrap();
                    let octree = Octree::build(&tape, 0);

                    assert_eq!(octree.cells.len(), 8);
                    let pos = CellIndex::default().pos(octree.verts[0]);
                    let mut mass_point = nalgebra::Vector3::zeros();
                    for v in &octree.verts[1..] {
                        mass_point += CellIndex::default().pos(*v);
                    }
                    mass_point /= (octree.verts.len() - 1) as f32;
                    assert!(
                        (pos - mass_point).norm() < EPSILON,
                        "bad vertex position at dx: {dx}, dy: {dy}, \
                         offset: {offset} => {pos:?} != {mass_point:?}"
                    );
                    let eval = tape.new_point_evaluator();
                    for v in &octree.verts {
                        let v = CellIndex::default().pos(*v);
                        let (r, _) = eval.eval(v.x, v.y, v.z, &[]).unwrap();
                        assert!(r.abs() < EPSILON, "bad value at {v:?}: {r}");
                    }
                }
            }
        }
    }

    #[test]
    fn test_cone_vert() {
        let ctx = BoundContext::new();
        let corner = nalgebra::Vector3::new(-1.0, -1.0, -1.0);
        let tip = nalgebra::Vector3::new(0.2, 0.3, 0.4);
        let shape = cone(&ctx, corner, tip, 0.1);
        let tape = shape.get_tape::<crate::vm::Eval>().unwrap();

        let eval = tape.new_point_evaluator();
        let (v, _) = eval.eval(tip.x, tip.y, tip.z, &[]).unwrap();
        assert!(v.abs() < 1e-6, "bad tip value: {v}");
        let (v, _) = eval.eval(corner.x, corner.y, corner.z, &[]).unwrap();
        assert!(v < 0.0, "bad corner value: {v}");

        let octree = Octree::build(&tape, 0);
        assert_eq!(octree.cells.len(), 8);
        assert_eq!(octree.verts.len(), 4);

        let pos = CellIndex::default().pos(octree.verts[0]);
        assert!(
            (pos - tip).norm() < 1e-3,
            "bad vertex position: expected {tip:?}, got {pos:?}"
        );
    }

    #[test]
    fn test_mesh_manifold() {
        for i in 0..256 {
            let ctx = BoundContext::new();
            let mut shape = vec![];
            for j in Corner::iter() {
                if i & (1 << j.index()) != 0 {
                    shape.push(sphere(
                        &ctx,
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
            let shape = shape.into_iter().fold(start, |acc, s| acc.min(s));

            // Now, we have our shape, which is 0-8 spheres placed at the
            // corners of the cell spanning [0, 0.25]
            let tape = shape.get_tape::<crate::vm::Eval>().unwrap();
            let octree = Octree::build(&tape, 2);

            let mesh = octree.walk_dual();
            if i != 0 && i != 255 {
                assert!(!mesh.vertices.is_empty());
                assert!(!mesh.triangles.is_empty());
            }
            if let Err(e) = check_for_vertex_dupes(&mesh) {
                panic!("mask {i:08b} has {e}");
            }
            if let Err(e) = check_for_edge_matching(&mesh) {
                panic!("mask {i:08b} has {e}");
            }
        }
    }

    fn check_for_vertex_dupes(mesh: &Mesh) -> Result<(), String> {
        let mut verts = mesh.vertices.clone();
        verts.sort_by_key(|k| (k.x.to_bits(), k.y.to_bits(), k.z.to_bits()));
        for i in 1..verts.len() {
            if verts[i - 1] == verts[i] {
                return Err("duplicate vertices".to_owned());
            }
        }
        Ok(())
    }

    fn check_for_edge_matching(mesh: &Mesh) -> Result<(), String> {
        let mut edges: BTreeMap<_, usize> = BTreeMap::new();
        for t in &mesh.triangles {
            for edge in [(t.x, t.y), (t.y, t.z), (t.z, t.x)] {
                *edges.entry(edge).or_default() += 1;
            }
        }
        for (&(a, b), &i) in &edges {
            if i != 1 {
                return Err(format!("duplicate edge ({a}, {b})"));
            }
            if !edges.contains_key(&(b, a)) {
                return Err("unpaired edges".to_owned());
            }
        }
        Ok(())
    }
}
