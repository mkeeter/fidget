use crate::eval::{types::Interval, Family, IntervalEval, Tape};

struct Leaf {
    corners: [f32; 8],
}

pub struct Octree {
    /// The top two bits determine cell types
    cells: Vec<usize>,
    leafs: Vec<Leaf>,
}

const X: usize = 1;
const Y: usize = 2;
const Z: usize = 4;

impl Octree {
    const CELL_TYPE_MASK: usize =
        0b11 << (std::mem::size_of::<usize>() * 8 - 2);
    const CELL_TYPE_BRANCH: usize =
        0b00 << (std::mem::size_of::<usize>() * 8 - 2);
    const CELL_TYPE_EMPTY: usize =
        0b10 << (std::mem::size_of::<usize>() * 8 - 2);
    const CELL_TYPE_FILLED: usize =
        0b11 << (std::mem::size_of::<usize>() * 8 - 2);
    const CELL_TYPE_LEAF: usize =
        0b01 << (std::mem::size_of::<usize>() * 8 - 2);

    pub fn build<I: Family>(tape: Tape<I>, depth: usize) -> Self {
        let i_handle = tape.new_interval_evaluator();
        let x = Interval::new(-1.0, 1.0);
        let y = Interval::new(-1.0, 1.0);
        let z = Interval::new(-1.0, 1.0);

        let mut out = Self {
            cells: vec![0; 8],
            leafs: vec![],
        };

        out.recurse(&i_handle, 0, x, y, z, depth);
        out
    }

    fn recurse<I: Family>(
        &mut self,
        i_handle: &IntervalEval<I>,
        index: usize,
        x: Interval,
        y: Interval,
        z: Interval,
        depth: usize,
    ) {
        let (i, r) = i_handle.eval(x, y, z, &[]).unwrap();
        if i.upper() < 0.0 {
            self.cells[index] = Self::CELL_TYPE_FILLED;
        } else if i.lower() > 0.0 {
            self.cells[index] = Self::CELL_TYPE_EMPTY;
        } else {
            let sub_tape = r.map(|r| r.simplify().unwrap());
            if depth == 0 {
                let tape = sub_tape.unwrap_or_else(|| i_handle.tape());
                let eval = tape.new_float_slice_evaluator();

                let (x_lo, x_hi) = (x.lower(), x.upper());
                let (y_lo, y_hi) = (y.lower(), y.upper());
                let (z_lo, z_hi) = (z.lower(), z.upper());
                let mut xs = [0.0; 8];
                let mut ys = [0.0; 8];
                let mut zs = [0.0; 8];
                for i in 0..8 {
                    xs[i] = if i & 1 == 0 { x_lo } else { x_hi };
                    ys[i] = if i & 1 == 0 { y_lo } else { y_hi };
                    zs[i] = if i & 1 == 0 { z_lo } else { z_hi };
                }
                let out = eval.eval(&xs, &ys, &zs, &[]).unwrap();
                let leaf_index = self.leafs.len();
                assert!(leaf_index & Self::CELL_TYPE_MASK == 0);

                self.leafs.push(Leaf {
                    corners: out.try_into().unwrap(),
                });
                self.cells[index] = leaf_index & Self::CELL_TYPE_LEAF;
            } else {
                let child = self.cells.len();
                for _ in 0..8 {
                    self.cells.push(0);
                }
                self.cells[index] = Self::CELL_TYPE_BRANCH | child;
                let (x_lo, x_hi) = x.split();
                let (y_lo, y_hi) = y.split();
                let (z_lo, z_hi) = z.split();
                for i in 0..8 {
                    let x = if i & 1 == 0 { x_lo } else { x_hi };
                    let y = if i & 2 == 0 { y_lo } else { y_hi };
                    let z = if i & 4 == 0 { z_lo } else { z_hi };
                    self.recurse(i_handle, child + i, x, y, z, depth - 1);
                }
            }
        }
    }

    pub fn walk_dual(&self) {
        self.dc_cell(0);
    }
}

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
/// The 8 octree cells are numbered equivalently, based on their corner vertex
#[allow(clippy::modulo_one, clippy::identity_op)]
impl Octree {
    fn dc_cell(&self, index: usize) {
        let i = self.cells[index];
        let ty = i & Self::CELL_TYPE_MASK;

        if ty == Self::CELL_TYPE_BRANCH {
            let c = i & !Self::CELL_TYPE_MASK;
            assert_eq!(c % 8, 0);
            for i in 0..8 {
                self.dc_cell(c | i);
            }
            for i in 0..4 {
                self.dc_face_x(
                    c | (2 * X * (i / X) + (i % X)) | 0,
                    c | (2 * X * (i / X) + (i % X)) | X,
                );
                self.dc_face_y(
                    c | (2 * Y * (i / Y) + (i % Y)) | 0,
                    c | (2 * Y * (i / Y) + (i % Y)) | Y,
                );
                self.dc_face_z(
                    c | (2 * Z * (i / Z) + (i % Z)) | 0,
                    c | (2 * Z * (i / Z) + (i % Z)) | Z,
                );
            }
            for i in 0..2 {
                self.dc_edge_x(
                    c | (X * i) | 0,
                    c | (X * i) | Y,
                    c | (X * i) | Y | Z,
                    c | (X * i) | Z,
                );
                self.dc_edge_y(
                    c | (Y * i) | 0,
                    c | (Y * i) | Z,
                    c | (Y * i) | X | Z,
                    c | (Y * i) | X,
                );
                self.dc_edge_z(
                    c | (Z * i) | 0,
                    c | (Z * i) | X,
                    c | (Z * i) | X | Y,
                    c | (Z * i) | Y,
                );
            }
        }
    }

    /// Checks whether the given cell is a leaf node
    pub fn is_leaf(&self, index: usize) -> bool {
        self.cells[index] & Self::CELL_TYPE_MASK != Self::CELL_TYPE_BRANCH
    }

    /// Looks up the given child of a cell.
    ///
    /// If the cell is a leaf node, returns that cell instead.
    pub fn child(&self, index: usize, child: usize) -> usize {
        assert!(child < 8);
        if self.is_leaf(index) {
            index
        } else {
            (self.cells[index] & Self::CELL_TYPE_MASK) + child
        }
    }

    /// Handles two cells which share a common `YZ` face
    ///
    /// `lo` is below `hi` on the `Z` axis
    fn dc_face_x(&self, lo: usize, hi: usize) {
        if self.is_leaf(lo) && self.is_leaf(hi) {
            return;
        }
        self.dc_face_x(self.child(lo, X), self.child(hi, 0));
        self.dc_face_x(self.child(lo, X | Y), self.child(hi, Y));
        self.dc_face_x(self.child(lo, X | Z), self.child(hi, Z));
        self.dc_face_x(self.child(lo, X | Y | Z), self.child(hi, Y | Z));
        for i in 0..2 {
            self.dc_edge_y(
                self.child(lo, (Y * i) | X),
                self.child(lo, (Y * i) | X | Z),
                self.child(hi, (Y * i) | Z),
                self.child(lo, (Y * i) | 0),
            );
            self.dc_edge_z(
                self.child(lo, (Z * i) | X),
                self.child(hi, (Z * i) | 0),
                self.child(hi, (Z * i) | Y),
                self.child(lo, (Z * i) | X | Y),
            );
        }
    }
    /// Handles two cells which share a common `XZ` face
    ///
    /// `lo` is below `hi` on the `Y` axis
    fn dc_face_y(&self, lo: usize, hi: usize) {
        if self.is_leaf(lo) && self.is_leaf(hi) {
            return;
        }
        self.dc_face_x(self.child(lo, Y), self.child(hi, 0));
        self.dc_face_x(self.child(lo, Y | X), self.child(hi, X));
        self.dc_face_x(self.child(lo, Y | Z), self.child(hi, Z));
        self.dc_face_x(self.child(lo, Y | X | Z), self.child(hi, X | Z));
        for i in 0..2 {
            self.dc_edge_x(
                self.child(lo, (X * i) | Y),
                self.child(hi, (X * i) | 0),
                self.child(hi, (X * i) | Z),
                self.child(lo, (X * i) | Y | Z),
            );
            self.dc_edge_z(
                self.child(lo, (Z * i) | Y),
                self.child(lo, (Z * i) | X | Y),
                self.child(hi, (Z * i) | X),
                self.child(hi, (Z * i) | 0),
            );
        }
    }
    /// Handles two cells which share a common `XY` face
    ///
    /// `lo` is below `hi` on the `Z` axis
    fn dc_face_z(&self, lo: usize, hi: usize) {
        if self.is_leaf(lo) && self.is_leaf(hi) {
            return;
        }
        self.dc_face_x(self.child(lo, Z), self.child(hi, 0));
        self.dc_face_x(self.child(lo, Z | X), self.child(hi, X));
        self.dc_face_x(self.child(lo, Z | Y), self.child(hi, Y));
        self.dc_face_x(self.child(lo, Z | X | Y), self.child(hi, X | Y));
        for i in 0..2 {
            self.dc_edge_x(
                self.child(lo, (X * i) | Z),
                self.child(lo, (X * i) | Y | Z),
                self.child(hi, (X * i) | Y),
                self.child(hi, (X * i) | 0),
            );
            self.dc_edge_y(
                self.child(lo, (Y * i) | Z),
                self.child(hi, (Y * i) | 0),
                self.child(hi, (Y * i) | X),
                self.child(lo, (Y * i) | X | Z),
            );
        }
    }
    /// Handles four cells that share a common `X` edge
    ///
    /// Cells positions are in the order `[0, Y, Z, Y | Z]`, i.e. a right-handed
    /// winding about `+X`.
    fn dc_edge_x(&self, a: usize, b: usize, c: usize, d: usize) {
        if [a, b, c, d].iter().all(|v| self.is_leaf(*v)) {
            // terminate!
        }
        for i in 0..2 {
            self.dc_edge_x(
                self.child(a, (i * Z) | Y | Z),
                self.child(b, (i * Z) | Z),
                self.child(c, (i * Z) | 0),
                self.child(d, (i * Z) | Y),
            )
        }
        todo!()
    }
    /// Handles four cells that share a common `Y` edge
    ///
    /// Cells positions are in the order `[0, Z, X, X | Z]`, i.e. a right-handed
    /// winding about `+Y`.
    fn dc_edge_y(&self, a: usize, b: usize, c: usize, d: usize) {
        if [a, b, c, d].iter().all(|v| self.is_leaf(*v)) {
            // terminate!
        }
        // Two calls to edgeproc
        todo!()
    }
    /// Handles four cells that share a common `Z` edge
    ///
    /// Cells positions are in the order `[0, X, Y, X | Y]`, i.e. a right-handed
    /// winding about `+Z`.
    fn dc_edge_z(&self, a: usize, b: usize, c: usize, d: usize) {
        if [a, b, c, d].iter().all(|v| self.is_leaf(*v)) {
            // terminate!
        }
        // Two calls to edgeproc
        todo!()
    }
}
