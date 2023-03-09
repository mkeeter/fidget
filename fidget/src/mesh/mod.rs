use crate::eval::{types::Interval, Family, IntervalEval, Tape};

struct Leaf {
    /// Corner mask, with set bits (1) for cube corners inside the surface
    ///
    /// "Inside" means < 0.0; exactly 0.0 is treated as outside.
    mask: u8,

    /// Intersection positions along active edges
    ///
    /// Intersections are numbered as `axis * 4 + next(axis) * 2 + prev(axis)`,
    /// (TODO is this correct?)
    /// where `prev(...)` and `next(...)` produce a right-handed coordinate
    /// system from the given axis.
    ///
    /// The intersection position is given as a fraction of the distance along
    /// the edge, where 0 is the minimum value on the relevant axis.
    intersections: [u16; 12],
}

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
            self.cells[cell.index] = Self::CELL_TYPE_FILLED;
        } else if i.lower() > 0.0 {
            self.cells[cell.index] = Self::CELL_TYPE_EMPTY;
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
                let leaf_index = self.leafs.len();

                // Make sure we haven't made too many leafs
                assert!(leaf_index & Self::CELL_TYPE_MASK == 0);
                assert_eq!(out.len(), 8);

                // Build a mask of active corners, which determines cell
                // topology / vertex count / active edges / etc.
                let mask = out
                    .iter()
                    .enumerate()
                    .filter(|(_i, &v)| v < 0.0)
                    .fold(0, |acc, (i, _v)| acc | (1 << i));

                // Pick fake intersections for now
                let mut intersections = [0; 12];
                for verts in CELL_TO_VERT_TO_EDGES[mask as usize] {
                    for e in *verts {
                        intersections[(e % 12) as usize] =
                            if e / 12 != 0 { 16384 } else { u16::MAX - 16384 }
                    }
                }
                self.leafs.push(Leaf {
                    mask,
                    intersections,
                });
                self.cells[cell.index] = leaf_index & Self::CELL_TYPE_LEAF;
            } else {
                let child = self.cells.len();
                for _ in 0..8 {
                    self.cells.push(0);
                }
                self.cells[cell.index] = Self::CELL_TYPE_BRANCH | child;
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

    pub fn walk_dual(&self) {
        let x = Interval::new(-1.0, 1.0);
        let y = Interval::new(-1.0, 1.0);
        let z = Interval::new(-1.0, 1.0);

        self.dc_cell(CellIndex {
            index: 0,
            x,
            y,
            z,
            depth: 0,
        });
    }
}

#[allow(clippy::modulo_one, clippy::identity_op)]
impl Octree {
    fn dc_cell(&self, cell: CellIndex) {
        let i = self.cells[cell.index];
        let ty = i & Self::CELL_TYPE_MASK;

        if ty == Self::CELL_TYPE_BRANCH {
            assert_eq!(cell.index % 8, 0);
            for i in 0..8 {
                self.dc_cell(self.child(cell, i));
            }
            for i in 0..4 {
                self.dc_face_x(
                    self.child(cell, (2 * X * (i / X) + (i % X)) | 0),
                    self.child(cell, (2 * X * (i / X) + (i % X)) | X),
                );
                self.dc_face_y(
                    self.child(cell, (2 * Y * (i / Y) + (i % Y)) | 0),
                    self.child(cell, (2 * Y * (i / Y) + (i % Y)) | Y),
                );
                self.dc_face_z(
                    self.child(cell, (2 * Z * (i / Z) + (i % Z)) | 0),
                    self.child(cell, (2 * Z * (i / Z) + (i % Z)) | Z),
                );
            }
            for i in 0..2 {
                self.dc_edge_x(
                    self.child(cell, (X * i) | 0),
                    self.child(cell, (X * i) | Y),
                    self.child(cell, (X * i) | Y | Z),
                    self.child(cell, (X * i) | Z),
                );
                self.dc_edge_y(
                    self.child(cell, (Y * i) | 0),
                    self.child(cell, (Y * i) | Z),
                    self.child(cell, (Y * i) | X | Z),
                    self.child(cell, (Y * i) | X),
                );
                self.dc_edge_z(
                    self.child(cell, (Z * i) | 0),
                    self.child(cell, (Z * i) | X),
                    self.child(cell, (Z * i) | X | Y),
                    self.child(cell, (Z * i) | Y),
                );
            }
        }
    }

    /// Checks whether the given cell is a leaf node
    fn is_leaf(&self, cell: CellIndex) -> bool {
        self.cells[cell.index] & Self::CELL_TYPE_MASK != Self::CELL_TYPE_BRANCH
    }

    /// Looks up the given child of a cell.
    ///
    /// If the cell is a leaf node, returns that cell instead.
    fn child(&self, cell: CellIndex, child: usize) -> CellIndex {
        assert!(child < 8);
        if self.is_leaf(cell) {
            cell
        } else {
            let index = (self.cells[cell.index] & Self::CELL_TYPE_MASK) + child;
            let (x, y, z) = cell.interval(child);
            CellIndex {
                index,
                x,
                y,
                z,
                depth: cell.depth + 1,
            }
        }
    }

    /// Handles two cells which share a common `YZ` face
    ///
    /// `lo` is below `hi` on the `X` axis
    fn dc_face_x(&self, lo: CellIndex, hi: CellIndex) {
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
                self.child(hi, (Y * i) | 0),
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
    fn dc_face_y(&self, lo: CellIndex, hi: CellIndex) {
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
                self.child(lo, (X * i) | Y | Z),
                self.child(hi, (X * i) | Z),
                self.child(hi, (X * i) | 0),
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
    fn dc_face_z(&self, lo: CellIndex, hi: CellIndex) {
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
    /// Cells positions are in the order `[0, Y, Y | Z, Z]`, i.e. a right-handed
    /// winding about `+X`.
    fn dc_edge_x(
        &self,
        a: CellIndex,
        b: CellIndex,
        c: CellIndex,
        d: CellIndex,
    ) {
        if [a, b, c, d].iter().all(|v| self.is_leaf(*v)) {
            // terminate!
        }
        for i in 0..2 {
            self.dc_edge_x(
                self.child(a, (i * X) | Y | Z),
                self.child(b, (i * X) | Z),
                self.child(c, (i * X) | 0),
                self.child(d, (i * X) | Y),
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
    ) {
        if [a, b, c, d].iter().all(|v| self.is_leaf(*v)) {
            // terminate!
        }
        for i in 0..2 {
            self.dc_edge_y(
                self.child(a, (i * Y) | X | Z),
                self.child(b, (i * Y) | X),
                self.child(c, (i * Y) | 0),
                self.child(d, (i * Y) | Z),
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
    ) {
        if [a, b, c, d].iter().all(|v| self.is_leaf(*v)) {
            // terminate!
        }
        for i in 0..2 {
            self.dc_edge_z(
                self.child(a, (i * Z) | X | Y),
                self.child(b, (i * Z) | Y),
                self.child(c, (i * Z) | 0),
                self.child(d, (i * Z) | X),
            )
        }
    }
}

include!(concat!(env!("OUT_DIR"), "/mdc_tables.rs"));
