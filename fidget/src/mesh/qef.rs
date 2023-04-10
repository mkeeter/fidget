use super::cell::{CellIndex, CellVertex};

/// Solver for a quadratic error function to position a vertex within a cell
pub struct QuadraticErrorSolver {
    /// A^T A term
    ata: nalgebra::Matrix3<f32>,

    /// A^T B term
    atb: nalgebra::Vector3<f32>,

    /// B^T B term
    btb: f32,

    /// Mass point of intersections is stored as XYZ / W, so that summing works
    mass_point: nalgebra::Vector4<f32>,
}

impl QuadraticErrorSolver {
    pub fn new() -> Self {
        Self {
            ata: nalgebra::Matrix3::zeros(),
            atb: nalgebra::Vector3::zeros(),
            btb: 0.0,
            mass_point: nalgebra::Vector4::zeros(),
        }
    }

    /// Adds a new intersection to the QEF
    ///
    /// `pos` is the position of the intersection and is accumulated in the mass
    /// point.  `grad` is the gradient at the surface, and is normalized in this
    /// function.
    pub fn add_intersection(
        &mut self,
        pos: nalgebra::Vector3<f32>,
        grad: nalgebra::Vector3<f32>,
    ) {
        self.mass_point += nalgebra::Vector4::new(pos.x, pos.y, pos.z, 1.0);
        let norm = grad.normalize();
        self.ata += norm * norm.transpose();
        self.atb += norm * norm.dot(&pos);
        self.btb += norm.dot(&pos).powi(2);
    }

    /// Solve the given QEF, minimizing towards the mass point
    ///
    /// Returns a vertex localized within the given cell, and adjusts the solver
    /// to increase the likelyhood that the vertex is bounded in the cell.
    pub fn solve(&self, cell: CellIndex) -> CellVertex {
        let center = self.mass_point.xyz() / self.mass_point.w as f32;
        let btb = self.btb
            + ((center.transpose() * self.ata - self.atb.transpose()) * center
                - center.transpose() * self.atb)[0];
        let atb = self.atb - self.ata * center;

        let svd = nalgebra::linalg::SVD::new(self.ata, true, true);
        // "Dual Contouring: The Secret Sauce" recomments a threshold of 0.1
        // when using normalized gradients, but I've found that fails on
        // things like the cone model.  Instead, we'll be a little more
        // clever: we'll pick the smallest epsilon that keeps the feature in
        // the cell without dramatically increasing QEF error.
        //
        // TODO: iterating by epsilons is a _little_ silly, because what we
        // actually care about is turning off the 0/1/2/3 lowest eigenvalues
        // in the solution matrix.
        const EPSILONS: &[f32] = &[1e-4, 1e-3, 1e-2];
        let mut prev = None;
        for (i, &epsilon) in EPSILONS.iter().enumerate() {
            let sol = svd.solve(&atb, epsilon);
            let pos = sol.map(|c| c + center).unwrap_or(center);
            let err = (pos.transpose() * self.ata * pos
                - 2.0 * pos.transpose() * atb)[0]
                + btb;

            // If this epsilon dramatically increases the error, then we'll
            // assume that the previous (out-of-cell) vertex was genuine and
            // use it.
            if let Some((_, prev_pos)) =
                prev.filter(|(prev_err, _)| err > prev_err * 2.0)
            {
                return prev_pos;
            }

            // If the matrix solution is in the cell, then we assume the
            // solution is good; we _also_ stop iterating if this is the
            // last possible chance.
            let pos = cell.relative(pos, err);
            if i == EPSILONS.len() - 1 || pos.valid() {
                return pos;
            }
            prev = Some((err, pos));
        }
        unreachable!();
    }
}
