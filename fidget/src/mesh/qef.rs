use super::cell::CellVertex;

/// Solver for a quadratic error function to position a vertex within a cell
#[derive(Copy, Clone, Debug, Default)]
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

impl std::ops::AddAssign for QuadraticErrorSolver {
    fn add_assign(&mut self, rhs: Self) {
        self.ata += rhs.ata;
        self.atb += rhs.atb;
        self.btb += rhs.btb;
        self.mass_point += rhs.mass_point;
    }
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

    #[cfg(test)]
    pub fn mass_point(&self) -> nalgebra::Vector4<f32> {
        self.mass_point
    }

    /// Adds a new intersection to the QEF
    ///
    /// `pos` is the position of the intersection and is accumulated in the mass
    /// point.  `grad` is the gradient at the surface, and is normalized in this
    /// function.
    pub fn add_intersection(
        &mut self,
        pos: nalgebra::Vector3<f32>,
        grad: nalgebra::Vector4<f32>,
    ) {
        // TODO: correct for non-zero distance value in grad.w
        self.mass_point += nalgebra::Vector4::new(pos.x, pos.y, pos.z, 1.0);
        let norm = grad.xyz().normalize();
        self.ata += norm * norm.transpose();
        self.atb += norm * norm.dot(&pos);
        self.btb += norm.dot(&pos).powi(2);
    }

    /// Solve the given QEF, minimizing towards the mass point
    ///
    /// Returns a vertex localized within the given cell, and adjusts the solver
    /// to increase the likelyhood that the vertex is bounded in the cell.
    ///
    /// Also returns the QEF error as the second item in the tuple
    pub fn solve(&self) -> (CellVertex, f32) {
        // This gets a little tricky; see
        // https://www.mattkeeter.com/projects/qef for a walkthrough of QEF math
        // and references to primary sources.
        let center = self.mass_point.xyz() / self.mass_point.w as f32;
        let atb = self.atb - self.ata * center;

        let svd = nalgebra::linalg::SVD::new(self.ata, true, true);
        // "Dual Contouring: The Secret Sauce" recomments a threshold of 0.1
        // when using normalized gradients, but I've found that fails on
        // things like the cone model.  Instead, we'll be a little more
        // clever: we'll pick the smallest epsilon that keeps the feature in
        // the cell without dramatically increasing QEF error.
        let mut prev = None;
        for i in 0..4 {
            let epsilon = if i == 3 {
                std::f32::INFINITY
            } else {
                svd.singular_values[2 - i]
            };
            let sol = svd.solve(&atb, epsilon);
            let pos = sol.map(|c| c + center).unwrap_or(center);
            // We'll clamp the error to a small > 0 value for ease of comparison
            let err = ((pos.transpose() * self.ata * pos
                - 2.0 * pos.transpose() * self.atb)[0]
                + self.btb)
                .max(1e-6);

            // If this epsilon dramatically increases the error, then we'll
            // assume that the previous (out-of-cell) vertex was genuine and
            // use it.
            if let Some(p) = prev.filter(|(_, prev_err)| err > prev_err * 2.0) {
                return p;
            }

            // If the matrix solution is in the cell, then we assume the
            // solution is good; we _also_ stop iterating if this is the
            // last possible chance.
            let pos = CellVertex { pos };
            if i == 3 {
                return (pos, err);
            }
            prev = Some((pos, err));
        }
        unreachable!();
    }
}
