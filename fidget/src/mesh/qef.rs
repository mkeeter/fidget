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
    /// to increase the likelihood that the vertex is bounded in the cell.
    ///
    /// Also returns the QEF error as the second item in the tuple
    pub fn solve(&self) -> (CellVertex, f32) {
        // This gets a little tricky; see
        // https://www.mattkeeter.com/projects/qef for a walkthrough of QEF math
        // and references to primary sources.
        let center = self.mass_point.xyz() / self.mass_point.w;
        let atb = self.atb - self.ata * center;

        let svd = nalgebra::linalg::SVD::new(self.ata, true, true);

        // nalgebra doesn't always actually order singular values (?!?)
        // https://github.com/dimforge/nalgebra/issues/1215
        let mut singular_values =
            svd.singular_values.data.0[0].map(ordered_float::OrderedFloat);
        singular_values.sort();
        singular_values.reverse();
        let singular_values = singular_values.map(|o| o.0);

        // Skip any eigenvalues that are small relative to the maximum
        // eigenvalue.  Without this filter, we can see failures in near-planar
        // situations.  This is very much a tuned value: our cone test needs all
        // three eigenvalues in [1.5633028, 1.430821, 0.0058764853] to succeed,
        // while the bear needs to use a rank-2 solver for [2.87, 0.13, 5.64e-7]
        const EIGENVALUE_CUTOFF_RELATIVE: f32 = 1e-3;
        let cutoff = singular_values[0].abs() * EIGENVALUE_CUTOFF_RELATIVE;
        let first_invalid_eigenvalue = (0..3)
            .find(|i| singular_values[*i].abs() < cutoff)
            .unwrap_or(3);

        // Brief guide to the value of `first_invalid_eigenvalue`:
        // 0 => all eigenvalues are invalid, use the center point
        // 1 => the first eigenvalue is valid, this must be planar
        // 2 => the first two eigenvalues are valid, this is a planar or an edge
        // 3 => all eigenvalues are valid, this is a planar, edge, or corner

        // "Dual Contouring: The Secret Sauce" recommends a threshold of 0.1
        // when using normalized gradients, but I've found that fails on
        // things like the cone model.  Instead, we'll be a little more
        // clever: we'll pick the smallest epsilon that keeps the feature in
        // the cell without dramatically increasing QEF error.
        let mut prev: Option<(CellVertex, f32)> = None;
        for i in 0..first_invalid_eigenvalue {
            let rank = first_invalid_eigenvalue - i;
            let epsilon = singular_values.get(rank).cloned().unwrap_or(0.0);
            let sol = svd.solve(&atb, epsilon);
            let pos = sol.map(|c| c + center).unwrap_or(center);
            // We'll clamp the error to a small > 0 value for ease of comparison
            let err = ((pos.transpose() * self.ata * pos
                - 2.0 * pos.transpose() * self.atb)[0]
                + self.btb)
                .max(1e-6);

            // If this epsilon dramatically increases the error, then we'll
            // assume that the previous (possibly out-of-cell) vertex was
            // genuine and use it.
            if let Some(p) = prev.filter(|(_, prev_err)| err > prev_err * 2.0) {
                return p;
            }

            prev = Some((CellVertex { pos }, err));
        }

        prev.unwrap()
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use nalgebra::{Vector3, Vector4};

    #[test]
    fn qef_rank2() {
        let mut q = QuadraticErrorSolver::new();
        q.add_intersection(
            Vector3::new(-0.5, -0.75, -0.75),
            Vector4::new(0.24, 0.12, 0.0, 0.0),
        );
        q.add_intersection(
            Vector3::new(-0.75, -1.0, -0.6),
            Vector4::new(0.0, 0.0, 0.31, 0.0),
        );
        q.add_intersection(
            Vector3::new(-0.50, -1.0, -0.6),
            Vector4::new(0.0, 0.0, 0.31, 0.0),
        );
        let (_out, err) = q.solve();
        assert_eq!(err, 1e-6);
    }

    #[test]
    fn qef_near_planar() {
        let mut q = QuadraticErrorSolver::new();
        q.add_intersection(
            Vector3::new(-0.5, -0.25, 0.4999981),
            Vector4::new(-0.66666776, -0.33333388, 0.66666526, -1.2516975e-6),
        );
        q.add_intersection(
            Vector3::new(-0.5, -0.25, 0.50),
            Vector4::new(-0.6666667, -0.33333334, 0.6666667, 0.0),
        );
        q.add_intersection(
            Vector3::new(-0.5, -0.25, 0.50),
            Vector4::new(-0.6666667, -0.33333334, 0.6666667, 0.0),
        );
        let (out, err) = q.solve();
        assert_eq!(err, 1e-6);
        let expected = Vector3::new(-0.5, -0.25, 0.5);
        assert!(
            (out.pos - expected).norm() < 1e-3,
            "expected {expected:?}, got {:?}",
            out.pos
        );
    }
}
