//! Compact operators: spectral theorem for compact self-adjoint operators.

use nalgebra::{DMatrix, DVector};
use crate::operator::BoundedLinearOperator;
use serde::{Deserialize, Serialize};

/// A compact operator (finite-rank approximation in finite dimensions).
/// In finite dimensions, every operator is compact, but we model the structure
/// for pedagogical purposes and for future extension to infinite dimensions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompactOperator {
    pub operator: BoundedLinearOperator,
}

impl CompactOperator {
    /// Create from a bounded linear operator.
    pub fn new(op: BoundedLinearOperator) -> Self {
        Self { operator: op }
    }

    /// Create a finite-rank operator from outer products: sum of α_i (u_i ⊗ v_i).
    pub fn finite_rank(
        coefficients: &[f64],
        u_vectors: &[DVector<f64>],
        v_vectors: &[DVector<f64>],
    ) -> Self {
        assert_eq!(coefficients.len(), u_vectors.len());
        assert_eq!(u_vectors.len(), v_vectors.len());
        let m = u_vectors[0].len();
        let n = v_vectors[0].len();
        let mut matrix = DMatrix::zeros(m, n);
        for (_i, (alpha, (u, v))) in coefficients.iter().zip(u_vectors.iter().zip(v_vectors.iter())).enumerate() {
            matrix += &(*alpha * u * v.transpose());
        }
        Self {
            operator: BoundedLinearOperator::new(matrix),
        }
    }

    /// The rank of the compact operator.
    pub fn rank(&self) -> usize {
        self.operator.rank()
    }

    /// Approximate by a rank-k operator using SVD truncation.
    pub fn approximate_rank_k(&self, k: usize) -> CompactOperator {
        let svd = self.operator.matrix.clone().svd(true, true);
        let n = self.operator.matrix.nrows();
        let m = self.operator.matrix.ncols();
        let mut matrix = DMatrix::zeros(n, m);
        for i in 0..k.min(svd.singular_values.len()) {
            let s = svd.singular_values[i];
            if let (Some(u), Some(v_t)) = (&svd.u, &svd.v_t) {
                let u_col = u.column(i);
                let v_row = v_t.row(i);
                matrix += &(s * &u_col * &v_row);
            }
        }
        CompactOperator::new(BoundedLinearOperator::new(matrix))
    }

    /// Check if self-adjoint.
    pub fn is_self_adjoint(&self, tol: f64) -> bool {
        self.operator.is_self_adjoint(tol)
    }

    /// Compute eigenvalues for self-adjoint compact operator.
    /// Returns eigenvalues sorted by absolute value (descending).
    pub fn eigenvalues(&self) -> Vec<f64> {
        let mut eigenvalues = self.operator.matrix.clone().symmetric_eigen().eigenvalues.iter().cloned().collect::<Vec<f64>>();
        eigenvalues.sort_by(|a, b| b.abs().partial_cmp(&a.abs()).unwrap());
        eigenvalues
    }

    /// Get eigenvectors corresponding to eigenvalues.
    pub fn eigen_decomposition(&self) -> (Vec<f64>, Vec<DVector<f64>>) {
        let eigen = self.operator.matrix.clone().symmetric_eigen();
        let eigenvalues: Vec<f64> = eigen.eigenvalues.iter().cloned().collect();
        let n = eigen.eigenvectors.nrows();
        let eigenvectors: Vec<DVector<f64>> = (0..n)
            .map(|i| eigen.eigenvectors.column(i).into())
            .collect();
        (eigenvalues, eigenvectors)
    }

    /// Verify spectral theorem: A = Σ λ_i (e_i ⊗ e_i) for self-adjoint compact operator.
    pub fn verify_spectral_theorem(&self, tol: f64) -> bool {
        if !self.is_self_adjoint(tol) {
            return false;
        }
        let (eigenvalues, eigenvectors) = self.eigen_decomposition();
        let n = self.operator.matrix.nrows();
        let mut reconstructed = DMatrix::zeros(n, n);
        for (i, lambda) in eigenvalues.iter().enumerate() {
            let e = &eigenvectors[i];
            reconstructed += &(*lambda * e * e.transpose());
        }
        let diff = &self.operator.matrix - &reconstructed;
        diff.iter().all(|x| x.abs() < tol)
    }

    /// Check if the operator is compact (always true in finite dim).
    pub fn is_compact(&self) -> bool {
        true
    }

    /// Apply to vector.
    pub fn apply(&self, v: &DVector<f64>) -> DVector<f64> {
        self.operator.apply(v)
    }

    /// Approximation error when truncating to rank k.
    pub fn approximation_error(&self, k: usize) -> f64 {
        let svd = self.operator.matrix.clone().svd(false, false);
        let tail: f64 = svd.singular_values.iter().skip(k).map(|s| s * s).sum();
        tail.sqrt()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compact_self_adjoint() {
        let m = DMatrix::from_row_slice(2, 2, &[3.0, 1.0, 1.0, 2.0]);
        let op = CompactOperator::new(BoundedLinearOperator::new(m));
        assert!(op.is_self_adjoint(1e-10));
    }

    #[test]
    fn test_compact_eigenvalues() {
        let m = DMatrix::from_row_slice(2, 2, &[3.0, 0.0, 0.0, 1.0]);
        let op = CompactOperator::new(BoundedLinearOperator::new(m));
        let eigs = op.eigenvalues();
        assert!((eigs[0] - 3.0).abs() < 1e-10);
        assert!((eigs[1] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_spectral_theorem() {
        let m = DMatrix::from_row_slice(2, 2, &[4.0, 1.0, 1.0, 3.0]);
        let op = CompactOperator::new(BoundedLinearOperator::new(m));
        assert!(op.verify_spectral_theorem(1e-8));
    }

    #[test]
    fn test_finite_rank() {
        let u1 = DVector::from_vec(vec![1.0, 0.0]);
        let v1 = DVector::from_vec(vec![1.0, 0.0]);
        let op = CompactOperator::finite_rank(&[2.0], &[u1], &[v1]);
        assert_eq!(op.rank(), 1);
        let v = DVector::from_vec(vec![3.0, 5.0]);
        let result = op.apply(&v);
        assert!((result[0] - 6.0).abs() < 1e-10);
        assert!((result[1] - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_rank_approximation() {
        // Full rank 2 operator
        let m = DMatrix::from_row_slice(2, 2, &[3.0, 1.0, 1.0, 2.0]);
        let op = CompactOperator::new(BoundedLinearOperator::new(m));
        let approx = op.approximate_rank_k(1);
        assert_eq!(approx.rank(), 1);
    }

    #[test]
    fn test_approximation_error() {
        let m = DMatrix::from_row_slice(3, 3, &[5.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 1.0]);
        let op = CompactOperator::new(BoundedLinearOperator::new(m));
        // Rank-2 approximation error should be 1.0
        let err = op.approximation_error(2);
        assert!((err - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_is_compact() {
        let op = CompactOperator::new(BoundedLinearOperator::identity(3));
        assert!(op.is_compact());
    }

    #[test]
    fn test_eigen_decomposition() {
        let m = DMatrix::from_row_slice(2, 2, &[2.0, 0.0, 0.0, 5.0]);
        let op = CompactOperator::new(BoundedLinearOperator::new(m));
        let (eigenvalues, eigenvectors) = op.eigen_decomposition();
        assert_eq!(eigenvalues.len(), 2);
        assert_eq!(eigenvectors.len(), 2);
    }
}
