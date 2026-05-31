//! Spectral theorem: eigenvalue decomposition for self-adjoint operators.

use nalgebra::{DMatrix, DVector};
use crate::operator::BoundedLinearOperator;
use serde::{Deserialize, Serialize};

/// Result of spectral decomposition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpectralDecomposition {
    /// Eigenvalues sorted by magnitude (descending).
    pub eigenvalues: Vec<f64>,
    /// Corresponding orthonormal eigenvectors.
    pub eigenvectors: Vec<DVector<f64>>,
    /// The original operator's dimension.
    pub dimension: usize,
}

impl SpectralDecomposition {
    /// Compute spectral decomposition of a self-adjoint operator.
    pub fn decompose(operator: &BoundedLinearOperator) -> Self {
        assert!(
            operator.is_self_adjoint(1e-8),
            "Spectral decomposition requires self-adjoint operator"
        );
        let eigen = operator.matrix.clone().symmetric_eigen();
        let n = eigen.eigenvalues.len();
        let mut pairs: Vec<(f64, DVector<f64>)> = (0..n)
            .map(|i| {
                let ev: f64 = eigen.eigenvalues[i];
                let vec: DVector<f64> = eigen.eigenvectors.column(i).into();
                (ev, vec)
            })
            .collect();
        pairs.sort_by(|a, b| b.0.abs().partial_cmp(&a.0.abs()).unwrap());
        SpectralDecomposition {
            eigenvalues: pairs.iter().map(|p| p.0).collect(),
            eigenvectors: pairs.iter().map(|p| p.1.clone()).collect(),
            dimension: n,
        }
    }

    /// Reconstruct the operator from spectral decomposition: A = Σ λ_i e_i e_i^T
    pub fn reconstruct(&self) -> DMatrix<f64> {
        let n = self.dimension;
        let mut matrix = DMatrix::zeros(n, n);
        for (lambda, e) in self.eigenvalues.iter().zip(self.eigenvectors.iter()) {
            matrix += &(*lambda * e * e.transpose());
        }
        matrix
    }

    /// Verify decomposition: check that A = Σ λ_i e_i e_i^T.
    pub fn verify(&self, original: &DMatrix<f64>, tol: f64) -> bool {
        let reconstructed = self.reconstruct();
        let diff = original - &reconstructed;
        diff.iter().all(|x| x.abs() < tol)
    }

    /// Compute the spectral radius (largest |eigenvalue|).
    pub fn spectral_radius(&self) -> f64 {
        self.eigenvalues[0].abs()
    }

    /// Check if the operator is positive semi-definite.
    pub fn is_positive_semidefinite(&self) -> bool {
        self.eigenvalues.iter().all(|&x| x >= -1e-10)
    }

    /// Check if the operator is positive definite.
    pub fn is_positive_definite(&self) -> bool {
        self.eigenvalues.iter().all(|&x| x > 1e-10)
    }

    /// Compute the spectral norm (same as operator norm for self-adjoint).
    pub fn spectral_norm(&self) -> f64 {
        self.spectral_radius()
    }

    /// Compute the trace from eigenvalues.
    pub fn trace(&self) -> f64 {
        self.eigenvalues.iter().sum()
    }

    /// Compute the determinant from eigenvalues.
    pub fn determinant(&self) -> f64 {
        self.eigenvalues.iter().product()
    }

    /// Apply a function to eigenvalues (functional calculus).
    pub fn functional_calculus<F>(&self, f: F) -> DMatrix<f64>
    where
        F: Fn(f64) -> f64,
    {
        let n = self.dimension;
        let mut matrix = DMatrix::zeros(n, n);
        for (lambda, e) in self.eigenvalues.iter().zip(self.eigenvectors.iter()) {
            matrix += &(f(*lambda) * e * e.transpose());
        }
        matrix
    }

    /// Compute A^p via functional calculus.
    pub fn power(&self, p: f64) -> DMatrix<f64> {
        self.functional_calculus(|x| x.powf(p))
    }

    /// Compute exp(A) via functional calculus.
    pub fn exp(&self) -> DMatrix<f64> {
        self.functional_calculus(|x| x.exp())
    }
}

/// Compute the spectral decomposition of a self-adjoint operator.
pub fn spectral_decomposition(matrix: &DMatrix<f64>) -> SpectralDecomposition {
    let op = BoundedLinearOperator::new(matrix.clone());
    SpectralDecomposition::decompose(&op)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_matrix() -> DMatrix<f64> {
        DMatrix::from_row_slice(2, 2, &[4.0, 1.0, 1.0, 3.0])
    }

    #[test]
    fn test_spectral_decomposition() {
        let m = sample_matrix();
        let decomp = spectral_decomposition(&m);
        assert_eq!(decomp.dimension, 2);
        assert_eq!(decomp.eigenvalues.len(), 2);
    }

    #[test]
    fn test_reconstruction() {
        let m = sample_matrix();
        let decomp = spectral_decomposition(&m);
        assert!(decomp.verify(&m, 1e-8));
    }

    #[test]
    fn test_spectral_radius() {
        let m = DMatrix::from_row_slice(2, 2, &[5.0, 0.0, 0.0, 3.0]);
        let decomp = spectral_decomposition(&m);
        assert!((decomp.spectral_radius() - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_positive_definite() {
        let m = DMatrix::from_row_slice(2, 2, &[5.0, 1.0, 1.0, 3.0]);
        let decomp = spectral_decomposition(&m);
        assert!(decomp.is_positive_definite());
    }

    #[test]
    fn test_not_positive_definite() {
        let m = DMatrix::from_row_slice(2, 2, &[-1.0, 0.0, 0.0, 3.0]);
        let decomp = spectral_decomposition(&m);
        assert!(!decomp.is_positive_definite());
        assert!(!decomp.is_positive_semidefinite());
    }

    #[test]
    fn test_trace_eigenvalues() {
        let m = sample_matrix();
        let decomp = spectral_decomposition(&m);
        let trace = decomp.trace();
        assert!((trace - 7.0).abs() < 1e-8);
    }

    #[test]
    fn test_determinant_eigenvalues() {
        let m = sample_matrix();
        let decomp = spectral_decomposition(&m);
        let det = decomp.determinant();
        // det = 4*3 - 1*1 = 11
        assert!((det - 11.0).abs() < 1e-6);
    }

    #[test]
    fn test_functional_calculus_square() {
        let m = sample_matrix();
        let decomp = spectral_decomposition(&m);
        let a_squared = decomp.power(2.0);
        let expected = &m * &m;
        let diff = &a_squared - &expected;
        assert!(diff.iter().all(|x| x.abs() < 1e-8));
    }

    #[test]
    fn test_exp_functional_calculus() {
        let m = DMatrix::from_row_slice(2, 2, &[1.0, 0.0, 0.0, 2.0]);
        let decomp = spectral_decomposition(&m);
        let exp_a = decomp.exp();
        assert!((exp_a[(0, 0)] - 1.0_f64.exp()).abs() < 1e-10);
        assert!((exp_a[(1, 1)] - 2.0_f64.exp()).abs() < 1e-10);
    }

    #[test]
    fn test_spectral_norm() {
        let m = DMatrix::from_row_slice(2, 2, &[5.0, 0.0, 0.0, 3.0]);
        let decomp = spectral_decomposition(&m);
        assert!((decomp.spectral_norm() - 5.0).abs() < 1e-10);
    }
}
