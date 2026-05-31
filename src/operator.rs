//! Bounded linear operators: norm, composition, adjoint.

use nalgebra::{DMatrix, DVector};
use serde::{Deserialize, Serialize};

/// A bounded linear operator represented as a matrix.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BoundedLinearOperator {
    /// Matrix representation of the operator.
    pub matrix: DMatrix<f64>,
}

impl BoundedLinearOperator {
    /// Create from a matrix.
    pub fn new(matrix: DMatrix<f64>) -> Self {
        Self { matrix }
    }

    /// Create a zero operator.
    pub fn zero(m: usize, n: usize) -> Self {
        Self {
            matrix: DMatrix::zeros(m, n),
        }
    }

    /// Create an identity operator.
    pub fn identity(n: usize) -> Self {
        Self {
            matrix: DMatrix::identity(n, n),
        }
    }

    /// Apply the operator to a vector.
    pub fn apply(&self, v: &DVector<f64>) -> DVector<f64> {
        &self.matrix * v
    }

    /// Compute the operator norm (spectral norm = largest singular value).
    pub fn operator_norm(&self) -> f64 {
        // SVD: largest singular value
        let svd = self.matrix.clone().svd(false, false);
        svd.singular_values[0]
    }

    /// Compose two operators: self ∘ other.
    pub fn compose(&self, other: &BoundedLinearOperator) -> BoundedLinearOperator {
        BoundedLinearOperator::new(&self.matrix * &other.matrix)
    }

    /// Compute the adjoint (transpose for real spaces).
    pub fn adjoint(&self) -> BoundedLinearOperator {
        BoundedLinearOperator::new(self.matrix.transpose())
    }

    /// Check if self-adjoint: A = A*.
    pub fn is_self_adjoint(&self, tol: f64) -> bool {
        let adj = self.adjoint();
        let diff = &self.matrix - &adj.matrix;
        diff.iter().all(|x| x.abs() < tol)
    }

    /// Check if the operator is bounded (always true for finite-dim, but verify norm is finite).
    pub fn is_bounded(&self) -> bool {
        self.operator_norm().is_finite()
    }

    /// Boundedness constant: ||A||.
    pub fn boundedness_constant(&self) -> f64 {
        self.operator_norm()
    }

    /// Linearity check: A(αx + βy) = αA(x) + βA(y).
    pub fn verify_linearity(
        &self,
        x: &DVector<f64>,
        y: &DVector<f64>,
        alpha: f64,
        beta: f64,
    ) -> bool {
        let lhs = self.apply(&(x * alpha + y * beta));
        let rhs = self.apply(x) * alpha + self.apply(y) * beta;
        let diff = &lhs - &rhs;
        diff.iter().all(|v| v.abs() < 1e-10)
    }

    /// Compute kernel (null space) dimension.
    pub fn nullity(&self) -> usize {
        let svd = self.matrix.clone().svd(true, true);
        svd.singular_values.iter().filter(|s| **s < 1e-10).count()
    }

    /// Compute rank.
    pub fn rank(&self) -> usize {
        let svd = self.matrix.clone().svd(true, true);
        svd.singular_values.iter().filter(|s| **s >= 1e-10).count()
    }

    /// Trace of the operator.
    pub fn trace(&self) -> f64 {
        self.matrix.trace()
    }

    /// Frobenius norm.
    pub fn frobenius_norm(&self) -> f64 {
        self.matrix.norm()
    }

    /// Number of rows.
    pub fn nrows(&self) -> usize {
        self.matrix.nrows()
    }

    /// Number of columns.
    pub fn ncols(&self) -> usize {
        self.matrix.ncols()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_identity_operator() {
        let id = BoundedLinearOperator::identity(3);
        let v = DVector::from_vec(vec![1.0, 2.0, 3.0]);
        let result = id.apply(&v);
        assert!((result - v).norm() < 1e-10);
    }

    #[test]
    fn test_zero_operator() {
        let zero = BoundedLinearOperator::zero(2, 3);
        let v = DVector::from_vec(vec![1.0, 2.0, 3.0]);
        let result = zero.apply(&v);
        assert!(result.iter().all(|x| x.abs() < 1e-10));
    }

    #[test]
    fn test_operator_norm() {
        // 2x2 matrix with known singular values
        let m = DMatrix::from_row_slice(2, 2, &[3.0, 0.0, 0.0, 4.0]);
        let op = BoundedLinearOperator::new(m);
        let norm = op.operator_norm();
        assert!((norm - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_composition() {
        let a = BoundedLinearOperator::new(DMatrix::from_row_slice(2, 2, &[2.0, 0.0, 0.0, 3.0]));
        let b = BoundedLinearOperator::new(DMatrix::from_row_slice(2, 2, &[1.0, 1.0, 0.0, 1.0]));
        let comp = a.compose(&b);
        let v = DVector::from_vec(vec![1.0, 1.0]);
        let result = comp.apply(&v);
        // a(b(v)) = a([2,1]) = [4, 3]
        assert!((result[0] - 4.0).abs() < 1e-10);
        assert!((result[1] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_adjoint() {
        let m = DMatrix::from_row_slice(2, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let op = BoundedLinearOperator::new(m);
        let adj = op.adjoint();
        assert_eq!(adj.matrix[(0, 0)], 1.0);
        assert_eq!(adj.matrix[(1, 0)], 2.0);
        assert_eq!(adj.matrix[(0, 1)], 4.0);
    }

    #[test]
    fn test_self_adjoint() {
        let m = DMatrix::from_row_slice(2, 2, &[2.0, 1.0, 1.0, 3.0]);
        let op = BoundedLinearOperator::new(m);
        assert!(op.is_self_adjoint(1e-10));
    }

    #[test]
    fn test_not_self_adjoint() {
        let m = DMatrix::from_row_slice(2, 2, &[1.0, 2.0, 3.0, 4.0]);
        let op = BoundedLinearOperator::new(m);
        assert!(!op.is_self_adjoint(1e-10));
    }

    #[test]
    fn test_linearity() {
        let m = DMatrix::from_row_slice(2, 2, &[1.0, 2.0, 3.0, 4.0]);
        let op = BoundedLinearOperator::new(m);
        let x = DVector::from_vec(vec![1.0, 0.0]);
        let y = DVector::from_vec(vec![0.0, 1.0]);
        assert!(op.verify_linearity(&x, &y, 2.0, 3.0));
    }

    #[test]
    fn test_is_bounded() {
        let op = BoundedLinearOperator::identity(5);
        assert!(op.is_bounded());
    }

    #[test]
    fn test_rank() {
        let m = DMatrix::from_row_slice(3, 3, &[1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0]);
        let op = BoundedLinearOperator::new(m);
        assert_eq!(op.rank(), 2);
    }

    #[test]
    fn test_nullity() {
        let m = DMatrix::from_row_slice(3, 3, &[1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0]);
        let op = BoundedLinearOperator::new(m);
        assert_eq!(op.nullity(), 1);
    }

    #[test]
    fn test_trace() {
        let m = DMatrix::from_row_slice(2, 2, &[1.0, 2.0, 3.0, 4.0]);
        let op = BoundedLinearOperator::new(m);
        assert!((op.trace() - 5.0).abs() < 1e-10);
    }
}
