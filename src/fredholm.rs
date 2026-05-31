//! Fredholm theory: Fredholm alternative, index.

use nalgebra::{DMatrix, DVector};
use crate::operator::BoundedLinearOperator;
use serde::{Deserialize, Serialize};

/// Fredholm operator properties and the Fredholm alternative.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FredholmAnalysis {
    /// The operator being analyzed.
    pub operator: BoundedLinearOperator,
}

impl FredholmAnalysis {
    /// Create a new Fredholm analysis for the given operator.
    pub fn new(op: BoundedLinearOperator) -> Self {
        Self { operator: op }
    }

    /// Compute the Fredholm index: index(A) = dim(ker(A)) - dim(coker(A)).
    /// For a matrix A: m×n, index = nullity(A) - (m - rank(A)).
    pub fn index(&self) -> i64 {
        let nullity = self.operator.nullity() as i64;
        let m = self.operator.nrows() as i64;
        let rank = self.operator.rank() as i64;
        let cokernel_dim = m - rank;
        nullity - cokernel_dim
    }

    /// Check if the operator is Fredholm (has finite index).
    /// In finite dimensions, all operators are Fredholm.
    pub fn is_fredholm(&self) -> bool {
        true
    }

    /// Check if the operator is Fredholm of index 0.
    pub fn is_index_zero(&self) -> bool {
        self.index() == 0
    }

    /// The Fredholm alternative for a compact operator K:
    /// Either (I - K)x = y has a unique solution for every y,
    /// or (I - K)x = 0 has a nontrivial solution.
    /// In finite dimensions, this checks if (I - A) is invertible.
    pub fn fredholm_alternative(&self) -> FredholmAlternative {
        let n = self.operator.nrows();
        assert_eq!(n, self.operator.ncols(), "Fredholm alternative requires square operator");
        let identity = DMatrix::identity(n, n);
        let i_minus_a = &identity - &self.operator.matrix;
        let i_minus_a_op = BoundedLinearOperator::new(i_minus_a);
        let nullity = i_minus_a_op.nullity();
        if nullity == 0 {
            FredholmAlternative::UniqueSolution
        } else {
            FredholmAlternative::NontrivialKernel { nullity }
        }
    }

    /// Solve (I - A)x = y if possible.
    pub fn solve_fredholm_equation(&self, y: &DVector<f64>) -> Option<DVector<f64>> {
        let n = self.operator.nrows();
        let identity = DMatrix::identity(n, n);
        let i_minus_a = &identity - &self.operator.matrix;
        let svd = i_minus_a.svd(true, true);
        match svd.solve(y, 1e-10) {
            Ok(x) => Some(x),
            Err(_) => None,
        }
    }

    /// Rank of the operator.
    pub fn rank(&self) -> usize {
        self.operator.rank()
    }

    /// Nullity (dimension of kernel).
    pub fn nullity(&self) -> usize {
        self.operator.nullity()
    }

    /// Cokernel dimension: m - rank(A) for m×n matrix.
    pub fn cokernel_dimension(&self) -> usize {
        self.operator.nrows() - self.operator.rank()
    }
}

/// Result of the Fredholm alternative.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FredholmAlternative {
    /// (I - K) is invertible, unique solution for every y.
    UniqueSolution,
    /// (I - K) has nontrivial kernel.
    NontrivialKernel { nullity: usize },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fredholm_index_identity() {
        let id = BoundedLinearOperator::identity(3);
        let analysis = FredholmAnalysis::new(id);
        assert_eq!(analysis.index(), 0);
    }

    #[test]
    fn test_fredholm_index_zero_map() {
        let zero = BoundedLinearOperator::zero(3, 3);
        let analysis = FredholmAnalysis::new(zero);
        // nullity = 3, cokernel = 3 - 0 = 3, index = 3 - 3 = 0
        assert_eq!(analysis.index(), 0);
    }

    #[test]
    fn test_fredholm_index_non_square() {
        // 3x2 matrix, rank 2
        let m = DMatrix::from_row_slice(3, 2, &[1.0, 0.0, 0.0, 1.0, 0.0, 0.0]);
        let op = BoundedLinearOperator::new(m);
        let analysis = FredholmAnalysis::new(op);
        // nullity = 0, cokernel = 3 - 2 = 1, index = -1
        assert_eq!(analysis.index(), -1);
    }

    #[test]
    fn test_is_fredholm() {
        let id = BoundedLinearOperator::identity(3);
        let analysis = FredholmAnalysis::new(id);
        assert!(analysis.is_fredholm());
    }

    #[test]
    fn test_fredholm_alternative_unique() {
        // Small operator: I - K is invertible
        let m = DMatrix::from_row_slice(2, 2, &[0.1, 0.0, 0.0, 0.2]);
        let op = BoundedLinearOperator::new(m);
        let analysis = FredholmAnalysis::new(op);
        match analysis.fredholm_alternative() {
            FredholmAlternative::UniqueSolution => (),
            _ => panic!("Expected unique solution"),
        }
    }

    #[test]
    fn test_fredholm_alternative_kernel() {
        // K = I, so I - K = 0 matrix (full kernel)
        let op = BoundedLinearOperator::identity(2);
        let analysis = FredholmAnalysis::new(op);
        match analysis.fredholm_alternative() {
            FredholmAlternative::NontrivialKernel { nullity } => {
                assert_eq!(nullity, 2);
            }
            FredholmAlternative::UniqueSolution => panic!("Expected nontrivial kernel"),
        }
    }

    #[test]
    fn test_solve_fredholm() {
        let m = DMatrix::from_row_slice(2, 2, &[0.1, 0.0, 0.0, 0.2]);
        let op = BoundedLinearOperator::new(m.clone());
        let analysis = FredholmAnalysis::new(op);
        let y = DVector::from_vec(vec![1.0, 1.0]);
        let x = analysis.solve_fredholm_equation(&y).expect("Should have solution");
        // Verify: (I - K)x = y
        let identity = DMatrix::identity(2, 2);
        let i_minus_k = &identity - &m;
        let check = &i_minus_k * &x;
        assert!((check - y).norm() < 1e-8);
    }

    #[test]
    fn test_rank() {
        let m = DMatrix::from_row_slice(2, 2, &[1.0, 0.0, 0.0, 0.0]);
        let op = BoundedLinearOperator::new(m);
        let analysis = FredholmAnalysis::new(op);
        assert_eq!(analysis.rank(), 1);
    }

    #[test]
    fn test_nullity() {
        let m = DMatrix::from_row_slice(2, 2, &[1.0, 0.0, 0.0, 0.0]);
        let op = BoundedLinearOperator::new(m);
        let analysis = FredholmAnalysis::new(op);
        assert_eq!(analysis.nullity(), 1);
    }

    #[test]
    fn test_cokernel_dimension() {
        let m = DMatrix::from_row_slice(3, 2, &[1.0, 0.0, 0.0, 1.0, 0.0, 0.0]);
        let op = BoundedLinearOperator::new(m);
        let analysis = FredholmAnalysis::new(op);
        assert_eq!(analysis.cokernel_dimension(), 1);
    }

    #[test]
    fn test_index_zero() {
        let id = BoundedLinearOperator::identity(4);
        let analysis = FredholmAnalysis::new(id);
        assert!(analysis.is_index_zero());
    }
}
