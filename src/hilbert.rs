//! Hilbert spaces: inner product spaces with completeness.

use nalgebra::DVector;
use serde::{Deserialize, Serialize};

/// A finite-dimensional Hilbert space with the standard inner product.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HilbertSpace {
    pub dim: usize,
}

impl HilbertSpace {
    /// Create a Hilbert space of given dimension.
    pub fn new(dim: usize) -> Self {
        Self { dim }
    }

    /// Standard inner product: <x, y> = x^T y
    pub fn inner_product(a: &DVector<f64>, b: &DVector<f64>) -> f64 {
        a.dot(b)
    }

    /// Norm induced by inner product.
    pub fn norm(v: &DVector<f64>) -> f64 {
        v.norm()
    }

    /// Distance between two vectors.
    pub fn distance(a: &DVector<f64>, b: &DVector<f64>) -> f64 {
        (a - b).norm()
    }

    /// Check if two vectors are orthogonal.
    pub fn are_orthogonal(a: &DVector<f64>, b: &DVector<f64>, tol: f64) -> bool {
        Self::inner_product(a, b).abs() < tol
    }

    /// Gram-Schmidt orthogonalization of a set of vectors.
    pub fn gram_schmidt(vectors: &[DVector<f64>]) -> Vec<DVector<f64>> {
        let mut orthogonal = Vec::new();
        for v in vectors {
            let mut u = v.clone();
            for prev in &orthogonal {
                let proj = Self::inner_product(&u, prev) / Self::inner_product(prev, prev);
                u -= &(prev * proj);
            }
            let n = u.norm();
            if n > 1e-12 {
                orthogonal.push(u / n);
            }
        }
        orthogonal
    }

    /// Orthogonal projection of v onto subspace spanned by basis.
    pub fn project(v: &DVector<f64>, basis: &[DVector<f64>]) -> DVector<f64> {
        let mut proj = DVector::zeros(v.len());
        for b in basis {
            let coeff = Self::inner_product(v, b) / Self::inner_product(b, b);
            proj += &(b * coeff);
        }
        proj
    }

    /// Orthogonal projection onto a unit-norm orthonormal basis.
    pub fn project_orthonormal(v: &DVector<f64>, basis: &[DVector<f64>]) -> DVector<f64> {
        let mut proj = DVector::zeros(v.len());
        for e in basis {
            proj += &(e * Self::inner_product(v, e));
        }
        proj
    }

    /// Compute orthonormal basis via Gram-Schmidt from standard-like vectors.
    pub fn standard_orthonormal_basis(dim: usize) -> Vec<DVector<f64>> {
        let mut vectors = Vec::new();
        for i in 0..dim {
            let mut v = DVector::zeros(dim);
            v[i] = 1.0;
            vectors.push(v);
        }
        vectors // Already orthonormal
    }

    /// Projection theorem: decompose v into projection onto W and v - proj (in W⊥).
    /// Returns (projection, orthogonal_complement).
    pub fn projection_theorem(
        v: &DVector<f64>,
        basis: &[DVector<f64>],
    ) -> (DVector<f64>, DVector<f64>) {
        let proj = Self::project_orthonormal(v, basis);
        let perp = v - &proj;
        (proj, perp)
    }

    /// Verify the projection theorem: proj ⊥ (v - proj).
    pub fn verify_projection_orthogonality(
        v: &DVector<f64>,
        basis: &[DVector<f64>],
    ) -> bool {
        let (proj, perp) = Self::projection_theorem(v, basis);
        Self::are_orthogonal(&proj, &perp, 1e-10)
    }

    /// Parseval's identity: ||v||² = Σ|<v, e_k>|² for orthonormal basis.
    pub fn parseval_identity(v: &DVector<f64>, basis: &[DVector<f64>]) -> bool {
        let norm_sq = v.norm().powi(2);
        let sum_sq: f64 = basis.iter().map(|e| Self::inner_product(v, e).powi(2)).sum();
        (norm_sq - sum_sq).abs() < 1e-10
    }

    /// Dimension.
    pub fn dimension(&self) -> usize {
        self.dim
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_inner_product() {
        let a = DVector::from_vec(vec![1.0, 2.0, 3.0]);
        let b = DVector::from_vec(vec![4.0, 5.0, 6.0]);
        // 1*4 + 2*5 + 3*6 = 32
        assert!((HilbertSpace::inner_product(&a, &b) - 32.0).abs() < 1e-10);
    }

    #[test]
    fn test_orthogonality() {
        let a = DVector::from_vec(vec![1.0, 0.0]);
        let b = DVector::from_vec(vec![0.0, 1.0]);
        assert!(HilbertSpace::are_orthogonal(&a, &b, 1e-10));
    }

    #[test]
    fn test_gram_schmidt() {
        let v1 = DVector::from_vec(vec![1.0, 1.0]);
        let v2 = DVector::from_vec(vec![1.0, 0.0]);
        let on = HilbertSpace::gram_schmidt(&[v1, v2]);
        assert_eq!(on.len(), 2);
        assert!(HilbertSpace::are_orthogonal(&on[0], &on[1], 1e-10));
        assert!((on[0].norm() - 1.0).abs() < 1e-10);
        assert!((on[1].norm() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_projection() {
        let v = DVector::from_vec(vec![3.0, 4.0]);
        let e1 = DVector::from_vec(vec![1.0, 0.0]);
        let proj = HilbertSpace::project(&v, &[e1]);
        assert!((proj[0] - 3.0).abs() < 1e-10);
        assert!((proj[1] - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_projection_theorem() {
        let v = DVector::from_vec(vec![3.0, 4.0, 5.0]);
        let e1 = DVector::from_vec(vec![1.0, 0.0, 0.0]);
        let e2 = DVector::from_vec(vec![0.0, 1.0, 0.0]);
        let (proj, perp) = HilbertSpace::projection_theorem(&v, &[e1, e2]);
        assert!((proj[0] - 3.0).abs() < 1e-10);
        assert!((proj[1] - 4.0).abs() < 1e-10);
        assert!((proj[2] - 0.0).abs() < 1e-10);
        assert!(HilbertSpace::are_orthogonal(&proj, &perp, 1e-10));
    }

    #[test]
    fn test_verify_projection_orthogonality() {
        let v = DVector::from_vec(vec![1.0, 2.0, 3.0]);
        let basis = HilbertSpace::standard_orthonormal_basis(3);
        // Project onto first 2 dimensions
        assert!(HilbertSpace::verify_projection_orthogonality(&v, &basis[..2]));
    }

    #[test]
    fn test_parseval_identity() {
        let v = DVector::from_vec(vec![1.0, 2.0, 3.0]);
        let basis = HilbertSpace::standard_orthonormal_basis(3);
        assert!(HilbertSpace::parseval_identity(&v, &basis));
    }

    #[test]
    fn test_standard_basis_orthonormal() {
        let basis = HilbertSpace::standard_orthonormal_basis(3);
        for i in 0..3 {
            assert!((basis[i].norm() - 1.0).abs() < 1e-10);
            for j in (i + 1)..3 {
                assert!(HilbertSpace::are_orthogonal(&basis[i], &basis[j], 1e-10));
            }
        }
    }

    #[test]
    fn test_hilbert_dimension() {
        let h = HilbertSpace::new(10);
        assert_eq!(h.dimension(), 10);
    }
}
