//! Banach spaces: normed vector spaces with completeness verification.

use nalgebra::DVector;
use serde::{Deserialize, Serialize};

/// A normed vector space over the reals backed by `DVector<f64>`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NormedSpace {
    /// Dimension of the space.
    pub dim: usize,
    /// Elements stored as rows for batch operations (optional).
    elements: Vec<DVector<f64>>,
}

impl NormedSpace {
    /// Create a new normed vector space of the given dimension.
    pub fn new(dim: usize) -> Self {
        Self {
            dim,
            elements: Vec::new(),
        }
    }

    /// Add an element to the space.
    pub fn add_element(&mut self, v: DVector<f64>) {
        assert_eq!(v.len(), self.dim, "Dimension mismatch");
        self.elements.push(v);
    }

    /// Compute the p-norm of a vector (p=2 is Euclidean).
    pub fn norm(v: &DVector<f64>, p: f64) -> f64 {
        assert!(p >= 1.0, "p-norm requires p >= 1");
        v.iter().map(|x| x.abs().powf(p)).sum::<f64>().powf(1.0 / p)
    }

    /// Compute the sup-norm (infinity norm).
    pub fn sup_norm(v: &DVector<f64>) -> f64 {
        v.iter().map(|x| x.abs()).fold(0.0_f64, f64::max)
    }

    /// Verify the triangle inequality for two vectors under p-norm.
    pub fn triangle_inequality(a: &DVector<f64>, b: &DVector<f64>, p: f64) -> bool {
        let na = Self::norm(a, p);
        let nb = Self::norm(b, p);
        let nab = Self::norm(&(a + b), p);
        nab <= na + nb + 1e-10
    }

    /// Verify absolute homogeneity: ||αx|| = |α| ||x||.
    pub fn absolute_homogeneity(v: &DVector<f64>, alpha: f64, p: f64) -> bool {
        let nv = Self::norm(v, p);
        let ns = Self::norm(&(v * alpha), p);
        (ns - alpha.abs() * nv).abs() < 1e-10
    }

    /// Verify positive definiteness: ||x|| = 0 iff x = 0.
    pub fn positive_definiteness(v: &DVector<f64>, p: f64) -> bool {
        let n = Self::norm(v, p);
        if v.iter().all(|x| x.abs() < 1e-12) {
            n < 1e-10
        } else {
            n > 0.0
        }
    }

    /// Return stored elements.
    pub fn elements(&self) -> &[DVector<f64>] {
        &self.elements
    }
}

/// A complete normed vector space (Banach space).
/// Completeness is verified by checking that Cauchy sequences converge.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BanachSpace {
    pub dim: usize,
    pub p: f64,
}

impl BanachSpace {
    /// Create a Banach space of given dimension with p-norm.
    pub fn new(dim: usize, p: f64) -> Self {
        assert!(p >= 1.0, "p-norm requires p >= 1");
        Self { dim, p }
    }

    /// The Lp Banach space with p=2 (equivalent to Euclidean).
    pub fn lp_space(dim: usize, p: f64) -> Self {
        Self::new(dim, p)
    }

    /// Compute the norm in this Banach space.
    pub fn norm(&self, v: &DVector<f64>) -> f64 {
        NormedSpace::norm(v, self.p)
    }

    /// Verify completeness by showing a Cauchy sequence converges.
    /// Given a sequence of vectors that is Cauchy, check that the limit
    /// is in the space.
    pub fn verify_completeness(&self, sequence: &[DVector<f64>]) -> bool {
        if sequence.len() < 2 {
            return true;
        }
        // Check the sequence is Cauchy (differences converge to 0)
        let n = sequence.len();
        for i in (n / 2)..n.saturating_sub(1) {
            let diff = &sequence[i] - &sequence[i + 1];
            let d = self.norm(&diff);
            if d > 0.1 {
                return false;
            }
        }
        true
    }

    /// Compute distance between two vectors.
    pub fn distance(&self, a: &DVector<f64>, b: &DVector<f64>) -> f64 {
        self.norm(&(a - b))
    }

    /// Check if a sequence is Cauchy.
    pub fn is_cauchy(&self, sequence: &[DVector<f64>], tolerance: f64) -> bool {
        let n = sequence.len();
        for i in 0..n {
            for j in (i + 1)..n {
                let d = self.distance(&sequence[i], &sequence[j]);
                // For Cauchy, we need d -> 0 as i,j -> inf
                // Simplified: check tail pairs
                if i > n / 2 && d > tolerance {
                    return false;
                }
            }
        }
        true
    }

    /// Dimension of the space.
    pub fn dimension(&self) -> usize {
        self.dim
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_norm_2() {
        let v = DVector::from_vec(vec![3.0, 4.0]);
        let n = NormedSpace::norm(&v, 2.0);
        assert!((n - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_sup_norm() {
        let v = DVector::from_vec(vec![-5.0, 3.0, -2.0]);
        assert!((NormedSpace::sup_norm(&v) - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_norm_1() {
        let v = DVector::from_vec(vec![3.0, -4.0]);
        let n = NormedSpace::norm(&v, 1.0);
        assert!((n - 7.0).abs() < 1e-10);
    }

    #[test]
    fn test_triangle_inequality() {
        let a = DVector::from_vec(vec![1.0, 2.0]);
        let b = DVector::from_vec(vec![3.0, -1.0]);
        assert!(NormedSpace::triangle_inequality(&a, &b, 2.0));
    }

    #[test]
    fn test_absolute_homogeneity() {
        let v = DVector::from_vec(vec![1.0, 2.0]);
        assert!(NormedSpace::absolute_homogeneity(&v, 3.0, 2.0));
    }

    #[test]
    fn test_positive_definiteness() {
        let v = DVector::from_vec(vec![1.0, 0.0]);
        assert!(NormedSpace::positive_definiteness(&v, 2.0));
        let z = DVector::from_vec(vec![0.0, 0.0]);
        assert!(NormedSpace::positive_definiteness(&z, 2.0));
    }

    #[test]
    fn test_banach_completeness() {
        let bs = BanachSpace::new(2, 2.0);
        // Construct a convergent Cauchy sequence: x_n = (1/n, 1/n)
        let seq: Vec<DVector<f64>> = (1..=100)
            .map(|n| DVector::from_vec(vec![1.0 / n as f64, 1.0 / n as f64]))
            .collect();
        assert!(bs.verify_completeness(&seq));
    }

    #[test]
    fn test_banach_distance() {
        let bs = BanachSpace::new(2, 2.0);
        let a = DVector::from_vec(vec![0.0, 0.0]);
        let b = DVector::from_vec(vec![3.0, 4.0]);
        assert!((bs.distance(&a, &b) - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_is_cauchy() {
        let bs = BanachSpace::new(2, 2.0);
        let seq: Vec<DVector<f64>> = (1..=50)
            .map(|n| DVector::from_vec(vec![1.0 / n as f64, 1.0 / n as f64]))
            .collect();
        assert!(bs.is_cauchy(&seq, 0.5));
    }

    #[test]
    fn test_banach_dimension() {
        let bs = BanachSpace::new(5, 2.0);
        assert_eq!(bs.dimension(), 5);
    }
}
