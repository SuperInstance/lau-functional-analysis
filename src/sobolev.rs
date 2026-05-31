//! Sobolev spaces basics: weak derivatives, embedding theorems (discrete approximation).

use nalgebra::DVector;
use serde::{Deserialize, Serialize};

/// Discrete approximation of a Sobolev space H^k on [0,1] with n grid points.
/// Represents functions by their values at equidistant grid points.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SobolevSpace {
    /// Number of grid points.
    pub n: usize,
    /// Sobolev order k (H^k).
    pub k: usize,
    /// Grid spacing.
    pub h: f64,
}

impl SobolevSpace {
    /// Create a Sobolev space H^k on [0,1] with n grid points.
    pub fn new(n: usize, k: usize) -> Self {
        assert!(n >= 2, "Need at least 2 grid points");
        Self {
            n,
            k,
            h: 1.0 / (n - 1) as f64,
        }
    }

    /// Grid points.
    pub fn grid(&self) -> Vec<f64> {
        (0..self.n).map(|i| i as f64 * self.h).collect()
    }

    /// Compute weak derivative (finite difference approximation).
    /// Returns the discrete derivative of order `order`.
    pub fn weak_derivative(&self, f: &DVector<f64>, order: usize) -> DVector<f64> {
        assert_eq!(f.len(), self.n);
        if order == 0 {
            return f.clone();
        }
        let mut current = f.clone();
        for _ in 0..order {
            current = self.finite_difference(&current);
        }
        current
    }

    /// First-order finite difference.
    fn finite_difference(&self, f: &DVector<f64>) -> DVector<f64> {
        let n = f.len();
        if n <= 1 {
            return DVector::zeros(1);
        }
        let mut deriv = DVector::zeros(n);
        // Forward difference at boundaries, central elsewhere
        deriv[0] = (f[1] - f[0]) / self.h;
        deriv[n - 1] = (f[n - 1] - f[n - 2]) / self.h;
        for i in 1..n - 1 {
            deriv[i] = (f[i + 1] - f[i - 1]) / (2.0 * self.h);
        }
        deriv
    }

    /// Sobolev norm ||f||_{H^k} = sqrt(Σ_{j=0}^{k} ||D^j f||_{L2}^2).
    pub fn sobolev_norm(&self, f: &DVector<f64>) -> f64 {
        let mut sum = 0.0;
        for j in 0..=self.k {
            let dj = self.weak_derivative(f, j);
            let l2_sq: f64 = dj.iter().map(|x| x * x).sum::<f64>() * self.h;
            sum += l2_sq;
        }
        sum.sqrt()
    }

    /// L2 inner product: <f, g> = ∫ f·g dx ≈ Σ f_i g_i h.
    pub fn l2_inner_product(&self, f: &DVector<f64>, g: &DVector<f64>) -> f64 {
        f.iter().zip(g.iter()).map(|(a, b)| a * b).sum::<f64>() * self.h
    }

    /// L2 norm.
    pub fn l2_norm(&self, f: &DVector<f64>) -> f64 {
        self.l2_inner_product(f, f).sqrt()
    }

    /// H^0 inner product (same as L2).
    pub fn h0_inner_product(&self, f: &DVector<f64>, g: &DVector<f64>) -> f64 {
        self.l2_inner_product(f, g)
    }

    /// Check if a function is in H^1 (has square-integrable weak derivative).
    pub fn is_in_h1(&self, f: &DVector<f64>) -> bool {
        let d1 = self.weak_derivative(f, 1);
        let d1_sq: f64 = d1.iter().map(|x| x * x).sum::<f64>() * self.h;
        d1_sq.is_finite()
    }

    /// Rellich-Kondrachov embedding (discrete version):
    /// In continuous case, H^1(Ω) ⊂⊂ L^2(Ω) for bounded Ω.
    /// We verify that H^1 norm >= L2 norm.
    pub fn verify_embedding(&self, f: &DVector<f64>) -> bool {
        let h1 = self.sobolev_norm(f);
        let l2 = self.l2_norm(f);
        h1 >= l2 - 1e-10
    }

    /// Poincaré inequality (discrete):
    /// ||f - f_avg||_{L2} ≤ C ||∇f||_{L2} for f with zero average.
    /// Returns (lhs, rhs) and whether inequality holds.
    pub fn poincare_inequality(&self, f: &DVector<f64>) -> (f64, f64, bool) {
        let avg: f64 = f.iter().sum::<f64>() / self.n as f64;
        let f_centered = f - DVector::from_element(self.n, avg);
        let lhs = self.l2_norm(&f_centered);
        let grad = self.weak_derivative(f, 1);
        let rhs = self.l2_norm(&grad);
        // Poincaré constant for [0,1] is 1/π
        let c = 1.0 / std::f64::consts::PI;
        (lhs, rhs, lhs <= c * rhs + 1e-10 || rhs < 1e-10)
    }

    /// Construct a smooth function (sine) on the grid.
    pub fn smooth_function(&self, freq: f64) -> DVector<f64> {
        DVector::from_iterator(
            self.n,
            (0..self.n).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 * self.h).sin()),
        )
    }

    /// Construct a step function on the grid.
    pub fn step_function(&self, threshold: f64) -> DVector<f64> {
        DVector::from_iterator(
            self.n,
            (0..self.n).map(|i| if (i as f64 * self.h) < threshold { 1.0 } else { 0.0 }),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sobolev_creation() {
        let s = SobolevSpace::new(100, 1);
        assert_eq!(s.n, 100);
        assert_eq!(s.k, 1);
    }

    #[test]
    fn test_grid() {
        let s = SobolevSpace::new(5, 1);
        let grid = s.grid();
        assert_eq!(grid.len(), 5);
        assert!((grid[0] - 0.0).abs() < 1e-10);
        assert!((grid[4] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_weak_derivative_constant() {
        let s = SobolevSpace::new(100, 1);
        let f = DVector::from_element(100, 3.0);
        let d = s.weak_derivative(&f, 1);
        assert!(d.iter().all(|x| x.abs() < 1e-8));
    }

    #[test]
    fn test_weak_derivative_linear() {
        let s = SobolevSpace::new(100, 1);
        // f(x) = x → f'(x) = 1
        let f = DVector::from_iterator(100, (0..100).map(|i| i as f64 * s.h));
        let d = s.weak_derivative(&f, 1);
        // Interior points should be close to 1
        for i in 5..95 {
            assert!((d[i] - 1.0).abs() < 0.05, "d[{}] = {}, expected ~1.0", i, d[i]);
        }
    }

    #[test]
    fn test_sobolev_norm_smooth() {
        let s = SobolevSpace::new(100, 1);
        let f = s.smooth_function(1.0);
        let norm = s.sobolev_norm(&f);
        assert!(norm > 0.0);
    }

    #[test]
    fn test_l2_inner_product_orthogonal() {
        let s = SobolevSpace::new(100, 1);
        let f1 = s.smooth_function(1.0);
        let f2 = s.smooth_function(2.0);
        let ip = s.l2_inner_product(&f1, &f2);
        // sin(2πx) and sin(4πx) are approximately orthogonal
        assert!(ip.abs() < 0.1, "Inner product = {}, expected ~0", ip);
    }

    #[test]
    fn test_l2_norm() {
        let s = SobolevSpace::new(100, 1);
        let f = DVector::from_element(100, 1.0);
        let norm = s.l2_norm(&f);
        assert!((norm - 1.0).abs() < 0.05, "L2 norm = {}, expected ~1.0", norm);
    }

    #[test]
    fn test_is_in_h1() {
        let s = SobolevSpace::new(100, 1);
        let f = s.smooth_function(1.0);
        assert!(s.is_in_h1(&f));
    }

    #[test]
    fn test_embedding() {
        let s = SobolevSpace::new(100, 1);
        let f = s.smooth_function(1.0);
        assert!(s.verify_embedding(&f));
    }

    #[test]
    fn test_smooth_function() {
        let s = SobolevSpace::new(100, 1);
        let f = s.smooth_function(1.0);
        assert_eq!(f.len(), 100);
        assert!(f.iter().all(|x| x.abs() <= 1.0));
    }

    #[test]
    fn test_step_function() {
        let s = SobolevSpace::new(100, 1);
        let f = s.step_function(0.5);
        assert_eq!(f.len(), 100);
        assert!(f[0] == 1.0);
        assert!(f[99] == 0.0);
    }
}
