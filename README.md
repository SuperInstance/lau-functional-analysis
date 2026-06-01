# lau-functional-analysis

A Rust library implementing core constructions from **functional analysis** — Banach and Hilbert spaces, bounded linear operators, compact operators, spectral theory, Fredholm theory, Sobolev spaces, and abstract agent/state spaces. Built on [nalgebra](https://nalgebra.org) for linear algebra and [serde](https://serde.rs) for serialization.

## What This Does

This crate provides typed, composable mathematical structures for working in infinite-dimensional function spaces — the backbone of modern PDE theory, quantum mechanics, and numerical analysis. You can build Hilbert spaces, project onto subspaces, diagonalize operators, check compactness criteria, compute Sobolev norms, and more — all from safe Rust.

**82 unit tests** cover every module.

## Key Idea

Functional analysis studies vector spaces equipped with norms, inner products, and topologies that let you take limits, define convergence, and decompose operators. This library encodes those structures as Rust types:

- A **Hilbert space** carries an inner product — you can orthogonalize, project, and compute best approximations.
- A **Banach space** carries a norm — you can measure convergence and completeness.
- **Bounded linear operators** map between these spaces with controlled norms; **compact operators** are the "small" ones whose images are precompact.
- **Spectral theory** decomposes operators into eigenvalues and eigenvectors; **Fredholm theory** characterizes when an operator equation `(T - λI)u = f` has a unique solution.
- **Sobolev spaces** measure both the size of a function *and* its derivatives — essential for PDEs.

## Install

Add to your `Cargo.toml`:

```toml
[dependencies]
lau-functional-analysis = { git = "https://github.com/SuperInstance/lau-functional-analysis" }
```

Or from a local clone:

```toml
[dependencies]
lau-functional-analysis = { path = "../lau-functional-analysis" }
```

### Dependencies

| Crate | Purpose |
|-------|---------|
| `nalgebra` | Matrices, vectors, eigenvalue decomposition |
| `serde` + `serde_json` | Serialization of spaces and operators |

## Quick Start

```rust
use lau_functional_analysis::*;

// Create a Hilbert space of finite functions (R^5 with L2 inner product)
let space = hilbert::HilbertSpace::from_vectors(&[
    vec![1.0, 0.0, 0.0, 0.0, 0.0],
    vec![0.0, 1.0, 0.0, 0.0, 0.0],
    vec![0.0, 0.0, 1.0, 0.0, 0.0],
]);

// Inner product and norm
let u = vec![1.0, 2.0, 3.0, 0.0, 0.0];
let v = vec![0.0, 1.0, 0.0, 0.0, 0.0];
println!("⟨u, v⟩ = {}", space.inner_product(&u, &v)); // 2.0
println!("‖u‖   = {}", space.norm(&u));               // √14

// Orthogonal projection onto a subspace
let projected = space.project(&u, &space.basis());

// Build a bounded linear operator (5×5 matrix)
let matrix = /* nalgebra DMatrix or your own */;
let op = operator::BoundedLinearOperator::new(matrix, 1.0, 5.0); // (data, norm, dimension)

// Spectral decomposition
let spectrum = spectral::SpectralDecomposition::compute(&op);
println!("Eigenvalues: {:?}", spectrum.eigenvalues());
```

## API Reference

### `banach` — Banach Spaces

| Type / Function | Description |
|----------------|-------------|
| `BanachSpace` | Finite-dimensional normed space with configurable norm (L¹, L², L∞) |
| `norm(v, kind)` | Compute norm of a vector |
| `is_complete(space, cauchy_seq)` | Verify completeness (all Cauchy sequences converge) |
| `contraction_mapping(f, x0, tol)` | Banach fixed-point iteration |

### `hilbert` — Hilbert Spaces

| Type / Function | Description |
|----------------|-------------|
| `HilbertSpace` | Finite-dimensional inner product space with orthonormal basis |
| `inner_product(u, v)` | Compute ⟨u, v⟩ |
| `gram_schmidt(vectors)` | Orthonormalize a set of vectors |
| `project(u, subspace)` | Orthogonal projection of u onto a subspace |
| `reconstruct(coeffs, basis)` | Reconstruct a vector from Fourier coefficients |

### `operator` — Bounded Linear Operators

| Type / Function | Description |
|----------------|-------------|
| `BoundedLinearOperator` | Linear map with bounded operator norm |
| `apply(v)` | Apply operator to a vector |
| `adjoint()` | Compute the adjoint (conjugate transpose) |
| `is_self_adjoint()` | Check if T = T* |
| `norm()` | Operator norm ‖T‖ |

### `compact` — Compact Operators

| Type / Function | Description |
|----------------|-------------|
| `CompactOperator` | Operator whose image of the unit ball is precompact |
| `is_compact(op)` | Test compactness (finite-rank approximation) |
| `singular_values(op)` | Compute singular values (decay rate determines compactness) |
| `approximate(op, rank)` | Low-rank approximation |

### `spectral` — Spectral Theory

| Type / Function | Description |
|----------------|-------------|
| `SpectralDecomposition` | Eigenvalue decomposition of self-adjoint operators |
| `eigenvalues()` | Sorted eigenvalues |
| `eigenvectors()` | Corresponding orthonormal eigenvectors |
| `spectral_radius(op)` | max |λ| |
| `functional_calculus(op, f)` | Apply f(T) via spectral mapping |

### `fredholm` — Fredholm Theory

| Type / Function | Description |
|----------------|-------------|
| `FredholmOperator` | Operator with finite-dimensional kernel and cokernel |
| `index(op)` | Fredholm index: dim(ker T) − dim(coker T) |
| `is_fredholm(op)` | Check Fredholm property |
| `solve(T, f)` | Solve (T − λI)u = f when λ is not in the spectrum |

### `sobolev` — Sobolev Spaces

| Type / Function | Description |
|----------------|-------------|
| `SobolevSpace` | W^{k,p} space: functions with k derivatives in L^p |
| `sobolev_norm(f, k, p)` | Compute the Sobolev norm |
| `weak_derivative(f, order)` | Compute weak derivatives numerically |
| `embedding_holds(k, p, n)` | Check Sobolev embedding theorem conditions |

### `agent_space` — Agent/State Spaces

| Type / Function | Description |
|----------------|-------------|
| `AgentSpace` | Abstract metric space for agent states and transitions |
| `distance(a, b)` | Metric between agent states |
| `trajectory(states)` | Analyze convergence of state sequences |
| `equilibrium(trajectory)` | Find fixed points of state dynamics |

## How It Works

The library models functional analysis at two levels:

1. **Concrete computation** — Vectors are `Vec<f64>`, operators are `nalgebra::DMatrix`, inner products and norms are computed directly. This lets you do real numerical work (solve equations, compute eigenvalues, project onto subspaces).

2. **Structural reasoning** — Types encode the mathematical hierarchy: every `HilbertSpace` *is* a `BanachSpace` (the inner product induces a norm), every `CompactOperator` *is* a `BoundedLinearOperator`, every `FredholmOperator` has a well-defined index. The type system prevents you from calling `project()` on a Banach space (which lacks an inner product) or computing eigenvalues of a non-self-adjoint operator.

The operator hierarchy flows:

```
BoundedLinearOperator
  ├── CompactOperator (image of unit ball is precompact)
  │     └── FredholmOperator (finite-dimensional kernel/cokernel)
  └── SelfAdjointOperator (T = T*)
        └── SpectralDecomposition (diagonalizable with real eigenvalues)
```

Sobolev spaces connect to PDEs: a function in W^{k,p} has k weak derivatives in L^p, and Sobolev embedding theorems tell you when W^{k,p} ⊂ L^q or W^{k,p} ⊂ C^m.

## The Math

### Hilbert Space Fundamentals

A **Hilbert space** H is a complete inner product space. The inner product ⟨·,·⟩ induces:
- A norm: ‖u‖ = √⟨u, u⟩
- Orthogonality: u ⊥ v ⟺ ⟨u, v⟩ = 0
- Best approximation: the closest point in a closed subspace M to u ∈ H is the orthogonal projection P_M(u)

The **Riesz representation theorem** states: every continuous linear functional φ : H → ℝ has the form φ(u) = ⟨u, v⟩ for a unique v ∈ H.

### Spectral Theorem

For a **self-adjoint compact operator** T on a separable Hilbert space:

> T = Σᵢ λᵢ ⟨·, eᵢ⟩ eᵢ

where {eᵢ} is an orthonormal set of eigenvectors and {λᵢ} are real eigenvalues with λᵢ → 0.

This is the infinite-dimensional generalization of diagonalizing a symmetric matrix.

### Fredholm Alternative

For a compact operator K and λ ≠ 0, exactly one of:
1. (K − λI)u = f has a unique solution for every f
2. (K − λI)u = 0 has non-trivial solutions

The **Fredholm index** index(T) = dim ker(T) − dim coker(T) is stable under compact perturbations.

### Sobolev Embedding

W^{k,p}(Ω) ⊂ L^q(Ω) when 1/p − k/n ≤ 1/q (n = dimension of Ω). In particular, if k > n/p, then W^{k,p} ⊂ C^m for m < k − n/p — functions are actually continuous (and differentiable), not just integrable.

## License

MIT
