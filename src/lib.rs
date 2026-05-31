//! # lau-functional-analysis
//!
//! Functional analysis library implementing Banach spaces, Hilbert spaces,
//! bounded linear operators, compact operators, spectral theory, Fredholm theory,
//! Sobolev spaces basics, and agent function spaces.

pub mod banach;
pub mod hilbert;
pub mod operator;
pub mod compact;
pub mod spectral;
pub mod fredholm;
pub mod sobolev;
pub mod agent_space;

pub use banach::*;
pub use hilbert::*;
pub use operator::*;
pub use compact::*;
pub use spectral::*;
pub use fredholm::*;
pub use sobolev::*;
pub use agent_space::*;
