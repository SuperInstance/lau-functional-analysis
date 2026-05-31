//! Agent function spaces: agents as elements of Hilbert spaces, similarity as inner product.

use nalgebra::DVector;
use crate::hilbert::HilbertSpace;
use serde::{Deserialize, Serialize};

/// An agent represented as a vector in a Hilbert space.
/// The vector encodes the agent's capabilities, preferences, or behavioral features.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Agent {
    /// Unique identifier.
    pub id: String,
    /// Feature vector in Hilbert space.
    pub features: DVector<f64>,
    /// Optional metadata.
    pub metadata: AgentMetadata,
}

/// Metadata for an agent.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentMetadata {
    /// Agent type or category.
    pub category: Option<String>,
    /// Confidence score.
    pub confidence: Option<f64>,
}

impl Default for AgentMetadata {
    fn default() -> Self {
        Self {
            category: None,
            confidence: None,
        }
    }
}

impl Agent {
    /// Create a new agent with the given ID and feature vector.
    pub fn new(id: impl Into<String>, features: DVector<f64>) -> Self {
        Self {
            id: id.into(),
            features,
            metadata: AgentMetadata::default(),
        }
    }

    /// Create an agent with metadata.
    pub fn with_metadata(
        id: impl Into<String>,
        features: DVector<f64>,
        category: impl Into<String>,
        confidence: f64,
    ) -> Self {
        Self {
            id: id.into(),
            features,
            metadata: AgentMetadata {
                category: Some(category.into()),
                confidence: Some(confidence),
            },
        }
    }

    /// Norm of the agent's feature vector.
    pub fn norm(&self) -> f64 {
        self.features.norm()
    }

    /// Normalize the agent's feature vector to unit length.
    pub fn normalize(&self) -> Self {
        let n = self.features.norm();
        let normalized = if n > 1e-12 {
            &self.features / n
        } else {
            self.features.clone()
        };
        Self {
            id: self.id.clone(),
            features: normalized,
            metadata: self.metadata.clone(),
        }
    }

    /// Feature dimension.
    pub fn dim(&self) -> usize {
        self.features.len()
    }
}

/// A collection of agents forming a function space.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentFunctionSpace {
    /// The agents in the space.
    pub agents: Vec<Agent>,
    /// Dimension of the feature space.
    pub dim: usize,
}

impl AgentFunctionSpace {
    /// Create an empty agent function space with given dimension.
    pub fn new(dim: usize) -> Self {
        Self {
            agents: Vec::new(),
            dim,
        }
    }

    /// Add an agent to the space.
    pub fn add_agent(&mut self, agent: Agent) {
        assert_eq!(agent.dim(), self.dim, "Agent dimension mismatch");
        self.agents.push(agent);
    }

    /// Compute cosine similarity between two agents.
    pub fn cosine_similarity(a: &Agent, b: &Agent) -> f64 {
        let ip = HilbertSpace::inner_product(&a.features, &b.features);
        let na = a.norm();
        let nb = b.norm();
        if na < 1e-12 || nb < 1e-12 {
            0.0
        } else {
            ip / (na * nb)
        }
    }

    /// Compute inner product similarity between two agents.
    pub fn inner_product_similarity(a: &Agent, b: &Agent) -> f64 {
        HilbertSpace::inner_product(&a.features, &b.features)
    }

    /// Find the k nearest neighbors of an agent in this space.
    pub fn nearest_neighbors(&self, agent: &Agent, k: usize) -> Vec<(usize, f64)> {
        let mut similarities: Vec<(usize, f64)> = self
            .agents
            .iter()
            .enumerate()
            .map(|(i, a)| (i, Self::cosine_similarity(agent, a)))
            .collect();
        similarities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        similarities.into_iter().take(k).collect()
    }

    /// Compute the centroid (mean) of all agents.
    pub fn centroid(&self) -> DVector<f64> {
        if self.agents.is_empty() {
            return DVector::zeros(self.dim);
        }
        let mut sum = DVector::zeros(self.dim);
        for agent in &self.agents {
            sum += &agent.features;
        }
        sum / self.agents.len() as f64
    }

    /// Project an agent onto the subspace spanned by a set of reference agents.
    /// Returns the projection coefficients and the projected agent.
    pub fn project_agent(
        agent: &Agent,
        basis_agents: &[Agent],
    ) -> (Vec<f64>, DVector<f64>) {
        // Orthonormalize basis agents via Gram-Schmidt
        let basis_vectors: Vec<DVector<f64>> =
            basis_agents.iter().map(|a| a.features.clone()).collect();
        let orthonormal = HilbertSpace::gram_schmidt(&basis_vectors);

        let mut coefficients = Vec::new();
        let mut projection = DVector::zeros(agent.dim());
        for e in &orthonormal {
            let c = HilbertSpace::inner_product(&agent.features, e);
            coefficients.push(c);
            projection += &(e * c);
        }
        (coefficients, projection)
    }

    /// Compute the similarity matrix for all agent pairs.
    pub fn similarity_matrix(&self) -> DVector<f64> {
        let n = self.agents.len();
        let mut matrix = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..n {
                matrix[i * n + j] = Self::cosine_similarity(&self.agents[i], &self.agents[j]);
            }
        }
        DVector::from_vec(matrix)
    }

    /// Cluster agents by projecting onto orthogonal subspaces.
    /// Returns cluster assignments based on maximum projection.
    pub fn cluster_by_projection(
        &self,
        prototypes: &[Agent],
    ) -> Vec<usize> {
        let proto_vecs: Vec<DVector<f64>> = prototypes.iter().map(|p| {
            let n = p.features.norm();
            if n > 1e-12 { &p.features / n } else { p.features.clone() }
        }).collect();

        self.agents
            .iter()
            .map(|agent| {
                let norm_agent = agent.normalize();
                proto_vecs
                    .iter()
                    .enumerate()
                    .map(|(i, p)| (i, HilbertSpace::inner_product(&norm_agent.features, p)))
                    .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                    .map(|(i, _)| i)
                    .unwrap_or(0)
            })
            .collect()
    }

    /// Number of agents.
    pub fn len(&self) -> usize {
        self.agents.len()
    }

    /// Check if space is empty.
    pub fn is_empty(&self) -> bool {
        self.agents.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_agent_creation() {
        let agent = Agent::new("agent-1", DVector::from_vec(vec![1.0, 0.0, 0.0]));
        assert_eq!(agent.id, "agent-1");
        assert_eq!(agent.dim(), 3);
    }

    #[test]
    fn test_agent_normalize() {
        let agent = Agent::new("agent-1", DVector::from_vec(vec![3.0, 4.0]));
        let normalized = agent.normalize();
        assert!((normalized.norm() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_cosine_similarity_identical() {
        let a = Agent::new("a", DVector::from_vec(vec![1.0, 0.0, 0.0]));
        let b = Agent::new("b", DVector::from_vec(vec![1.0, 0.0, 0.0]));
        let sim = AgentFunctionSpace::cosine_similarity(&a, &b);
        assert!((sim - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        let a = Agent::new("a", DVector::from_vec(vec![1.0, 0.0]));
        let b = Agent::new("b", DVector::from_vec(vec![0.0, 1.0]));
        let sim = AgentFunctionSpace::cosine_similarity(&a, &b);
        assert!(sim.abs() < 1e-10);
    }

    #[test]
    fn test_cosine_similarity_opposite() {
        let a = Agent::new("a", DVector::from_vec(vec![1.0, 0.0]));
        let b = Agent::new("b", DVector::from_vec(vec![-1.0, 0.0]));
        let sim = AgentFunctionSpace::cosine_similarity(&a, &b);
        assert!((sim + 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_nearest_neighbors() {
        let mut space = AgentFunctionSpace::new(3);
        space.add_agent(Agent::new("a", DVector::from_vec(vec![1.0, 0.0, 0.0])));
        space.add_agent(Agent::new("b", DVector::from_vec(vec![0.0, 1.0, 0.0])));
        space.add_agent(Agent::new("c", DVector::from_vec(vec![0.9, 0.1, 0.0])));

        let query = Agent::new("q", DVector::from_vec(vec![1.0, 0.0, 0.0]));
        let nn = space.nearest_neighbors(&query, 2);
        assert_eq!(nn.len(), 2);
        // First should be agent "a" (identical)
        assert_eq!(nn[0].0, 0);
    }

    #[test]
    fn test_centroid() {
        let mut space = AgentFunctionSpace::new(2);
        space.add_agent(Agent::new("a", DVector::from_vec(vec![1.0, 0.0])));
        space.add_agent(Agent::new("b", DVector::from_vec(vec![-1.0, 0.0])));
        let c = space.centroid();
        assert!(c[0].abs() < 1e-10);
        assert!(c[1].abs() < 1e-10);
    }

    #[test]
    fn test_project_agent() {
        let agent = Agent::new("query", DVector::from_vec(vec![3.0, 4.0, 5.0]));
        let basis1 = Agent::new("b1", DVector::from_vec(vec![1.0, 0.0, 0.0]));
        let basis2 = Agent::new("b2", DVector::from_vec(vec![0.0, 1.0, 0.0]));
        let (coeffs, proj) = AgentFunctionSpace::project_agent(&agent, &[basis1, basis2]);
        assert_eq!(coeffs.len(), 2);
        assert!((coeffs[0] - 3.0).abs() < 1e-10);
        assert!((coeffs[1] - 4.0).abs() < 1e-10);
        // Projection should be [3, 4, 0]
        assert!((proj[2] - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_cluster_by_projection() {
        let mut space = AgentFunctionSpace::new(2);
        space.add_agent(Agent::new("a1", DVector::from_vec(vec![1.0, 0.0])));
        space.add_agent(Agent::new("a2", DVector::from_vec(vec![0.9, 0.1])));
        space.add_agent(Agent::new("b1", DVector::from_vec(vec![0.0, 1.0])));
        space.add_agent(Agent::new("b2", DVector::from_vec(vec![0.1, 0.9])));

        let prototypes = vec![
            Agent::new("proto-a", DVector::from_vec(vec![1.0, 0.0])),
            Agent::new("proto-b", DVector::from_vec(vec![0.0, 1.0])),
        ];
        let clusters = space.cluster_by_projection(&prototypes);
        assert_eq!(clusters[0], 0);
        assert_eq!(clusters[1], 0);
        assert_eq!(clusters[2], 1);
        assert_eq!(clusters[3], 1);
    }

    #[test]
    fn test_inner_product_similarity() {
        let a = Agent::new("a", DVector::from_vec(vec![1.0, 2.0]));
        let b = Agent::new("b", DVector::from_vec(vec![3.0, 4.0]));
        let sim = AgentFunctionSpace::inner_product_similarity(&a, &b);
        assert!((sim - 11.0).abs() < 1e-10);
    }

    #[test]
    fn test_agent_with_metadata() {
        let agent = Agent::with_metadata("agent-1", DVector::from_vec(vec![1.0, 0.0]), "type-a", 0.95);
        assert_eq!(agent.metadata.category.as_deref(), Some("type-a"));
        assert!((agent.metadata.confidence.unwrap() - 0.95).abs() < 1e-10);
    }
}
