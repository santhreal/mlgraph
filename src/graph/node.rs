use crate::graph::{NodeId, TensorId};
use crate::op::Op;

/// A computation node in the graph.
///
/// A node represents a single operation (primitive or fused) that reads
/// input tensors and produces output tensors.
#[derive(Debug, Clone)]
pub struct Node {
    pub(crate) id: NodeId,
    pub(crate) name: String,
    pub(crate) op: Op,
    pub(crate) inputs: Vec<TensorId>,
    pub(crate) outputs: Vec<TensorId>,
}

impl Node {
    /// Create a new computation node.
    pub fn new(id: NodeId, name: impl Into<String>, op: Op, inputs: Vec<TensorId>, outputs: Vec<TensorId>) -> Self {
        Self {
            id,
            name: name.into(),
            op,
            inputs,
            outputs,
        }
    }

    /// Unique identifier for this node.
    pub fn id(&self) -> NodeId {
        self.id
    }

    /// Human-readable name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// The operation performed by this node.
    pub fn op(&self) -> &Op {
        &self.op
    }

    /// Tensors consumed by this node.
    pub fn inputs(&self) -> &[TensorId] {
        &self.inputs
    }

    /// Tensors produced by this node.
    pub fn outputs(&self) -> &[TensorId] {
        &self.outputs
    }
}
