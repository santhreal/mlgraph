//! The computation graph  -  nodes, tensors, and edges.
//!
//! A `Graph` is an acyclic dataflow graph where `Node`s perform operations
//! and `TensorMeta` edges carry shape/dtype metadata. No actual tensor data
//! is stored  -  this is a compile-time optimization graph.

mod builder;
mod node;
mod tensor;

use std::collections::{BinaryHeap, HashMap};

use crate::dtype::DType;
pub use node::Node;
pub use tensor::TensorMeta;

/// Unique identifier for a node in the computation graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct NodeId(pub(crate) u32);

/// Unique identifier for a tensor (edge) in the computation graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct TensorId(pub(crate) u32);

impl std::fmt::Display for NodeId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "n{}", self.0)
    }
}

impl std::fmt::Display for TensorId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "t{}", self.0)
    }
}

/// The computation graph.
///
/// This is a directed acyclic graph (DAG) where nodes are operations and
/// edges are tensors. It tracks the topological structure and metadata needed
/// for optimization passes, but does not execute the operations.
#[derive(Debug, Clone)]
pub struct Graph {
    pub(crate) nodes: HashMap<NodeId, Node>,
    pub(crate) tensors: HashMap<TensorId, TensorMeta>,
    pub(crate) next_node_id: u32,
    pub(crate) next_tensor_id: u32,
    pub(crate) name: String,
    pub(crate) inputs: Vec<TensorId>,
    pub(crate) outputs: Vec<TensorId>,
}

impl Graph {
    /// Create a new empty graph with the given name.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            nodes: HashMap::new(),
            tensors: HashMap::new(),
            next_node_id: 0,
            next_tensor_id: 0,
            name: name.into(),
            inputs: Vec::new(),
            outputs: Vec::new(),
        }
    }

    /// The name of this graph.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Add a graph-level input tensor (no producer node).
    pub fn input(&mut self, name: &str, shape: &[usize], dtype: DType) -> TensorId {
        let id = self.alloc_tensor_id();
        self.tensors.insert(
            id,
            TensorMeta {
                id,
                shape: shape.to_vec(),
                dtype,
                producer: None,
                consumers: Vec::new(),
                name: name.to_string(),
            },
        );
        self.inputs.push(id);
        id
    }

    /// Allocate a new unique node ID.
    pub fn alloc_node_id(&mut self) -> NodeId {
        let id = self.next_node_id;
        self.next_node_id += 1;
        NodeId(id)
    }

    /// Allocate a new unique tensor ID.
    pub fn alloc_tensor_id(&mut self) -> TensorId {
        let id = self.next_tensor_id;
        self.next_tensor_id += 1;
        TensorId(id)
    }

    /// Mark tensor(s) as graph-level outputs.
    pub fn mark_output(&mut self, tensor_id: TensorId) {
        if !self.outputs.contains(&tensor_id) {
            self.outputs.push(tensor_id);
        }
    }

    /// Look up a tensor by id.
    pub fn tensor(&self, id: TensorId) -> Option<&TensorMeta> {
        self.tensors.get(&id)
    }

    /// Look up a mutable tensor by id.
    pub fn tensor_mut(&mut self, id: TensorId) -> Option<&mut TensorMeta> {
        self.tensors.get_mut(&id)
    }

    /// Look up a node by id.
    pub fn node_by_id(&self, id: NodeId) -> Option<&Node> {
        self.nodes.get(&id)
    }

    /// Iterate over all nodes in the graph (unordered).
    pub fn nodes(&self) -> impl Iterator<Item = &Node> {
        self.nodes.values()
    }

    /// Iterate over all tensors in the graph (unordered).
    pub fn tensors(&self) -> impl Iterator<Item = &TensorMeta> {
        self.tensors.values()
    }

    /// Number of nodes in the graph.
    pub fn num_nodes(&self) -> usize {
        self.nodes.len()
    }

    /// Number of tensors in the graph.
    pub fn num_tensors(&self) -> usize {
        self.tensors.len()
    }

    /// Graph-level input tensor IDs.
    pub fn graph_inputs(&self) -> &[TensorId] {
        &self.inputs
    }

    /// Graph-level output tensor IDs.
    pub fn graph_outputs(&self) -> &[TensorId] {
        &self.outputs
    }

    /// Update the producer of a tensor.
    pub fn update_tensor_producer(&mut self, tensor_id: TensorId, new_producer: Option<NodeId>) {
        if let Some(tensor) = self.tensors.get_mut(&tensor_id) {
            tensor.producer = new_producer;
        }
    }

    pub(crate) fn insert_node(&mut self, node: Node) {
        // Register the node as a consumer of each of its input tensors so
        // successor/topological queries observe the new edges. `add_node` does
        // this for the builder path; fusion inserts pre-built nodes here and
        // relied on it happening, but it did not, leaving the fused node absent
        // from its inputs' consumer lists.
        for &input_id in &node.inputs {
            if let Some(tensor) = self.tensors.get_mut(&input_id) {
                if !tensor.consumers.contains(&node.id) {
                    tensor.consumers.push(node.id);
                }
            }
        }
        self.nodes.insert(node.id, node);
    }

    pub(crate) fn successors(&self, node_id: NodeId) -> Vec<NodeId> {
        let mut succs = Vec::new();
        if let Some(node) = self.nodes.get(&node_id) {
            for output_id in &node.outputs {
                if let Some(tensor) = self.tensors.get(output_id) {
                    succs.extend(tensor.consumers.iter().copied());
                }
            }
        }
        succs
    }

    /// Topologically sorted node IDs (sources first, sinks last).
    pub fn topological_order(&self) -> Vec<NodeId> {
        let mut in_degree: HashMap<NodeId, usize> = HashMap::new();
        for node in self.nodes.values() {
            in_degree.entry(node.id).or_insert(0);
            for output_tid in &node.outputs {
                if let Some(tensor) = self.tensors.get(output_tid) {
                    for consumer_nid in &tensor.consumers {
                        *in_degree.entry(*consumer_nid).or_insert(0) += 1;
                    }
                }
            }
        }

        // A max-heap keeps the traversal deterministic (largest NodeId first,
        // matching the previous sort()+pop() behavior) while giving O(N log N)
        // instead of re-sorting the whole queue on every insertion.
        let mut queue: BinaryHeap<NodeId> = in_degree
            .iter()
            .filter(|&(_, &deg)| deg == 0)
            .map(|(&nid, _)| nid)
            .collect();

        let mut order = Vec::with_capacity(self.nodes.len());

        while let Some(nid) = queue.pop() {
            order.push(nid);
            if let Some(node) = self.nodes.get(&nid) {
                for output_tid in &node.outputs {
                    if let Some(tensor) = self.tensors.get(output_tid) {
                        for consumer_nid in &tensor.consumers {
                            if let Some(deg) = in_degree.get_mut(consumer_nid) {
                                *deg -= 1;
                                if *deg == 0 {
                                    queue.push(*consumer_nid);
                                }
                            }
                        }
                    }
                }
            }
        }

        order
    }
}
