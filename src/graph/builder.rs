
use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::graph::{Graph, Node, NodeId, TensorId, TensorMeta};
use crate::op::Op;

impl Graph {
    /// Add a computation node to the graph.
    ///
    /// Performs shape inference from the input tensors and creates output
    /// tensor metadata automatically. Returns the output tensor IDs.
    ///
    /// # Errors
    ///
    /// Returns [`Error::UnknownId`] if any input tensor does not exist,
    /// or [`Error::ShapeMismatch`] if shape inference fails.
    pub fn node(
        &mut self,
        name: &str,
        op: Op,
        inputs: &[TensorId],
    ) -> Result<Vec<TensorId>> {
        // Validate all inputs exist and collect their shapes.
        let input_shapes: Vec<Vec<usize>> = inputs
            .iter()
            .map(|tid| {
                self.tensors
                    .get(tid)
                    .map(|t| t.shape.clone())
                    .ok_or(Error::UnknownId {
                        kind: "tensor",
                        id: tid.0,
                    })
            })
            .collect::<Result<Vec<_>>>()?;

        let input_shape_refs: Vec<&[usize]> =
            input_shapes.iter().map(Vec::as_slice).collect();
        let output_shapes = op.infer_shapes(&input_shape_refs)?;

        // Determine dtype from first input (all tensors in a graph share dtype
        // for now  -  mixed precision is a future extension).
        let dtype = inputs
            .first()
            .and_then(|tid| self.tensors.get(tid).map(|t| t.dtype))
            .unwrap_or(DType::F32); // fallback if no inputs

        let node_id = self.alloc_node_id();

        // Create output tensors
        let mut output_ids = Vec::with_capacity(output_shapes.len());
        let output_len = output_shapes.len();
        for (i, shape) in output_shapes.into_iter().enumerate() {
            let tensor_id = self.alloc_tensor_id();
            let tensor_name = if output_len == 1 {
                name.to_string()
            } else {
                format!("{name}.{i}")
            };

            self.tensors.insert(
                tensor_id,
                TensorMeta {
                    id: tensor_id,
                    shape,
                    dtype,
                    producer: Some(node_id),
                    consumers: Vec::new(),
                    name: tensor_name,
                },
            );
            output_ids.push(tensor_id);
        }

        // Register node
        self.nodes.insert(
            node_id,
            Node::new(node_id, name, op, inputs.to_vec(), output_ids.clone()),
        );

        // Update consumers of the input tensors
        for input_id in inputs {
            if let Some(tensor) = self.tensors.get_mut(input_id) {
                tensor.consumers.push(node_id);
            }
        }

        Ok(output_ids)
    }

    /// Add a fused node that replaces a sequence of existing nodes.
    ///
    /// This is a low-level builder method used by fusion passes. It rewires
    /// the graph edges from the old nodes to the new node, but does NOT
    /// remove the old nodes (the pass must do that via `remove_node`).
    ///
    /// The `inputs` are the tensors that flow INTO the fused subgraph.
    /// The `output_mapping` maps from (old tensor ID produced by a node being fused)
    /// to (new tensor ID produced by the new fused node).
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidGraph`] if any input or output mapping is invalid.
    pub fn add_fused_node(
        &mut self,
        name: &str,
        op: Op,
        inputs: &[TensorId],
        output_mapping: &[(TensorId, TensorId)], // old -> new
    ) -> Result<NodeId> {
        let node_id = self.alloc_node_id();

        let mut output_ids = Vec::new();
        for &(old_id, new_id) in output_mapping {
            output_ids.push(new_id);

            // Clone the old tensor's metadata but update ID and producer
            let old_tensor =
                self.tensors.get(&old_id).ok_or(Error::UnknownId {
                    kind: "tensor",
                    id: old_id.0,
                })?;

            let new_tensor = TensorMeta {
                id: new_id,
                shape: old_tensor.shape.clone(),
                dtype: old_tensor.dtype,
                producer: Some(node_id),
                // Consumers will be rewired below
                consumers: old_tensor.consumers.clone(),
                name: format!("{name}.out"),
            };

            self.tensors.insert(new_id, new_tensor);
        }

        self.nodes.insert(
            node_id,
            Node::new(node_id, name, op, inputs.to_vec(), output_ids),
        );

        // Update consumers of the inputs to point to the new node
        for input_id in inputs {
            if let Some(tensor) = self.tensors.get_mut(input_id) {
                tensor.consumers.push(node_id);
            }
        }

        // Rewire all nodes that consumed the old outputs to consume the new outputs
        for &(old_id, new_id) in output_mapping {
            // Find all nodes that consumed the old output
            let consumers = if let Some(old_t) = self.tensors.get(&old_id) {
                old_t.consumers.clone()
            } else {
                Vec::new()
            };

            for consumer_nid in consumers {
                if let Some(consumer_node) = self.nodes.get_mut(&consumer_nid) {
                    for input in &mut consumer_node.inputs {
                        if *input == old_id {
                            *input = new_id;
                        }
                    }
                }
            }
        }

        // If the old output was a graph-level output, update it.
        for i in 0..self.outputs.len() {
            for &(old_id, new_id) in output_mapping {
                if self.outputs[i] == old_id {
                    self.outputs[i] = new_id;
                }
            }
        }

        Ok(node_id)
    }

    /// Remove a node from the graph.
    ///
    /// This removes the node and its output tensors from the graph. It also
    /// removes the node from the consumer lists of its input tensors.
    ///
    /// Note: This does NOT check if the node's outputs are still being consumed
    /// by other nodes. The caller must ensure the graph remains valid.
    pub fn remove_node(&mut self, node_id: NodeId) {
        if let Some(node) = self.nodes.remove(&node_id) {
            // Remove this node from its inputs' consumers list
            for input_id in &node.inputs {
                if let Some(input_tensor) = self.tensors.get_mut(input_id) {
                    input_tensor.consumers.retain(|&c| c != node_id);
                }
            }

            // Remove all output tensors
            for output_id in &node.outputs {
                self.tensors.remove(output_id);
                // Also remove from graph outputs if present
                self.outputs.retain(|&id| id != *output_id);
            }
        }
    }
}
