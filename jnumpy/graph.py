from __future__ import annotations
from typing import List, Mapping, Optional

import numpy as np

from jnumpy.core import Placeholder, Tensor, Op


class TensorGraph:
    

    def __init__(self, exec_mode: str = 'eager', resolve_namespace: bool = True):
        """Initializes a Tensor graph.

        Args:
            exec_mode (str, optional): What computation paradigm the Tensors follow. 
                'eager' means that computations are executed immediately
                'lazy' means that computations are deferred until the result is needed
                Defaults to 'eager'.
            resolve_namespace (bool, optional): Whether to resolve the names of Tensors
                to ensure uniqueness. Defaults to True.
        """
        self.exec_mode = exec_mode
        self.nodes = list()  # List[Tensor]
        self.edges = list()  # List[Tuple[Tensor (src), Tensor (dst)]]

    @staticmethod
    def default() -> TensorGraph:
        """Returns the default graph"""
        # In the future, you will be able to change the working graph 
        # using context and `with` expressions
        return eval('_DEFAULT_GRAPH')  

    def add_node(self, tensor):
        """Adds a tensor to the graph.

        Args:
            tensor (Tensor): The tensor to add to the graph.
        """
        self.nodes.append(tensor)

    def add_edge(self, src_tensor, dst_tensor):
        """Adds an edge from `src_tensor` to `dst_tensor`.

        Args:
            src_tensor (Tensor): The source tensor.
            dst_tensor (Tensor): The destination tensor.
        """
        self.edges.append((src_tensor, dst_tensor))

    def eval(graph, 
        tensors: Optional[List[Tensor]] = None,
        recompute_parents: bool = True) -> List[np.ndarray]:
        """Refreshes the numerical values of `tensors` and all their parents in the graph.
        This method is usually called in `lazy` execution mode or by `backprop`.

        If the graph is cyclic, each node is only updated once.

        Args:
            tensors (List[Tensor], optional): Tensors to retrieve values of. If `tensors`
                are not provided, all tensors in the graph are refreshed.
            recompute_parents (bool, optional): Whether to recompute the values
                of the parents of `tensors`. Defaults to True.

        Returns:
            List[np.ndarray]: Values of the tensors in the order supplied.
        """
        
        # refresh all tensors if no tensors are specified
        if tensors is None:
            tensors = graph.nodes

        # identify minimial subgraph to compute
        
        prev_eval_nodes = []
        eval_nodes = tensors
        if recompute_parents:
            while eval_nodes != prev_eval_nodes:
                prev_eval_nodes = eval_nodes
                for node in eval_nodes:
                    if isinstance(node, Op):
                        eval_nodes.extend([node for node in node.input_tensors
                                             if node not in eval_nodes])
            del prev_eval_nodes

        # store node values
        vals = { tensor: None for tensor in eval_nodes }
        
        # update node values
        while not all(vals is not None for vals in vals.values()):

            # maybe update tensor value
            for tensor in eval_nodes:
                if isinstance(tensor, Op):
                    input_values = tuple(vals[input_tensor] for input_tensor in tensor.input_tensors)

                    if all(value is not None for value in input_values):
                        vals[tensor] = tensor.forward(input_values)

        # maybe write tensor values
        if graph.exec_mode == 'eager':
            for tensor in eval_nodes:
                tensor.val = vals[tensor]

        # return requested tensor values
        return [vals[tensor] for tensor in tensors]
        

    def backprop(graph,
        input_tensors: Optional[List[Tensor]], 
        output_tensors: Optional[List[Tensor]],
        output_gradients: Optional[List[np.ndarray]] = None) -> List[np.ndarray]:
        """Backpropagates the `output_gradients` through the graph from the 
        `output_tensors` to the `input_tensors`. If neither `input_tensors` 
        nor `output_tensors` are provided, the entire graph is backpropagated.

        Args:
            input_tensors (List[Tensor]): Tensors to compute the gradients for.
            output_tensors (List[Tensor]): Tensors whose gradients are known.
            output_gradients (List[np.ndarray], optional): Gradients of the output tensors.
                If not supplied, they are assumed to be all ones.

        Returns:
            List[np.ndarray]: Gradients of the input tensors in the order supplied.
        """

        # identify subgraph to backprop
        prev_nodes = []
        bp_nodes = output_tensors
        while prev_nodes != bp_nodes:
            prev_nodes = bp_nodes
            for node in bp_nodes:
                if isinstance(node, Op) and node not in input_tensors:
                    bp_nodes.extend([node for node in node.input_tensors
                                       if node not in bp_nodes])
        del prev_nodes

        # get foreward pass values 
        if graph.exec_mode == 'lazy':
            vals = {
                tensor: val for tensor, val 
                in zip(bp_nodes, graph.eval(bp_nodes, recompute_parents=True))
            }
        elif graph.exec_mode == 'eager':
            vals = {tensor: tensor.val for tensor in output_tensors}
        else:
            raise ValueError(f'Unknown execution mode: {graph.exec_mode}')

        # maybe build output gradients
        if output_gradients is None:
            assert graph.exec_mode == 'eager', \
                '`output_gradients` must be manually specified when the graph is not in eager execution mode'
            output_gradients = [np.ones_like(tensor.val) for tensor in output_tensors]

        # build graph to track gradients
        grads = { tensor: None for tensor in bp_nodes }
        # assign output gradients to tensors on the packpropagation graph
        for tensor, gradient in zip(output_tensors, output_gradients):            
            grads[tensor] = gradient

        # backpropagate until every node has a gradient
        while not all(grad is not None for grad in grads.values()):

            # backpropagate
            for tensor in bp_nodes:
                
                # see if child tensors have gradients
                children = [dst for src, dst in graph.edges if src==tensor]
                if len(children) > 0 and all(grads[child] is not None for child in children):
                    
                    # compute average gradient (when a tensor has more than one child)
                    grads[tensor] = np.mean([grads[child] for child in children], axis=0)

                    # if tensor is an op, backpropagate gradients to parents
                    if isinstance(tensor, Op):
                        input_grads = tensor.reverse_grad(
                            inputs=tuple(vals[input_tensor] for input_tensor in tensor.input_tensors),
                            output=vals[tensor], 
                            top_grad=grads[tensor])
                        for input_tensor, grad in zip(tensor.input_tensors, input_grads):
                            grads[input_tensor] = grad

        # return gradients
        return [tensor.grad for tensor in input_tensors]


_DEFAULT_TENSORGRAPH = TensorGraph()