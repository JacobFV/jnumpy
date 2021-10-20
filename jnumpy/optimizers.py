from typing import List, Optional

import numpy as np

from jnumpy.core import Tensor, Var, Op
from jnumpy.graph import TensorGraph


class Optimizer:
    
    def minimize(self, tensor: Tensor):
        pass

    def grads(self, 
        input_tensors: List[Tensor], 
        output_tensors: List[Tensor],
        output_gradients: Optional[List[np.ndarray]] = None) -> List[np.ndarray]:
        """Backpropagates the `output_gradients` through the graph from the 
        `output_tensors` to the `input_tensors`. 
        
        This method may be used by subclasses to implement the `minimize` method.

        Args:
            input_tensors (List[Tensor]): Tensors to compute the gradients for.
            output_tensors (List[Tensor]): Tensors whose gradients are known.
            output_gradients (List[np.ndarray], optional): Gradients of the output tensors.
                If not supplied, they are assumed to be all ones.

        Returns:
            List[np.ndarray]: Gradients of the input tensors in the order supplied.
        """
        

    def apply_gradients(self, tensors: List[Tensor], gradients: List[np.ndarray]):
        """Applies the gradients to the tensors.

        Args:
            tensors (List[Tensor]): Tensors to apply the gradients to.
            gradients (List[np.ndarray]): Gradients of the tensors.
        """
        raise NotImplementedError('Subclasses should implement this method.')


class SGD(Optimizer):
    
    def __init__(self, lr: float = 0.001):
        
        self.lr = lr
        self.debug = False
        
        super().__init__()
    
    def minimize(self, t: Tensor = None, loss_fn: Callable = None, 
        loss_fn_args: tuple = (), loss_fn_kwargs: dict = {}):
        """Minimizes a tensor's value. This function supports two paradigms:
        1) give me the tensor `t` and I'll optimize over all trainable 
            `Var` tensors to minimize it.
        2) give me a `loss_fn` and I'll optimize over all trainable `Var` 
            tensors to minimize the loss function.
        The first takes precedence over the second (i.e.: if `t` is not `None`).
        However the second is useful if you're working in EAGER mode since
        `Op` tensor values don't update following backpropagation. 

        Args:
            t (Tensor): tensor to minimize.
            loss_fn (Callable): loss function to minimize.
            loss_fn_args (tuple, optional): arguments to pass to the loss 
                function. Defaults to ().
            loss_fn_kwargs (dict, optional): keyword arguments to pass to the
                loss function. Defaults to {}.

        Returns:
            nothing.
        """
        
        if EXECUTION_MODE == ExecutionMode.STATIC:
            raise 'STATIC execution mode not enabled'

        if t is None:
            t = loss_fn(*loss_fn_args, **loss_fn_kwargs)

        self.bprop(t_out=t, output_grad=-np.ones_like(t.val))

        
    def bprop(self, t_out: Tensor, output_grad: V):
        """Iteratively backpropagates through a tensor.
        
        Currently, this recursive approach implements a DFS.
        
        Ideally though, it should use some vertex-marking algorithm
        and iterate over the entire tensor graph. You should either
        explicitly or implicitly pass the tensor graph around. Using
        a tensor graph would be useful for foreward propagation too.
        Also STATIC mode wouldn't be too hard to implement if the 
        graph was already clearly defined. 
        
        Finally, the tree-topology assumption of DFS results in 
        the same variable getting updated twice when it is jointly
        backpropagated and gradient updated. Ideally, backpropagtion
        should take place in the graph class. Then optimizers can
        support optimization tricks like momentum and returning
        named gradients or supplying output/internal gradients.
        
        I really need to make a graph class with the functions
        graph.refresh(nodes=None) 
        graph.bprop(nodes=None, input_grad=None)
        # nodes=None for all or you can specify specific nodes which are iteratively removed

        Copilt had a great idea:

            # minimize over all trainable variables
            for v in t.trainable_vars:
                v.value -= self.lr * v.grad
        elif loss_fn is not None:
            # minimize over all trainable variables
            for v in t.trainable_vars:
                v.value -= self.lr * v.grad
        else:
            raise 'no tensor or loss function provided'

        This made me think: store the gradients on the nodes and
        then update the values using the optimizer.
        """
        
        assert isinstance(t_out, (Var, Op))
        
        if self.debug:
            print(f'bp {t_out} output_grads:')
            print(output_grad)
        
        # iteratively called
        if isinstance(t_out, Var):
            if t_out.trainable:
                #print('output_grad', output_grad.shape)
                if self.debug:                    
                    print('t_out.val (old)', t_out.val.shape)
                    print(t_out.val)
                
                t_out.val = t_out.val + (self.lr * output_grad)
                if self.debug:
                    print('t_out.val (new)', t_out.val.shape)
                    print(t_out.val)
            
        elif isinstance(t_out, Op):
            input_grads = t_out.reverse_grad(
                inputs=tuple(t.val for t in t_out.input_ts),
                outputs=(t_out.val,), 
                top_grads=(output_grad,))
            
            for input_t, input_grad in zip(t_out.input_ts, input_grads):
                self.bprop(t_out=input_t, output_grad=input_grad)
        
        return


class Adam(Optimizer):
    
    def __init__(self, t: Tensor, lr: float, beta1: float, beta2: float, eps: float):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        
        self.m = np.zeros(t.shape)
        self.v = np.zeros(t.shape)
        
        self.t = 0
        
        super().__init__(t)
    
    def foreward(self, inputs: Vs) -> Vs:
        X = inputs[0]
        
        self.t += 1
        
        self.m = self.beta1 * self.m + (1 - self.beta1) * X
        self.v = self.beta2 * self.v + (1 - self.beta2) * (X**2)
        
        m_hat = self.m / (1 - self.beta1 ** self.t)
        v_hat = self.v / (1 - self.beta2 ** self.t)
        
        Y = self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
        
        return (Y,)
    
class RMSprop(Optimizer):
    
    def __init__(self, t: Tensor, lr: float, beta: float, eps: float):
        self.lr = lr
        self.beta = beta
        self.eps = eps
        
        self.m = np.zeros(t.shape)
        
        self.t = 0
        
        super().__init__(t)
    
    def foreward(self, inputs: Vs) -> Vs:
        X = inputs[0]
        
        self.t += 1
        
        self.m = self.beta * self.m + (1 - self.beta) * (X**2)
        
        m_hat = self.m / (1 - self.beta ** self.t)
        
        Y = self.lr * X / (np.sqrt(m_hat) + self.eps)
        
        return (Y,)


all_opimizers = {
    'sgd': SGD
}