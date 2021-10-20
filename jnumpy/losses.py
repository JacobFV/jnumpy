import jnumpy.ops


class Loss:

    def __call__(self, ytrue, ypred):
        raise NotImplementedError('subclasses should implement this method')


class MAE(Loss):

    def __call__(self, ytrue, ypred):
        return jnumpy.ops.ReduceSum(jnumpy.ops.Abs(ytrue - ypred))


class MSE(Loss):

    def __call__(self, ytrue, ypred):
        return jnumpy.ops.ReduceSum((ytrue - ypred) ** 2)


class CrossEntropy(Loss):

    def __init__(self, 
        binary: bool = False, 
        sparse: bool = False, 
        from_logits: bool = False):
        """Cross entropy loss. 
        l = p(x)log(q(x)) + (1-p(x))log(1-q(x))

        Args:
            binary (bool, optional): If the . Defaults to False.
            sparse (bool, optional): [description]. Defaults to False.
            from_logits (bool, optional): [description]. Defaults to False.
        """
        self.binary = binary
        self.sparse = sparse
        self.from_logits = from_logits

    def __call__(self, ytrue, ypred):
        return jnumpy.ops.ReduceSum(
            
        )


all_losses = {
    'mse': mse
}