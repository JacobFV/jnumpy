

from typing import Callable, Mapping

from numpy.lib.arraysetops import isin

from jnumpy.opt import Optimizer


class NN(Op):
    """This is not currently implemented"""
    
    def __init__(self):
        self._built = False
        self.loss_fn = None
    
    def __call__(self, X: V):
        
        if not self._built:
            self.build(X.shape)
            
        return self.foreward(X)
    
    def build(self, input_shape: tuple):
        raise NotImplementedError('subclasses should implement this method')
        
    def foreward(self, X: V):
        raise NotImplementedError('subclasses should implement this method')
    
    def compile(self, 
        loss_fn: Union[str, Callable], 
        metrics: List[Union[str, Callable]] = None, 
        optimizer: Union[str, Optimizer] = None):
        """Prepares a model for training

        Args:
            loss_fn (str): The loss function to use. See `jnumpy.losses` for options.
            metrics (List[str], optional): A list of metrics to compute during training. 
                Defaults to None. See `jnumpy.metrics` for options.
            optimizer (str, optional): The optimizer to use. Defaults to SGD. See 
                `jnumpy.optimizers` for more options.
        """
        
        self.compiled = True

        if isinstance(loss_fn, str):
            loss_fn = all_losses[loss_fn]
        self.loss_fn = loss_fn

        self.metrics = []
        for metric in metrics:
            if isinstance(metric, str):
                metric = all_metrics[metric]
            self.metrics.append(metric)

        if optimizer is None:
            optimizer = 'sgd'
        if isinstance(optimizer, str):
            optimizer = all_optimizers[optimizer]()
        self.optimizer = optimizer
    
    def fit(nn, 
        X: np.ndarray,
        Y: np.ndarray,
        epochs: int, 
        steps_per_epoch: int,
        batch_size: int,
        val_ds=None,
        val_steps_per_epoch=None,
        val_batch_size=None,
        callbacks=None, 
        verbose=1) -> Mapping[int, Mapping[str, float]]:
        """Trains a model on a dataset.

        Args:
            X (np.ndarray): The input data.
            Y (np.ndarray): The target data.
            epochs (int): The number of epochs to train for.
            steps_per_epoch (int): The number of steps per epoch.
            batch_size (int): The number of samples per batch.
            val_ds (Optional[np.ndarray], Optional[np.ndarray]): The validation data.
            val_steps_per_epoch (Optional[int]): The number of validation steps per epoch.
            val_batch_size (Optional[int]): The number of validation samples per batch.
            callbacks (Optional[List[Callback]]): A list of callbacks to use during training.
            verbose (int): The verbosity level.

        Returns:
            Mapping[int, Mapping[str, float]]: A mapping of epochs to a mapping of metrics to values.
        """

        history = {'loss': [], 'val_loss': []}
        for epoch in range(epochs):
            if verbose > 0:
                print('Epoch {}/{}'.format(epoch + 1, epochs))
            for step, (x, y) in enumerate(ds):
                if verbose > 1:
                    print('Step {}/{}'.format(step + 1, steps_per_epoch))
                opt.zero


class Sequential(NN):
    """
    This is a sequential model.
    """
    
    def __init__(self):
        super().__init__()
        self.layers = []
    
    def build(self, input_shape: tuple):
        self._built = True
        
        for layer in self.layers:
            layer.build(input_shape)
            input_shape = layer.output_shape
    
    def foreward(self, X: V):
        for layer in self.layers:
            X = layer.foreward(X)
        return X
    
    def add(self, layer: Layer):
        self.layers.append(layer)
    
    def compile(self, loss_fn: str):
        super().compile(loss_fn)
        for layer in self.layers:
            layer.compile(loss_fn)
    
    def fit(self, X: V, Y: V):
        for layer in self.layers:
            X = layer.foreward(X)
        return X
    
    def summary(self):
        for layer in self.layers:
            layer.summary()