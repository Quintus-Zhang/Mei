import itertools
import numpy as np


class ParamsSearch(object):
    def __init__(self, lr, dropout, other_hidden_layers, layer_size, batch_size, epochs,
                 shapes, kernel_initializer, optimizer, losses, activation, last_activation):
        self.lr = lr
        self.dropout = dropout
        self.other_hidden_layers = other_hidden_layers
        self.layer_size = layer_size
        self.batch_size = batch_size
        self.epochs = epochs
        self.shapes = shapes
        self.kernel_initializer = kernel_initializer
        self.optimizer = optimizer
        self.losses = losses
        self.activation = activation
        self.last_activation = last_activation

        # order matters here
        self.params_name = list(self.__dict__.keys())
        self.int_type_params = ['other_hidden_layers', 'layer_size', 'batch_size', 'epochs']
        self.params_grid = self._params_grid()

    def _params_grid(self):
        params_grid = [dict(zip(self.params_name, combo)) for combo in self._axes()]
        return params_grid

    def _axes(self):
        return itertools.product(*(self._axis(key, self.__dict__[key]) for key in self.params_name))

    def _axis(self, key, specs):
        raise NotImplementedError("")


# TODO: need to be tested before use
class ParamsGridSearch(ParamsSearch):
    def __init__(self, lr, dropout, other_hidden_layers, layer_size, batch_size, epochs,
                 shapes, kernel_initializer, optimizer, losses, activation, last_activation):
        super().__init__(lr, dropout, other_hidden_layers, layer_size, batch_size, epochs,
                         shapes, kernel_initializer, optimizer, losses, activation, last_activation)

    def _axis(self, key, specs):
        if type(specs) is tuple:
            start, end, step = specs
            return list(range(start, end, step))   # TODO: log-scale or not
        elif type(specs) is list:
            return specs
        else:
            pass    # TODO:


class ParamsRandomSearch(ParamsSearch):
    def __init__(self, lr, dropout, other_hidden_layers, layer_size, batch_size, epochs,
                 shapes, kernel_initializer, optimizer, losses, activation, last_activation):
        super().__init__(lr, dropout, other_hidden_layers, layer_size, batch_size, epochs,
                         shapes, kernel_initializer, optimizer, losses, activation, last_activation)

    def _axis(self, key, specs):
        np.random.seed(1)
        if type(specs) is tuple:
            if key in self.int_type_params:
                return np.random.randint(*specs)
            else:
                if key is 'lr':
                    return 10**np.random.uniform(*specs)
                else:
                    return np.random.uniform(*specs)
        elif type(specs) is list:
            return specs
        else:
            pass
