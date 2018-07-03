import itertools
import numpy as np


class ParamsSearch(object):
    def __init__(self, params):
        self.lr = params['lr']
        self.dropout = params['dropout']
        self.other_hidden_layers = params['other_hidden_layers']
        self.layer_size = params['layer_size']
        self.batch_size = params['batch_size']
        self.epochs = params['epochs']
        self.shapes = params['shapes']
        self.kernel_initializer = params['kernel_initializer']
        self.optimizer = params['optimizer']
        self.losses = params['losses']
        self.activation = params['activation']
        self.last_activation = params['last_activation']

        self.params_name = list(params.keys())
        self.int_type_params = ['other_hidden_layers', 'layer_size', 'batch_size', 'epochs']

        # self._params_grid() --> self._axes() --> self.axis()
        self.params_grid = self._params_grid()

    def _params_grid(self):
        params_grid = [dict(zip(self.params_name, combo)) for combo in self._axes()]
        return params_grid

    def _axes(self):
        # iterator of combs
        return itertools.product(*(self._axis(key, self.__dict__[key]) for key in self.params_name))

    def _axis(self, key, specs):
        raise NotImplementedError("")


# TODO: need to be tested before use
class ParamsGridSearch(ParamsSearch):
    def __init__(self, params):
        super().__init__(params)

    def _axis(self, key, specs):
        if type(specs) is tuple:
            start, end, step = specs
            return list(range(start, end, step))   # TODO: log-scale or not
        elif type(specs) is list:
            return specs
        else:
            pass    # TODO:


class ParamsRandomGridSearch(ParamsSearch):
    def __init__(self, params):
        super().__init__(params)

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


class ParamsRandomSearch(ParamsSearch):
    def __init__(self, params, n_iter=1):
        # order matters here
        self.n_iter = n_iter
        super().__init__(params)

    def _axes(self):
        # iterator of combs
        return zip(*(self._axis(key, self.__dict__[key]) for key in self.params_name))
        # unpack list of np array before zipping it !!!

    def _axis(self, key, specs):
        if type(specs) is tuple:
            if key in self.int_type_params:
                return np.random.randint(*specs, self.n_iter)
            else:
                if key is 'lr':
                    return 10**np.random.uniform(*specs, self.n_iter)
                else:
                    return np.random.uniform(*specs, self.n_iter)
        elif type(specs) is list:
            return np.random.choice(specs, self.n_iter)
        else:
            pass
