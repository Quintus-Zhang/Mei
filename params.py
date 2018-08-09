import itertools
import numpy as np


class ParamsSearch(object):
    def __init__(self, params, model):
        """
        :param params: a Dict, stores the parameters for search
        :param model: a String, 'NN' or 'XGB'
        """
        if model == 'NN':
            self.lr = params['lr']
            self.dropout = params['dropout']
            self.other_hidden_layers = params['other_hidden_layers']
            self.layer_size = params['layer_size']
            self.batch_size = params['batch_size']
            self.epochs = params['epochs']
            self.shapes = params['shapes']
            self.pos_weight = params['pos_weight']
            self.kernel_initializer = params['kernel_initializer']
            self.optimizer = params['optimizer']
            self.losses = params['losses']
            self.activation = params['activation']
            self.last_activation = params['last_activation']

            self.int_type_params = ['other_hidden_layers', 'layer_size', 'batch_size', 'epochs']

        elif model == 'XGB':
            self.eta = params['eta']
            self.max_depth = params['max_depth']
            self.min_child_weight = params['min_child_weight']
            self.subsample = params['subsample']
            self.colsample_bytree = params['colsample_bytree']
            self.eval_metric = params['eval_metric']
            self.objective = params['objective']
            self.silent = params['silent']

            self.int_type_params = ['max_depth', 'min_child_weight']

        self.params_name = list(params.keys())

        # self._params_grid() --> self._axes() --> self.axis()
        self.params_grid = self._params_grid()

    def _params_grid(self):
        """ Generate the parameters grid

        :return: a List of Dict, each element is a parameter combo, e.g. {'lr': 0.01, 'dropout': 0.5, ...}
        """
        params_grid = [dict(zip(self.params_name, combo)) for combo in self._axes()]
        return params_grid

    def _axes(self):
        raise NotImplementedError("")

    def _axis(self, key, specs):
        raise NotImplementedError("")


class ParamsGridSearch(ParamsSearch):
    def __init__(self, params):
        super().__init__(params)

    def _axes(self):
        """ Create combinations of the parameters for search

        :return: iterator of combos, each element is a combo such as [0.01, 0.5, ...]
        """
        # iterator of combs
        return itertools.product(*(self._axis(key, self.__dict__[key]) for key in self.params_name))

    def _axis(self, key, specs):
        """ Discretize the search range for a parameter

        :param key: no use here
        :param specs: a tuple or list, the range specs, e.g. (start, end, step) or [a, b, c, ...]
        :return: a list, grid for single parameter
        """
        if type(specs) is tuple:
            start, end, step = specs
            return list(range(start, end, step))
        elif type(specs) is list:
            return specs
        else:
            pass


class ParamsRandomSearch(ParamsSearch):
    def __init__(self, params, n_iter=1, model='NN'):
        # order matters here
        self.n_iter = n_iter
        super().__init__(params, model)

    def _axes(self):
        """ Create combinations of the parameters for search

        :return: iterator of combos, each element is a combo such as [0.01, 0.5, ...]
        """
        # iterator of combs
        return zip(*(self._axis(key, self.__dict__[key]) for key in self.params_name))
        # unpack list of np array before zipping it,
        # because zip() accept multiple iterables instead of a single iterable containing iterables to zip

    def _axis(self, key, specs):
        """ Discretize the search range for a parameter

        :param key: no use here
        :param specs: a tuple or list, the range specs, e.g. (start, end) or [a, b, c, ...]
        :return: a list, grid for single parameter
        """
        if type(specs) is tuple:
            if key in self.int_type_params:
                return np.random.randint(*specs, self.n_iter)
            else:
                if key is 'lr' or key is 'eta':
                    return 10**np.random.uniform(*specs, self.n_iter)
                else:
                    return np.random.uniform(*specs, self.n_iter)
        elif type(specs) is list:
            return np.random.choice(specs, self.n_iter)
        else:
            pass
