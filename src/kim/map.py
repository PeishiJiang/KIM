"""The general mapping class."""

# Author: Peishi Jiang <shixijps@gmail.com>

from .data import Data
from .mapping_model import train_ensemble, loss_mse
from .mapping_model import MLP, MLP2

import json
import random
import pickle
import itertools
from pathlib import PosixPath, Path

import jax
import jax.numpy as jnp
import equinox as eqx

from jaxlib.xla_extension import Device
from typing import Optional
from jaxtyping import Array


class Maps(object):

    def __init__(self, data: Data):
        pass

    def train(self):
        pass

    def predict(self):
        pass

    def save(self):
        pass

    def load(self):
        pass


class Map(object):
    """The class for one mapping training, prediction, saving and loading.
       Ensemble training is supported through either serial or parallel way,
       using joblib.
       

    Arguments:
    ----------
    x (array-like): the predictors with shape (Ns, Nx)
    y (array-like): the predictands with shape (Ns, Ny)
    model_type (eqx._module._ModuleMeta): the equinox model class
    n_model (int): the number of ensemble models
    ensemble_type (str): the ensemble type, either 'single', 'serial' or 'parallel'.
    model_hp_choices (dict): the tunable model hyperparameters, in dictionary format 
        {key: [value1, value2,...]}. The model hyperparameters must follow the arguments
        of the specified model_type
    model_hp_fixed (dict): the fixed model hyperparameters, in dictionary format 
        {key: value}. The model hyperparameters must follow the arguments
        of the specified model_type
    optax_hp_choices (dict): the tunable optimizer hyperparameters, in dictionary format 
        {key: [value1, value2,...]}. The optimizer hyperparameters must follow the arguments
        of the specified optax optimizer. Key hyperparameters: 
        'optimizer_type' (str), 'nsteps' (int), and 'loss_func' (callable)
    optax_hp_fixed (dict): the fixed optimizer hyperparameters, in dictionary format 
        {key: value}. The optimizer hyperparameters must follow the arguments
        of the specified model_type. Key hyperparameters: 
        'optimizer_type' (str), 'nsteps' (int), and 'loss_func' (callable)
    dl_hp_choices (dict): the tunable dataloader hyperparameters, in dictionary format 
        {key: [value1, value2,...]}. The optimizer hyperparameters must follow the arguments
        of make_big_data_loader. Key hyperparameters: 'batch_size' (int) and 'num_train_sample' (int)
    dl_hp_fixed (dict): the fixed dataloader hyperparameters, in dictionary format 
        {key: value}. The optimizer hyperparameters must follow the arguments
        of make_big_data_loader. Key hyperparameters: 'batch_size' (int) and 'num_train_sample' (int
    training_parallel (bool): whether to perform parallel training
    ens_seed (Optional[int], optional): the random seed for generating ensemble configurations.
    parallel_config (Optional[dict], optional): the parallel training configurations following
        the arguments of joblib.Parallel
    device (Optional[Device], optional): the computing device to be set


    Attributes
    ----------
    self.x (array_like): argument copy
    self.y (array_like): argument copy
    self.n_model (int): argument copy
    self.training_parallel (bool) : argument copy
    self.model_type (eqx._module._ModuleMeta): argument copy
    self.ensemble_type (str): argument copy
    self.model_hp_choices (dict): argument copy
    self.model_hp_fixed (dict): argument copy
    self.optax_hp_choices (dict):argument copy
    self.optax_hp_fixed (dict): argument copy
    self.dl_hp_choices (dict): argument copy
    self.dl_hp_fixed (dict): argument copy
    self.training_parallel (bool): argument copy
    self.ens_seed (Optional[int], optional): argument copy
    self.parallel_config (Optional[dict], optional): argument copy
    self.device (Optional[Device], optional): argument copy
    self.trained (bool) : whether the mapping has been trained
    self.loaded_from_other_sources (bool) : whether the mapping is loaded from other sources.
    self.Ns (int): number of samples
    self.Nx (int): number of input features
    self.Ny (int): number of output features
    self.model_configs (list): model hyperparameters for all ensemble models
    self.optax_configs (list): optimizer hyperparameters for all ensemble models
    self.dl_configs (list): dataloader hyperparameters for all ensemble models
    self.model_ens (list): list of trained model ensemble
    self.loss_train_ens (list): list of the training losses over steps
    self.loss_test_ens (list): list of the test losses over steps

    """

    def __init__(
        self, x: Array, y: Array, model_type: eqx._module._ModuleMeta=MLP, 
        n_model: int=0, ensemble_type: str='single',
        model_hp_choices: dict={}, model_hp_fixed: dict={}, 
        optax_hp_choices: dict={}, optax_hp_fixed: dict={},
        dl_hp_choices: dict={}, dl_hp_fixed: dict={},
        training_parallel: bool=True,
        ens_seed: Optional[int]=None,
        parallel_config: Optional[dict]=None,
        device: Optional[Device]=None
    ):
        # TODO: Need a great way to pass the computational device
        # somehow coupled to the parallel training
        # for now, the parallel training uses joblib through multiple CPUs
        self.x, self.y = x, y
        self.training_parallel = training_parallel
        self.parallel_config = parallel_config
        self.device = device
        self.trained = False
        self.loaded_from_other_sources = False

        # Set up the random seed for ensemble generation
        random.seed(ens_seed)

        # Get the data dimensions
        assert self.x.shape[0] == self.y.shape[0]
        self.Ns, self.Nx, self.Ny = x.shape[0], x.shape[1], y.shape[1]
    
        # Get model configs
        self.model_type = model_type
        self.ensemble_type = ensemble_type
        self.model_hp_choices = model_hp_choices
        self.model_hp_fixed = model_hp_fixed
        self.optax_hp_choices = optax_hp_choices
        self.optax_hp_fixed = optax_hp_fixed
        self.dl_hp_choices = dl_hp_choices
        self.dl_hp_fixed = dl_hp_fixed
        self.n_model_init = n_model
        self._get_model_configs()

    @property
    def n_model(self):
        return len(self._model_configs)

    @property
    def model_configs(self):
        return self._model_configs

    @property
    def optax_configs(self):
        return self._optax_configs

    @property
    def dl_configs(self):
        return self._dl_configs

    @property
    def model_ens(self):
        if self.trained:
            return self._model_ens
        else:
            print("Mapping has not been trained yet.")

    @property
    def loss_train_ens(self):
        if self.trained:
            return self._loss_train_ens
        else:
            print("Mapping has not been trained yet.")

    @property
    def loss_test_ens(self):
        if self.trained:
            return self._loss_test_ens
        else:
            print("Mapping has not been trained yet.")

    def _get_model_configs(self):
        # Check key configs
        # TODO: A naming convention should be implemented in KIM.
        # e.g., the input and output parameters used in the DNN models.
        # Numbers of model inputs and outputs should be fixed
        self.model_hp_fixed['in_size'] = self.Nx
        self.model_hp_fixed['out_size'] = self.Ny

        if 'optimizer_type' not in self.optax_hp_fixed and \
            'optimizer_type' not in self.optax_hp_choices:
            self.optax_hp_fixed['optimizer_type'] = 'Adam'
        if 'nsteps' not in self.optax_hp_fixed and \
            'nsteps' not in self.optax_hp_choices:
            self.optax_hp_fixed['nsteps'] = 100
        if 'loss_func' not in self.optax_hp_fixed and \
            'loss_func' not in self.optax_hp_choices:
            self.optax_hp_fixed['loss_func'] = loss_mse

        if 'batch_size' not in self.dl_hp_fixed and \
            'batch_size' not in self.dl_hp_choices:
            self.dl_hp_fixed['batch_size'] = 32
        if 'num_train_sample' not in self.dl_hp_fixed and \
            'num_train_sample' not in self.dl_hp_choices:
            self.dl_hp_fixed['num_train_sample'] = self.Ns

        # Generate ensemble configurations
        n_model, model_configs, optax_configs, dl_configs = generate_ensemble_configs(
            self.model_hp_choices, self.model_hp_fixed,
            self.optax_hp_choices, self.optax_hp_fixed,
            self.dl_hp_choices, self.dl_hp_fixed,
            self.n_model_init, self.ensemble_type,
        )
        # self.n_model = n_model
        self._model_configs = model_configs
        self._optax_configs = optax_configs
        self._dl_configs = dl_configs
        # _, self.model_configs = generate_ensemble_configs(
        #     self.model_hp_choices, self.model_hp_fixed, self.n_model, self.ensemble_type
        # )
        # _, self.optax_configs = generate_ensemble_configs(
        #     self.optax_hp_choices, self.optax_hp_fixed, self.n_model, self.ensemble_type
        # )
        # self.n_model, self.dl_configs = generate_ensemble_configs(
        #     self.dl_hp_choices, self.dl_hp_fixed, self.n_model, self.ensemble_type
        # )

    def train(self, verbose: int=0):
        """Mapping training.

        Args:
            verbose (int): the verbosity level (0: normal, 1: debug)
        """
        if self.trained:
            raise Exception("The mapping has already been trained!")

        model_ens, loss_train_ens, loss_test_ens = train_ensemble(
            self.x, self.y, self.model_type, 
            self.model_configs, self.optax_configs, self.dl_configs, 
            self.training_parallel, self.parallel_config, verbose
        )
        self._model_ens = model_ens
        self._loss_train_ens = loss_train_ens
        self._loss_test_ens = loss_test_ens
        self.trained = True

    def predict(self, x: Array):
        """Prediction using the trained mapping.

        Args:
            x (Array): predictors with shape (Ns,...,Nx)

        """
        assert x.shape[-1] == self.Nx  # The same dimension
        assert len(x.shape) >= 2  # At least 2 dimensions with the leading batch dimension

        # Perform predictions on all models
        y_ens = []
        for i in range(self.n_model):
            y = jax.vmap(self.model_ens[i])(x)
            y_ens.append(y)
        y_ens = jnp.array(y_ens)
        
        # Calculate mean
        y_mean = jnp.array(y_ens).mean(axis=0)
        # print(y_mean)

        # Calculate weighted mean based on loss
        loss_ens = self.loss_test_ens if len(self.loss_test_ens)>0 else self.loss_train_ens
        loss = jnp.array([l_all[-1] for l_all in loss_ens])
        weights = 1./loss / jnp.sum(1./loss)
        weighted_product = jax.vmap(lambda w, y: w*y, in_axes=(0,0))
        y_ens_w = weighted_product(weights, y_ens)
        y_mean_w = y_ens_w.sum(axis=0)

        return y_ens, y_mean, y_mean_w

    def save(self, rootpath: PosixPath=Path("./")):
        """Save the trained mapping to specified location, including:
            - trained models
            - model/optax/dl configurations
            - loss values for both training and test sets

        Args:
            rootpath (PosixPath): the root path where mappings will be saved

        """
        if not self.trained:
            raise Exception("Mapping has not been trained yet.")

        if not rootpath.exists():
            rootpath.mkdir(parents=True)

        # Dump overall configurations
        f_overall_configs = rootpath / "configs.pkl"
        overall_configs = {
            "n_model": self.n_model,
            "ensemble_type": self.ensemble_type,
            "training_parallel": self.training_parallel,
            "parallel_config": self.parallel_config,
            "device": self.device,
            "Ns": self.Ns, "Nx": self.Nx, "Ny": self.Ny,
            "model_type": self.model_type,
        }
        with open(f_overall_configs, "wb") as f:
            pickle.dump(overall_configs, f)

        # Dump each model, its configuration, and its loss values
        for i, model in enumerate(self.model_ens):
            model_dir = rootpath / str(i)
            if not model_dir.exists():
                model_dir.mkdir(parents=True)

            f_model = model_dir / "model.eqx"
            f_configs = model_dir / "configs.pkl"
            f_loss = model_dir / "loss.pkl"

            # Save the trained model
            model_configs = self.model_configs[i]
            save_model(f_model, model_configs, self.model_ens[i])

            # Save the configuration
            configs = {
                "model_configs": self.model_configs[i],
                "optax_configs": self.optax_configs[i],
                "dl_configs": self.dl_configs[i],
            }
            with open(f_configs, "wb") as f:
                pickle.dump(configs, f)

            # Save its loss values
            loss = {
                "train": self.loss_train_ens[i],
                "test": self.loss_test_ens[i]
            }
            with open(f_loss, "wb") as f:
                pickle.dump(loss, f)

    def load(self, rootpath: PosixPath=Path("./")):
        """Save the trained mapping to specified location.

        Args:
            rootpath (PosixPath): the root path where mappings will be loaded
        """
        if self.trained:
            raise Exception("Mapping has already been trained.")

        # Load the overall configuration
        f_overall_configs = rootpath / "configs.pkl"
        with open(f_overall_configs, "rb") as f:
            overall_configs = pickle.load(f)
        n_model = overall_configs["n_model"]
        self.ensemble_type = overall_configs["ensemble_type"]
        self.training_parallel = overall_configs["training_parallel"]
        self.parallel_config = overall_configs["parallel_config"]
        self.device = overall_configs["device"]
        Ns, Nx, Ny = overall_configs["Ns"], overall_configs["Nx"], overall_configs["Ny"]
        self.model_type = overall_configs["model_type"]

        assert Nx == self.Nx
        assert Ny == self.Ny
        
        # Load each model, its configuration, and its loss values
        model_ens = []
        model_configs, optax_configs, dl_configs = [], [], []
        loss_train_ens, loss_test_ens = [], []
        for i in range(n_model):
            f_model = rootpath / str(i) / "model.eqx"
            f_configs = rootpath / str(i) / "configs.pkl"
            f_loss = rootpath / str(i) / "loss.pkl"

            # Save the trained model
            m = load_model(f_model, self.model_type)
            model_ens.append(m)

            # Save the configuration
            with open(f_configs, "rb") as f:
                configs = pickle.load(f)
            model_configs.append(configs["model_configs"])
            optax_configs.append(configs["optax_configs"])
            dl_configs.append(configs["dl_configs"])

            # Save its loss values
            with open(f_loss, "rb") as f:
                loss = pickle.load(f)
            loss_train_ens.append(loss["train"])
            loss_test_ens.append(loss["test"])

        self._model_ens = model_ens
        self._model_configs = model_configs
        self._optax_configs = optax_configs
        self._dl_configs = dl_configs
        self._loss_train_ens = loss_train_ens
        self._loss_test_ens = loss_test_ens

        self.loaded_from_other_sources = True
        self.trained = True


def generate_ensemble_configs(
    model_hp_choices: dict, model_hp_fixed: dict,
    optax_hp_choices: dict, optax_hp_fixed: dict,
    dl_hp_choices: dict, dl_hp_fixed: dict,
    n_model: int, ens_type: str,
):
    hp_all = [(model_hp_choices, model_hp_fixed), 
              (optax_hp_choices, optax_hp_fixed),
              (dl_hp_choices, dl_hp_fixed)]

    # Check there is no overlapped keys between hp_choices and hp_fixed
    for hp_choices, hp_fixed  in hp_all: 
        hp_keys1 = list(hp_choices.keys())
        hp_keys2 = list(hp_fixed.keys())
        assert all(i not in hp_keys1 for i in hp_keys2)
        for key, value in hp_choices.items():
            assert isinstance(value, list)
        for key, value in hp_fixed.items():
            assert isinstance(value, float) | isinstance(value, int) | \
                isinstance(value, str) | callable(value)

    # Generate the ensemble configs
    model_configs, optax_configs, dl_configs = [], [], []
    if ens_type == 'single':
        n_model = 1
        model_configs = [model_hp_fixed]
        optax_configs = [optax_hp_fixed]
        dl_configs = [dl_hp_fixed]

    elif ens_type == 'ens_random':
        # Get the configurations for each ensemble member
        for i in range(n_model):
            config_three = []
            for hp_choices, hp_fixed in hp_all:
                config = {}
                # Fixed configurations
                for key, value in hp_fixed.items():
                    config[key] = value
                # Tuned configurations
                for key, choices in hp_choices.items():
                    value = random.sample(choices, 1)[0]
                    config[key] = value
                config_three.append(config)
            model_configs.append(config_three[0])
            optax_configs.append(config_three[1])
            dl_configs.append(config_three[2])

    elif ens_type == 'ens_grid':
        # Get all the combinations of tuned configurations
        hp_choices_three = {
            **model_hp_choices, **optax_hp_choices, **dl_hp_choices
        }
        keys_c, options_c = zip(*hp_choices_three.items())
        combinations = list(itertools.product(*options_c))
        n_model = len(combinations)
        # Get the configurations for each ensemble member
        for i in range(n_model):
            config_three = []
            tuned_config = dict(zip(keys_c, combinations[i]))
            for hp_choices, hp_fixed in hp_all:
                config = {}
                # Fixed configurations
                for key, value in hp_fixed.items():
                    config[key] = value
                # Tuned configurations
                for key, choices in hp_choices.items():
                    config[key] = tuned_config[key]
                config_three.append(config)
            model_configs.append(config_three[0])
            optax_configs.append(config_three[1])
            dl_configs.append(config_three[2])

    else:
        raise Exception('Unknown ensemble type %s' % ens_type)
    
    return n_model, model_configs, optax_configs, dl_configs

def save_model(filename, hyperparams, model):
    with open(filename, "wb") as f:
        hyperparam_str = json.dumps(hyperparams)
        # hyperparam_str = json.dumps({})
        f.write((hyperparam_str + "\n").encode())
        eqx.tree_serialise_leaves(f, model)

def load_model(filename, model_type):
    with open(filename, "rb") as f:
        hyperparams = json.loads(f.readline().decode())
        # print("hyperparameters: ")
        # print(hyperparams)
        # print("Model type: ")
        # print(model_type)
        model = model_type(**hyperparams)
        return eqx.tree_deserialise_leaves(f, model)

# def generate_ensemble_configs(
#     hp_choices: dict, hp_fixed: dict, n_model: int, ens_type: str
# ):
#     # Check there is no overlapped keys between hp_choices and hp_fixed
#     hp_keys1 = list(hp_choices.keys())
#     hp_keys2 = list(hp_fixed.keys())
#     assert all(i not in hp_keys1 for i in hp_keys2)
#     for key, value in hp_choices.items():
#         assert isinstance(value, list)
#     for key, value in hp_fixed.items():
#         assert isinstance(value, float) | isinstance(value, int) | \
#             isinstance(value, str) | callable(value)

#     # Generate the ensemble configs
#     configs = []
#     if ens_type == 'single':
#         n_model = 1
#         configs = [hp_fixed]

#     elif ens_type == 'ens_random':
#         for i in range(n_model):
#             config = {}
#             # Fixed configurations
#             for key, value in hp_fixed.items():
#                 config[key] = value
#             # Tuned configurations
#             for key, choices in hp_choices.items():
#                 value = random.sample(choices, 1)[0]
#                 config[key] = value
#             configs.append(config)

#     elif ens_type == 'ens_grid':
#         # Get all the combinations of tuned configurations
#         keys_c, options_c = zip(*hp_choices.items())
#         combinations = list(itertools.product(*options_c))
#         n_model = len(combinations)
#         for i in range(n_model):
#             # Tuned configurations
#             config = dict(zip(keys_c, combinations[i]))
#             # Fixed configurations
#             for key, value in hp_fixed.items():
#                 config[key] = value
#             configs.append(config)

#     else:
#         raise Exception('Unknown ensemble type %s' % ens_type)
    
#     return n_model, configs