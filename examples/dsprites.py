import torch
from torch.utils.data import TensorDataset
import numpy as np
import math
from scipy.stats import unitary_group, ortho_group
from itertools import permutations

from utils.data_handler import DataHandler

import disentanglement_lib as dl
from disentanglement_lib.data.ground_truth import dsprites

class DSprites():
    """
    Creates multiple datasets with varying factors of variation for the standard 
    representation learning dataset `DSprites` from "beta-VAE: Learning Basic 
    Visual Concepts with a Constrained Variational Framework". This can be 
    downloaded from https://github.com/deepmind/dsprites-dataset.

    This program assumes a folder ./dsprites with the corresponding dataset.
    This uses the DisentanglementLib under the Apache 2.0 licence.

    The ground-truth factors of variation are (in the default setting):
    0 - shape (3 different values) in (0,1,2)
    1 - scale (6 different values) in [0.5, 1]
    2 - orientation (40 different values) in [0, 2 pi]
    3 - position x (32 different values) in [0, 1]
    4 - position y (32 different values) in [0, 1]
    
    (There is a first factor which has only one value and is ignored here.)

    Here we create separate datasets where the factors of variation are:

    0 - (0,2,3)
    1 - (1,2,3)
    2 - (1,3,4)
    3 - (0,1,2,3,4)

    The other latent variables remain fixed. This choice is such that the 
    maximum dataset sizes are comparable.

    Args:
        *kwargs:
            num_datapoints (int): The size of each individual dataset.

    """
    def __init__(self, num_datapoints=50000):
        self.num_datapoints = num_datapoints
        self.fixed_factors = np.repeat([[0, 0, 0, 0, 0, 0]], self.num_datapoints, axis=0) # not including the 0 at start
        self.dlib_data = dsprites.DSprites()
        data_list = self.create_dataset()
        self.data = DataHandler(4, datasets=data_list)

    def create_dataset(self):
        """
        Creates the datasets for training from `disentanglement_lib`.

        Returns:
            data_list (list): List of `torch.TensorDataset`.
        """
        data_list = []
        num_interventions = 3
        random_state = np.random.RandomState()

        # intervened data
        # for i in range(num_interventions):
            # latent_factors = self.dlib_data.state_space.sample_latent_factors(self.num_datapoints, random_state)
            # latent_factors[:, :1+i] = self.fixed_factors[:, :1+i]
            # latent_factors[:, i+4:] = self.fixed_factors[:, i+4:]
            # imgs = self.dlib_data.sample_observations_from_factors(latent_factors,random_state)

        fixed_indices = [[0,2,5],[0,1,5],[0,1,3]] # make sure sizes of datasets are comparable
        for i_list in fixed_indices:
            latent_factors = self.dlib_data.state_space.sample_latent_factors(self.num_datapoints, random_state)
            latent_factors[:, i_list] = self.fixed_factors[:, i_list]
            imgs = self.dlib_data.sample_observations_from_factors(latent_factors,random_state)
            # imgs = imgs.reshape(-1, np.prod(imgs.shape[1:])) # TODO: remove flatten

            data_list.append(TensorDataset(torch.from_numpy(imgs.reshape(len(imgs),1,64,64)))) # TODO: ok to reshape?

        # full data
        latent_factors = self.dlib_data.state_space.sample_latent_factors(self.num_datapoints, random_state)
        imgs = self.dlib_data.sample_observations_from_factors(latent_factors,random_state)
        # imgs = imgs.reshape(-1, np.prod(imgs.shape[1:])) # TODO: remove flatten

        data_list.append(TensorDataset(torch.from_numpy(imgs.swapaxes(1,3).swapaxes(2,3)))) # TODO: NOT ok to reshape: https://stackoverflow.com/questions/53623472/how-do-i-display-a-single-image-in-pytorch/55196345

        return data_list

    def save_hypothesis(self, file_name):
        """
        Creates and saves data for hypothesis testing of a representation.

        Args:
            file_name (str): The name of the file saved in ./data
        """
        fixed_factor = np.array([[0, 0, 0, 0, 0, 0]])
        random_state = np.random.RandomState()
        for index, n in enumerate(self.dlib_data.factor_sizes):
            if index == 0:
                continue
            data = torch.zeros(n, *self.dlib_data.data_shape)
            for value in range(n):
                factor = fixed_factor.copy()
                img = self.dlib_data.sample_observations_from_factors(factor, random_state)
                data[value] = torch.from_numpy(img)
            torch.save(data, "./data/" + file_name + "_" + str(index-1) + ".pth")

if __name__ == "__main__":
    dsprites_data = DSprites()
    dsprites_data.data.save("dsprites")
    dsprites_data.save_hypothesis("dsprites_hypothesis")
