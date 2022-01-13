import torch
from torch.utils.data import TensorDataset

from . import io

class DataHandler(object):
    """
    Data handler for intervention-based Autoencoder.

    TODO: Always handle TensorDataset

    Args:
        num_interventions (int): Number of interventions.
        *args (torch.Tensor or torch.TensorDataset): The full datasets.
    """
    def __init__(self, num_interventions, datasets=torch.Tensor([])):
        for data in datasets:
            assert type(data) == torch.Tensor or type(data) == TensorDataset, 'Data is not `torch.Tensor` or `TensorDataset`.'
        self.datasets = ConcatDataset(*datasets)
        self.num_interventions = num_interventions

    def load(self, file_name):
        """
        Loads all data files corresponding to different interventions.

        TODO: Always handle TensorDataset

        Args:
            file_name (str): Name of file as which data is to be loaded.
        """
        input_data = []
        for i in range(self.num_interventions):
            data = torch.load(io.data_path + file_name + '_' + str(i) + '.pth')
            if type(data) == torch.Tensor:
                if i == 0:
                    # assuming data has always the same size.
                    input_data = torch.empty((self.num_interventions, *data.size()))
                input_data[i] = data
            elif type(data) == TensorDataset:
                input_data.append(data)
        self.datasets = ConcatDataset(*input_data)
    
    def save(self, file_name):
        """
        Saves all data files corresponding to different interventions separately.

        Args:
            file_name (str): Name of file as which data is to be saved.
        """
        assert self.num_interventions == len(self.datasets.datasets), 'Data does not match number of interventions.'
        for i in range(self.num_interventions):
            torch.save(self.datasets.datasets[i], io.data_path + file_name + '_' + str(i) + '.pth')

class ConcatDataset(torch.utils.data.Dataset):
    """
    Concatenated datasets for datasets corresponding to different interventions.
    Assumes one-dimensional arrays as data.

    TODO: Always handle TensorDataset    

    Args:
        datasets (torch.Tensor or torch.TensorDataset): All datasets for different interventions.
    """
    def __init__(self, *datasets):
        self.datasets = datasets

        # `list` of `int`: shape of the data, assuming one dimensional arrays.
        # TODO: Always handle TensorDataset (this is not needed...)
        self.sizes = [data.size()[1] if type(data)==torch.Tensor else None for data in datasets]

    def __getitem__(self, i):
        return tuple(d[i] for d in self.datasets)

    def __len__(self):
        return min(len(d) for d in self.datasets)
