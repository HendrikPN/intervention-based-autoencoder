import torch
from torch.utils.data import TensorDataset
import numpy as np
import math
from scipy.stats import unitary_group, ortho_group
from itertools import permutations

from utils.data_handler import DataHandler

class QCausality(object):
    """
    Changing causal structure between interventions.
    Data is the latent representation.
    """
    def __init__(self, num_qubits=1, num_parties=2, num_datapoints=100000):
        self.num_qubits = num_qubits
        self.num_datapoints = num_datapoints
        self.num_parties = num_parties

        data_list = self.create_dataset()
        self.data = DataHandler(math.factorial(self.num_parties) + 1, datasets=data_list)

    def create_dataset(self):
        """
        """
        data_list = []
        dim_exp = 2**(2*self.num_qubits)
        # the starting state (all plus)
        state_in = np.ones(2**self.num_qubits) * 1/np.sqrt(2)**self.num_qubits
        # each intervention is a new permutation
        for i, perm in enumerate(permutations(range(self.num_parties))):
            # input
            exp_in = torch.empty(self.num_datapoints, 2 ** (2*self.num_qubits)* self.num_parties)
            # target
            exp_out = torch.empty(self.num_datapoints, 2 ** self.num_qubits)
            # input to decoder is a bunch of flattened matrices
            for d in range(self.num_datapoints):
                mats = []
                for p in range(self.num_parties):
                    ortho = self._random_ortho()
                    mats += [ortho]
                    exp_in[d, p*dim_exp:((1+p)*dim_exp)] = torch.from_numpy(ortho.flatten())
                # target is the matrices applied to the input state
                out = state_in.copy()
                for j in perm:
                    out = mats[j] @ out
                exp_out[d] = torch.from_numpy(out)

            data_list.append(TensorDataset(exp_in, exp_out))

        # add non-causal
        # input
        exp_in = torch.empty(self.num_datapoints, 2 ** (2*self.num_qubits)* self.num_parties)
        # target
        exp_out = torch.empty(self.num_datapoints, 2 ** self.num_qubits)
        for d in range(self.num_datapoints):
            mats = []
            for p in range(self.num_parties):
                ortho = self._random_ortho()
                mats += [ortho]
                exp_in[d, p*dim_exp:((1+p)*dim_exp)] = torch.from_numpy(ortho.flatten())

            mat = np.zeros((2**self.num_qubits, 2**self.num_qubits))
            for perm in permutations(range(self.num_parties)):
                mat_i = np.eye(2**self.num_qubits)
                for j in perm:
                    mat_i = mats[j] @ mat_i
                mat += 1/np.sqrt(math.factorial(self.num_parties)) * mat_i

            out = mat @ state_in.copy()
            exp_out[d] = torch.from_numpy(out)
        
        data_list.append(TensorDataset(exp_in, exp_out))

        return data_list

    def _random_unitary(self):
        """
        Draws a random unitary matrixe.

        Returns:
            unitary (np.ndarray): The random unitary
        """
        unitary = unitary_group.rvs(2**self.num_qubits)

        return unitary

    def _random_ortho(self):
        """
        Draws a random orthogonal matrix.

        Returns:
            ortho (np.ndarray): The random orthogonal matrix
        """
        ortho = ortho_group.rvs(2**self.num_qubits)

        return ortho

if __name__ == "__main__":
    # Create data
    tomography = QCausality(num_qubits=1, num_parties=2, num_datapoints=100000)
    # Saves data
    tomography.data.save('qcausal')