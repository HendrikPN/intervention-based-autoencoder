import torch
import numpy as np
from scipy.stats import unitary_group
from functools import reduce

from utils.data_handler import DataHandler

class StateTomography(object):
    """
    Creates various datasets corresponding to local and global measurements on
    random mixed multi-qubit quantum states.

    Each local measurements and the global measurement are understood as 
    different interventions to the observation.

    Args:
        num_qubits (int): Number of qubits.
        num_measure (int): Number of different measurements performed on each qubit.
        num_datapoints (int): Number of datapoints collected for each measurement set.
    """
    def __init__(self, num_qubits=2, num_measure=75, num_datapoints=100000):
        self.num_qubits = num_qubits
        self.num_measure = num_measure
        self.num_datapoints = num_datapoints
        self.measurements = self._create_measurements()
        raw_data, self.true_data = self.create_dataset()
        self.data = DataHandler(self.num_qubits+1, datasets=raw_data)

    def create_dataset(self):
        """
        Creates `num_qubits + 1` different datasets of size 
        `(num_datapoints, num_measure * (num_qubits+1))` where each datapoint of
        the ith dataset (i<`num_qubits`) corresponds to `num_measure` local 
        measurement probabilities on the ith qubit and 0th everywhere else. If 
        i=`num_qubit` each datapoint corresponds to a collection of all 
        `num_measure * (num_qubits)` measurement probabilities and one set of 
        `num_measure` global measurements.

        Returns:
            data (torch.Tensor): A collection of all datasets.
            true_data (torch.Tensor): The underlying state for each datapoint.
        """
        # each intervention (local vs. global projection) gets its own dataset.
        data = torch.empty((self.num_qubits + 1, self.num_datapoints, self.num_measure * (self.num_qubits + 1)))
        # for each datapoint there is one true underlying state
        true_data = np.empty((self.num_datapoints, 2**self.num_qubits, 2**self.num_qubits), dtype=complex)
        for l in range(self.num_datapoints):
            # create random mixed state
            rho = self._random_mixed_state()
            # get datapoint
            datapoint = self._create_datapoint(rho)

            # restructure datapoint in accordance to dataset structure # TODO: improve
            global_measure = torch.zeros(self.num_measure * (self.num_qubits+1))
            for i in range(self.num_qubits):
                local_measure = torch.zeros(self.num_measure * (self.num_qubits+1))
                local_measure[self.num_measure * i:self.num_measure * (i+1)] = datapoint[i]
                global_measure[self.num_measure * i:self.num_measure * (i+1)] = datapoint[i]
                data[i, l] = local_measure
            
            global_measure[self.num_measure * self.num_qubits:] = datapoint[self.num_qubits]
            data[self.num_qubits, l] = global_measure

            # collect true data
            true_data[l] = rho
        
        return data, true_data

    def save_truth(self, file_name):
        """
        Saves the true state underlying the data.
        """
        torch.save(torch.from_numpy(self.true_data), 'data/' + file_name + '.pth')

    def save_measurements(self, file_name):
        """
        Saves the measurement bases that were used to generate the data.
        """
        torch.save(torch.from_numpy(self.measurements), 'data/' + file_name + '.pth')
    
    def save_hypothesis(self, file_name):
        """
        Creates and saves data for hypothesis testing of a representation.

        For each qubit we generate a density matrix

        ..math::
            \rho = \frac{1}{2}(I + \vec{a}\vec{\sigma})

        and set :math:`\vec{a}=(x,0,0),(0,y,0),(0,0,z)` where we vary x,y,z
        between -1 and 1 in 0.2 steps.
        """
        # pauli matrices for Bloch representation
        pauli_mats = [np.array([[0,1],[1,0]], dtype=complex), # pauli-x
                      np.array([[0,-1.j],[1.j,0]], dtype=complex), # pauli-y
                      np.array([[1,0],[0,-1]], dtype=complex) # pauli-z
                     ]
        axis = 3 # x,y,z
        num_datapoint = 11 # -1., -0.8, ..., 0.8, 1.
        data = torch.empty((self.num_qubits, axis, num_datapoint, self.num_measure * (self.num_qubits + 1)))
        # generate data: for each qubit, one density matrix with variable amplitudes of one pauli matrix in Bloch rep
        for q in range(self.num_qubits):
            for axis, pauli in enumerate(pauli_mats):
                for index, var in enumerate(np.arange(-1., 1.2, 0.2)):
                    # create test density matrix
                    rho = [1/2*np.eye(2) for q in range(self.num_qubits)]
                    local_rho = 1/2* (np.eye(2) + var * pauli)
                    rho[q] = local_rho
                    rho = reduce(np.kron, rho)

                    # create datapoint
                    datapoint = self._create_datapoint(rho) #TODO: NO?
                    # restructure datapoint
                    local_measure = torch.zeros(self.num_measure * (self.num_qubits+1))
                    local_measure[self.num_measure * q:self.num_measure * (q+1)] = datapoint[q]
                    #add data
                    data[q,axis,index] = local_measure

        # save data
        torch.save(data, 'data/' + file_name + '.pth')

    # ----------------- helper methods -----------------------------------------

    def _create_datapoint(self, rho):
        """
        Creates one datapoint consisting of measurement results for some 
        pre-specified bases on a input density matrix. 

        For each qubit there is a set of `num_measure` different local 
        measurement bases. In addition, there is one set of `num_measure` 
        arbitrary global measurement bases.

        Args:
            rho (np.ndarray): Input density matrix.

        Returns:
            datapoint (torch.Tensor): Datapoint of size (#qubits+1,#measurements)
        """
        # empty datapoint. For each set of local and global projection, we collect a datapoint.
        datapoint = torch.empty((self.num_qubits + 1, self.num_measure))
        
        # collect local measurements
        for i in range(self.num_qubits):
            for j in range(self.num_measure):
                datapoint[i,j] = self._measure(rho, self.measurements[i,j])
        
        # collect collective measurements
        for j in range(self.num_measure):
            datapoint[self.num_qubits,j] = self._measure(rho, self.measurements[self.num_qubits,j])

        return datapoint

    def _measure(self, rho, projection):
        """
        Measures the density matrix `rho` with `projection`.

        Returns: 
            result (float): Probabiltiy to measure `projection`.
        """
        result = np.trace(rho@projection)
        assert(np.imag(result) < 1e-3)
        return np.real(result)

    def _create_measurements(self):
        """
        Creates the set of all measurements. The first `num_qubits` elements are
        local measurements.

        Returns:
            measurements (np.ndarray): The projections that form the measurements.
        """
        measurements = np.empty((self.num_qubits + 1, self.num_measure, 2**self.num_qubits, 2**self.num_qubits),dtype=complex)

        # collect all local measurements
        for i in range(self.num_qubits):
            for j in range(self.num_measure):
                proj = [np.eye(2) for k in range(self.num_qubits)]
                state = unitary_group.rvs(2)[:, 0]#.conj().T#TODO what?
                proj[i] = np.outer(state.conj(), state)
                measurements[i,j] = reduce(np.kron, proj)
        
        # collect global measurements
        for j in range(self.num_measure):
            state = unitary_group.rvs(2**self.num_qubits)[:, 0]
            proj = np.outer(state.conj(), state)
            measurements[self.num_qubits, j] = proj
        
        return measurements

    def _random_mixed_state(self):
        """
        Draws a random mixed state as the partial trace of a pure state.

        Returns:
            mixed_rho (np.ndarray): The random mixed state.
        """
        pure_state = unitary_group.rvs(2 * 2**self.num_qubits)[:, 0]
        pure_rho = np.outer(pure_state.conj(), pure_state)
        mixed_rho = np.trace(
                             pure_rho.reshape(
                                              2**self.num_qubits, 
                                              2, 
                                              2**self.num_qubits, 
                                              2), 
                             axis1=1, 
                             axis2=3
                            )
        return mixed_rho

if __name__=="__main__":
    # Create data
    tomography = StateTomography(num_qubits=2, num_measure=75, num_datapoints=1000)
    # Saves data
    tomography.data.save('Xtomography')
    # Saves the mixed states that yielded the data
    tomography.save_truth('Xtomography_truth')
    # Saves the set of measurement bases chosen at the beginning
    tomography.save_measurements('Xtomography_measurements')
    # Saves the paramterized data for hypothesis testing.
    tomography.save_hypothesis('Xtomography_hypothesis')
