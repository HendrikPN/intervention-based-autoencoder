import torch
import torch.nn as nn
from torch.autograd import Variable

class FilterEncoder(nn.Module):
    """
    A dense neural network which has a low-dimensional output 
    representing latent variables of an abstract representation with adjustable
    dimensionality through :class:`Selection` neurons.

    Args:
        dim_input (int): The size of the input.
        dim_dense (:obj:`list` of :obj:`int`): Number of neurons per layer.
        dim_latent (int): The size of the output.
    """
    def __init__(self, dim_input, dim_dense, dim_latent):
        super(FilterEncoder, self).__init__()
        # Building input layer.
        self.input = nn.Sequential(
                            nn.Linear(dim_input, dim_dense[0]),
                            nn.ELU()
                        )
            
        # Building hidden, dense layers.
        self.encoding = nn.ModuleList()
        for l in range(len(dim_dense)-1):
            modules = []
            modules.append(nn.Linear(dim_dense[l], dim_dense[l+1]))
            if l != len(dim_dense)-2:
                modules.append(nn.ELU())
            self.encoding.append(
                nn.Sequential(*modules)
            )

        # Building dense output/abstraction layer.
        self.abstraction = nn.Linear(dim_dense[-1], dim_latent)

        # Building selection neurons.
        self.selection = Selection(dim_latent)

    def forward(self, x):
        """
        The forward pass through the encoder with selection neurons. 

        Args:
            x (torch.Tensor): The input array.
        
        Returns:
            out (torch.Tensor): The latent representation of the encoder.
        """
        if x.size(0) == 1:
            raise ValueError(f'Input batch size is incorrect: {x.size()}. Requires batch size > 1.')
        # (0) Create random batch for selection neurons.
        rand_batch = torch.randn((x.size(0), self.selection.selectors.size(0)))
        # (i) Input
        x = self.input(x)
        # (ii) Dense
        for dense in self.encoding:
            x = dense(x)
        # (iii) Abstraction
        x = self.abstraction(x)
        # (iv) Pass through selection neurons.
        out = self.selection(x, rand_batch)
        return out

class FilterDecoder(nn.Module):
    """
    A dense neural network which has a low-dimensional input 
    representing latent variables of an abstract representation and producing
    a high-dimensional output.

    Args:
        dim_latent (int): The size of the output.
        dim_dense (:obj:`list` of :obj:`int`): Number of neurons per layer.
        dim_output (int): The size of the input.
    """
    def __init__(self, dim_latent, dim_dense, dim_output):
        super(FilterDecoder, self).__init__()
        # Building input layer.
        self.unabstraction = nn.Sequential(
                             nn.Linear(dim_latent, dim_dense[0]),
                             nn.ELU()
                            )
            
        # Building hidden, dense decoding layers.
        dim_hidden = dim_dense + [dim_output]
        self.decoding = nn.ModuleList()
        for l in range(len(dim_hidden)-1):
            modules = []
            modules.append(nn.Linear(dim_hidden[l], dim_hidden[l+1]))
            if l != len(dim_hidden)-2:
                modules.append(nn.ELU())
            else:
                modules.append(nn.Sigmoid()) # TODO: Remove!
            self.decoding.append(
                nn.Sequential(*modules)
            )

        # Building selection neurons.
        self.selection = Selection(dim_latent)

    def forward(self, x):
        """
        The forward pass through the encoder. 

        Args:
            x (torch.Tensor): The input array.
        
        Returns:
            out (torch.Tensor): The latent representation of the encoder.
        """
        # (0) Create random batch.
        rand_batch = torch.randn((x.size(0), self.selection.selectors.size(0)))
        # (i) Input
        x = self.selection(x, rand_batch)
        x = self.unabstraction(x)
        # (ii) Dense
        for dense in self.decoding:
            x = dense(x)

        return x
    
class iAutoencoder(nn.Module):
    """
    A dense autoencoder consisting of a :class:`FilterEncoder` and :class:`FilterDecoder`.
    
    Args:
        dim_input (int): The size of the input.
        dim_dense_enc (:obj:`list` of :obj:`int`): Sizes of hidden, dense layers for Encoder.
        dim_dense_dec (:obj:`list` of :obj:`int`): Sizes of hidden, dense layers for Decoders.
        dim_latent (int): The size of the latent representation space.
        num_interventions (int): The number of interventions, i.e., decoders.
    """
    def __init__(self, dim_input, dim_dense_enc, dim_dense_dec, dim_latent, num_interventions):
        super(iAutoencoder, self).__init__()

        #:class:`FilterEncoder`: Dense encoder.
        self.encoder = FilterEncoder(dim_input, dim_dense_enc, dim_latent)
        #:class:`nn.ModuleList` of :class:`FilterDecoder`
        self.decoders = nn.ModuleList()
        for i in range(num_interventions):
            self.decoders.append(FilterDecoder(dim_latent, dim_dense_dec, dim_input))
    
    def forward(self, x, intervention):
        """
        The forward pass through the iAE.

        Args:
            x (torch.Tensor): The input array of shape (batch_size, #channel, x-size, y-size).
            intervention (int): The used decoder labeled by its intervention.

        Returns:
            dec_out (torch.Tensor): The decoder output.
        """

        latent = self.encoder(x)
        dec_out = self.decoders[intervention](latent)

        return dec_out

class Selection(nn.Module):
    """
    Selection neurons to sample from a latent representation for a decoder agent.
    An abstract representation :math:`l_i` is disturbed by a value :math:`r_i` sampled from a normal 
    standard distribution which is scaled by the selection neuron :math:`s_i`.

    ..math::
        n_i \sim l_i + \sigma_{l_i} \times \exp(s_i) \times r_i

    where :math:`\sigma_{l_i}` is the standard deviation over the batch. 
    If the selection neuron has a low (i.e. negative) value, the latent variable is passed to the agent. 
    If the selection neuron has a high value (i.e. close to zero), the latent variable is rendered useless to the agent.

    Args:
        num_selectors (int): Number of selection neurons, i.e. latent variables.

        **kwargs:
            init_selectors (float): Initial value for selection neurons. Default: -10.
    """

    def __init__(self, num_selectors, init_selectors=-10.):
        super(Selection, self).__init__()
        select = torch.Tensor([init_selectors for _ in range(num_selectors)])
        # torch.nn.parameter.Parameter: The selection neurons.
        self.selectors = nn.Parameter(select)

    def forward(self, x, rand):
        """
        The forward pass for the selection neurons.

        Args:
            x (torch.Tensor): The input array of shape (batch_size, size_latent).
            rand (torch.Tensor): Random samples from standard normal distribution of size (batch_size, size_latent).

            **kwargs:
                std_dev (:class:`torch.Tensor` or :class:`NoneType`): The standard deviation calculated throughout 
                                                                      episodes. Needs to be specified for prediction. 
                                                                      Default: None.
        
        Returns:
            sample (torch.Tensor): Sample from a distribution around latent variables.
        """
        selectors = self.selectors.expand_as(x)
        std = x.std(dim=0).expand_as(x)
        sample = x + std * torch.exp(selectors) * rand
        return sample
