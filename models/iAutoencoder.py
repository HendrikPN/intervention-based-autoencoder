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
        if x.is_cuda:
            rand_batch = torch.cuda.FloatTensor(x.size(0), self.selection.selectors.size(0)).normal_() # TODO CUDA
        else:
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
        The forward pass through the decoder. 

        Args:
            x (torch.Tensor):The latent representation of the encoder.
        
        Returns:
            out (torch.Tensor): The input array.
        """
        # (0) Create random batch.
        if x.is_cuda:
            rand_batch = torch.cuda.FloatTensor(x.size(0), self.selection.selectors.size(0)).normal_() # TODO CUDA
        else:
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

    def __init__(self, num_selectors, init_selectors=-2.):
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

class FilterConvEncoder(nn.Module):
    """
    A convolutional neural network which has a low-dimensional output 
    representing latent variables of an abstract representation with adjustable
    dimensionality through :class:`Selection` neurons.

    Args:
        dim_img (:obj:`list` of :obj:`int`): The size of the input image.
        dim_channels (:obj:`list` of :obj:`int`): Number of channels per convolutional layer.
        kernel_sizes (:obj:`list` of :obj:`int`): Size of kernel per convolutional layer.
        dim_dense (:obj:`list` of :obj:`int`): Number of neurons per dense layer.
        dim_latent (int): The size of the output.
    """
    def __init__(self, dim_img, dim_channels, kernel_sizes, dim_dense, dim_latent):
        super(FilterConvEncoder, self).__init__()
        
        # Build convolutional layers
        self.convs = nn.ModuleList()
        channels = [dim_img[0]] + dim_channels
        for i in range(len(dim_channels)):
            self.convs.append(
                nn.Sequential(
                    nn.Conv2d(channels[i], channels[i+1], kernel_sizes[i]),
                    nn.ELU(),
                    nn.MaxPool2d(2, 2)
                )
            )
        
        # Build standard FilterEncoder from dense layers
        input_size = self._get_conv_size(dim_img)
        self.filter_encoder = FilterEncoder(input_size, dim_dense, dim_latent)
        # Selection neurons from `FilterEncoder`
        self.selection = self.filter_encoder.selection
    
    def _get_conv_size(self, shape):
        """
        Given the shape of the input image, calculate the size of the output over the convolution.
        The size of a layer is calculated as (n + f + 2p)/s + 1 where n=input size, f=kernel size, 
        p=padding and s=stride.

        Args:
            shape (:obj:`list` of :obj:`int`): The input array of shape (#channels, x-size, y-size).
        """
        x = Variable(torch.rand(1, *shape))
        for conv in self.convs:
            x = conv(x)
        x = x.view(x.size(0), -1)
        conv_size = x.data.size(1)

        return conv_size

    def forward(self, x):
        """
        The forward pass through the convolutional encoder with selection neurons. 

        Args:
            x (torch.Tensor): The input array.
        
        Returns:
            out (torch.Tensor): The latent representation of the encoder.
        """
        # convolutional layers
        for conv in self.convs:
            x = conv(x)
        # flatten output
        x = x.view(x.size(0), -1)
        # dense layers
        x = self.filter_encoder(x)

        return x

class FilterConvDecoder(nn.Module):
    """
    A convolutional neural network which has a low-dimensional input 
    representing latent variables of an abstract representation and producing
    a high-dimensional image output.

    Args:
        dim_latent (int): The size of the input.
        dim_dense (:obj:`list` of :obj:`int`): Number of neurons per dense layer.
        dim_channels (:obj:`list` of :obj:`int`): Number of channels per convolutional layer.
        kernel_sizes (:obj:`list` of :obj:`int`): Size of kernel per convolutional layer.
        dim_img (:obj:`list` of :obj:`int`): The size of the output image.
    """
    def __init__(self, dim_latent, dim_dense, dim_channels, kernel_sizes, dim_img):
        super(FilterConvDecoder, self).__init__()

        # Build standard FilterDecoder
        self.filter_decoder = FilterDecoder(dim_latent, dim_dense[:-1], dim_dense[-1])
        # Selection neurons from `FilterDecoder`
        self.selection = self.filter_decoder.selection

        # Build convolutional layers
        self.convsTransposed = nn.ModuleList()
        channels = [dim_dense[-1]] + dim_channels
        for i in range(len(dim_channels)):
            self.convsTransposed.append(
                nn.Sequential(
                    nn.ConvTranspose2d(channels[i], channels[i+1], kernel_sizes[i]),
                    nn.ELU() if i != len(dim_channels)-1 else nn.Sigmoid()
                )
            )

        # Check output image size
        conv_shape = self._get_conv_shape((1, dim_dense[-1]))
        if not conv_shape == torch.Size([1, *dim_img]):
            raise ValueError(f"""
                             The transposed convolution does not produce an image of size 
                             {(1, *dim_img)}, instead got {(1, *conv_shape)}. 
                             The image size per transposed convolution is calculated 
                             as s*(n-1)+f-2p.
                             """
                            )

    def _get_conv_shape(self, shape):
        """
        Given the shape of the input image, calculate the size of the output over the convolution.
        The size of a layer is calculated as s * (n-1) + f - 2p where n=input size, f=kernel size, 
        p=padding and s=stride.

        Args:
            shape (:obj:`list` of :obj:`int`): The input array of shape (#channels, x-size, y-size).
        """
        x = Variable(torch.rand(*shape, 1, 1))
        for conv in self.convsTransposed:
            x = conv(x)
        conv_shape = x.data.size()
        
        return conv_shape

    def forward(self, x):
        """
        The forward pass through the decoder. 

        Args:
            x (torch.Tensor): The latent representation of the encoder.
        
        Returns:
            out (torch.Tensor): The output image.
        """
        # dense layers
        x = self.filter_decoder(x)
        # unflatten output
        x = x.view(x.size(0), x.size(1), 1, 1)
        # transposed convolutional layers
        for conv in self.convsTransposed:
            x = conv(x)

        return x

class iConvAE(nn.Module):
    """
    A convolutional autoencoder consisting of a :class:`FilterConvEncoder` and :class:`FilterConvDecoder`.
    
    Args:
        dim_img (int): The size of the input.
        dim_latent (int): The size of the latent representation space.
        num_interventions (int): The number of interventions, i.e., decoders.
        dim_dense_enc (:obj:`list` of :obj:`int`): Sizes of hidden, dense layers for Encoder.
        dim_channels_enc (:obj:`list` of :obj:`int`): Sizes of conv channels for Encoder.
        kernel_sizes_enc (:obj:`list` of :obj:`int`): Kernel dimension of conv channels for Encoder.
        dim_dense_dec (:obj:`list` of :obj:`int`): Sizes of hidden, dense layers for Decoders.
        dim_channels_dec (:obj:`list` of :obj:`int`): Sizes of conv channels for Decoders.
        kernel_sizes_dec (:obj:`list` of :obj:`int`): Kernel dimension of conv channels for Decoders.
    """
    def __init__(self, dim_img, dim_latent, num_interventions,
                       dim_dense_enc, dim_channels_enc, kernel_sizes_enc, 
                       dim_dense_dec, dim_channels_dec, kernel_sizes_dec):
        super(iConvAE, self).__init__()

        #:class:`FilterConvEncoder`: Convolutional encoder.
        self.encoder = FilterConvEncoder(dim_img, dim_channels_enc, kernel_sizes_enc, dim_dense_enc, dim_latent)
        #:class:`nn.ModuleList` of :class:`FilterConvDecoder`
        self.decoders = nn.ModuleList()
        for i in range(num_interventions):
            self.decoders.append(FilterConvDecoder(dim_latent, dim_dense_dec, dim_channels_dec, kernel_sizes_dec, dim_img))
    
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
