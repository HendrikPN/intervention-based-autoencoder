import torch

class Config:
    ####################################
    # Data configurations
    ####################################

    DATA_FILE = 'dsprites' # root name of the data file (not including labels and file type)

    NUM_INTERVENTIONS = 4 # number of total interventions

    DEVICE = "cuda" # device to run it on

    ####################################
    # Model configurations
    ####################################

    LOAD_MODEL = False # whether to load a model

    LOAD_MODEL_FILE = 'iae' # root file name for the model

    LOAD_OPTIM = False # whether to load an optimizer

    LOAD_OPTIM_FILE = 'iae_optimizer' # root file name for the optimizer

    ENC_DIM = [512, 256] # dimensions of the fully-connected layers of the encoder

    DEC_DIM = [256, 512] # dimensions of the fully-connected layers of all decoders

    LATENT_DIM = 3 # number of neurons in the latent space 

    LEARNING_RATE = 0.0006#0.0008 # 0.001 # learning rate for training

    LOSS = torch.nn.MSELoss() #torch.nn.MSELoss()# # loss function

    ####################################
    # Convolutional model configurations
    ####################################

    CONV = False

    ENC_DIM = [512, 256] # dimensions of the fully-connected layers of the encoder

    ENC_CHANNELS = [5, 4]

    ENC_KERNELS = [4, 4]

    DEC_DIM = [256, 512] # dimensions of the fully-connected layers of all decoders

    DEC_CHANNELS = [32, 1]

    DEC_KERNELS = [12, 53]

    ####################################
    # Training configurations
    ####################################

    NUM_EPOCHS = 500 # number of epochs

    BATCH_SIZE = 300 # training batch size

    DISCOUNTS_REC_LOSS = [1.,1.,0.58,1.0]#[1. for i in range(NUM_INTERVENTIONS)] #[1.,1.,0.55,1.0] # discount for the recreation loss of each intervention

    DISCOUNT_MIN_LOSS = 0.#0.003 # discount for the minimization loss

    DISCOUNT_SEL_LOSS = 0.003#0.004#0.0034#0.0096 # discount for the disentangling (or selection) loss

    ####################################
    # Output configurations
    ####################################

    SAVE_LOSS = True # whether to save all losses

    LOSS_FILE = 'results_loss' # root file name for the losses

    SAVE_FILTERS = True # whether to save all filters

    MIN_FILE = 'results_min' # root file name for the minimization filters

    SEL_FILE = 'results_sel' # root file name for the selection filters

    SAVE_MODEL = True # whether to save the model and optimizer

    SAVE_MODEL_FILE = 'iae' # root file name for the model

    SAVE_OPTIM_FILE = 'iae_optimizer' # root file name for the optimizer

    LOG = True # whether to log the progress in the terminal
