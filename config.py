import torch

class Config:
    ####################################
    # Data configurations
    ####################################

    DATA_FILE = 'qcausal' # root name of the data file (not including labels and file type)

    NUM_INTERVENTIONS = 3 # number of total interventions

    ####################################
    # Model configurations
    ####################################

    LOAD_MODEL = False # whether to load a model

    LOAD_MODEL_FILE = 'iae' # root file name for the model

    LOAD_OPTIM = False # whether to load an optimizer

    LOAD_OPTIM_FILE = 'iae_optimizer' # root file name for the optimizer

    ENC_DIM = [512, 256] # dimensions of the fully-connected layers of the encoder

    DEC_DIM = [100, 100, 256] # dimensions of the fully-connected layers of all decoders

    LATENT_DIM = 8 # number of neurons in the latent space 

    LEARNING_RATE = 0.0006 # learning rate for training

    LOSS = torch.nn.MSELoss() # loss function

    ####################################
    # Training configurations
    ####################################

    NUM_EPOCHS = 40 # number of epochs

    BATCH_SIZE = 100 # training batch size

    DISCOUNTS_REC_LOSS = [1. for i in range(NUM_INTERVENTIONS)] # discount for the recreation loss of each intervention

    DISCOUNT_MIN_LOSS = 0.00001 # discount for the minimization loss

    DISCOUNT_SEL_LOSS = 0.00005 # discount for the disentangling (or selection) loss

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
