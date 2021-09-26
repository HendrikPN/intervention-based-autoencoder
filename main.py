import sys, os
import torch
import numpy as np

from config import Config
from examples.utils.data_handler import DataHandler
from models.iAutoencoder import iAutoencoder

####################################
# Delete previous save files
####################################
if Config.SAVE_FILTERS:
    if os.path.exists('results/' + Config.SEL_FILE + '.txt'):
        os.remove('results/' + Config.SEL_FILE + '.txt')
    if os.path.exists('results/' + Config.MIN_FILE + '.txt'):
        os.remove('results/' + Config.MIN_FILE + '.txt')
if Config.SAVE_LOSS:
    if os.path.exists('results/' + Config.LOSS_FILE + '.txt'):
        os.remove('results/' + Config.LOSS_FILE + '.txt')
    if os.path.exists('results/' + Config.LOSS_FILE + '_rec' + '.txt'):
        os.remove('results/' + Config.LOSS_FILE + '_rec' + '.txt')

####################################
# Get training data
####################################

# load data
data_handler = DataHandler(Config.NUM_INTERVENTIONS)
data_handler.load(Config.DATA_FILE)
#int: The size of the input data.
DATA_SIZE = data_handler.datasets.sizes[0]
num_interventions = len(data_handler.datasets.datasets)
assert Config.NUM_INTERVENTIONS == num_interventions, 'Your number of interventions in your dataset is not what you expected.'

# data loader
train_loader = torch.utils.data.DataLoader(data_handler.datasets,
                                           batch_size=Config.BATCH_SIZE, 
                                           shuffle=True,
                                           num_workers=1, 
                                           pin_memory=True
                                          )

####################################
# Define model and optimizer
####################################

# intervention-based autoencoder model
iae = iAutoencoder(DATA_SIZE, Config.ENC_DIM, Config.DEC_DIM, Config.LATENT_DIM, Config.NUM_INTERVENTIONS)
# load model if desired
if Config.LOAD_MODEL:
    loaded_dict = torch.load('results/models/' + Config.LOAD_MODEL_FILE + '.pth')
    iae.load_state_dict(loaded_dict)

# optimizer
optimizer = torch.optim.Adam(iae.parameters(), lr=Config.LEARNING_RATE)#, amsgrad=True, weight_decay=1e-5)
# load optimizer if desired
if Config.LOAD_OPTIM:
    loaded_dict = torch.load('results/models/' + Config.LOAD_OPTIM_FILE + '.pth')
    optimizer.load_state_dict(loaded_dict)

####################################
# Run
####################################

# result data
data_loss = []
data_loss_rec = []
data_min = []
data_sel = []

# run iAE
for e in range(Config.NUM_EPOCHS):
    for l, (data_multi) in enumerate(train_loader):
        optimizer.zero_grad()
        loss_sum = 0.
        for i, data in enumerate(data_multi):
            # output of iAE
            out = iae(data, i)
            # loss for input recreation
            loss_rec = Config.LOSS(out, data) * Config.DISCOUNTS_REC_LOSS[i]
            # print(loss_rec)
            # loss for representation minimization
            loss_min = -iae.encoder.selection.selectors.sum() * 1./Config.LATENT_DIM
            # print(loss_min)
            # loss for selection minimization under interventions
            loss_sel = -iae.decoders[i].selection.selectors.sum() * 1./Config.LATENT_DIM
            # print(loss_sel)
            # combined loss
            loss = (loss_rec + \
                loss_min * Config.DISCOUNT_MIN_LOSS + \
                loss_sel * Config.DISCOUNT_SEL_LOSS) * 1./Config.NUM_INTERVENTIONS
            loss.backward()
            loss_sum += loss.detach()
            # collect decoder specific data every 10 steps
            if l%10 == 0:
                data_loss_rec += [loss_rec.detach().item()]
                data_min += [iae.encoder.selection.selectors.data.tolist()]
                data_sel += [iae.decoders[i].selection.selectors.data.tolist()]
        # collect overall loss every ten steps
        if l%10 == 0:
            data_loss += [loss_sum.item()]
        # run gradient
        optimizer.step()

    ####################################
    # Log and save intermediate results
    ####################################
    # log some info
    if Config.LOG and e%1 == 0:
        print(f'End of epoch: {e}')
        print(f'Current total loss: {loss_sum}')
        print(f'Current recreation losses: {[d for d in data_loss_rec[-num_interventions:]]}')
        print(f'Current global filter: {data_min[-1]}')
        for j in reversed(range(1, num_interventions+1)):
            print(f'Current local filter {num_interventions-j}: {data_sel[-j]}')
        sys.stdout.flush()
    # save losses
    if Config.SAVE_LOSS:
        with open('results/' + Config.LOSS_FILE + '.txt', 'a') as loss_file, \
             open('results/' + Config.LOSS_FILE + '_rec' + '.txt', 'a') as loss_rec_file:
            # overall loss
            for item1 in data_loss:
                loss_file.write("%s\n" % item1)
            # recreation loss
            for i, item2 in enumerate(data_loss_rec):
                label = i%num_interventions
                loss_rec_file.write("[Intervention: %s], %s\n" % (label, item2))
            data_loss = []
            data_loss_rec = []
    # save filters
    if Config.SAVE_FILTERS:
        with open('results/' + Config.SEL_FILE + '.txt', 'a') as sel_file, \
             open('results/' + Config.MIN_FILE + '.txt', 'a') as min_file:
            for i, (item1, item2) in enumerate(zip(data_sel, data_min)):
                label = i%num_interventions
                sel_file.write("[Intervention: %s], %s\n" % (label, item1))
                min_file.write("%s\n" % item2)
            data_sel = []
            data_min = []

####################################
# Save models
####################################

if Config.SAVE_MODEL:
    torch.save(iae.state_dict(), 'results/models/' + Config.SAVE_MODEL_FILE + '.pth')
    torch.save(optimizer.state_dict(), 'results/models/' + Config.SAVE_OPTIM_FILE + '.pth')
