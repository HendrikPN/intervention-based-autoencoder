import sys, os
import torch
import numpy as np

from config import Config
from examples.utils.data_handler import DataHandler
from models.iAutoencoder import StrFDecoder

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
# DATA_SIZE = data_handler.datasets.sizes[0]
num_interventions = len(data_handler.datasets.datasets)
num_causal = len(Config.DEC_DIM)
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
sfdec = StrFDecoder(Config.LATENT_DIM, Config.DEC_DIM, 2)
# optimizer
optimizer = torch.optim.Adam(sfdec.parameters(), lr=Config.LEARNING_RATE)#, amsgrad=True, weight_decay=1e-5)
####################################
# Run
####################################

# result data
data_loss = []
data_loss_rec = []
data_sel = []
# run iAE
for e in range(Config.NUM_EPOCHS):
    for l, (data_multi) in enumerate(train_loader):
        optimizer.zero_grad()
        loss_sum = 0.
        for i, data in enumerate(data_multi):
            # skip different interventions (only one decoder)
            if i != 1:
                continue
            # separate input/output data
            input_data = data[0]
            output_data = data[1]
            # output of sfDec
            out = sfdec(input_data)
            # loss for input recreation
            loss_rec = Config.LOSS(out, output_data) * Config.DISCOUNTS_REC_LOSS[i]
            # print(loss_rec)
            # loss for selection minimization under interventions and causal order
            loss_sel = 0
            for selector in sfdec.selection:
                loss_sel += -selector.selectors.sum() * 1./Config.LATENT_DIM * 1./num_causal
            # print(loss_sel)
            # combined loss
            loss = (loss_rec + \
                loss_sel * Config.DISCOUNT_SEL_LOSS) #* 1./Config.NUM_INTERVENTIONS
            loss.backward()
            loss_sum += loss.detach()
            # collect decoder specific data every 10 steps
            if l%10 == 0:
                data_loss_rec += [loss_rec.detach().item()]
                data_sel += [sfdec.selection[k].selectors.data.tolist() for k in range(num_causal)]
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
        print(f'Current recreation losses: {data_loss_rec[-1]}')
        # for j in reversed(range(1, num_interventions+1)):
        #     print(f'Current local filter {num_interventions-j}: {data_sel[-j]}')
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
                label = 0#i%num_interventions
                loss_rec_file.write("[Intervention: %s], %s\n" % (label, item2))
            data_loss = []
            data_loss_rec = []
    # save filters
    if Config.SAVE_FILTERS:
        with open('results/' + Config.SEL_FILE + '.txt', 'a') as sel_file:
            for i, item1 in enumerate(data_sel):
                label = i%num_causal#num_interventions
                sel_file.write("[Causal Pos: %s], %s\n" % (label, item1))
            data_sel = []
            data_min = []

####################################
# Save models
####################################

if Config.SAVE_MODEL:
    torch.save(sfdec.state_dict(), 'results/models/' + Config.SAVE_MODEL_FILE + '.pth')
    torch.save(optimizer.state_dict(), 'results/models/' + Config.SAVE_OPTIM_FILE + '.pth')
