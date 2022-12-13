import sys, os
import torch
import numpy as np
import torchvision

from config import Config
from examples.utils.data_handler import DataHandler

to_PIL = torchvision.transforms.ToPILImage()

# load data
data_handler = DataHandler(Config.NUM_INTERVENTIONS)
data_handler.load(Config.DATA_FILE, device=Config.DEVICE) # TODO: CUDA
#int: The size of the input data.
DATA_SIZE = data_handler.datasets.sizes
num_interventions = len(data_handler.datasets.datasets)
assert Config.NUM_INTERVENTIONS == num_interventions, 'Your number of interventions in your dataset is not what you expected.'

# data loader
train_loader = torch.utils.data.DataLoader(data_handler.datasets,
                                           batch_size=Config.BATCH_SIZE, 
                                           shuffle=True,
                                           num_workers=0
                                        #    pin_memory=True # TODO: CUDA
                                          )

counter = 0
for l, (data_multi) in enumerate(train_loader):
    for i, data in enumerate(data_multi):
        if i==3:
            counter += 1
            img = to_PIL(torch.reshape(data[0][0], (1,64,64)))
            # img.show()
        if counter > 10:
            break
    if counter > 10:
        break

HYPOTHESIS = 2
data = torch.load('examples/data/' + Config.DATA_FILE + '_hypothesis_' + str(HYPOTHESIS) + '.pth')
# for i in range(NUM_HYPOTHESIS):
    # data = torch.load('../examples/data/' + Config.DATA_FILE + '_hypothesis_' + str(i) + '.pth')
counter = 0
for d in data:
    img = to_PIL(torch.reshape(d, (1,64,64)))
    img.show()
    counter += 1
    if counter > 10:
        break
