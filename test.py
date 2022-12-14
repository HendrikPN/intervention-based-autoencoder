import sys, os
import torch
import torchvision
import numpy as np
from IPython.display import Image
import time

from config import Config
from examples.utils.data_handler import DataHandler


data_handler = DataHandler(Config.NUM_INTERVENTIONS)
data_handler.load(Config.DATA_FILE, device="cpu")

to_pil = torchvision.transforms.ToPILImage()

tensor = data_handler.datasets
print(type(tensor))

train_loader = torch.utils.data.DataLoader(data_handler.datasets,
                                           batch_size=1, 
                                           shuffle=True,
                                           num_workers=0
                                          )
for l, (data_multi) in enumerate(train_loader):
    print(type(data_multi))
    for i, data in enumerate(data_multi):
        print(type(data[0]))


        print(data[0][0].size())
        # print(data[0][0][np.where(data[0][0].numpy() > 0)])
        img = to_pil(data[0][0])

        img.show()

        time.sleep(1)