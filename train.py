import torch
from torch import nn, optim
from torch.autograd import Variable
from torch.utils import data

import time
from model import VoxelNet
from loss import VoxelNetLoss
from config import config as cfg
from data.dataset import KittiDataset
import pathlib, os
import numpy as np

PATH = "./model"

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight.data)
        m.bias.data.zero_()

def detection_collate(batch):
  voxel_features = []
  voxel_coords = []
  pos_equal_one = []
  neg_equal_one = []
  targets = []
  images = []
  calibs = []
  ids = []

  for i, sample in enumerate(batch):
    voxel_features.append(sample['voxel_features'])
    voxel_coords.append(
        np.pad(sample['voxel_coords'], ((0,0), (1,0)), mode='constant',
               constant_values=i
        )
    )
    pos_equal_one.append(sample['pos_equal_one'])
    neg_equal_one.append(sample['neg_equal_one'])
    targets.append(sample['target'])
    images.append(sample['rgb'])
    calibs.append(sample['calib'])
    ids.append(sample['file_id'])

    return np.concatenate(voxel_features), np.concatenate(voxel_coords), \
           np.array(pos_equal_one), np.array(neg_equal_one), \
           np.array(targets), images, calibs, ids

# model
net = VoxelNet()
device = torch.device("cuda")

print("\n## Hardware type: {}\n".format(device.type))
net.to(device)

# training mode
net.train()

# initialization
print('Initializing weights...')
net.apply(weights_init)

# define optimizer
optimizer = optim.SGD(net.parameters(), lr=0.01)

# loss function
loss = VoxelNetLoss(alpha=1.5, beta=1)



# dataset
print("## data path:", pathlib.Path(os.getcwd()).absolute()/"data/training")
dataset=KittiDataset(pathlib.Path(os.getcwd()).absolute()/"data/training", cfg)
data_loader = data.DataLoader(dataset, batch_size=cfg.N, shuffle=True, 
                              num_workers=2, collate_fn=detection_collate,
                              pin_memory=False)
# training process
batch_iterator = None
epoch_size = len(dataset) // cfg.N
print('## epoch size', epoch_size)

for iteration in range(10000):
  if (not batch_iterator) or (iteration % epoch_size == 0):
    batch_iterator = iter(data_loader)

  voxel_features, voxel_coords, pos_equal_one, neg_equal_one, targets, images, calibs, ids= next(batch_iterator)
  
  voxel_features = Variable(torch.cuda.FloatTensor(voxel_features))
  pos_equal_one = Variable(torch.cuda.FloatTensor(pos_equal_one))
  neg_equal_one = Variable(torch.cuda.FloatTensor(neg_equal_one))
  targets = Variable(torch.cuda.FloatTensor(targets))

  optimizer.zero_grad()

  t0 = time.time()
  probablitity_score_map, regression_map = net(voxel_features, voxel_coords)

  # calculate loss
  conf_loss, regression_loss = loss(regression_map, probablitity_score_map, pos_equal_one, neg_equal_one, targets)
  loss_sum = conf_loss + regression_loss
  # backward
  loss_sum.backward()
  optimizer.step()

  t1 = time.time()
  print('Timer: %.4f sec.' % (t1 - t0))
  # print(loss_sum)
  # print(conf_loss)
  print(regression_loss)
  print('iter ' + repr(iteration) + ' || Loss: %.4f || Conf Loss: %.4f || Regression Loss: %.4f' % \
        (loss_sum.item(), conf_loss.item(), regression_loss.item()))

torch.save(net, PATH)

