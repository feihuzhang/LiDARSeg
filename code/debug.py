import os
import torch

import sys

import provider
import numpy as np

from torch.autograd import Variable
import math
dimension = 3
scale = 32
full_scale = 4*scale
spatialSize = [full_scale, full_scale, full_scale]

locations = np.random.randn(1, 20, 3)
features = np.random.randn(1, 20, 3)
targets = np.ones([1, 20])
masks = np.ones([1, 20])
batch = provider.transform_for_sparse(locations, features, targets, masks, scale, full_scale=spatialSize)
print(batch['x'])
print(locations)
