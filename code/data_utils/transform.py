
#import scipy, scipy.ndimage
#import sparseconvnet as scn
from torch.nn.modules.module import Module
import torch
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
from torch.autograd import Function

class PcNormalizeFunction(Function):
    @staticmethod
    def forward(ctx, points):
        batch_size, npoint, _ = points.size()
        rem = torch.zeros(batch_size, 4, dtype=torch.float32)
        for i in range(batch_size):
            pc = points[i, :, :3]
            centroid = pc.mean(axis=0)
            pc = pc - centroid
            m = torch.max(torch.sqrt(torch.sum(pc ** 2, axis=1)))
            pc = pc / m
            points[i,:,:3] = pc
            rem[i, :3] = centroid
            rem[i, 3] = m
        return {'x': points, 'norm': rem} 
    def backward(ctx, gradOutput): 
        return None

class PcNormalize(Module):
    def __init__(self):
        super(PcNormalize, self).__init__()
    
    def forward(self, points):
        result = PcNormalizeFunction.apply(points)
        return result

class TransformFunction(Function):
    @staticmethod
    def forward(ctx, points, feas, masks, scale=20, full_scale=4096):
        batch_size, npoints, channel = points.size()
        points = scale * points[:,:,:3]
        locs, feats = [], []
        offsets = torch.zeros(batch_size, 3)
        for i in range(batch_size):
            a = points[i, :, :]
            b = feas[i, :, :]
            m = a.min(0)#[min_x, min_y, min_z]
            M = a.max(0)#[max_x, max_y, max_z]
            q = M - m
            #if range M-m > full_scale; offset = -m-random_crop; if M-m < full_scale, offset = -m + random_crop
            #voxel range [0, 4095]. Centering the points.
            offset = -m + np.clip(full_scale - M + m - 0.001, 0, None) * np.random.rand(3) + np.clip(full_scale - M + m + 0.001, None, 0)*np.random.rand(3)
            a += offset
            idxs = (a.min(1) >= 0) * (a.max(1) < full_scale) * (masks[i * npoints: (i + 1) * npoints] > 0) # remove outliers if any of the [x, y, z] out of [0, full_scale]
            a = a[idxs]
            b = b[idxs]
            masks[i * npoints: (i + 1) * npoints] *= idxs
#            temp[idxs]
#            masks
            a = torch.from_numpy(a).long()
            locs.append(torch.cat([a, torch.LongTensor(a.shape[0], 1).fill_(i)], 1)) #[x, y, z, idx of frames]
            feats.append(torch.from_numpy(b) + torch.randn(3) * 0.1)
            offsets[i, :, :] = offset
#            labels.append(torch.from_numpy(c))
        locs = torch.cat(locs, 0)#list to tensor [x, y, z, idx]
        feats = torch.cat(feats, 0)
#        labels = torch.cat(labels, 0)
        return {'x': [locs, feats], 'offset': offsets}
            
    @staticmethod
    def backward(ctx, gradOutput): 
        return None, None, None, None

#from pointnet [x y z] to sparse conv [x, y, z]
# xyz = points[i, :, :3] * norm[i, 3] + norm[i, :3]
# xyz = xyz * scale + offsets[i,:]
 

class Transform(Module):
    def __init__(self, scale=20, full_scale=4096):
        super(Transform, self).__init__()
        self.scale = scale
        self.full_scale = full_scale
    
    def forward(self, points, masks):
        x = TransformFunction.apply(points, masks, self.scale, self.full_scale)
        return x



