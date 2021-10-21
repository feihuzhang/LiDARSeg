from torch.nn.modules.module import Module
import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
from ..functions import *
import torch.nn.functional as F
from ..functions.fusion import QueryFunction
from ..functions.fusion import MergeFunction






class SparseTransform(Module):
    def __init__(self):
        super(SparseTransform, self).__init__()

    def rotatexy_flip_shift(self, points, angle=30, flip=True, flip_ratio=0.25, var=[0.1, 0.1, 0.05]):
        if flip and np.random.uniform() < flip_ratio:
            points[:,1] = - points[:,1]
        rotation_angle = (np.random.uniform()-0.5) * angle/180. * np.pi
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = torch.Tensor([[cosval, sinval, 0],
                                        [-sinval, cosval, 0],
                                        [0, 0, 1]]).to(points.device)

        points[:,:3] = torch.matmul(points[:,:3], rotation_matrix)
        points[:, 0] += np.random.normal(0, var[0])
        points[:, 1] += np.random.normal(0, var[1])
        points[:, 2] += np.random.normal(0, var[2])
        return points

    def get_feature(self, pc, normalize=True, radius=60):
        points = torch.cuda.FloatTensor(pc.size(0), 7).zero_()
        points[:, 0:3] = pc[:, 0:3]
        if normalize:
            temp = pc[:,0:4]
            points[:, 3:7] = (temp-torch.mean(temp, 0))/torch.std(temp, 0)
 
            return points
 
        points[:, 5:7] = pc[:, 2:4]
        points[:, 3:5] = pc[:, 0:2] / radius  
        return points

    def forward(self, points, targets, masks, scale, full_scale, training=True):
 
        batch_size, npoints, _ = points.size()
#        locations = scale * locations[:,:,:3]
        locs, feats, labels = [], [], []
        offsets = torch.cuda.FloatTensor(batch_size, 3).zero_()
        org_points = torch.cuda.FloatTensor(batch_size, npoints, 7).zero_()
        for i in range(batch_size):
            pc = points[i, :, :]
            if training:
                pc = self.rotatexy_flip_shift(pc)
            pc = self.get_feature(pc)
            org_points[i, :, :] = pc
            

            a = scale * pc[:, :3]
            b = pc[:, 3:]
            c = targets[i, :]
            m, _ = a.min(0)#[min_x, min_y, min_z]
            M, _ = a.max(0)#[max_x, max_y, max_z]
            q = M - m
            if training:
                offset = -m + torch.clamp(full_scale - q - 0.001, 0, None) * torch.rand(3).cuda() + torch.clamp(full_scale - q + 0.001, None, 0)*torch.rand(3).cuda()
                a += offset
                m, _ = a.min(1)
                idxs = (m >= 0) * (a[:, 0] < full_scale[0])* (a[:, 1] < full_scale[1]) * (a[:, 2] < full_scale[2]) * (masks[i, :] > 0)
            else:
                offset = -m + torch.clamp(full_scale - q - 0.001, 0, None)/2 + torch.clamp(full_scale - q + 0.001, None, 0)/2
                a += offset
                idxs = masks[i, :] > 0  
                a[a<0] = 0
                tag = a[:, 0] >= full_scale[0]
                a[tag, 0] = full_scale[0]-1
                tag = a[:, 1] >= full_scale[1]
                a[tag, 1] = full_scale[1]-1
                tag = a[:, 2] >= full_scale[2]
                a[tag, 2] = full_scale[2]-1

            a = a[idxs]
            b = b[idxs]
            masks[i, :] *= idxs
            c = c[idxs]

            a = a.int()
            locs.append(torch.cat([a, torch.cuda.IntTensor(a.shape[0], 1).fill_(i)], 1))
            feats.append(b)
            labels.append(c)
            offsets[i, :] = offset
        locs = torch.cat(locs, 0).int().contiguous()
        feats = torch.cat(feats, 0).float().contiguous()
        labels = torch.cat(labels, 0).long().contiguous()
        masks = masks.int().contiguous()
#        locs, feas = input_layer(locs, feats)
        org_locs = locs
        
        locs, feats = MergeFunction.apply(locs, feats, True)

        return [locs, feats], [org_locs, org_points], labels, masks, offsets  

    def forward2(self, locations, features, targets, masks, scale, full_scale, training=True):
 
        batch_size, npoints, _ = features.size()
        locations = scale * locations[:,:,:3]
        locs, feats, labels = [], [], []
        offsets = torch.cuda.FloatTensor(batch_size, 3).zero_()
        for i in range(batch_size):
            a = locations[i, :, :]
            b = features[i, :, :]
            c = targets[i, :]
            m, _ = a.min(0)#[min_x, min_y, min_z]
            M, _ = a.max(0)#[max_x, max_y, max_z]
            q = M - m
            if training:
                offset = -m + torch.clamp(full_scale - q - 0.001, 0, None) * torch.rand(3).cuda() + torch.clamp(full_scale - q + 0.001, None, 0)*torch.rand(3).cuda()
                a += offset
                m, _ = a.min(1)
                idxs = (m >= 0) * (a[:, 0] < full_scale[0])* (a[:, 1] < full_scale[1]) * (a[:, 2] < full_scale[2]) * (masks[i, :] > 0)
            else:
                offset = -m + torch.clamp(full_scale - q - 0.001, 0, None)/2 + torch.clamp(full_scale - q + 0.001, None, 0)/2
                a += offset
                idxs = masks[i, :] > 0  
                a[a<0] = 0
                tag = a[:, 0] >= full_scale[0]
                a[tag, 0] = full_scale[0]-1
                tag = a[:, 1] >= full_scale[1]
                a[tag, 1] = full_scale[1]-1
                tag = a[:, 2] >= full_scale[2]
                a[tag, 2] = full_scale[2]-1

            a = a[idxs]
            b = b[idxs]
            masks[i, :] *= idxs
            c = c[idxs]

            a = a.int()
            locs.append(torch.cat([a, torch.cuda.IntTensor(a.shape[0], 1).fill_(i)], 1))
            feats.append(b)
            labels.append(c)
            offsets[i, :] = offset
        locs = torch.cat(locs, 0).int().contiguous()
        feats = torch.cat(feats, 0).float().contiguous()
        labels = torch.cat(labels, 0).long().contiguous()
        masks = masks.int().contiguous()
#        locs, feas = input_layer(locs, feats)
        
        coords, feats = MergeFunction.apply(locs, feats, True)

        return [coords, locs], feats, labels, masks, offsets    


class FocalLoss(Module):
    def __init__(self, gamma=2, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float,int)): self.alpha = torch.Tensor([alpha,1-alpha])
       # if isinstance(alpha,list): 
        else: self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)

        logpt = F.log_softmax(input)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            if self.alpha.device != input.device:
                self.alpha = self.alpha.to(input.device)
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()

class Query(Module):
    def __init__(self):
        super(Query, self).__init__()
    def forward(self, coords, feature, points):
#        print(coords)
#        print(points)
     
        points = points.to(feature.device)
        coords = coords.to(feature.device)
        index = QueryFunction.apply(coords, points)
        result = feature[index.long()]
        return result.contiguous()
class Attention(Module):
    def __init__(self, batchsize, in_channel=20, inner_channel=20, knn=9, repeat=2):
        super(Attention, self).__init__()
        self.knn = knn
        self.repeat = repeat
        self.batchsize = batchsize
        self.softmax = nn.Softmax(dim=1)
        self.conv = nn.Sequential(nn.Linear(in_channel, inner_channel),
                                    nn.BatchNorm1d(inner_channel))
        self.bn_relu = nn.Sequential(nn.BatchNorm1d(inner_channel),
                                    nn.ReLU(inplace=True))
        self.refine = nn.Sequential(nn.Linear(in_channel+inner_channel, inner_channel),
                                    nn.BatchNorm1d(inner_channel),
                                    nn.ReLU(inplace=True))
#                                    nn.Linear(nClasses*2, nClasses))
    def forward(self, coords, points, feature):
#        print(coords)
#        print(points)
#        points = points.to(feature.device)
#        coords = coords.to(feature.device)
        x = self.conv(feature)
        weights = torch.cuda.FloatTensor(feature.size(0), self.knn).zero_()
        output = torch.cuda.FloatTensor(x.size(0), x.size(1)).zero_()
#        temp = torch.cuda.FloatTensor(feature.size(0), feature.size(1)).zero_()
        batches = coords[:, 3].contiguous()
        index = KnnFunction.apply(batches, points, self.knn, self.batchsize)
        for i in range(self.knn):
            weights[:, i]=torch.sum(x*x[index[:, i].long()], 1)
        weights = self.softmax(weights)
        for i in range(self.knn):
            output += x[index[:,i]]*weights[:, i]
        if self.repeat>1:
            temp = output.clone()
            output = output.zero_()
            for i in range(self.knn):
                output += temp[index[:,i]]*weights[:,i]
#        result = feature[index.long()]
        output = torch.cat([self.bn_relu(output), feature], 1)
        output = self.refine(output)
        return output #.contiguous()

class Attention2(Module):
    def __init__(self, batchsize=2, in_channel=20, inner_channel=20, knn=9, repeat=2):
        super(Attention2, self).__init__()
        self.knn = knn
        self.repeat = repeat
        self.batchsize = batchsize
        self.softmax = nn.Softmax(dim=1)
#        self.conv1 = nn.Sequential(nn.Linear(in_channel, inner_channel),
#                                    nn.BatchNorm1d(inner_channel),
#                                    nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(nn.Conv1d(inner_channel+3, inner_channel, kernel_size=1, bias=False),
                                    nn.BatchNorm1d(inner_channel))

        self.bn_relu = nn.Sequential(nn.BatchNorm1d(inner_channel),
                                    nn.ReLU(inplace=True))
        self.refine = nn.Sequential(nn.Linear(in_channel+inner_channel, in_channel),
                                    nn.BatchNorm1d(in_channel),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(in_channel, in_channel))
#                                    nn.Linear(nClasses*2, nClasses))
    def forward(self, coords, points, feature):
#        self.batchsize = coords[coords.size(0)-1, coords.size(1)-1].item()
        B, C = points.size()
#        x = self.conv1(feature)
#        weights = torch.cuda.FloatTensor(feature.size(0), self.knn).zero_()
#        output = torch.cuda.FloatTensor(x.size(0), x.size(1)).zero_()
#        temp = torch.cuda.FloatTensor(feature.size(0), feature.size(1)+3, self.knn).zero_()
        batches = coords[:, 3].contiguous()
        self.batchsize = batches[coords.size(0)-1].item()+1
#        print(self.batchsize)
#        quit()
        index = KnnFunction.apply(batches, points, self.knn, self.batchsize)
#        print(index)
        xyz = points[index, :].transpose(1, 2)
        xyz -= points.view(B, C, 1)
#        print(torch.sum(xyz, 0))
        x = torch.cat([feature[index,:].transpose(1, 2), xyz], 1)

        x = self.conv2(x)
        weight = torch.sum(x * x[:, :, 0].view(B, -1, 1), 1)
        weights = self.softmax(weight)
        x = torch.sum(x * weight.view(-1, 1, self.knn), 2)
        
 #       for i in range(self.knn):
 #           weights[:, i]=torch.sum(x*x[index[:, i].long()], 1)

#        for i in range(self.knn):
#            output += x[index[:,i]]*weights[:, i]
        for iter in range (self.repeat-1):
            x = torch.sum(x[index,:] * weight.view(-1, self.knn, 1), 1)

        x = torch.cat([self.bn_relu(x), feature], 1)
        x = self.refine(x)
        return x #.contiguous()

class Merge(Module):
    def __init__(self, data=False):
        super(Merge, self).__init__()
        self.data_layer = data
    def forward(self, coords, feature):
        locs, feas = MergeFunction.apply(coords, feature, self.data_layer)
        return locs.cpu(), feas


