# *_*coding:utf-8 *_*
import os
import json
import warnings
import numpy as np
from torch.utils.data import Dataset
warnings.filterwarnings('ignore')

def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / m
    return pc

def random_point_dropout(pc, seg, ins, max_dropout_ratio=0.05):
    ''' pc: Npointx3 '''
    dropout_ratio =  np.random.random()*max_dropout_ratio # 0~0.875
    nondrop_idx = np.where(np.random.random((pc.shape[0])) > dropout_ratio)[0] # 
    if len(nondrop_idx)>0:
        pc = pc[nondrop_idx,:] # set to the first point
        seg = seg[nondrop_idx]
        ins = ins[nondrop_idx]
    return pc, seg, ins

def augmentation(points, var=[0.1, 0.1, 0.05]):
    #augmentation
    m = np.eye(3) + np.random.randn(3, 3) * 0.1
    m[0][0] *= np.random.randint(0, 2) * 2 - 1
    theta = np.random.rand() * 2 * math.pi
    m = np.matmul(m, [[math.cos(theta), math.sin(theta), 0], [-math.sin(theta), math.cos(theta), 0], [0, 0, 1]])
    points[:, :3] = np.matmul(points[:, :3], m) # rotation
    points[:, 0] += np.random.normal(0, var[0])
    points[:, 1] += np.random.normal(0, var[1])
    points[:, 2] += np.random.normal(0, var[2])
    return points


def get_feature(pc, normalize=True, radius=60):
    points = np.zeros([pc.shape[0], 7], 'float32')
    points[:, 0:3] = pc[:, 0:3]
    if normalize:
        temp = pc[:,0:4]
        points[:, 3:7] = (temp-np.mean(temp, 0))/np.std(temp, 0)
#        print(np.mean(points, 0), np.std(points, 0))
#        quit()
        return points
 
    points[:, 5:7] = pc[:, 2:4]
    points[:, 3:5] = pc[:, 0:2] / radius  
#    points[:, 3] = np.arctan2(pc[:, 1], pc[:, 0]) / (2*np.pi)
#    points[:, 4] = np.hypot(pc[:, 0], pc[:, 1]) / radius
    return points


def open_label(filename):
    """ Open label file and fill in attributes
    """
    label = np.fromfile(filename, dtype=np.int32)
    label = label.reshape((-1))
#    print(filename)
#    print(label.shape)
    sem_label = label & 0xFFFF  # semantic label in lower half
    inst_label = label >> 16    # instance id in upper half
 
    return sem_label, inst_label

def open_scan(filename):
    """ Open raw points file and fill in attributes
    """
  
    # if all goes well, open pointcloud
    scan = np.fromfile(filename, dtype=np.float32)
#    print(filename)
#    print(scan.shape)
#    return scan
    scan = scan.reshape((-1, 4))

    # put in attribute
#    points = scan[:, 0:3]    # get xyz
#    remissions = scan[:, 3]  # get remission

    return scan

def rotatexy_flip_shift(points, angle=30, flip=True, flip_ratio=0.25, var=[0.1, 0.1, 0.05]):

#    for k in range(batch_data.shape[0]):
    if flip and np.random.uniform() < flip_ratio:
        points[:,1] = - points[:,1]
    rotation_angle = (np.random.uniform()-0.5) * angle/180. * np.pi
    cosval = np.cos(rotation_angle)
    sinval = np.sin(rotation_angle)
    rotation_matrix = np.array([[cosval, sinval, 0],
                                [-sinval, cosval, 0],
                                [0, 0, 1]])

    points[:,:3] = np.dot(points[:,:3], rotation_matrix)
    points[:, 0] += np.random.normal(0, var[0])
    points[:, 1] += np.random.normal(0, var[1])
    points[:, 2] += np.random.normal(0, var[2])
    return points


class KittiDataset(Dataset):
    def __init__(self,root = '/media/feihu/Storage/kitti_point_cloud/semantic_kitti/', file_list='./lists/train.list', npoints=2500, training=True, augment=True):
        self.npoints = npoints
        self.root = root
        self.augment = augment
        self.training = training
        f = open(file_list, 'r')
        self.file_list = f.readlines()

    def __getitem__(self, index):
        current_file = self.file_list[index]
        filename = self.root +'data_odometry_velodyne/' + current_file[:len(current_file) - 14] + 'velodyne/' + current_file[len(current_file) - 7: len(current_file) - 1] + '.bin'
        labelname = self.root +'data_odometry_labels/' + current_file[0: len(current_file) - 1] + '.label'
        point_set = open_scan(filename)
        seg, ins = open_label(labelname)     
#        print(np.mean(point_set, 0)) 
#        print(np.std(point_set, 0))  
        if self.training and self.augment:
#            point_set = rotatexy_flip(data)
            point_set, seg, ins = random_point_dropout(point_set, seg, ins)
#            point_set = rotatexy_flip_shift(point_set)
#        point_set = get_feature(point_set)

        valid = len(seg)
        mask = np.ones([self.npoints], dtype='int')
        if self.npoints < valid:
            choice = np.random.choice(valid, self.npoints, replace=True)
        # resample
            point_set = point_set[choice, :]
            seg = seg[choice]
            ins = ins[choice]
            return point_set, seg, ins, mask
        else:
            points = np.zeros([self.npoints, point_set.shape[1]], dtype='float32')
            segment = np.zeros([self.npoints], dtype='int')
            instance = np.zeros([self.npoints], dtype='int')
            choice = np.random.choice(valid, self.npoints-valid, replace=True)
            points[:valid, :] = point_set
            points[valid:, :] = point_set[choice, :]
            segment[:valid] = seg
            segment[valid:] = seg[choice]
            instance[:valid]= ins
            instance[valid:] = ins[choice]
            mask[valid:] = 0
#            points[:, 0] = np.arange(self.npoints)/30.
#            points[:, 1] = 
            return points, segment, instance, mask

    def __len__(self):
        return len(self.file_list)



