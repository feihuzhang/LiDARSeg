import torch.utils.data as data
import numpy as np
import random
from sklearn.neighbors import NearestNeighbors


# bbox [classtype, center_x, center_y, center_z, length, width, height, theta/angle(-PI,PI)]
# points [x, y, z, intensity]
# obj_points [obj_id, point_id, point_id, ... point_id]
def train_transform(points, bbox, obj_points, rotate=30, flip=True):
    num, _ = np.shape(points)
    obj_num, _ = np.shape(bbox)
    if flip and random.randint(1, 100) >= 50:# flip accross x-axis (the forward direction)
        points[:, 1] = -points[:, 1]
        bbox[:, 2] = -bbox[:, 2]
        bbox[:, 7] = -bbox[:, 7] + 2 * np.pi
    theta = (random.randint(0, rotate * 2000) / 1000. - rotate) / 180 * np.pi
    R = [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]

    xy = np.transpose(points[:, 0: 2])
    
    xy = np.matmul(R, xy)
    points[:, 0: 2] = np.transpose(xy)
    
    xy = np.transpose(bbox[:, 1: 3])
    xy = np.matmul(R, xy)
    bbox[:, 1: 3] = np.transpose(xy)
    bbox[:, 7] += theta
    
 
def knn_encoding(points, centers, k=32):
#    neigh = NearestNeighbors(algorithm='auto', leaf_size=30, metric='minkowski', metric_params=None, n_jobs=None, n_neighbors=3, p=2, radius=1.0)

    neigh = NearestNeighbors(n_neighbors=k)
    neigh.fit(points[:, 0: 2])
    _, k_index = neigh.kneighbors(centers.reshape([centers.shape[0] * centers.shape[1], 2]))
    temp = points[k_index, :].reshape([centers.shape[0], centers.shape[1], k, 4])
#    k_centers = np.reshape(k_centers, [centers.shape[0], centers.shape[1], k])
#    _, k_points = neigh.kneighbors(points[:, 0: 2])
    dense = np.transpose(temp, (2, 3, 1, 0))
    data = np.zeros([6, k, cneters.shape[0], centers.shape[1]] dtype='float')
    np.transpose(np.reshape(centers.repeat(k), [centers.shape[0],centers[1], 2, k], (2, 3, 0, 1))
    np.transpose(centers, (1, 2, 0))
    data[3: 5,:,:,:] = dense[0: 2, :, :, :] - data[1: 3, :, :, :] 
    data[5: 7,:,:,:] = dense[2: 4, :, :, :]
    return k_centers, k_points
    
def label_transform(points, bbox, point_labels, mask, ranges=[640, 640], resolution=0.2):
    foreground = points[point_labels[:,1], :3] #x, y, z
    obj = bbox[point_labels[:,0], :4] #class type, center_x, center_y
    heights = bbox[point_labels[:,0], 6:7] #height
    max_heights = obj[:, :, 4] + heights/2
    min_heights = obj[:, :, 4] - heights/2
    labels = np.zeros([mask.shape[0], mask.shape[1], 6])
    
    
    index = np.int(ranges[0] / 2 * resolution - foreground[:, 0: 2]) / resolution)
    labels[index] = 

 
def load_label(file_path, current_file):
    f = open(file_path + current_file[0: len(current_file) - 4] + 'txt', 'r')
    bbox=[]
    points=[]
    obj_id = 0
    for x in f:
       temp = np.array(x.split(), dtype='f')
       bbox.append(temp[0: 8])
       _, length = np.shape(temp)
       obj_id += 1
       points_num = length - 8
       temp_points = np.ones([points_num, 2], dtype='float')
       temp_points[:, 0] = obj_id # obj id
       temp_points[:, 1] = temp[8: length] # points id
       if np.shape(bbox)[0] > 0:
           points = np.concatenate((points, temp_points), 0)
       else:
           points = temp_points
          
    return bbox, points

def load_data(file_path, current_file):

    points = np.loadtxt(file_path + current_file[0: len(current_file) - 1], dtype='f', delimiter=' ')

    return points



class DatasetFromList(data.Dataset): 
    def __init__(self, data_path, file_list, ranges=[640, 640], resolution=0.2, num_sample=64, rotate=30, flip=True, max_num=200000):
        super(DatasetFromList, self).__init__()
        #self.image_filenames = [join(image_dir, x) for x in listdir(image_dir) if is_image_file(x)]
        f = open(file_list, 'r')
        self.data_path = data_path
        self.num_sample = num_sample
        self.file_list = f.readlines()
        self.training = training
        self.resolution = resolution
        self.height = ranges[0]
        self.width = ranges[1]
        self.rotate = rotate
        self.flip = flip
        self.max_num = max_num
        self.centers=np.zeros([ranges[0], ranges[1], 2])
        temp = np.zeros(ranges[0])
        temp[:] = range(ranges[0])
        self.centers[:,:,0] = temp.repeat(ranges[1]).reshape([ranges[0],ranges[1]])
        self.centers[:,:,1] = self.centers[:,:,0].transpose()
        self.centers = ranges[0]*resolution/2. - (self.centers + 0.5) * resolution

    def __getitem__(self, index):
    #    print self.file_list[index]
        points = load_data(self.data_path, self.file_list[index])
        bbox, obj_points = load_label(self.data_path, self.file_list[index])


        if self.training:
            points, bbox, obj_points = train_transform(points, bbox, obj_points, self.rotate, self.flip)

        k_centers, k_points = knn_index(points, self.centers, self.num_sample)

        data = np.zeros([self.max_num, 4], dtype='float') -1000
        label1 = np.zeros([500, 8], dtype='float') -1
        label2 = np.zeros([self.max_num, 2], dtype='float') -1

        data[0:np.shape(points)[0], :] = points
        label1[0: np.shape(bbox)[0], :] = bbox
        label2[0: np.shape(obj_points)[0], :] = obj_points
#        data = np.zeros(7, self.num_sample, self.height, self.width, dtype='float')
#        label = np.zeros(10, ranges[0], ranges[1], dtype='float')
#        ext.Projection(points, bbox, obj_points, data, label)
#        order = np.zeros(121203, 3)

        return data, label1, label2 # points, bbox, point label


    def __len__(self):
        return len(self.file_list)
