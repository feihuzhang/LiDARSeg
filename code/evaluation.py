
import argparse
import os
from data_utils.kitti import KittiDataset
import torch
import datetime
import logging
from pathlib import Path
import sys
import importlib
import shutil
from tqdm import tqdm
import provider
import numpy as np
import torch.nn.parallel as parallel
from data_utils.transform import PcNormalize, Transform
from torch.autograd import Variable
import MinkowskiEngine as ME
import MinkowskiEngine.MinkowskiFunctional as MF
import time
from models.unet import  UNet
import math
import yaml
from np_iou import iouEval
import torch.nn as nn
from libs.modules.fusion import Query, Merge, SparseTransform

#torch.backends.cudnn.enabled = False
#torch.backends.cudnn.benchmark = True


def parse_args():
    parser = argparse.ArgumentParser('Model')
    parser.add_argument('--model', type=str, default='unet', help='model name [default: pointnet2_part_seg_msg]')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch Size during training [default: 16]')
    parser.add_argument('--epoch',  default=251, type=int, help='Epoch to run [default: 251]')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='Initial learning rate [default: 0.001]')
    parser.add_argument('--gpu', type=str, default='0,1', help='GPU to use [default: GPU 0]')
    parser.add_argument('--optimizer', type=str, default='Adam', help='Adam or SGD [default: Adam]')
    parser.add_argument('--log_dir', type=str, default=None, help='Log path [default: None]')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='weight decay [default: 1e-4]')
    parser.add_argument('--npoint', type=int,  default=130000, help='Point Number [default: 2048]')
    parser.add_argument('--normal', action='store_true', default=False, help='Whether to use normal information [default: False]')
    parser.add_argument('--step_size', type=int,  default=20, help='Decay step for lr decay [default: every 20 epochs]')
    parser.add_argument('--lr_decay', type=float,  default=0.5, help='Decay rate for lr decay [default: 0.5]')
    parser.add_argument('--num_gpus', type=int,  default=2, help='number of gpus')

    return parser.parse_args()

alpha = [3.36683823e+00, 4.57367840e+00, 2.24514928e-02, 4.50004005e-02, 
         1.78850105e-01, 2.69537690e-01, 4.54167396e-02, 2.12545001e-02,
         3.87960694e-03, 1.94847152e+01, 1.42262602e+00, 1.39878799e+01,
         3.38113242e-01, 1.29423702e+01, 6.41163489e+00, 2.71404251e+01,
         6.91301788e-01, 8.69588554e+00, 2.94130345e-01, 6.40105657e-02]

alpha = np.array(alpha)
alpha = np.log(alpha.max()/alpha) + 1.
alpha = alpha/alpha.mean()

dimension = 3
reps = 1 #Conv block repetition factor
m = 32 #Unet number of features
nPlanes = [m, 2*m, 3*m, 4*m, 5*m, 6*m, 6*m, 6*m] #UNet number of features per level
num_classes = 20
dimension = 3
scale = 32
full_scale = 128*2*scale
spatialSize = torch.Tensor([full_scale, full_scale, full_scale/4])
scale = torch.Tensor([scale, scale, scale])


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))

DATA = yaml.safe_load(open('config/semantic-kitti.yaml', 'r'))

  # get number of interest classes, and the label mappings
class_strings = DATA["labels"]
class_remap = DATA["learning_map"]
class_inv_remap = DATA["learning_map_inv"]
class_ignore = DATA["learning_ignore"]
nr_classes = len(class_inv_remap)

ignore = []
for cl, ign in class_ignore.items():
    if ign:
        x_cl = int(cl)
        ignore.append(x_cl)
        print("Ignoring xentropy class ", x_cl, " in IoU evaluation")

  # create evaluator

input_layer = SparseTransform()

output_layer = Query()

def main(args):
    def log_string(str):
#        logger.info(str)
        print(str)

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    '''CREATE DIR'''
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    experiment_dir = Path('./log/')
    experiment_dir.mkdir(exist_ok=True)
    experiment_dir = experiment_dir.joinpath('part_seg')
    experiment_dir.mkdir(exist_ok=True)
    if args.log_dir is None:
        experiment_dir = experiment_dir.joinpath(timestr)
    else:
        experiment_dir = experiment_dir.joinpath(args.log_dir)
    experiment_dir.mkdir(exist_ok=True)
    checkpoints_dir = experiment_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = experiment_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)

    '''LOG'''
    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, args.model))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    root = '/media/feihu/Storage/kitti_point_cloud/semantic_kitti/'
#    file_list = '/media/feihu/Storage/kitti_point_cloud/semantic_kitti/train2.list'
    val_list = '/media/feihu/Storage/kitti_point_cloud/semantic_kitti/val2.list'
#    TRAIN_DATASET = KittiDataset(root = root, file_list=file_list, npoints=args.npoint, training=True, augment=True)
#    trainDataLoader = torch.utils.data.DataLoader(TRAIN_DATASET, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=2)
    TEST_DATASET = KittiDataset(root = root, file_list=val_list, npoints=args.npoint, training=False, augment=False)
    testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=args.batch_size,shuffle=False, drop_last=True, num_workers=2)
#    log_string("The number of training data is: %d" % len(TRAIN_DATASET))
    log_string("The number of test data is: %d" %  len(TEST_DATASET))
#    num_classes = 16



    num_devices = args.num_gpus #torch.cuda.device_count()
#    assert num_devices > 1, "Cannot detect more than 1 GPU."
#    print(num_devices)
    devices = list(range(num_devices))
    target_device = devices[0]

#    MODEL = importlib.import_module(args.model)

    net = UNet(4, 20, nPlanes)

#    net = MODEL.get_model(num_classes, normal_channel=args.normal)
    net = net.to(target_device)



    try:
        checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth')
        start_epoch = checkpoint['epoch']
        net.load_state_dict(checkpoint['model_state_dict'])
        log_string('Use pretrain model')
    except:
        log_string('No existing model, starting training from scratch...')
        quit()

    if 1:

        with torch.no_grad():
            net.eval()
            evaluator = iouEval(num_classes, ignore)
            
            evaluator.reset()
#            for iteration, (points, target, ins, mask) in tqdm(enumerate(testDataLoader), total=len(testDataLoader), smoothing=0.9):
            for iteration, (points, target, ins, mask) in enumerate(testDataLoader):
                evaone = iouEval(num_classes, ignore)
                evaone.reset()
                cur_batch_size, NUM_POINT, _ = points.size()

                if iteration > 128:
                    break   

                inputs, targets, masks = [], [], []
                coords = []
                for i in range(num_devices):
                    start = int(i*(cur_batch_size/num_devices))
                    end = int((i+1)*(cur_batch_size/num_devices))
                    with torch.cuda.device(devices[i]):
                        pc = points[start:end,:,:].to(devices[i])
                        #feas = points[start:end,:,3:].to(devices[i])
                        targeti = target[start:end,:].to(devices[i])
                        maski = mask[start:end,:].to(devices[i])

                        locs, feas, label, maski, offsets = input_layer(pc, targeti, maski, scale.to(devices[i]), spatialSize.to(devices[i]), True)
#                        print(locs.size(), feas.size(), label.size(), maski.size(), offsets.size())
                        org_coords = locs[1]
                        label = Variable(label, requires_grad=False)


                        inputi = ME.SparseTensor(feas.cpu(), locs[0].cpu()) 
                        inputs.append([inputi.to(devices[i]), org_coords])
                        targets.append(label)
                        masks.append(maski)

                replicas = parallel.replicate(net, devices)
                outputs = parallel.parallel_apply(replicas, inputs, devices=devices)
                   
                seg_pred = outputs[0].cpu()
                mask = masks[0].cpu()
                target = targets[0].cpu()
                loc = locs[0].cpu()
                for i in range(1, num_devices):
                    seg_pred = torch.cat((seg_pred, outputs[i].cpu()), 0)
                    mask = torch.cat((mask, masks[i].cpu()), 0)
                    target = torch.cat((target, targets[i].cpu()), 0)

                seg_pred = seg_pred[target>0, :]
                target = target[target>0]
                _, seg_pred = seg_pred.data.max(1)#[1]

                target = target.data.numpy()

                evaluator.addBatch(seg_pred, target)

                evaone.addBatch(seg_pred, target)
                cur_accuracy = evaone.getacc()
                cur_jaccard, class_jaccard = evaone.getIoU()
                print('%.4f %.4f'%(cur_accuracy, cur_jaccard))


            m_accuracy = evaluator.getacc()
            m_jaccard, class_jaccard = evaluator.getIoU()

            
            log_string('Validation set:\n'
                'Acc avg {m_accuracy:.3f}\n'
                'IoU avg {m_jaccard:.3f}'.format(m_accuracy=m_accuracy, m_jaccard=m_jaccard))
  # print also classwise
            for i, jacc in enumerate(class_jaccard):
                if i not in ignore:
                    log_string('IoU class {i:} [{class_str:}] = {jacc:.3f}'.format(
                        i=i, class_str=class_strings[class_inv_remap[i]], jacc=jacc))


if __name__ == '__main__':
    args = parse_args()
    main(args)

