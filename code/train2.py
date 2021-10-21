
import argparse
import os
from data_utils.debug import KittiDataset
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
from models.hourglass2 import  HourGlass
import math
import yaml
from np_iou import iouEval
import torch.nn as nn
    




def parse_args():
    parser = argparse.ArgumentParser('Model')
    parser.add_argument('--model', type=str, default='pointnet2_part_seg_msg3', help='model name [default: pointnet2_part_seg_msg]')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch Size during training [default: 16]')
    parser.add_argument('--epoch',  default=251, type=int, help='Epoch to run [default: 251]')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='Initial learning rate [default: 0.001]')
    parser.add_argument('--gpu', type=str, default='0,1', help='GPU to use [default: GPU 0]')
    parser.add_argument('--optimizer', type=str, default='Adam', help='Adam or SGD [default: Adam]')
    parser.add_argument('--log_dir', type=str, default=None, help='Log path [default: None]')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='weight decay [default: 1e-4]')
    parser.add_argument('--npoint', type=int,  default=160000, help='Point Number [default: 2048]')
    parser.add_argument('--normal', action='store_true', default=False, help='Whether to use normal information [default: False]')
    parser.add_argument('--step_size', type=int,  default=20, help='Decay step for lr decay [default: every 20 epochs]')
    parser.add_argument('--lr_decay', type=float,  default=0.5, help='Decay rate for lr decay [default: 0.5]')
    parser.add_argument('--num_gpus', type=int,  default=1, help='number of gpus')

    return parser.parse_args()

dimension = 3
reps = 1 #Conv block repetition factor
m = 32 #Unet number of features
nPlanes = [m, 2*m, 3*m, 4*m, 5*m, 6*m, 6*m, 6*m] #UNet number of features per level
num_classes = 20
dimension = 3
scale = 32
full_scale = 128*2*scale
spatialSize = [full_scale, full_scale, full_scale]


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))
#torch.backends.cudnn.enabled = False
#torch.backends.cudnn.benchmark = True
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



def main(args):
    def log_string(str):
        logger.info(str)
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
    file_list = '/media/feihu/Storage/kitti_point_cloud/semantic_kitti/train.list'
    val_list = '/media/feihu/Storage/kitti_point_cloud/semantic_kitti/val.list'
    TRAIN_DATASET = KittiDataset(root = root, file_list=file_list, npoints=args.npoint, training=True, augment=True)
    trainDataLoader = torch.utils.data.DataLoader(TRAIN_DATASET, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=8)
    TEST_DATASET = KittiDataset(root = root, file_list=val_list, npoints=args.npoint, training=False, augment=False)
    testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=args.batch_size,shuffle=False, drop_last=True, num_workers=8)
    log_string("The number of training data is: %d" % len(TRAIN_DATASET))
    log_string("The number of test data is: %d" %  len(TEST_DATASET))
#    num_classes = 16

    '''MODEL LOADING'''


    shutil.copy('models/%s.py' % args.model, str(experiment_dir))
    shutil.copy('models/pointnet_util.py', str(experiment_dir))

    num_devices = args.num_gpus #torch.cuda.device_count()
#    assert num_devices > 1, "Cannot detect more than 1 GPU."
#    print(num_devices)
    devices = list(range(num_devices))
    target_device = devices[0]

#    MODEL = importlib.import_module(args.model)

    net = HourGlass(4, 20, nPlanes)

#    net = MODEL.get_model(num_classes, normal_channel=args.normal)
    net = net.to(target_device)



    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv2d') != -1:
            if m.weight is not None:
                torch.nn.init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias.data, 0.0)
        elif classname.find('Linear') != -1:
            if m.weight is not None:
                torch.nn.init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias.data, 0.0)

    try:
        checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth')
        start_epoch = checkpoint['epoch']
        net.load_state_dict(checkpoint['model_state_dict'])
        log_string('Use pretrain model')
    except:
        log_string('No existing model, starting training from scratch...')
        start_epoch = 0
        net = net.apply(weights_init)

    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            net.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.decay_rate
        )
    else:
        optimizer = torch.optim.SGD(net.parameters(),
            lr = 1e-1,
            momentum = 0.9,
            weight_decay = 1e-4,
            nesterov=True)
#        optimizer = torch.optim.SGD(net.parameters(), lr=args.learning_rate, momentum=0.9)

    def bn_momentum_adjust(m, momentum):
        if isinstance(m, torch.nn.BatchNorm2d) or isinstance(m, torch.nn.BatchNorm1d):
            m.momentum = momentum

    LEARNING_RATE_CLIP = 1e-5
    MOMENTUM_ORIGINAL = 0.1
    MOMENTUM_DECCAY = 0.5
    MOMENTUM_DECCAY_STEP = args.step_size

    best_acc = 0
    global_epoch = 0
    best_class_avg_iou = 0
    best_inctance_avg_iou = 0

#    criterion = MODEL.get_loss()
    criterion = nn.CrossEntropyLoss()
    criterions = parallel.replicate(criterion, devices)

        # The raw version of the parallel_apply
#    replicas = parallel.replicate(net, devices)
#    input_coding = scn.InputLayer(dimension, torch.LongTensor(spatialSize), mode=4)

    for epoch in range(start_epoch,args.epoch):
        log_string('Epoch %d (%d/%s):' % (global_epoch + 1, epoch + 1, args.epoch))
        '''Adjust learning rate and BN momentum'''

        lr = max(args.learning_rate * (args.lr_decay ** (epoch // args.step_size)), LEARNING_RATE_CLIP)
        lr = args.learning_rate * \
            math.exp((1 - epoch) * args.lr_decay)

        log_string('Learning rate:%f' % lr)

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
#        for param_group in optimizer.param_groups:
#            param_group['lr'] = lr

        mean_correct = []
        if 0:
            momentum = MOMENTUM_ORIGINAL * (MOMENTUM_DECCAY ** (epoch // MOMENTUM_DECCAY_STEP))
            if momentum < 0.01:
                momentum = 0.01
            print('BN momentum updated to: %f' % momentum)
            net = net.apply(lambda x: bn_momentum_adjust(x,momentum))

        '''learning one epoch'''
        net.train()



        for iteration, data in tqdm(enumerate(trainDataLoader), total=len(trainDataLoader), smoothing=0.9):

            optimizer.zero_grad()
#        for iteration, data in enumerate(trainDataLoader):
            if iteration > 100:
                break
            points, target, ins, mask = data

            valid = mask > 0
            total_points = valid.cpu().sum()

            points = points.data.numpy()
#            print(total_points)
            inputs, targets, masks = [], [], []
            for i in range(num_devices):
                start = int(i*(args.batch_size/num_devices))
                end = int((i+1)*(args.batch_size/num_devices))
                batch = provider.transform_for_sparse(points[start:end,:,:3], points[start:end,:,3:], target[start:end,:].data.numpy(), mask[start:end,:].data.numpy(), scale, spatialSize)
                batch['x'][1]=batch['x'][1].type(torch.FloatTensor)
                batch['x'][0]=batch['x'][0].type(torch.IntTensor)
                batch['y']=batch['y'].type(torch.LongTensor)
                label = Variable(batch['y'], requires_grad=False)
                maski = batch['mask'].type(torch.IntTensor)
                

 
 #               print(inputi.size(), batch['x'][1].size())

                with torch.cuda.device(devices[i]):
                    inputi = ME.SparseTensor(batch['x'][1], batch['x'][0]) #input_coding(batch['x'])
                    inputs.append(inputi.to(devices[i]))
                    targets.append(label.to(devices[i]))
                    masks.append(maski.contiguous().to(devices[i]))


            replicas = parallel.replicate(net, devices)
            predictions = parallel.parallel_apply(replicas, inputs, devices=devices)

            count = 0
#            print("end ...") 
            prediction1=[]
            prediction2=[]
            labels = []
            match = 0
            
            for i in range(num_devices):
                temp = predictions[i]['output1'].F#.view(-1, num_classes)
                temp = temp[targets[i] > 0, :]
                prediction1.append(temp)
                temp = predictions[i]['output2'].F#view(-1, num_classes)
                temp = temp[targets[i] > 0, :]
                prediction2.append(temp)
                temp = targets[i]
                temp = temp[targets[i]>0]
                labels.append(temp)
 #               print(prediction2[i].size(), prediction1[i].size(), targets[i].size())
                outputi = prediction2[i].contiguous().view(-1, num_classes)
                num_points = labels[i].size(0)
                count += num_points
                
                _, pred_choice = outputi.data.max(1)#[1]
#                print(pred_choice)
                correct = pred_choice.eq(labels[i].data).cpu().sum()
                match += correct.item()
                mean_correct.append(correct.item() / num_points)
#            print(prediction2, labels)
            losses1 = parallel.parallel_apply(criterions, tuple(zip(prediction1, labels)), devices=devices)
            losses2 = parallel.parallel_apply(criterions, tuple(zip(prediction2, labels)), devices=devices)
            loss = 0.6 * parallel.gather(losses1, target_device, dim=0).mean() + parallel.gather(losses2, target_device, dim=0).mean()
            loss.backward()
            optimizer.step()
#            assert(count1 == count2 and total_points == count1)
            log_string("===> Epoch[{}]({}/{}) Valid points:{}/{} Loss: {:.4f} Accuracy: {:.4f}".format(epoch, iteration, len(trainDataLoader), count, total_points, loss.item(), match/count))
#            sys.stdout.flush()
        train_instance_acc = np.mean(mean_correct)
        log_string('Train accuracy is: %.5f' % train_instance_acc)

#        continue

        with torch.no_grad():
            net.eval()
            evaluator = iouEval(num_classes, ignore)
            
            evaluator.reset()
            for iteration, (points, target, ins, mask) in tqdm(enumerate(testDataLoader), total=len(testDataLoader), smoothing=0.9):
                cur_batch_size, NUM_POINT, _ = points.size()
#                points, label, target, mask = points.float().cuda(), label.long().cuda(), target.long().cuda(), mask.float().cuda()
                if iteration > 64:
                    break   
                if 0:
                    points = points.data.numpy()
                    points[:,:, 0:3], norm = provider.pc_normalize(points[:,:, :3], mask.data.numpy())
                    points = torch.Tensor(points)

                points = points.data.numpy()
                inputs, targets, masks, locs = [], [], [], []
                for i in range(num_devices):
                    start = int(i*(cur_batch_size/num_devices))
                    end = int((i+1)*(cur_batch_size/num_devices))
                    batch = provider.transform_for_sparse(points[start:end,:,:3], points[start:end,:,3:], target[start:end,:].data.numpy(), mask[start:end,:].data.numpy(), scale, spatialSize)
                    batch['x'][1]=batch['x'][1].type(torch.FloatTensor)
                    batch['y']=batch['y'].type(torch.LongTensor)
                    label = Variable(batch['y'], requires_grad=False)
                    maski = batch['mask'].type(torch.IntTensor)
                    input = input_coding(batch['x'])
                    locs.append(batch['x'][0])
                    with torch.cuda.device(devices[i]):
                        inputs.append(input.to(devices[i]))
                        targets.append(label.to(devices[i]))
                        masks.append(maski.contiguous().to(devices[i]))
                replicas = parallel.replicate(net, devices)
                outputs = parallel.parallel_apply(replicas, inputs, devices=devices)

#                net = net.eval()
#                seg_pred = classifier(points, to_categorical(label, num_classes))
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

  # when I am done, print the evaluation
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


        log_string('Epoch %d test Accuracy: %f  mean avg mIOU: %f' % (epoch+1, m_accuracy, m_jaccard))
        if (m_jaccard >= best_class_avg_iou):
#            logger.info('Save model...')
            log_string('Saveing model...')
            savepath = str(checkpoints_dir) + '/best_model.pth'
            log_string('Saving at %s'% savepath)
            state = {
                'epoch': epoch,
                'train_acc': train_instance_acc,
                'test_acc': m_accuracy,
                'class_avg_iou': m_jaccard,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }
            torch.save(state, savepath)
#            log_string('Saving model....')

        if m_accuracy > best_acc:
            best_acc = m_accuracy
        if m_jaccard > best_class_avg_iou:
            best_class_avg_iou = m_jaccard
       
        log_string('Best accuracy is: %.5f'%best_acc)
        log_string('Best class avg mIOU is: %.5f'%best_class_avg_iou)
    
        global_epoch+=1

if __name__ == '__main__':
    args = parse_args()
    main(args)

