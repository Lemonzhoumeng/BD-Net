import argparse
import logging
import os
import random
import shutil
import sys
import time
import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn import BCEWithLogitsLoss
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import make_grid
from tqdm import tqdm
from itertools import cycle
from dataloaders import utils
from dataloaders.dataset_semi import BaseDataSets, RandomGenerator
from networks.net_factory import net_factory
from utils import losses, metrics, ramps
from val_2D import test_single_volume, test_single_volume_ds
from torch.cuda.amp import GradScaler, autocast
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from dgcl import (
    compute_dgcl_loss,
    FeatureMemory, 
    )
parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='../data/ACDC', help='Name of Experiment')
parser.add_argument('--exp', type=str,
                    default='5%_pce_ce_ceboun_contra_5zhe', help='experiment_name')
parser.add_argument('--fold', type=str,
                    default='fold2', help='cross validation')
parser.add_argument('--sup_type', type=str,
                    default='scribble', help='supervision type')
parser.add_argument('--model', type=str,
                    default='BoundaryUNet', help='model_name')
parser.add_argument('--num_classes', type=int,  default=4,
                    help='output channel of network')
parser.add_argument('--max_iterations', type=int,
                    default=60000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=4,
                    help='batch_size per gpu')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.03,
                    help='segmentation network learning rate')
parser.add_argument('--ema_decay', type=float,  default=0.99, help='ema_decay')
parser.add_argument('--patch_size', type=list,  default=[256, 256],
                    help='patch size of network input')
parser.add_argument('--seed', type=int,  default=2022, help='random seed')
parser.add_argument('-g', '--gpu', type=int, default=0)

args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
model_file1 = args.model_file1
model_file2 = args.model_file2
torch.cuda.empty_cache()
def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return 1.0 * ramps.sigmoid_rampup(epoch, 60)
def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    #alpha = min(1 - 1 / (global_step + 1), alpha)
    alpha = 0.5
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)
    print(alpha)
def train(args, snapshot_path):
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size
    max_iterations = args.max_iterations
    macro_model = net_factory(net_type=args.model, in_chns=1, class_num=num_classes)
    if torch.cuda.is_available():
        macro_model = macro_model.cuda()

    macro_model.train()
    db_train_labeled = BaseDataSets(base_dir=args.root_path, num=4, labeled_type="labeled",fold=args.fold, split="train", sup_type="label", transform=transforms.Compose([
        RandomGenerator(args.patch_size)
    ]))

    db_train_unlabeled = BaseDataSets(base_dir=args.root_path, num=4, labeled_type="unlabeled", fold=args.fold, split="train",sup_type="scribble", transform=transforms.Compose([
        RandomGenerator(args.patch_size)]))
    db_val = BaseDataSets(base_dir=args.root_path,
                          fold=args.fold, split="val", )
    db_test = BaseDataSets(base_dir=args.root_path,
                          fold=args.fold, split="test", )                      
    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)
    
    trainloader_labeled = DataLoader(db_train_labeled, batch_size=args.batch_size//2, shuffle=True,
                                     num_workers=16, pin_memory=True, worker_init_fn=worker_init_fn)
    trainloader_unlabeled = DataLoader(db_train_unlabeled, batch_size=args.batch_size//2, shuffle=True,
                                       num_workers=0, pin_memory=True, worker_init_fn=worker_init_fn)
    valloader = DataLoader(db_val, batch_size=1, shuffle=False,
                           num_workers=1)
    testloader = DataLoader(db_test, batch_size=1, shuffle=False,
                           num_workers=1)
    macro_model.train()
    optimizer = optim.SGD(macro_model.parameters(), lr=base_lr,
                          momentum=0.9, weight_decay=0.0001)
    ce_loss = CrossEntropyLoss(ignore_index=4)
    dice_loss = losses.pDLoss(num_classes, ignore_index=4)
    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("{} iterations per epoch".format(len(trainloader_unlabeled)))
    iter_num = 0
    num_labeled = len(trainloader_labeled)
    num_unlabeled = len(trainloader_unlabeled)
    max_epoch = max_iterations // num_labeled + num_unlabeled + 1
    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)
    memobank = FeatureMemory(memory_per_class=10000, n_classes=4, k=[8,16,32])
    print(memobank.fts_memory[0].size())
    torch.distributed.init_process_group
    for epoch_num in iterator:
        for data_labeled, data_unlabeled in zip(trainloader_labeled, trainloader_unlabeled):
            sampled_batch_labeled, sampled_batch_unlabeled = data_labeled, data_unlabeled
            volume_batch, label_batch = sampled_batch_labeled['image'], sampled_batch_labeled['label']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()
            mask = label_batch.cpu()
            target_boundary = np.zeros((label_batch.shape[0], 256,256), dtype=np.uint8)
            for i in range(label_batch.shape[0]):
                label_image = (mask[i]).to(torch.uint8).numpy()
                edges = np.zeros_like(label_image, dtype=np.uint8)
                for class_id in range(1, 4):
                    class_mask = np.where(label_image == class_id, 255, 0).astype(np.uint8)
                    class_edges = cv2.Canny(class_mask, 0, 255)
                    edges = np.maximum(edges, class_edges)
                target_boundary[i] = edges   
            target_boundary[target_boundary == 255] = 1  
            target_boundary[target_boundary == 0] = 0 
            target_boundary = torch.tensor(target_boundary)
            volume_batch_scribble, label_batch_scribble  = sampled_batch_unlabeled['image'], sampled_batch_unlabeled['label']
            volume_batch_scribble, label_batch_scribblel = volume_batch_scribble.cuda(), label_batch_scribble.cuda()
            outputs_macro, fts, rep, boundary, boundary1, boundary2 = macro_model(volume_batch)
            outputs_scribble_macro, fts_scribble, rep_scribble, boundary_scribble, boundary1_scribble, boundary2_scribble = macro_model(volume_batch_scribble)
            outputs_macro_soft = F.softmax(outputs_macro, dim=1)
            #logits_t, label_t = torch.max(outputs_micro_soft, dim=1)
        
            # get pseudo label from micro model of scribble images
            outputs_macro_soft = F.softmax(outputs_scribble_macro, dim=1)
            logits_t_micro, label_t_micro = torch.max(outputs_macro_soft, dim=1)
            fts_bi_s = torch.cat((fts, fts_scribble), dim=0)
            rep_bi_s = torch.cat((rep, rep_scribble), dim=0)
            label_contra_memo = torch.cat([label_batch.long(), label_t_micro.long()], dim=0)
            h = 256
            w = 256
            fts_bi_s = F.interpolate(fts_bi_s.float(),size=(h,w),mode="nearest")
            rep_bi_s = F.interpolate(rep_bi_s.float(),size=(h,w),mode="nearest")
            #label_contra_memo = F.interpolate(rep_bi_s.float(),size=(h,w),mode="nearest")
            #if memobank.check_if_full():
            if memobank.check_if_full():
                with autocast():
                    contra_loss = compute_dgcl_loss(                        
                        rep = rep_bi_s,                    
                        fts = fts_bi_s.float().detach(),
                        memo = memobank,
                        label = label_contra_memo.detach(),
                        k_low_thresh=200,
                        k_den_cal=[8,16,32])
                if contra_loss is None:
                    contra_loss = 0*rep_bi_s.sum()
            else:
                with autocast():
                    contra_loss = 0*rep_bi_s.sum()
    
            loss_ce = ce_loss(outputs_macro, label_batch[:].long())
            outputs_macro_soft = torch.softmax(outputs_macro, dim=1)
            loss_ce_supervised = 0.5 * (loss_ce + dice_loss(outputs_macro_soft,
                          label_batch.unsqueeze(1)))
            outputs_scribble_macro,label_batch_scribble = outputs_scribble_macro.cuda(),label_batch_scribble.cuda()
            
            loss_ce_scribble = ce_loss(outputs_scribble_macro, label_batch_scribble[:].long())
            loss_pce = loss_ce_scribble
            
            loss_boun = F.binary_cross_entropy(boundary, target_boundary.cuda().unsqueeze(1).to(torch.float))
            
            rot_times = random.randrange(0, 4)
            rotated_volume_batch = torch.rot90(volume_batch_scribble, rot_times, [2, 3])
            noise = torch.clamp(torch.randn_like(
               rotated_volume_batch) * 0.1, -0.2, 0.2)

             # use pseudo label get pseudo boundary label
            target_boundary_scribble = np.zeros((label_t_micro.shape[0], 256,256), dtype=np.uint8)
            
            mask_scribble = label_t_micro.cpu()
            for i in range(label_t_micro.shape[0]):
                label_image = (mask_scribble[i]).to(torch.uint8).numpy()
                edges = np.zeros_like(label_image, dtype=np.uint8)
                for class_id in range(1, 4):
                    class_mask = np.where(label_image == class_id, 255, 0).astype(np.uint8)
                    class_edges = cv2.Canny(class_mask, 0, 255)
                    edges = np.maximum(edges, class_edges)
                target_boundary_scribble[i] = edges 
            
            target_boundary_scribble[target_boundary_scribble == 255] = 1  
            target_boundary_scribble[target_boundary_scribble == 0] = 0 
            
            target_boundary_scribble = torch.tensor(target_boundary_scribble)
            label = label_batch_scribble.cpu().numpy()
            
            loss_boun_scribble = F.binary_cross_entropy(mask*boundary_scribble, mask*target_boundary_scribble.cuda().unsqueeze(1).to(torch.float))
            loss_boun_all = loss_boun + loss_boun_scribble 
            loss = loss_ce_supervised+loss_pce+ loss_boun + contra_loss
            print('loss: %f loss_pce : %f loss_ce_supervised: %f loss_boun:%f loss_boun_scribble :%f loss_boun_all :%f contra_loss:%f' % (loss, loss_pce,loss_ce_supervised,loss_boun,loss_boun_scribble,loss_boun_all,contra_loss))
            optimizer.zero_grad()
             # Update memory bank
            memobank.update(fts_bi_s.float().detach(), rep_bi_s.float().detach(), label_contra_memo.detach()) 
            loss.backward(retain_graph=True)
            optimizer.step()
            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_
    
            iter_num = iter_num + 1

            logging.info(
                'iteration %d : loss : %f, loss_pce: %f, loss_ce_supervised:%f, loss_boun:%f' %
                (iter_num, loss.item(), loss_pce.item(),loss_ce_supervised.item(),loss_boun.item()))
    
            if iter_num % 20 == 0:
                image = volume_batch_scribble[1, 0:1, :, :]
                image = (image - image.min()) / (image.max() - image.min())
                writer.add_image('train/Image', image, iter_num)
                outputs_macro = torch.argmax(torch.softmax(
                    outputs_macro, dim=1), dim=1, keepdim=True)
                writer.add_image('train/Prediction',
                                 outputs_macro[1, ...] * 50, iter_num)
                labs = label_batch_scribble[1, ...].unsqueeze(0) * 50
                writer.add_image('train/GroundTruth', labs, iter_num)
    
            if iter_num > 0 and iter_num % 200 == 0:
                macro_model.eval()
                metric_list = 0.0
                for i_batch, sampled_batch in enumerate(valloader):
                    metric_i = test_single_volume(
                        sampled_batch["image"], sampled_batch["label"], macro_model, classes=num_classes)
                    metric_list += np.array(metric_i)
                metric_list = metric_list / len(db_val)
                for class_i in range(num_classes-1):
                    writer.add_scalar('info/val_{}_dice'.format(class_i+1),
                                      metric_list[class_i, 0], iter_num)
                    writer.add_scalar('info/val_{}_hd95'.format(class_i+1),
                                      metric_list[class_i, 1], iter_num)
    
                performance = np.mean(metric_list, axis=0)[0]
    
                mean_hd95 = np.mean(metric_list, axis=0)[1]
                writer.add_scalar('info/val_mean_dice', performance, iter_num)
                writer.add_scalar('info/val_mean_hd95', mean_hd95, iter_num)
    
                if performance > best_performance:
                    best_performance = performance
                    save_mode_path = os.path.join(snapshot_path,
                                                  'iter_{}_dice_{}.pth'.format(
                                                      iter_num, round(best_performance, 4)))
                    save_best = os.path.join(snapshot_path,
                                             '{}_best_model.pth'.format(args.model))
                    torch.save(macro_model.state_dict(), save_mode_path)
                    torch.save(macro_model.state_dict(), save_best)
    
                logging.info(
                    'iteration %d : mean_dice : %f mean_hd95 : %f' % (iter_num, performance, mean_hd95))
                macro_model.train()
    
            if iter_num % 3000 == 0:
                save_mode_path = os.path.join(
                    snapshot_path, 'iter_' + str(iter_num) + '.pth')
                torch.save(macro_model.state_dict(), save_mode_path)
                logging.info("save model to {}".format(save_mode_path))
        if iter_num >= max_iterations:
            iterator.close()
            break
    writer.close()
    return "Training Finished!"


if __name__ == "__main__":
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    snapshot_path = "../macro_model/{}_{}/{}".format(
        args.exp, args.fold, args.sup_type)
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    if os.path.exists(snapshot_path + '/code'):
        shutil.rmtree(snapshot_path + '/code')
    shutil.copytree('.', snapshot_path + '/code',
                    shutil.ignore_patterns(['.git', '__pycache__']))

    logging.basicConfig(filename=snapshot_path+"/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    train(args, snapshot_path)
