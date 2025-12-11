import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--root_train_path', type=str, default='/autodl-fs/data/NCTCRCHE/train/', help='dataset root dir')
parser.add_argument('--root_val_path', type=str, default='/autodl-fs/data/NCTCRCHE/val/', help='dataset root dir')
parser.add_argument('--root_test_path', type=str, default='/autodl-fs/data/NCTCRCHE/test/', help='dataset root dir')
parser.add_argument('--csv_file_train', type=str, default='/PDMatchEnv/PDMatch/data/NCTCRCHE/train.csv', help='training set csv file')
parser.add_argument('--csv_file_val', type=str, default='/PDMatchEnv/PDMatch/data/NCTCRCHE/val.csv', help='validation set csv file')
parser.add_argument('--csv_file_test', type=str, default='/PDMatchEnv/PDMatch/data/NCTCRCHE/test.csv', help='testing set csv file')
parser.add_argument('--epochs', type=int,  default=50, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=48, help='batch_size per gpu')
parser.add_argument('--labeled_bs', type=int, default=12, help='number of labeled data per batch')
parser.add_argument('--labeled_ratio', type=float, default=0.001, help='ratio of labeled')
parser.add_argument('--pretrain', type=str,  default=None, help='pretrain') # '/PDMatchEnv/PDMatch/code/pretrain/pretrain_nct.pth'
parser.add_argument('--exp', type=str,  default='PD', help='model_name')
parser.add_argument('--drop_rate', type=float, default=0.01, help='dropout rate')
parser.add_argument('--base_lr', type=float,  default=1e-4, help='maximum epoch number to train')
parser.add_argument('--deterministic', type=int,  default=1, help='whether use deterministic training')
parser.add_argument('--seed', type=int,  default=7900, help='random seed')
parser.add_argument('--gpu', type=str,  default='0,1', help='GPU to use')
parser.add_argument('--start_epoch', type=int,  default=0, help='start_epoch')
parser.add_argument('--global_step', type=int,  default=0, help='global_step')
parser.add_argument('--label_uncertainty', type=str,  default='U-Ones', help='label type')
parser.add_argument('--consistency_start', type=int,  default=20, help='consistency_start')

args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
import sys
# from tensorboardX import SummaryWriter
import shutil
import logging
import time
import random
import numpy as np
import pandas as pd
from tensorboardX import SummaryWriter
import torch
import torch.optim as optim
from torchvision import transforms
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torchvision.utils import make_grid

from networks.models import DenseNet121
from utils import losses, ramps
from utils.metrics import compute_AUCs
from utils.metric_logger import MetricLogger
from dataloaders import  dataset
from dataloaders.dataset import TwoStreamBatchSampler
from utils.util import get_timestamp
from utils.epoch_metrics import epoch_metrics
from functions.getPrototype import update_prototypes_batch,update_prototypes_epoch
from functions.getDBL import update_maw,update_daw
from functions.getEdgeList import PDEV_MPUS_masks,compute_dists
from collections import defaultdict

snapshot_path = "../model/" + args.exp + "/"

batch_size = args.batch_size * len(args.gpu.split(','))
base_lr = args.base_lr
labeled_bs = args.labeled_bs * len(args.gpu.split(','))

if args.deterministic:
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)


def ensure_numpy(x):
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        elif isinstance(x, np.ndarray):
            return x
        else:
            raise ValueError(f"Unsupported input type: {type(x)}")
def get_labeled_num(labeled_num, labeled_bs,minnum=10):
    min_required = minnum* labeled_bs
    if labeled_num < min_required:
        labeled_num = ((labeled_num + labeled_bs - 1) // labeled_bs) * labeled_bs
    return int(labeled_num)

def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)

if __name__ == "__main__":
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
        os.makedirs(snapshot_path + './checkpoint')
    if os.path.exists(snapshot_path + '/code'):
        shutil.rmtree(snapshot_path + '/code')
    shutil.copytree('.', snapshot_path + '/code', shutil.ignore_patterns(['.git','__pycache__']))

    logging.basicConfig(filename=snapshot_path+"/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    def create_model(ema=False):
        # Network definition
        net = DenseNet121(out_size=dataset.N_CLASSES, mode=args.label_uncertainty, drop_rate=args.drop_rate)
        if len(args.gpu.split(',')) > 1:
            net = torch.nn.DataParallel(net)
        model = net.cuda()
        if ema:
            for param in model.parameters():
                param.detach_()
        return model

    model = create_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.base_lr, 
                                 betas=(0.9, 0.999), weight_decay=5e-4)
    iter_num = args.global_step
    lr_ = base_lr
    model.train()
    D=1024
    protos = torch.zeros(dataset.N_CLASSES, D).cuda()
    maw = torch.ones(dataset.N_CLASSES, dtype=torch.long).cuda() 
    # load pretrain
    if args.pretrain:
        assert os.path.isfile(args.pretrain), f"=> no checkpoint found at '{args.pretrain}'"
        checkpoint = torch.load(args.pretrain)
        args.start_epoch = checkpoint['epoch']
        args.global_step = checkpoint['global_step']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_ = base_lr * (0.9 ** args.start_epoch)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_
        if 'rng_state' in checkpoint:
            random.setstate(checkpoint['rng_state']['random'])
            np.random.set_state(checkpoint['rng_state']['np'])
            torch.set_rng_state(checkpoint['rng_state']['torch'])
            torch.cuda.set_rng_state_all(checkpoint['rng_state']['cuda'])
        if 'labeled_idxs' in checkpoint and 'unlabeled_idxs' in checkpoint:
            labeled_idxs = checkpoint['labeled_idxs']
            unlabeled_idxs = checkpoint['unlabeled_idxs']
        logging.info(f"=> loaded checkpoint '{args.pretrain}")
    
    # dataset
    normalize = transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])

    train_dataset = dataset.GetDataset(root_dir=args.root_train_path,
                                            csv_file=args.csv_file_train,
                                            transform=dataset.TransformTwice(transforms.Compose([
                                                transforms.Resize((224, 224)),
                                                transforms.RandomAffine(degrees=10, translate=(0.02, 0.02)),
                                                transforms.RandomHorizontalFlip(),
                                                transforms.RandomVerticalFlip(),
                                                transforms.ToTensor(),
                                                normalize,
                                            ])))
    val_dataset = dataset.GetDataset(root_dir=args.root_val_path,
                                          csv_file=args.csv_file_val,
                                          transform=transforms.Compose([
                                              transforms.Resize((224, 224)),
                                              transforms.ToTensor(),
                                              normalize,
                                          ]))
    test_dataset = dataset.GetDataset(root_dir=args.root_test_path,
                                          csv_file=args.csv_file_test,
                                          transform=transforms.Compose([
                                              transforms.Resize((224, 224)),
                                              transforms.ToTensor(),
                                              normalize,
                                          ]))
    train_num= len(pd.read_csv(args.csv_file_train))
    labeled_num=get_labeled_num(args.labeled_ratio*train_num,args.labeled_bs)
    labeled_idxs = list(range(labeled_num))
    unlabeled_idxs = list(range(labeled_num,train_num))
    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, batch_size, batch_size-labeled_bs)

    def worker_init_fn(worker_id):
        random.seed(args.seed+worker_id)
        
    train_dataloader = DataLoader(dataset=train_dataset, batch_sampler=batch_sampler,
                                  num_workers=8, pin_memory=True, worker_init_fn=worker_init_fn)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=batch_size,
                                shuffle=False, num_workers=8, pin_memory=True, worker_init_fn=worker_init_fn)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size,
                                shuffle=False, num_workers=8, pin_memory=True)
    
    model.train()

    major_classes=[6]
    topk=int((dataset.N_CLASSES+1)/2)
    loss_dbl=losses.DBLLoss(major_classes)
    
    writer = SummaryWriter(snapshot_path+'/log')
    
    #train
    for epoch in range(args.start_epoch, args.epochs):
        time1 = time.time()
        iter_max = len(train_dataloader) 
        feats_labeled_all=[]
        labels_labeled_all=[]
        for i, (_,_, (image_batch, strong_image_batch), label_batch) in enumerate(train_dataloader):
            time2 = time.time()
            image_batch, strong_image_batch,label_batch = image_batch.cuda(),strong_image_batch.cuda(), label_batch.cuda()
            lab_labels=label_batch[:labeled_bs]

            activations, outputs = model(image_batch)
            _, strong_outputs = model(strong_image_batch)
            
            lab_outputs=outputs[:labeled_bs]
            unlab_outputs=outputs[labeled_bs:]
            lab_activations=activations[:labeled_bs]
            unlab_activations=activations[labeled_bs:]
            strong_lab_outputs=strong_outputs[:labeled_bs]
            strong_unlab_outputs=strong_outputs[labeled_bs:]
            unlab_probs = F.softmax(unlab_outputs, dim=1)
            
            if epoch ==args.consistency_start: 
                maw=update_maw(lab_labels, maw)
            
            if epoch > args.consistency_start: 
                kmeans_protos=update_prototypes_batch(kmeans_protos,cluster_nums_tensor,lab_activations,lab_labels)
                topk_class_idx,topk_dists=compute_dists(unlab_activations, kmeans_protos,cluster_nums_tensor,topk,dataset.N_CLASSES,D)
                km_masks,km_labels=PDEV_MPUS_masks(unlab_probs,major_classes,topk_class_idx,topk_dists)
                daw=update_daw(kmeans_protos,cluster_nums_tensor,unlab_activations[km_masks],km_labels,unlab_probs[km_masks],major_classes,dataset.N_CLASSES,D)
                if km_masks.any():  
                    labels_combined = torch.cat([lab_labels, km_labels], dim=0)
                    maw=update_maw(labels_combined, maw)
                    loss_pseudo =loss_dbl(strong_unlab_outputs[km_masks], km_labels,maw,daw)
                else:
                    maw=update_maw(lab_labels, maw)
                    loss_pseudo=0.0
            else:
                loss_pseudo=0.0
                
            if epoch <=args.consistency_start:
                loss_classification =loss_dbl(lab_outputs, lab_labels)+loss_dbl(strong_lab_outputs, lab_labels)
                loss=loss_classification
            else:  
                loss_classification =loss_dbl(lab_outputs, lab_labels,maw)+loss_dbl(strong_lab_outputs, lab_labels,maw)
                loss = loss_classification  +loss_pseudo
                  
            if epoch >=args.consistency_start:
                feats_labeled = ensure_numpy(lab_activations)
                labels_labeled = ensure_numpy(lab_labels)
                protos_numpy = ensure_numpy(protos)
                feats_labeled_all.append(feats_labeled)
                labels_labeled_all.append(labels_labeled)
                
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            iter_num = iter_num + 1
            if (i % 10 == 0) or (i%(iter_max-1)==0):
                logging.info("\nEpoch: {}, iteration: {}/{}, ==> train <===, loss: {:.6f}, classification loss: {:.6f}, loss_pseudo: {:.6f}"
                            .format(epoch, i, iter_max, loss,  loss_classification,loss_pseudo))
                
        timestamp = get_timestamp()
        
        if (epoch >= args.consistency_start):
            feats_labeled_all_np= np.concatenate(feats_labeled_all, axis=0)
            labels_labeled_all_np= np.concatenate(labels_labeled_all, axis=0)
            kmeans_protos, cluster_nums_tensor=update_prototypes_epoch(feats_labeled_all_np, labels_labeled_all_np, losses.CLASS_NUM,args.seed,topk )
            feats_labeled_all.clear()
            labels_labeled_all.clear()

        #val
        AUROCs, Accus, Senss, Specs = epoch_metrics(model, val_dataloader,dataset.CLASS_NAMES)  
        AUROC_avg = np.array(AUROCs).mean()
        Accus_avg = np.array(Accus).mean()
        Senss_avg = np.array(Senss).mean()
        Specs_avg = np.array(Specs).mean()
        
        # test 
        test_AUROCs, test_Accus, test_Senss, test_Specs = epoch_metrics(model, test_dataloader,dataset.CLASS_NAMES)  
        test_AUROC_avg = np.array(test_AUROCs).mean()
        test_Accus_avg = np.array(test_Accus).mean()
        test_Senss_avg = np.array(test_Senss).mean()
        test_Specs_avg = np.array(test_Specs).mean()
        logging.info("\nTEST Student: Epoch: {}, iteration: {}".format(epoch, i))
        logging.info("\nTEST TEST Accus: {:6f}, TEST Senss: {:6f}, TEST Specs: {:6f}, AUROC: {:6f}"
                    .format(test_Accus_avg, test_Senss_avg, test_Specs_avg,test_AUROC_avg ))
        logging.info("Accus: " + " ".join(["{}:{:.6f}".format(dataset.CLASS_NAMES[i], v) for i,v in enumerate(test_Accus)]))
        logging.info("Senss: " + " ".join(["{}:{:.6f}".format(dataset.CLASS_NAMES[i], v) for i,v in enumerate(test_Senss)]))
        logging.info("Specs: " + " ".join(["{}:{:.6f}".format(dataset.CLASS_NAMES[i], v) for i,v in enumerate(test_Specs)]))
        logging.info("AUROCs: " + " ".join(["{}:{:.6f}".format(dataset.CLASS_NAMES[i], v) for i,v in enumerate(test_AUROCs)]))
        
        # save model
        save_mode_path = os.path.join(snapshot_path + 'checkpoint/', 'epoch_' + str(epoch+1) + '.pth')
        torch.save({
            'epoch': epoch + 1,
            'global_step': iter_num,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'rng_state': {
                'random': random.getstate(),
                'np': np.random.get_state(),
                'torch': torch.get_rng_state(),
                'cuda': torch.cuda.get_rng_state_all()
            },
            'labeled_idxs': labeled_idxs,
            'unlabeled_idxs': unlabeled_idxs
        }, save_mode_path)
        logging.info("save model to {}".format(save_mode_path))
        lr_ = lr_ * 0.9
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_
    save_mode_path = os.path.join(snapshot_path, 'iter_'+str(iter_num+1)+'.pth')
    torch.save(model.state_dict(), save_mode_path)
    logging.info("save model to {}".format(save_mode_path))
    writer.close()
