import glob
import json
import os
import pdb
import pprint
import random
import time

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data.sampler
import tqdm
from torch.autograd import Variable
from torchsummary import summary

import backbone
import configs
import data.feature_loader as feat_loader
import wandb
from data.datamgr import SetDataManager
from io_utils import (get_assigned_file, get_best_file,
                      model_dict, parse_args)
from methods.CTX import CTX
from methods.transformer import FewShotTransformer

global device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train(base_loader, val_loader, model, optimization, num_epoch, params):
    if optimization == 'Adam':
        optimizer = torch.optim.Adam(
            model.parameters(), lr=params.learning_rate, weight_decay=params.weight_decay)
    elif optimization == 'AdamW':
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=params.learning_rate, weight_decay=params.weight_decay)
    elif optimization == 'SGD':
        optimizer = torch.optim.SGD(
            model.parameters(), lr=params.learning_rate, momentum=params.momentum, weight_decay=params.weight_decay)
    else:
        raise ValueError('Unknown optimization, please define by yourself')

    max_acc = 0

    for epoch in range(num_epoch):
        model.train()

        model.train_loop(epoch, num_epoch, base_loader,
                         params.wandb,  optimizer)
        with torch.no_grad():
            model.eval()

            if not os.path.isdir(params.checkpoint_dir):
                os.makedirs(params.checkpoint_dir)

            acc = model.val_loop(val_loader, epoch, params.wandb)
            if acc > max_acc:  
                print("best model! save...")
                max_acc = acc
                outfile = os.path.join(params.checkpoint_dir, 'best_model.tar')
                torch.save(
                    {'epoch': epoch, 'state': model.state_dict()}, outfile)
                # if params.wandb:
                #     wandb.save(outfile)

            if (epoch % params.save_freq == 0) or (epoch == num_epoch-1):
                outfile = os.path.join(
                    params.checkpoint_dir, '{:d}.tar'.format(epoch))
                torch.save(
                    {'epoch': epoch, 'state': model.state_dict()}, outfile)
        print()

    return model

def direct_test(test_loader, model, params):

    correct = 0
    count = 0
    acc = []

    iter_num = len(test_loader)
    with tqdm.tqdm(total=len(test_loader)) as pbar:
        for i, (x, _) in enumerate(test_loader):
            scores = model.set_forward(x)
            pred = scores.data.cpu().numpy().argmax(axis=1)
            y = np.repeat(range(params.n_way), pred.shape[0]//params.n_way)
            acc.append(np.mean(pred == y)*100)
            pbar.set_description(
                'Test       | Acc {:.6f}'.format(np.mean(acc)))
            pbar.update(1)

    acc_all = np.asarray(acc)
    acc_mean = np.mean(acc_all)
    acc_std = np.std(acc_all)
    return acc_mean, acc_std

def seed_func():
    seed = 4040 
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(10)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def change_model(model_name):
    if model_name == 'Conv4':
        model_name = 'Conv4NP'
    elif model_name == 'Conv6':
        model_name = 'Conv6NP'
    elif model_name == 'Conv4S':
        model_name = 'Conv4SNP'
    elif model_name == 'Conv6S':
        model_name = 'Conv6SNP'
    return model_name

if __name__ == '__main__':
    
    params = parse_args()
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(vars(params))
    print()
        
    project_name = "Few-Shot_TransFormer"            
    
    if params.dataset == 'Omniglot': params.n_query = 15
    if params.wandb:

        wandb_name = params.method + "_" + params.backbone + "_" + params.dataset + \
            "_" + str(params.n_way) + "w" + str(params.k_shot) + "s"
        
        if params.train_aug:
            wandb_name += "_aug"
        if params.feti and 'ResNet' in params.backbone:
            wandb_name += "_feti"
        wandb_name += "_" + params.datetime
        
        wandb.init(project=project_name, name=wandb_name,
                   config=params, id=params.datetime)
        print()

    if params.dataset == 'cross':
        base_file = configs.data_dir['miniImagenet'] + 'all.json'
        val_file = configs.data_dir['CUB'] + 'val.json'
    elif params.dataset == 'cross_char':
        base_file = configs.data_dir['Omniglot'] + 'noLatin.json'
        val_file = configs.data_dir['emnist'] + 'val.json'
    else:
        base_file = configs.data_dir[params.dataset] + 'base.json'
        val_file = configs.data_dir[params.dataset] + 'val.json'

    if params.dataset == "CIFAR":
        image_size = 112 if 'ResNet' in params.backbone else 64
    else:
        image_size = 224 if 'ResNet' in params.backbone else 84

    if params.dataset in ['Omniglot', 'cross_char']:
        if params.backbone == 'Conv4': params.backbone = 'Conv4S'
        if params.backbone == 'Conv6': params.backbone = 'Conv6S'

    optimization = params.optimization

    if params.method in ['FSTF_softmax', 'FSTF_cosine', 'CTX_softmax', 'CTX_cosine']:

        few_shot_params = dict(
            n_way=params.n_way, k_shot=params.k_shot, n_query = params.n_query)
        base_datamgr = SetDataManager(
            image_size, n_episode=params.n_episode, **few_shot_params)
        base_loader = base_datamgr.get_data_loader(
            base_file, aug=params.train_aug)

        val_datamgr = SetDataManager(
            image_size, n_episode=params.n_episode, **few_shot_params)
        val_loader = val_datamgr.get_data_loader(
            val_file, aug=False)
       
        seed_func()

        if params.method in ['FSTF_softmax', 'FSTF_cosine']:
            variant = 'cosine' if params.method == 'FSTF_cosine' else 'softmax'
            
            def feature_model():
                if params.dataset in ['Omniglot', 'cross_char']:
                    params.backbone = change_model(params.backbone)
                return model_dict[params.backbone](params.feti, params.dataset, flatten=True) if 'ResNet' in params.backbone else model_dict[params.backbone](params.dataset, flatten=True)

            model = FewShotTransformer(feature_model, variant=variant, **few_shot_params)
            
        elif params.method in ['CTX_softmax', 'CTX_cosine']:
            variant = 'cosine' if params.method == 'CTX_cosine' else 'softmax'
            input_dim = 512 if "ResNet" in params.backbone else 64
            def feature_model():
                if params.dataset in ['Omniglot', 'cross_char']:
                    params.backbone = change_model(params.backbone)
                return model_dict[params.backbone](params.feti, params.dataset, flatten=False) if 'ResNet' in params.backbone else model_dict[params.backbone](params.dataset, flatten=False)

            model = CTX(feature_model, variant=variant, input_dim=input_dim, **few_shot_params)
    else:
        raise ValueError('Unknown method')

    model = model.to(device)
    
    
    params.checkpoint_dir = '%scheckpoints/%s/%s_%s' % (
        configs.save_dir, params.dataset, params.backbone, params.method)
    if params.train_aug:
        params.checkpoint_dir += '_aug'
    if params.feti and 'ResNet' in params.backbone:
        params.checkpoint_dir += '_feti'

    params.checkpoint_dir += '_%dway_%dshot' % (
        params.n_way, params.k_shot)
    
    if not os.path.isdir(params.checkpoint_dir):
        os.makedirs(params.checkpoint_dir)
    
    print("===================================")
    print("Train phase: ")
    
    model = train(base_loader, val_loader,  model, optimization, params.num_epoch, params)


######################################################################

    print("===================================")
    print("Test phase: ")

    iter_num = params.test_iter
    
    split = params.split
    if params.dataset == 'cross':
        if split == 'base':
            testfile = configs.data_dir['miniImagenet'] + 'all.json'
        else:
            testfile = configs.data_dir['CUB'] + split + '.json'
    elif params.dataset == 'cross_char':
        if split == 'base':
            testfile = configs.data_dir['Omniglot'] + 'noLatin.json'
        else:
            testfile = configs.data_dir['emnist'] + split + '.json'
    else:
        testfile = configs.data_dir[params.dataset] + split + '.json'
        
        
    if params.save_iter != -1:
        modelfile = get_assigned_file(params.checkpoint_dir, params.save_iter)
    else:
        modelfile = get_best_file(params.checkpoint_dir)
        
    test_datamgr = SetDataManager(
        image_size, n_episode=iter_num,  **few_shot_params)
    test_loader = test_datamgr.get_data_loader(testfile, aug=False)
 
    acc_all = []
   
    model = model.to(device)

    root = os.getcwd()

    if params.save_iter != -1:
        modelfile = get_assigned_file(params.checkpoint_dir, params.save_iter)
    else:
        modelfile = get_best_file(params.checkpoint_dir)
    if modelfile is not None:
        tmp = torch.load(modelfile)
        model.load_state_dict(tmp['state'])

    split = params.split
    if params.save_iter != -1:
        split_str = split + "_" + str(params.save_iter)
    else:
        split_str = split
    
    acc_mean, acc_std = direct_test(test_loader, model, params)
        
    print('%d Test Acc = %4.2f%% +- %4.2f%%' %
            (iter_num, acc_mean, 1.96 * acc_std/np.sqrt(iter_num)))
    
    if params.wandb:
        wandb.log({'Test Acc': acc_mean})
            
    with open('./record/results.txt', 'a') as f:

        timestamp = params.datetime

        aug_str = '-aug' if params.train_aug else ''
        aug_str += '-feti' if params.feti and 'ResNet' in params.backbone else ''

        if params.backbone == "Conv4SNP": 
            params.backbone = "Conv4"
        elif params.backbone == "Conv6SNP":
            params.backbone = "Conv6"
        exp_setting = '%s-%s-%s%s-%sw%ss' % (params.dataset, params.backbone, 
                params.method, aug_str, params.n_way, params.k_shot)
        
        acc_str = 'Test Acc = %4.2f%% +- %4.2f%%' % (acc_mean, 1.96 * acc_std/np.sqrt(iter_num))
        
        f.write('Time: %s   Setting: %s %s \n' % (timestamp, exp_setting.ljust(50), acc_str))
        
    wandb.finish()