import numpy as np
import os
import glob
import argparse
import backbone
import datetime
model_dict = dict(
            Conv4 = backbone.Conv4,
            Conv4S = backbone.Conv4S,
            Conv4NP = backbone.Conv4NP,
            Conv4SNP = backbone.Conv4SNP,
            Conv6 = backbone.Conv6,
            Conv6S = backbone.Conv6S,
            Conv6NP=backbone.Conv6NP,
            Conv6SNP = backbone.Conv6SNP,
            ResNet12 = backbone.ResNet12,
            ResNet18 = backbone.ResNet18,
            ResNet34 = backbone.ResNet34) 

def parse_args():
    parser = argparse.ArgumentParser(description= 'few-shot script' )
    parser.add_argument('--dataset'         , default='miniImagenet', help='CIFAR/CUB/miniImagenet/cross/Omniglot/cross_char/Yoga/')
    parser.add_argument('--backbone'        , default='ResNet18',      help='backbone: Conv{4|6} / ResNet{12|18|34}')
    parser.add_argument('--method'          , default='FSCT_cosine',   help='CTX_softmax/CTX_cosine/FSCT_softmax/FSCT_cosine') 
    parser.add_argument('--n_way'           , default=5, type=int,  help='number of categories')
    parser.add_argument('--n_query'         , default=16, type=int,  help='number of query samples per category')
    parser.add_argument('--k_shot'          , default=5, type=int,  help='number of labeled data per category') 
    parser.add_argument('--train_aug'       , type=int, default=0, help='[1:0] - [True:False]; perform data augmentation or not during training')
    parser.add_argument('--n_episode'       , default=200, type=int, help='number of iteration (episode) per epoch for training/validating')
    parser.add_argument('--FETI'            , default=0, type=int, help='[1:0] - [True:False]; Use pre-trained model on ImageNet subset that is trained non-overlapped with mini-ImgeNet test set. Only support ResNet backbone')
    parser.add_argument('--test_iter'       , default=600, type=int, help ='Number of iteration (episode) for testing') 
    parser.add_argument('--learning_rate'   , default=1e-3, type=float, help='learning rate')
    parser.add_argument('--weight_decay'    , default=1e-5, type=float, help='weight decay')
    parser.add_argument('--momentum'        , type=float, default=0.9, help='momentum')
    parser.add_argument('--optimization'    , type=str, default='AdamW', help='Optimization algorithms. Support Adam, AdamW, SGD')
    parser.add_argument('--wandb'           , type=int, default=0, help='[1:0] - [True:False]; Wandb Log, only for train.py and train_save_test.py')
    parser.add_argument('--datetime'        , default = str("{:%Y%m%d@%H%M%S}".format(datetime.datetime.now())), help='Execute time log')

    parser.add_argument('--save_freq'       , default=50, type=int, help='Save frequency')
    parser.add_argument('--num_epoch'       , default=50, type=int, help ='Stopping epoch')
    parser.add_argument('--split'           , default='novel', help='base/val/novel, only for train.py and train_save_test.py')
                                              # default novel, but you can also test base/val class accuracy if you want 
    parser.add_argument('--save_iter'       , default=-1, type=int,help ='save feature from the model trained in x epoch, use the best model if x is -1')
    return parser.parse_args()


def get_assigned_file(checkpoint_dir,num):
    assign_file = os.path.join(checkpoint_dir, '{:d}.tar'.format(num))
    return assign_file

def get_best_file(checkpoint_dir):    
    best_file = os.path.join(checkpoint_dir, 'best_model.tar')
    if os.path.isfile(best_file):
        return best_file
