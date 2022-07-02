import numpy as np
from os import listdir
from os.path import isfile, isdir, join
import os
import json
import random
import re

cwd = os.getcwd()
data_path = join(cwd, 'Dataset')
savedir = './'
dataset_list = ['base', 'val', 'novel']

cl = -1
folderlist = []  # to store label??

datasetmap = {'base': 'train', 'val': 'val', 'novel': 'test'}

filelists = {'base': {}, 'val': {}, 'novel': {}}
filelists_flat = {'base': [], 'val': [], 'novel': []}
labellists_flat = {'base': [], 'val': [], 'novel': []}

for dataset in dataset_list:
    with open(datasetmap[dataset] + ".csv", "r") as lines:
        for i, line in enumerate(lines):
            if i == 0:
                continue
            fid, _, label = re.split(',|\.', line)
            label = label.replace('\n', '')
            if not label in filelists[dataset]:
                folderlist.append(label)
                filelists[dataset][label] = []  # new label
                fnames = listdir(join(data_path, label))
            name = fid + '.jpg'
            fname = join(data_path, label, name)
            filelists[dataset][label].append(fname)

    for key, filelist in filelists[dataset].items():
        cl += 1
        random.shuffle(filelist)
        filelists_flat[dataset] += filelist
        labellists_flat[dataset] += np.repeat(cl, len(filelist)).tolist()

for dataset in dataset_list:
    fo = open(savedir + dataset + ".json", "w")
    fo.write('{"label_names": [')
    fo.writelines(['"%s",' % item for item in folderlist])
    fo.seek(0, os.SEEK_END)
    fo.seek(fo.tell()-1, os.SEEK_SET)
    fo.write('],')

    fo.write('"image_names": [')
    fo.writelines(['"%s",' % item for item in filelists_flat[dataset]])
    fo.seek(0, os.SEEK_END)
    fo.seek(fo.tell()-1, os.SEEK_SET)
    fo.write('],')

    fo.write('"image_labels": [')
    fo.writelines(['%d,' % item for item in labellists_flat[dataset]])
    fo.seek(0, os.SEEK_END)
    fo.seek(fo.tell()-1, os.SEEK_SET)
    fo.write(']}')

    fo.close()
    print("%s -OK" % dataset)
