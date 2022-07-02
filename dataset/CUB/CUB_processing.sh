#!/usr/bin/env bash
# download dataset from http://www.vision.caltech.edu/datasets/cub_200_2011/
tar -zxvf CUB_200_2011.tgz
python write_CUB_filelist.py
