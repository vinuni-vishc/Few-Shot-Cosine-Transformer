#!/usr/bin/env bash
DATADIR=./Dataset
mkdir -p $DATADIR
unzip data_part1.zip -d $DATADIR
unzip data_part2.zip -d $DATADIR
unzip data_part3.zip -d $DATADIR
unzip data_part4.zip -d $DATADIR
unzip data_part5.zip -d $DATADIR
mv $DATADIR/data_part1/* $DATADIR/
mv $DATADIR/data_part2/* $DATADIR/
mv $DATADIR/data_part3/* $DATADIR/
mv $DATADIR/data_part4/* $DATADIR/
mv $DATADIR/data_part5/* $DATADIR/
rmdir $DATADIR/data_part1
rmdir $DATADIR/data_part2
rmdir $DATADIR/data_part3
rmdir $DATADIR/data_part4
rmdir $DATADIR/data_part5

python write_yoga_filelist.py
