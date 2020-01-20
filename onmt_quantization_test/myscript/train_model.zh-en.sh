#!/bin/bash

CUDA_VISIBLE_DEVICES=6 python train.py -config myconfig/config-transformer-base-1GPU.yml
