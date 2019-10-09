#!/bin/bash

CUDA_VISIBLE_DEVICES=2 python script_generate_adversarial_imagenet5000.py --dataset-dir ~/yantao/imagenet5000 --adv-method dr -tm inception_v3 --inc3-attacklayer 12 --step-size 4 --steps 100