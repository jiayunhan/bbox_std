#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python script_generate_adversarial_imagenet5000.py --adv-method dr -tm inception_v3 --inc3-attacklayer 7 --step-size 4 --steps 100