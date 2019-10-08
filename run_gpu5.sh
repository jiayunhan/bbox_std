#!/bin/bash

CUDA_VISIBLE_DEVICES=5 python script_generate_adversarial_imagenet5000.py --adv-method dr -tm inception_v3 --inc3-attacklayer 6 --step-size 4 --steps 100