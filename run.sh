#!/bin/bash

GPU_IDS=(0 1 2 3)
ATTACK_LAYER=(4 5 6 7)

for ((i=0; i<4; i++))
do
    echo CUDA_VISIBLE_DEVICES=${GPU_IDS[$i]} python script_generate_adversarial_imagenet5000.py --dataset-dir ~/imagenet5000 --adv-method dr -tm resnet152 --res152-attacklayer ${ATTACK_LAYER[$i]} --step-size 2 --steps 500
done
