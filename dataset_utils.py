import numpy as np 
import os
import shutil

import pdb

def generate_imagenet_testdata(input_dir, output_dir, num_imgs=100):
    class_dir_list = os.listdir(input_dir)
    selected_image_names = np.random.choice(class_dir_list, num_imgs)
    for temp_image_name in selected_image_names:
        shutil.copyfile(os.path.join(input_dir, temp_image_name), os.path.join(output_dir, temp_image_name))


if __name__ == "__main__":
    generate_imagenet_testdata("/home/yantao/datasets/ILSVRC/Data/DET/test/", "/home/yantao/datasets/imagenet_100image/")