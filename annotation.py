import os
import random

import numpy as np
from PIL import Image
from tqdm import tqdm

trainval_percent    = 1
train_percent       = 0.85

Image_path      = 'weizmann_horse_db'

if __name__ == "__main__":
    random.seed(0)
    segfilepath     = os.path.join(Image_path, 'mask')
    saveBasePath    = os.path.join(Image_path, 'annotation')
    
    temp_seg = os.listdir(segfilepath)
    total_seg = []
    for seg in temp_seg:
        if seg.endswith(".png"):
            total_seg.append(seg)

    num     = len(total_seg)  
    list    = range(num)  
    all      = int(num*trainval_percent)  
    tr      = int(all*train_percent)  
    trainval= random.sample(list,all)  
    train   = random.sample(trainval,tr)  
    
    print("train and val size",all)
    print("traub suze",tr)
    ftrainval   = open(os.path.join(saveBasePath,'trainval.txt'), 'w')  
    ftest       = open(os.path.join(saveBasePath,'test.txt'), 'w')  
    ftrain      = open(os.path.join(saveBasePath,'train.txt'), 'w')  
    fval        = open(os.path.join(saveBasePath,'val.txt'), 'w')  
    
    for i in list:  
        name = total_seg[i][:-4]+'\n'  
        if i in trainval:  
            ftrainval.write(name)  
            if i in train:  
                ftrain.write(name)  
            else:  
                fval.write(name)  
        else:  
            ftest.write(name)  
    
    ftrainval.close()  
    ftrain.close()  
    fval.close()  
    ftest.close()
    classes_nums        = np.zeros([256], np.int)
    for i in tqdm(list):
        name            = total_seg[i]
        png_file_name   = os.path.join(segfilepath, name)
        png             = np.array(Image.open(png_file_name), np.uint8)
        classes_nums += np.bincount(np.reshape(png, [-1]), minlength=256)
            
    

