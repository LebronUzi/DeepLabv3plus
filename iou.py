import os
import numpy as np
from PIL import Image
import torch
import cv2

def img_colormap2label(img):
    colormap=[[0,0,0],[128,0,0]]
    colormap2label=torch.zeros(img.shape**3)
    for i,colormap in enumerate(colormap):
        colormap2label[(colormap[0]*256+colormap[1])*256+colormap[2]]=i
    return colormap2label

def img_label_indices(img,colormap2label):
    idx=(img[:,:,0]*256+img[:,:,1])*256+img[:,:,2]
    return colormap2label[idx]

def fast_hist(a, b, n):
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)  

def per_class_iu(hist):
    return np.diag(hist) / np.maximum((hist.sum(1) + hist.sum(0) - np.diag(hist)), 1) 

def per_class_PA(hist):
    return np.diag(hist) / np.maximum(hist.sum(1), 1) 

def mask_to_boundary(mask, dilation_ratio=0.02):
    """
    Convert binary mask to boundary mask.
    :param mask (numpy array, uint8): binary mask
    :param dilation_ratio (float): ratio to calculate dilation = dilation_ratio * image_diagonal
    :return: boundary mask (numpy array)
    """
    h, w = mask.shape
    img_diag = np.sqrt(h ** 2 + w ** 2) # 计算图像对角线长度
    dilation = int(round(dilation_ratio * img_diag))
    if dilation < 1:
        dilation = 1
    # Pad image so mask truncated by the image border is also considered as boundary.
    new_mask = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    kernel = np.ones((3, 3), dtype=np.uint8)
    new_mask_erode = cv2.erode(new_mask, kernel, iterations=dilation)
    
    # 因为之前向四周填充了0, 故而这里不再需要四周
    mask_erode = new_mask_erode[1 : h + 1, 1 : w + 1]
    
    # G_d intersects G in the paper.
    return mask - mask_erode

def compute_mIoU(gt_dir, pred_dir, png_name_list, num_classes):  
    print('Num classes', num_classes)  
    
    hist = np.zeros((num_classes, num_classes))#混淆矩阵
    hist_boundary = np.zeros((num_classes, num_classes))#混淆矩阵
    
    gt_imgs     = [os.path.join(gt_dir, x + ".png") for x in png_name_list]  
    pred_imgs   = [os.path.join(pred_dir, x + ".jpg") for x in png_name_list]  

    for ind in range(len(gt_imgs)): 
        
        pred = np.array(Image.open(pred_imgs[ind]))  #预测
        label = np.array(Image.open(gt_imgs[ind]))   #标签
        pred_boundary = mask_to_boundary(pred)
        label_boundary = mask_to_boundary(label)
        pred= np.clip(pred,0,1)
        label= np.clip(label,0,1)
        pred_boundary= np.clip(pred_boundary,0,1)
        label_boundary= np.clip(label_boundary,0,1)

        if len(label.flatten()) != len(pred.flatten()):  
            continue
        
        
        hist += fast_hist(label.flatten(), pred.flatten(), num_classes)
        hist_boundary += fast_hist(label_boundary.flatten(), pred_boundary.flatten(), num_classes)
          
        
    IoUs    = per_class_iu(hist)
    mPA     = per_class_PA(hist)
    BIoUs   = per_class_iu(hist_boundary)
    print('mIoU: ' + str(round(np.nanmean(IoUs) * 100, 2)) + '; mPA: ' + str(round(np.nanmean(mPA) * 100, 2))+'; Boundary IoU: ' + str(round(np.nanmean(BIoUs) * 100, 2))) 



if __name__ == "__main__":

    num_classes     = 2
    name_classes    = ["background","horse"]

    Img_path  = 'weizmann_horse_db'

    image_ids       = open(os.path.os.path.join(Img_path, "annotation/trainval.txt"),'r').read().splitlines() 
    gt_dir          = os.path.os.path.join(Img_path, "mask")
    pred_dir        = os.path.os.path.join(Img_path, 'predict_iou')

    print("Get miou.")
    compute_mIoU(gt_dir, pred_dir, image_ids, num_classes)
    print("Get miou done.")
    
    