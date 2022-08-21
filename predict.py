import numpy as np
from PIL import Image
import torch
import copy
import torch.nn.functional as F
import cv2

from PIL import Image
from model import DeepLab

def cvtColor(image):
    if len(np.shape(image)) == 3 and np.shape(image)[2] == 3:
        return image 
    else:
        image = image.convert('RGB')
        return image 
    
def resize_image(image, size):#不失真的调整图片大小
    iw, ih  = image.size#原始大小
    w, h    = size#期待图片大小

    scale   = min(w/iw, h/ih)
    new_w      = int(iw*scale)
    new_h      = int(ih*scale)

    image   = image.resize((new_w,new_h), Image.BICUBIC)
    new_image = Image.new('RGB', size, (128,128,128))
    new_image.paste(image, ((w-new_w)//2, (h-new_h)//2))

    return new_image, new_w, new_h

def preprocess_input(image):
    image /= 255.0
    return image

class DeeplabV3(object):
    _defaults = {
        "model_path"        : './logs/ep090-loss0.084-val_loss0.074.pth',
        "num_classes"       : 2,
        "backbone"          : "mobilenet",
        #输入图像大小
        "input_shape"       : [512, 512],
        #backbone下采样的倍数
        "downsample_factor" : 16,
        #   mix_type = 0的时候代表原图与生成的图按比例进行混合
        #   mix_type = 1的时候代表仅保留生成的图
        "mix_type"          : 1,
        "cuda"              : True,
    }

    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        for name, value in kwargs.items():
            setattr(self, name, value)
        self.colors = [(0, 0, 0), (128, 0, 0)]
        device      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.net = DeepLab(num_classes=self.num_classes, backbone=self.backbone, downsample_factor=self.downsample_factor, pretrained=False)
        self.net.load_state_dict(torch.load(self.model_path, map_location=device))
        
        self.net    = self.net.eval()
        if self.cuda:
            self.net = self.net.cuda()


    def predict_image(self, image):
        image       = cvtColor(image)
        old_img     = copy.deepcopy(image)#保存原始图像
        #原始大小
        orininal_h  = np.array(image).shape[0]
        orininal_w  = np.array(image).shape[1]
        #避免图像失真
        image_data, nw, nh  = resize_image(image, (self.input_shape[1],self.input_shape[0]))
        #归一化然后调整成torch存储的格式，最后加上batch维度
        image_data  = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, np.float32)), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()
            predict = self.net(images)[0]
            predict = F.softmax(predict.permute(1,2,0),dim = -1).cpu().numpy()
            #首先删除为了保证图片不失真加的框
            predict = predict[int((self.input_shape[0] - nh) // 2) : int((self.input_shape[0] - nh) // 2 + nh), \
                    int((self.input_shape[1] - nw) // 2) : int((self.input_shape[1] - nw) // 2 + nw)]
            #然后变回原始大小
            predict = cv2.resize(predict, (orininal_w, orininal_h), interpolation = cv2.INTER_LINEAR)
            #找出可能性最大的类型
            predict = predict.argmax(axis=-1)
            
        if self.mix_type == 0:
            seg_img = np.reshape(np.array(self.colors, np.uint8)[np.reshape(predict, [-1])], [orininal_h, orininal_w, -1])
            image   = Image.fromarray(np.uint8(seg_img))
            image   = Image.blend(old_img, image, 0.7)

        elif self.mix_type == 1:
            seg_img = np.reshape(np.array(self.colors, np.uint8)[np.reshape(predict, [-1])], [orininal_h, orininal_w, -1])
            image   = Image.fromarray(np.uint8(seg_img))
            
        return image,Image.fromarray(np.uint8(predict))
            
deeplab=DeeplabV3()
mode = "predict"#选择预测模式
image_path = ".\weizmann_horse_db\horse"
image_out_path   = ".\weizmann_horse_db\predict_iou"

if mode == "predict":#预测一张图片查看效果
    while True:
        img = input('Input image filename:')
        try:
            image = Image.open(img)
        except:
            print('Open Error! Try again!')
            continue
        else:
            r_image,pr = deeplab.predict_image(image)
            max=np.max(np.array(pr))
            print(max)
            r_image.show()
    
elif mode == "dir_predict":#预测整个数据
    import os
    from tqdm import tqdm

    img_names = os.listdir(image_path)
    for img_name in tqdm(img_names):
        if img_name.lower().endswith(('.png', '.jpg', )):
            image_path_one  = os.path.join(image_path, img_name)
            image       = Image.open(image_path_one)
            rgb_image,predict     = deeplab.predict_image(image)#rgb_imgge是三通道，pr是索引图
            if not os.path.exists(image_out_path):
                os.makedirs(image_out_path)
            rgb_image.save(os.path.join(image_out_path, img_name))