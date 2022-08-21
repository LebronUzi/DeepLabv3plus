import torch
from torch import nn
from torch.nn import functional as F
from Mobilnetv2 import MobileNetV2


class encoder_mobilenet(nn.Module):#以Mobilnet为backbone的encoder部分
    def __init__(self, downsample_factor=8):
        super(encoder_mobilenet, self).__init__()
        from functools import partial
        
        model           = MobileNetV2()
        self.features   = model.features[:-1]#不要最后一层

        self.total_idx  = len(self.features)
        self.down_idx   = [2, 4, 7, 14]#下采样的block的位置

        if downsample_factor == 8:
            for i in range(self.down_idx[-2], self.down_idx[-1]):
                self.features[i].apply(
                    partial(self._nostride_dilate, dilate=2)
                )
            for i in range(self.down_idx[-1], self.total_idx):
                self.features[i].apply(
                    partial(self._nostride_dilate, dilate=4)
                )
        elif downsample_factor == 16:
            for i in range(self.down_idx[-1], self.total_idx):
                self.features[i].apply(
                    partial(self._nostride_dilate, dilate=2)
                )
        
    def _nostride_dilate(self, m, dilate):#保证图片大小不变
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            if m.stride == (2, 2):
                m.stride = (1, 1)
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate//2, dilate//2)
                    m.padding = (dilate//2, dilate//2)
            else:
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate, dilate)
                    m.padding = (dilate, dilate)

    def forward(self, x):
        low_level_features = self.features[:4](x)
        x = self.features[4:](low_level_features)
        return low_level_features, x 
    
    
    
class ASPPConv(nn.Sequential):#空洞卷积层
    def __init__(self, in_channels, out_channels, dilation):
        super(ASPPConv, self).__init__(
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )


class ASPPPooling(nn.Sequential):#先池化到1*1然后再上采样
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        size = x.shape[-2:]
        for feature in self:
            x = feature(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)#双线性插值


class ASPP(nn.Module):#利用ASPP结构获得深层特征
    def __init__(self, in_channels, dilation_list, out_channels = 256):
        super(ASPP, self).__init__()
        modules = [
            nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False),
                          nn.BatchNorm2d(out_channels),
                          nn.ReLU())
        ]#第一个普通的1×1卷积
        
        rates = tuple(dilation_list)
        for rate in rates:
            modules.append(ASPPConv(in_channels, out_channels, rate))#第二到四个空洞卷积

        modules.append(ASPPPooling(in_channels, out_channels))#最后一个池化层

        self.convs = nn.ModuleList(modules)

        self.conv_cat = nn.Sequential(
            nn.Conv2d(len(self.convs) * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )#1×1普通卷积
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res_list = []
        for conv in self.convs:
            res_list.append(conv(x))
        res = torch.cat(res_list, dim=1)#在深度方向进行叠加
        result=self.conv_cat(res)
        return result#得到深层特征
    
    
class DeepLab(nn.Module):
    def __init__(self, num_classes, backbone="mobilenet", pretrained=True, downsample_factor=16):
        super(DeepLab, self).__init__()
        if backbone=='mobilenet':
            self.backbone = encoder_mobilenet(downsample_factor=downsample_factor)
            in_channels = 320#输入ASPP前的通道数
            low_level_channels = 24#浅层通道数
        
        self.aspp = ASPP(in_channels,[6,12,18])            
       
        #浅层特征的卷积
        self.shortcut_conv = nn.Sequential(
            nn.Conv2d(low_level_channels, 48, 1),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True)
        )		
        #融合后的卷积
        self.cat_conv = nn.Sequential(
            nn.Conv2d(48+256, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Conv2d(256, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
        )
        #调整通道深度为分类数
        self.cls_conv = nn.Conv2d(256, num_classes, 1, stride=1)

    def forward(self, x):
        size = x.shape[-2:]
        low_level_features, x = self.backbone(x)
        x = self.aspp(x)
        low_level_features = self.shortcut_conv(low_level_features)
        #深层特征上采样
        x = F.interpolate(x, size=(low_level_features.size(2), low_level_features.size(3)), mode='bilinear', align_corners=True)
        #深度方向连接
        x = self.cat_conv(torch.cat((x, low_level_features), dim=1))
        #普通卷积调整通道
        x = self.cls_conv(x)
        #双线性插值
        x = F.interpolate(x, size=size, mode='bilinear', align_corners=True)
        
        return x