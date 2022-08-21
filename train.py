import numpy as np
import os
import torch
import torch.backends.cudnn as cudnn
import math

from PIL import Image
from model import DeepLab
from functools import partial
from tqdm import tqdm
from torch.utils.data import DataLoader

from Mydataset import DeeplabDataset,dataset_collate

    
    
def weights_init(net, init_type='normal', init_gain=0.02):#初始化参数
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and classname.find('Conv') != -1:
            if init_type == 'normal':
                torch.nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                torch.nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                torch.nn.init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
        elif classname.find('BatchNorm2d') != -1:
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
            torch.nn.init.constant_(m.bias.data, 0.0)
    print('initialize network with %s type' % init_type)
    net.apply(init_func)
    
    
def get_lr_scheduler(lr_decay_type, lr, min_lr, total_iters, warmup_iters_ratio = 0.1, warmup_lr_ratio = 0.1, no_aug_iter_ratio = 0.3, step_num = 10):#学习率变化
    def yolox_warm_cos_lr(lr, min_lr, total_iters, warmup_total_iters, warmup_lr_start, no_aug_iter, iters):
        if iters <= warmup_total_iters:
            lr = (lr - warmup_lr_start) * pow(iters / float(warmup_total_iters), 2) + warmup_lr_start
        elif iters >= total_iters - no_aug_iter:
            lr = min_lr
        else:
            lr = min_lr + 0.5 * (lr - min_lr) * (
                1.0 + math.cos(math.pi* (iters - warmup_total_iters) / (total_iters - warmup_total_iters - no_aug_iter))
            )
        return lr

    def step_lr(lr, decay_rate, step_size, iters):
        if step_size < 1:
            raise ValueError("step_size must above 1.")
        n       = iters // step_size
        out_lr  = lr * decay_rate ** n
        return out_lr

    if lr_decay_type == "cos":
        warmup_total_iters  = min(max(warmup_iters_ratio * total_iters, 1), 3)
        warmup_lr_start     = max(warmup_lr_ratio * lr, 1e-6)
        no_aug_iter         = min(max(no_aug_iter_ratio * total_iters, 1), 15)
        func = partial(yolox_warm_cos_lr ,lr, min_lr, total_iters, warmup_total_iters, warmup_lr_start, no_aug_iter)
    else:
        decay_rate  = (min_lr / lr) ** (1 / (step_num - 1))
        step_size   = total_iters / step_num
        func = partial(step_lr, lr, decay_rate, step_size)

    return func


def set_optimizer_lr(optimizer, lr_scheduler_func, epoch):
    lr = lr_scheduler_func(epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def CE_Loss(inputs, target, cls_weights, num_classes=21):#用交叉熵计算每一个像素的损失
    n, c, h, w = inputs.size()
    nt, ht, wt = target.size()
    if h != ht and w != wt:
        inputs = torch.nn.functional.interpolate(inputs, size=(ht, wt), mode="bilinear", align_corners=True)

    temp_inputs = inputs.permute(3,1,2).contiguous().view(-1, c)
    temp_target = target.view(-1)

    CE_loss  = torch.nn.CrossEntropyLoss(weight=cls_weights, ignore_index=num_classes)(temp_inputs, temp_target)
    return CE_loss

def Dice_loss(inputs, target, beta=1, smooth = 1e-5):
    n, c, h, w = inputs.size()
    nt, ht, wt, ct = target.size()
    if h != ht and w != wt:
        inputs = torch.nn.functional.interpolate(inputs, size=(ht, wt), mode="bilinear", align_corners=True)
        
    temp_inputs = torch.softmax(inputs.permute(3,1,2).contiguous().view(n, -1, c),-1)
    temp_target = target.view(n, -1, ct)

    #   计算dice loss
    tp = torch.sum(temp_target[...,:-1] * temp_inputs, axis=[0,1])
    fp = torch.sum(temp_inputs                       , axis=[0,1]) - tp
    fn = torch.sum(temp_target[...,:-1]              , axis=[0,1]) - tp

    score = ((1 + beta ** 2) * tp + smooth) / ((1 + beta ** 2) * tp + beta ** 2 * fn + fp + smooth)
    dice_loss = 1 - torch.mean(score)
    return dice_loss

if __name__ == "__main__":
    #基础设置
    Cuda            = True
    num_classes     = 2
    backbone        = "mobilenet"
    pretrained      = False
    model_path      = "pretrained/deeplab_mobilenetv2.pth"#预训练权重
    downsample_factor   = 16
    input_shape         = [512, 512]
    #Epoch和batch_size设置
    Freeze_Train        = True
    Init_Epoch          = 0
    Freeze_Epoch        = 50
    Freeze_batch_size   = 8
    UnFreeze_Epoch      = 100
    Unfreeze_batch_size = 4
    #梯度下降和学习率设置
    Init_lr             = 7e-3
    Min_lr              = Init_lr * 0.01
    optimizer_type      = "sgd"
    momentum            = 0.9
    weight_decay        = 1e-4
    lr_decay_type       = 'cos'
    #训练权重保存
    save_period         = 5
    save_dir            = './logs'
    #图片路径
    image_path  =  './weizmann_horse_db'    
    #gpu使用情况
    device          = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    local_rank      = 0
    num_workers         = 1
    
    cls_weights     = np.ones([num_classes], np.float32)
    model   = DeepLab(num_classes=num_classes, backbone=backbone, downsample_factor=downsample_factor, pretrained=pretrained)
    #初始化
    if not pretrained:
        weights_init(model)
    #加载预训练模型    
    if model_path != '':
        model_dict      = model.state_dict()
        pretrained_dict = torch.load(model_path, map_location = device)
        load_key, no_load_key, temp_dict = [], [], {}
        for k, v in pretrained_dict.items():
            if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
                temp_dict[k] = v
                load_key.append(k)
            else:
                no_load_key.append(k)
        model_dict.update(temp_dict)
        model.load_state_dict(model_dict)
    
    
    model_train     = model
    if Cuda:
        cudnn.benchmark = True
        model_train = model_train.cuda()
    #   读取数据集
    with open(os.path.join(image_path, "annotation/train.txt"),"r") as f:
        train_lines = f.readlines()
    with open(os.path.join(image_path, "annotation/val.txt"),"r") as f:
        val_lines = f.readlines()
    num_train   = len(train_lines)
    num_val     = len(val_lines)
    train_dataset   = DeeplabDataset(train_lines, input_shape, num_classes, True, image_path)
    val_dataset     = DeeplabDataset(val_lines, input_shape, num_classes, False, image_path)
    
    if True:
        UnFreeze_flag = False
        #冻结部分的训练参数
        if Freeze_Train:
            for param in model.backbone.parameters():
                param.requires_grad = False
        batch_size = Freeze_batch_size if Freeze_Train else Unfreeze_batch_size

        #冻结训练时候的学习率
        nbs             = 16
        lr_limit_max    = 5e-4 if optimizer_type == 'adam' else 1e-1
        lr_limit_min    = 3e-4 if optimizer_type == 'adam' else 5e-4
        Init_lr_fit     = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
        Min_lr_fit      = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)
        lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)
        
        #optimizer
        optimizer = {
            'adam'  : torch.optim.Adam(model.parameters(), Init_lr_fit, betas = (momentum, 0.999), weight_decay = weight_decay),
            'sgd'   : torch.optim.SGD(model.parameters(), Init_lr_fit, momentum = momentum, nesterov=True, weight_decay = weight_decay)
        }[optimizer_type]
        #处理数据集不能被batch_size整除问题
        epoch_step      = num_train // batch_size
        epoch_step_val  = num_val // batch_size
        #Dataloader参数
        train_sampler   = None
        val_sampler     = None
        shuffle         = True
        dataset_train             = DataLoader(train_dataset, shuffle = shuffle, batch_size = batch_size, num_workers = num_workers, pin_memory=True,
                                    drop_last = True, collate_fn = dataset_collate, sampler=train_sampler)
        dataset_val         = DataLoader(val_dataset  , shuffle = shuffle, batch_size = batch_size, num_workers = num_workers, pin_memory=True, 
                                    drop_last = True, collate_fn = dataset_collate, sampler=val_sampler)
        eval_callback   = None
        

        for epoch in range(Init_Epoch, UnFreeze_Epoch):
            #解冻后的训练参数
            if epoch >= Freeze_Epoch and not UnFreeze_flag and Freeze_Train:
                batch_size = Unfreeze_batch_size
                #解冻后的学习率
                nbs             = 16
                lr_limit_max    = 5e-4 if optimizer_type == 'adam' else 1e-1
                lr_limit_min    = 3e-4 if optimizer_type == 'adam' else 5e-4
                Init_lr_fit     = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
                Min_lr_fit      = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)
                lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)
                    
                for param in model.backbone.parameters():
                    param.requires_grad = True
                            
                epoch_step      = num_train // batch_size
                epoch_step_val  = num_val // batch_size


                dataset_train             = DataLoader(train_dataset, shuffle = shuffle, batch_size = batch_size, num_workers = num_workers, pin_memory=True,
                                            drop_last = True, collate_fn = dataset_collate, sampler=train_sampler)
                dataset_val         = DataLoader(val_dataset  , shuffle = shuffle, batch_size = batch_size, num_workers = num_workers, pin_memory=True, 
                                            drop_last = True, collate_fn = dataset_collate, sampler=val_sampler)

                UnFreeze_flag = True

            set_optimizer_lr(optimizer, lr_scheduler_func, epoch)

            #训练集
            total_loss      = 0
            model_train.train()
            print('Start Train')
            pbar = tqdm(total=epoch_step,desc=f'Epoch {epoch + 1}/{epoch}',postfix=dict,mininterval=0.3)
            for iteration, batch in enumerate(dataset_train):
                if iteration >= epoch_step: 
                    break
                imgs, pngs, labels = batch

                with torch.no_grad():
                    weights = torch.from_numpy(cls_weights)
                    if Cuda:
                        imgs    = imgs.cuda(local_rank)
                        pngs    = pngs.cuda(local_rank)
                        labels  = labels.cuda(local_rank)
                        weights = weights.cuda(local_rank)

                optimizer.zero_grad()
                outputs = model_train(imgs)
                
                loss = CE_Loss(outputs, pngs, weights, num_classes = num_classes)
                dice_loss = Dice_loss(outputs, labels)
                loss      = loss + dice_loss
                loss.backward()
                optimizer.step()
                total_loss      += loss.item()
            
            #验证集
            val_loss        = 0
            best_loss       = 1
            model_train.eval()
            print('Finish Train')
            print('Start Validation')
            for iteration, batch in enumerate(dataset_val):
                if iteration >= epoch_step_val:
                    break
                imgs, pngs, labels = batch
                with torch.no_grad():
                    weights = torch.from_numpy(cls_weights)
                    if Cuda:
                        imgs    = imgs.cuda(local_rank)
                        pngs    = pngs.cuda(local_rank)
                        labels  = labels.cuda(local_rank)
                        weights = weights.cuda(local_rank)

                    outputs     = model_train(imgs)
                    loss = CE_Loss(outputs, pngs, weights, num_classes = num_classes)
                    dice_loss = Dice_loss(outputs, labels)
                    loss  = loss + dice_loss
                    val_loss    += loss.item()
            
            
        
            if (epoch + 1) % save_period == 0 or epoch + 1 == epoch:
                torch.save(model.state_dict(), os.path.join(save_dir, 'ep%03d-loss%.3f-val_loss%.3f.pth' % (epoch + 1, total_loss / epoch_step, val_loss / epoch_step_val)))

            if val_loss<best_loss:
                print('Save best model to best_epoch_weights.pth')
                torch.save(model.state_dict(), os.path.join(save_dir, "best_epoch_weights.pth"))
                best_loss =val_loss
                
            torch.save(model.state_dict(), os.path.join(save_dir, "last_epoch_weights.pth"))
                    
                    
                   