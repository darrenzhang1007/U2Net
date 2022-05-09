import glob
import os
import logging
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

from data_loader import RandomCrop, RescaleT, SalObjDataset, ToTensorLab
from model import U2NET, U2NETP
from utils import dice, set_n_get_device, save_checkpoint, set_logger
from loss import muti_bce_loss_fusion


# 1. 设置超参数
image_ext = '.png'
label_ext = '.png'
ite_num = 0
running_loss = 0.0
running_tar_loss = 0.0
ite_num4val = 0
save_frq = 2000  # save the model every 2000 iterations
early_stopping_round = 50
epoch_num = 100
model_name = 'u2net'  # 'u2netp'
batch_size_val = 1
batch_size_train = 12
model_dir = os.path.join(os.getcwd(), 'saved_models', model_name + os.sep)
# pre_model_dir = os.path.join(os.getcwd(), 'saved_models', model_name, 'u2net_bce_itr_54400_train_0.053013_tar_0.005284.pth')
pre_model_dir = r'F:\Segmentation\SemiSeg_CPS_Torch_Darren\pretrained_model\u2net.pth'
device = set_n_get_device("0", data_device_id="cuda:0")  # use the first GPU

# 2. set the directory of training dataset
data_dir = os.path.join(os.getcwd(), 'datasets' + os.sep)
train_image_dir = os.path.join('imgs', 'train' + os.sep)
train_label_dir = os.path.join('labels', 'train' + os.sep)
val_image_dir = os.path.join('imgs', 'val' + os.sep)
val_label_dir = os.path.join('labels', 'val' + os.sep)

train_img_name_list = glob.glob(data_dir + train_image_dir + '*' + image_ext)
train_label_name_list = glob.glob(data_dir + train_label_dir + '*' + image_ext)
val_img_name_list = glob.glob(data_dir + val_image_dir + '*' + image_ext)
val_label_name_list = glob.glob(data_dir + val_label_dir + '*' + image_ext)

assert len(train_img_name_list) == len(train_label_name_list), "please check your number of training data imgs and labels"
assert len(val_img_name_list) == len(val_label_name_list), "please check your number of training data imgs and labels"

print("---")
print("train images: ", len(train_img_name_list))
print("train labels: ", len(train_label_name_list))
print("---")

train_num = len(train_img_name_list)
val_num = len(val_img_name_list)

train_dataset = SalObjDataset(
    img_name_list=train_img_name_list,
    lbl_name_list=train_label_name_list,
    transform=transforms.Compose([
        RescaleT(640),
        RandomCrop(512),
        ToTensorLab(flag=0)]))

val_dataset = SalObjDataset(
    img_name_list=val_img_name_list,
    lbl_name_list=val_label_name_list,
    transform=transforms.Compose([ToTensorLab(flag=0)]))

train_dataloader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True, num_workers=0)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size_val, shuffle=False, num_workers=0)

# ------- 3. define model --------
# define the net
if(model_name == 'u2net'):
    net = U2NET(3, 1)
elif(model_name == 'u2netp'):
    net = U2NETP(3, 1)

if torch.cuda.is_available():
    net.load_state_dict(torch.load(pre_model_dir))
    net.cuda()

# ------- 4. define optimizer --------
print("---define optimizer...")
optimizer = optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max',
                                                       factor=0.5, patience=4,
                                                       verbose=False, threshold=0.0001,
                                                       threshold_mode='rel', cooldown=0,
                                                       min_lr=0, eps=1e-08)


# ------- 5. training process --------
print("---start training")
best_val_metric = -0.1
optimizer.zero_grad()
for epoch in range(0, epoch_num):
    net.train()
    train_loss_list, train_metric_list = [], []
    for i, data in enumerate(train_dataloader):
        train_img, train_label = data['image'], data['label']

        img = train_img.to(device=device, dtype=torch.float)
        label = train_label.to(device=device, dtype=torch.float)

        # forward + backward + optimize
        d0, d1, d2, d3, d4, d5, d6 = net(img)
        loss2, loss = muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, label)
        train_dice = dice(d0, label)

        train_loss_list.append(loss)
        train_metric_list.append(train_dice)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # print statistics
        running_loss += loss.data.item()
        running_tar_loss += loss2.data.item()

        # del temporary outputs and loss
        del d0, d1, d2, d3, d4, d5, d6, loss2, loss

        print("[epoch: %3d/%3d, batch: %5d/%5d, ite: %d] train loss: %3f, tar: %3f " % (
            epoch + 1, epoch_num, (i + 1) * batch_size_train, train_num, ite_num, running_loss / ite_num4val, running_tar_loss / ite_num4val))

    train_loss = np.mean(train_loss_list)
    train_metric = np.mean(train_metric_list)

    net.eval()
    with torch.no_grad():
        val_loss_list, val_metric_list = [], []
        for i, data in enumerate(val_dataloader):
            val_img, val_label = data['image'], data['label']
            img = val_img.to(device=device, dtype=torch.float)
            label = val_label.to(device=device, dtype=torch.float)

            d0, d1, d2, d3, d4, d5, d6 = net(img)
            loss2, loss = muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, label)
            val_dice = dice(d0, label)
            val_loss_list.append(loss)
            val_metric_list.append(val_dice)
    val_loss = np.mean(val_loss_list)
    val_metric = np.mean(val_metric_list)

    # Adjust learning_rate
    scheduler.step(val_metric)

    # force to at least train N epochs
    if epoch >= -1:
        if val_metric > best_val_metric:
            best_val_metric = val_metric
            is_best = True
            diff = 0
        else:
            is_best = False
            diff += 1
            if diff > early_stopping_round:
                logging.info('Early Stopping: val_metric does not increase %d rounds' % early_stopping_round)
                break
    else:
        is_best = False

    # save checkpoint
    checkpoint_dict = \
        {
            'epoch': epoch,
            'state_dict': net.state_dict(),
            'optim_dict': optimizer.state_dict(),
            'metrics': {'train_loss': train_loss, 'val_loss': val_loss,
                        'train_metric': train_metric, 'val_metric': val_metric}
        }
    save_checkpoint(checkpoint_dict, is_best=is_best, checkpoint=model_dir)

    if epoch > -1:
        logging.info('[EPOCH %05d]train_loss, train_metric: %0.5f, %0.5f; val_loss, val_metric: %0.5f, %0.5f; time elapsed: %0.1f min' %
                        (epoch, train_loss.item(), train_metric.item(), val_loss.item(), val_metric.item(), (time.time()-t0)/60))
