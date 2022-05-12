import glob
import logging
import os
import time

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

from data_loader import RandomCrop, RescaleT, SalObjDataset, ToTensorLab
from loss import muti_bce_loss_fusion
from model import U2NET, U2NETP
from utils import (dice, save_checkpoint, seed_everything, set_logger,
                   set_n_get_device)

# import sys
# sys.path.append('../')


######### Define the training process #########
def run_training(train_dl, val_dl):
    set_logger(LOG_PATH)
    # ------- 3. define model --------
    # define the net
    if(MODEL_NAME == 'u2net'):
        net = U2NET(3, 1)
    elif(MODEL_NAME == 'u2netp'):
        net = U2NETP(3, 1)

    if torch.cuda.is_available():
        net.load_state_dict(torch.load(PRE_MODEL_DIR))
        net.to(device=DEVICE)

    # ------- 4. define optimizer --------
    logging.info('define optimizer')
    optimizer = optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max',
                                                        factor=0.5, patience=4,
                                                        verbose=False, threshold=0.0001,
                                                        threshold_mode='rel', cooldown=0,
                                                        min_lr=0, eps=1e-08)

    # ------- 5. training process --------
    logging.info('start training')
    diff = 0  # 记录模型持续优化的epoch数
    best_val_loss = 1000
    optimizer.zero_grad()
    for epoch in range(0, NUM_EPOCHS):
        t0 = time.time()
        net.train()
        train_loss_list, train_metric_list = [], []
        for i, data in enumerate(train_dl):
            train_img, train_label = data['image'], data['label']

            img = train_img.to(device=DEVICE, dtype=torch.float)
            label = train_label.to(device=DEVICE, dtype=torch.float)

            # forward + backward + optimize
            d0, d1, d2, d3, d4, d5, d6 = net(img)
            loss2, loss = muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, label)
            # train_dice = dice(d0, label)

            train_loss_list.append(loss.item())
            # train_metric_list.append(train_dice)

            #grandient accumulation step=2
            acc_step = GRADIENTACCSTEP
            _train_loss = loss / acc_step
            _train_loss.backward()
            if (i+1)%acc_step==0:
                optimizer.step()
                optimizer.zero_grad()

            # del temporary outputs and loss
            del d0, d1, d2, d3, d4, d5, d6, loss2, loss

        train_loss = np.mean(train_loss_list)
        train_metric = np.mean(train_metric_list)

        net.eval()
        with torch.no_grad():
            val_loss_list, val_metric_list = [], []
            for i, data in enumerate(val_dl):
                val_img, val_label = data['image'], data['label']
                img = val_img.to(device=DEVICE, dtype=torch.float)
                label = val_label.to(device=DEVICE, dtype=torch.float)

                d0, d1, d2, d3, d4, d5, d6 = net(img)
                loss2, loss = muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, label)
                # val_dice = dice(d0, label)
                val_loss_list.append(loss.item())
                # val_metric_list.append(val_dice)
        val_loss = np.mean(val_loss_list)
        val_metric = np.mean(val_metric_list)

        # Adjust learning_rate
        scheduler.step(val_loss)

        # force to at least train N epochs
        if epoch >= -1:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                is_best = True
                diff = 0
            else:
                is_best = False
                diff += 1
                if diff > EARLY_STOPPING_ROUND:
                    logging.info('Early Stopping: val_metric does not increase %d rounds' % EARLY_STOPPING_ROUND)
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
        save_checkpoint(checkpoint_dict, is_best=is_best, checkpoint=MODEL_DIR)

        if epoch > -1:
            logging.info('[EPOCH %05d]train_loss: %0.5f,train_metric: %0.5f; val_loss: %0.5f,  val_metric: %0.5f; time elapsed: %0.1f min' %
                            (epoch, train_loss.item(), train_metric.item(), val_loss.item(), val_metric.item(), (time.time()-t0)/60))

if __name__ == '__main__':
    SEED = 1234
    seed_everything(SEED)
    DEVICE = set_n_get_device("0", data_device_id="cuda:0")  # use the first GPU
    IMG_SIZE = 512
    NUM_WORKERS = 4
    NUM_EPOCHS = 30
    DEBUG = False # if True, load 100 samples, False
    GRADIENTACCSTEP = 1  # 梯度累积参数
    PRE_TRAINED = False

    IMAGE_EXT = '.png'
    LABEL_EXT = '.png'
    MODEL_NAME = 'u2netp'  # 'u2netp'
    BATCH_SIZE_VAL = 1
    BATCH_SIZE_TRAIN = 16
    EARLY_STOPPING_ROUND = 50

    CREATE_TIME = time.strftime('%Y-%m-%d_%H-%M-%S',time.localtime(time.time()))  # 时间戳
    MODEL_DIR = os.path.join(os.getcwd(), 'saved_models', MODEL_NAME, CREATE_TIME)
    LOG_PATH = os.path.join(MODEL_DIR, '%s_%s.log' %(MODEL_NAME, IMG_SIZE))
    PRE_MODEL_DIR = r'F:\Segmentation\SemiSeg_CPS_Torch_Darren\pretrained_model\u2netp.pth'
    
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR, exist_ok=True)

    ######### 3. Load data #########
    # train_dl, val_dl = prepare_trainset(BATCH_SIZE, NUM_WORKERS, SEED, IMG_SIZE, debug, nonempty_only=False)#True: Only using nonempty-mask!
    data_dir = os.path.join(os.getcwd(), 'datasets' + os.sep)
    train_image_dir = os.path.join('imgs', 'train' + os.sep)
    train_label_dir = os.path.join('labels', 'train' + os.sep)
    val_image_dir = os.path.join('imgs', 'val' + os.sep)
    val_label_dir = os.path.join('labels', 'val' + os.sep)

    train_img_name_list = glob.glob(data_dir + train_image_dir + '*' + IMAGE_EXT)
    train_label_name_list = glob.glob(data_dir + train_label_dir + '*' + LABEL_EXT)
    val_img_name_list = glob.glob(data_dir + val_image_dir + '*' + IMAGE_EXT)
    val_label_name_list = glob.glob(data_dir + val_label_dir + '*' + LABEL_EXT)

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

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE_TRAIN, shuffle=True, num_workers=0)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE_VAL, shuffle=False, num_workers=0)

    ######### 4. Run the training process #########
    run_training(train_dataloader, val_dataloader)
    
    print('------------------------\nComplete SEED=%d\n------------------------'%SEED)
