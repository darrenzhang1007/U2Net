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
from utils import dice, seed_everything, set_n_get_device, save_checkpoint, set_logger, load_checkpoint
from loss import muti_bce_loss_fusion


# import sys
# sys.path.append('../')

import numpy as np
import os
import logging
import time
import torch
import argparse

######### Define the training process #########
def run_training(train_dl, val_dl, multi_gpu=[0, 1]):
    set_logger(LOG_PATH)
    logging.info('\n\n')
    # ------- 3. define model --------
    # define the net
    if(model_name == 'u2net'):
        net = U2NET(3, 1)
    elif(model_name == 'u2netp'):
        net = U2NETP(3, 1)

    if torch.cuda.is_available():
        net.load_state_dict(torch.load(pre_model_dir))
        net.cuda(device=device)

    # ------- 4. define optimizer --------
    print("---define optimizer...")
    optimizer = optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max',
                                                        factor=0.5, patience=4,
                                                        verbose=False, threshold=0.0001,
                                                        threshold_mode='rel', cooldown=0,
                                                        min_lr=0, eps=1e-08)

    if pre_trained:
        logging.info('pre_trained: '+last_checkpoint_path)
        net, _ = load_checkpoint(last_checkpoint_path, net)

    # ------- 5. training process --------
    print("---start training")
    diff = 0  # 记录模型持续优化的epoch数
    best_val_metric = -0.1
    optimizer.zero_grad()
    for epoch in range(0, epoch_num):
        net.train()
        train_loss_list, train_metric_list = [], []
        for i, data in enumerate(train_dl):
            train_img, train_label = data['image'], data['label']

            img = train_img.to(device=device, dtype=torch.float)
            label = train_label.to(device=device, dtype=torch.float)

            # forward + backward + optimize
            d0, d1, d2, d3, d4, d5, d6 = net(img)
            loss2, loss = muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, label)
            train_dice = dice(d0, label)

            train_loss_list.append(loss.data.item())
            train_metric_list.append(train_dice)

            #grandient accumulation step=2
            acc_step = GradientAccStep
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
                img = val_img.to(device=device, dtype=torch.float)
                label = val_label.to(device=device, dtype=torch.float)

                d0, d1, d2, d3, d4, d5, d6 = net(img)
                loss2, loss = muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, label)
                val_dice = dice(d0, label)
                val_loss_list.append(loss.data.item())
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='====Model Parameters====')
    parser.add_argument('--SEED', type=int, default=1234)
    params = parser.parse_args()
    SEED = params.SEED
    print('SEED=%d'%SEED)
    MODEL = 'u2net'
    print('====MODEL ACHITECTURE: %s===='%MODEL)
    device = set_n_get_device("0", data_device_id="cuda:0")  # use the first GPU
    multi_gpu = None # [0,1] use 2 gpus; None single gpu
    # multi_gpu = [0, 1, 3, 4, 5, 7]
    debug = False # if True, load 100 samples, False
    IMG_SIZE = 512 #1024#768#512#256
    BATCH_SIZE = 16
    GradientAccStep = 1  # 梯度累积参数
    NUM_WORKERS = 4
    pre_trained, last_checkpoint_path = False, './checkpoint/%s_%s_v1_seed%s/best.pth.tar'%(MODEL, IMG_SIZE, SEED)
    checkpoint_path = './checkpoint/%s_%s_v1_seed%s'%(MODEL, IMG_SIZE, SEED)
    LOG_PATH = './logging/%s_%s_v1_seed%s.log'%(MODEL, IMG_SIZE, SEED)#

    NUM_EPOCHS = 30
    early_stopping_round = 5
    LearningRate = 0.2
    seed_everything(SEED)

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

    ######### 3. Load data #########
    # train_dl, val_dl = prepare_trainset(BATCH_SIZE, NUM_WORKERS, SEED, IMG_SIZE, debug, nonempty_only=False)#True: Only using nonempty-mask!
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

    ######### 4. Run the training process #########
    run_training(train_dataloader, val_dataloader, multi_gpu=multi_gpu)

    print('------------------------\nComplete SEED=%d\n------------------------'%SEED)
