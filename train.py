import os
import random
import argparse
import logging
import torch
import cv2
import torch.nn.functional as F
from torchvision import transforms
from models.emonet_split import EmoNet
from datasets.data_augmentation import DataAugmentor
from datasets.data import dataloader
from datasets.sampler import ImbalancedDatasetSampler
from utils.utils_logging import AverageMeter, init_logging
from utils.utils_callbacks import CallBackEvaluation, CallBackLogging, CallBackModelCheckpoint
from losses.losses import pearsonr,concordance_cc
from utils.utlis_pts import *
torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser(description='parameters for face emotion recognition network')

parser.add_argument('--batchsize', type=int, default=64, metavar='N', help='batch size')
parser.add_argument('--epochs', type=int, default=50, metavar='N', help='training epochs')
parser.add_argument('--nclasses', type=int, default=5, choices=[5,8])
parser.add_argument('--num_workers', type=int, default=4, metavar='N', help='5,8')

parser.add_argument('--optim', type=str, default='adamw', choices=['adam', 'sgd','adamw'])
parser.add_argument('--wd', type=float, default=0)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--s',  type=float, default=260.0)
parser.add_argument('--pts', action='store_true', help='plot alignment on face images')

parser.add_argument('--kd', action='store_true', help='train with kd')
parser.add_argument('--kd_w', type=float)
parser.add_argument('--path', type=str, default='', help='teacher mode path')

def kl_loss(x_s, y_t, T = 4):
    p = F.log_softmax(x_s / T, dim=1)
    q = F.softmax(y_t / T, dim=1)
    l_kl = F.kl_div(p, q, reduction='sum') * (T ** 2) / x_s.shape[0]
    return l_kl

def main():
    args = parser.parse_args()
    manualSeed = random.randint(1, 10000)
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    torch.cuda.manual_seed_all(manualSeed)
    cur_path = os.path.abspath(os.curdir)
    n_expression = args.nclasses
    image_size = 256

    if args.new: assert args.sample
    if args.kd:
        args.output = cur_path.replace('emonet_code','Results')+'/emonet/KD/'+'E_'+str(n_expression)+'_optim_'+str(args.optim)+\
                      'wd'+str(args.wd)+'lr'+str(args.lr)+'_scale_'+str(args.s)+'_kdw_'+str(args.kd_w)
    else:
        args.output = cur_path.replace('emonet_code','Results')+'/emonet/'+'E_'+str(n_expression)+'_optim_'+str(args.optim)+\
                      'wd'+str(args.wd)+'lr'+str(args.lr)+'_scale_'+str(args.s)

    if not os.path.exists(args.output):
        os.makedirs(args.output, exist_ok=True)

    log_root = logging.getLogger()
    init_logging(log_root, args.output)
    transform_image = transforms.Compose([transforms.ToTensor()])
    # rotation -15-15
    # scale 0.75-1.25
    transform_shape_train = DataAugmentor(image_size, image_size, random_translation = 10, random_rotation=15,
                                      random_scaling=0.25, random_seed=manualSeed, flipping_probability=0.5,
                                      scale_=args.s)

    transform_shape_valid = DataAugmentor(image_size, image_size, scale_=args.s)
    transform_shape_valid_flip = DataAugmentor(image_size, image_size, mirror=True, flipping_probability=1.0, scale_=args.s)
    print('loading training set')
    train_data = dataloader(subset='train', transform_image_shape=transform_shape_train,transform_image=transform_image, n_expression=n_expression)
    # weights = train_data.weight

    train_loader = torch.utils.data.DataLoader(train_data, sampler=ImbalancedDatasetSampler(train_data),
                                                       batch_size=args.batchsize, num_workers=args.num_workers, pin_memory=True,drop_last=True)

    print('loading validation set')
    valid_data = dataloader(subset='valid', transform_image_shape=transform_shape_valid, transform_image=transform_image, n_expression=n_expression)
    valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=args.batchsize, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    valid_data_flip = dataloader(subset='valid', transform_image_shape=transform_shape_valid_flip, transform_image=transform_image, n_expression=n_expression)
    valid_loader_flip = torch.utils.data.DataLoader(valid_data_flip, batch_size=args.batchsize, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    print('loading testing set')
    test_data = dataloader(subset='test', transform_image_shape=transform_shape_valid, transform_image=transform_image, n_expression=n_expression)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batchsize, shuffle=False, num_workers=1, pin_memory=True)
    test_data_flip = dataloader(subset='test', transform_image_shape=transform_shape_valid_flip, transform_image=transform_image, n_expression=n_expression)
    test_loader_flip = torch.utils.data.DataLoader(test_data_flip, batch_size=args.batchsize, shuffle=False, num_workers=1, pin_memory=True)

    # model initilization
    if args.kd:
        model_T = EmoNet(n_expression=n_expression)
        state_dict = torch.load(args.path,map_location='cpu')
        model_T.load_state_dict(state_dict)
        model_T = torch.nn.DataParallel(model_T).cuda()
        model_T.eval()

    model = EmoNet(n_expression=n_expression)
    model = torch.nn.DataParallel(model).cuda()
    params = list(model.parameters())
    sub_params = [p for p in params if p.requires_grad]

    print('num of params', sum(p.numel() for p in sub_params))
    if args.optim == 'adamw':
        optimizer = torch.optim.AdamW(sub_params, lr=args.lr, betas=(0.9, 0.999), weight_decay=args.wd)
    else:
        raise ValueError()

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[15, 30, 45], gamma=0.1, last_epoch=-1)
    total_step = int(len(train_data) / args.batchsize * args.epochs)
    callback_logging = CallBackLogging(400,total_step, args.batchsize, None)
    callback_checkpoint = CallBackModelCheckpoint(args.output)
    callback_validation_valid = CallBackEvaluation(valid_loader,valid_loader_flip, subset='valid')
    callback_validation_test = CallBackEvaluation(test_loader,test_loader_flip, subset='test')

    if args.kd:
        _ = callback_validation_test(0, model_T)
        model_T.eval()

    # training
    global_step = 0
    losses = AverageMeter()
    acces = AverageMeter()
    losses_au = AverageMeter()
    for epoch in range(args.epochs):
        model.train()
        model.module.feature.eval()

        for index, data in enumerate(train_loader):
            global_step += 1
            images = data['image'].cuda()
            valence = data['valence'].squeeze().cuda()
            arousal = data['arousal'].squeeze().cuda()
            label = data['expression'].cuda()
            outputs = model(images)
            if args.pts and index<10 and epoch==0:
                pts = get_pts(outputs['heatmap'][0])
                preds = pts * 4
                img_pts = preds[17:, :]
                img_show_ = np.asarray(images[0].permute(1,2,0).cpu().data)*255
                img_show = LineDrawer_51(img_show_, img_pts.astype('int32'))
                impath = args.output+'/debug/'
                if not os.path.exists(impath):
                    os.makedirs(impath)
                cv2.imwrite('%s/%d.jpg' % (impath, index), img_show[:,:,-1::])

            loss_c = F.cross_entropy(outputs['expression'], label)
            loss_mse = F.mse_loss(outputs['valence'], valence.detach())+F.mse_loss(outputs['arousal'], arousal.detach())
            loss_pcc = 1-(pearsonr(outputs['valence'], valence.detach())+pearsonr(outputs['arousal'], arousal.detach()))/2
            loss_ccc = 1-(concordance_cc(outputs['valence'].squeeze(), valence.detach())+concordance_cc(outputs['arousal'].squeeze(), arousal.detach()))/2
            if args.kd:
                with torch.no_grad():
                    outputs_T = model_T(images)
                loss_kd = kl_loss(outputs['expression'], outputs_T['expression'].detach(), T=1)

            # shake shake?
            alpha = torch.rand(1).cuda()
            alpha.requires_grad = False
            beta = torch.rand(1).cuda()
            beta.requires_grad = False
            gamma = torch.rand(1).cuda()
            gamma.requires_grad = False
            loss = loss_c + alpha/(alpha+beta+gamma)*loss_mse + beta/(alpha+beta+gamma)*loss_pcc + gamma/(alpha+beta+gamma)*loss_ccc

            if args.kd:
                loss = loss+loss_kd*args.kd_w

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.update(loss_c.detach().item(), 1)
            if args.kd:
                losses_au.update(loss_kd.detach().item(), 1)
            else:
                losses_au.update(loss_pcc.detach().item(), 1)
            batch_size = images.size(0)
            _, pred = torch.max(outputs['expression'].detach(), 1)
            correct = torch.eq(pred, label)
            acc = correct.float().sum()/float(batch_size)
            acces.update(acc.detach().item(), batch_size)
            callback_logging(global_step, losses, losses_au,acces, epoch, optimizer)
        val_results = callback_validation_valid(epoch, model)
        test_results = callback_validation_test(epoch, model)
        callback_checkpoint(epoch, model)
        scheduler.step()

if __name__ == '__main__':
    main()