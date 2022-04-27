import argparse
import os

import pandas as pd
import torch
import torch.optim as optim
try:
  from thop import profile, clever_format
  USE_THOP = True
except:
  print("Not using thop.")
  USE_THOP = False
from torch.utils.data import DataLoader
from tqdm import tqdm
import math
import numpy as np
import h5py

import utils
from model import Model

import torchvision
import matplotlib
from matplotlib import pyplot as plt

import pdb

try:
  import wandb
  USE_WANDB = True
except Exception as e:
  print('Exception:', e)
  print('Not using wandb. \n\n')
  USE_WANDB = False


if torch.cuda.is_available():
  torch.backends.cudnn.benchmark = True
  device = 'cuda'
else:
  device = 'cpu'

def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

# train for one epoch to learn unique features
def train(net, data_loader, train_optimizer):
    net.train()
    total_loss, total_num, train_bar = 0.0, 0, tqdm(data_loader)
    bt_cnt = 0
    for data_tuple in train_bar:
        (pos_1, pos_2), _ = data_tuple
        pos_1, pos_2 = pos_1.cuda(non_blocking=True), pos_2.cuda(non_blocking=True)
        feature_1, out_1 = net(pos_1)
        feature_2, out_2 = net(pos_2)
        # Barlow Twins
        
        # normalize the representations along the batch dimension
        out_1_norm = (out_1 - out_1.mean(dim=0)) / out_1.std(dim=0)
        out_2_norm = (out_2 - out_2.mean(dim=0)) / out_2.std(dim=0)
        
        # cross-correlation matrix
        c = torch.matmul(out_1_norm.T, out_2_norm) / batch_size

        # loss
        if symloss:
            if loss_no_on_diag:
              on_diag = torch.tensor([0.0]).to(device)
            else:
              if corr_neg_one_on_diag is False:
                on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
              else:
                on_diag = torch.diagonal(c).add_(1).pow_(2).sum()

            if loss_no_off_diag:
              off_diag = torch.tensor([0.0]).to(device)
            else:
              if corr_neg_one is False:
                  # the loss described in the original Barlow Twin's paper
                  # encouraging off_diag to be zero
                  off_diag = off_diagonal(c).pow_(2).sum()
              else:
                  # inspired by HSIC
                  # encouraging off_diag to be negative ones
                  off_diag = off_diagonal(c).add_(1).pow_(2).sum()
            loss = on_diag + lmbda * off_diag      
        else:
            on_diag = torch.tensor([0.0]).to(device)
            off_diag = torch.tensor([0.0]).to(device)
            loss = torch.linalg.matrix_norm(c - torch.eye(c.size(dim=0)), ord=1)

        

        train_optimizer.zero_grad()
        loss.backward()
        train_optimizer.step()

        total_num += batch_size
        total_loss += loss.item() * batch_size
        if corr_neg_one is True:
            off_corr = -1
        else:
            off_corr = 0

        bt_cnt += 1
        if USE_WANDB and bt_cnt % 20 == 0:
          wandb.log({
            'loss':loss.item(),
            'on_diag': on_diag.item(),
            'off_diag': off_diag.item(),
            'on_diag_avg': on_diag.item() / args.feature_dim,
            'off_diag_avg': off_diag.item() / (args.feature_dim * (args.feature_dim-1)),
            })

        train_bar.set_description('Train Epoch: [{}/{}] Loss: {:.4f} off_corr:{} lmbda:{:.4f} bsz:{} f_dim:{} dataset: {}'.format(\
                                epoch, epochs, total_loss / total_num, off_corr, lmbda, batch_size, feature_dim, dataset))

    return total_loss / total_num


# test for one epoch, use weighted knn to find the most similar images' label to assign the test image
def test(net, memory_data_loader, test_data_loader, epoch):
    net.eval()
    total_top1, total_top5, total_num, feature_bank, target_bank = 0.0, 0.0, 0, [], []
    with torch.no_grad():
        # generate feature bank and target bank
        for data_tuple in tqdm(memory_data_loader, desc='Feature extracting'):
            (data, _), target = data_tuple
            target_bank.append(target)
            feature, out = net(data.cuda(non_blocking=True))
            feature_bank.append(feature)
        # [D, N]
        feature_bank = torch.cat(feature_bank, dim=0).t().contiguous()
        # [N]
        feature_labels = torch.cat(target_bank, dim=0).contiguous().to(feature_bank.device)
        # loop test data to predict the label by weighted knn search
        test_bar = tqdm(test_data_loader)
        on_diag_total, off_diag_total = 0, 0
        on_diag_f_total, off_diag_f_total = 0, 0
        bt_cnt = 0
        for data_tuple in test_bar:
            (data, _), target = data_tuple
            data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)
            feature, out = net(data)

            total_num += data.size(0)
            # compute cos similarity between each feature vector and feature bank ---> [B, N]
            sim_matrix = torch.mm(feature, feature_bank)
            # [B, K]
            sim_weight, sim_indices = sim_matrix.topk(k=k, dim=-1)
            # [B, K]
            sim_labels = torch.gather(feature_labels.expand(data.size(0), -1), dim=-1, index=sim_indices)
            sim_weight = (sim_weight / temperature).exp()

            # counts for each class
            one_hot_label = torch.zeros(data.size(0) * k, NUM_CLS, device=sim_labels.device)
            # [B*K, C]
            one_hot_label = one_hot_label.scatter(dim=-1, index=sim_labels.view(-1, 1), value=1.0)
            # weighted score ---> [B, C]
            pred_scores = torch.sum(one_hot_label.view(data.size(0), -1, NUM_CLS) * sim_weight.unsqueeze(dim=-1), dim=1)

            pred_labels = pred_scores.argsort(dim=-1, descending=True)
            total_top1 += torch.sum((pred_labels[:, :1] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()
            total_top5 += torch.sum((pred_labels[:, :5] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()

            # added by BB
            bt_cnt += 1

            # normalize the representations along the batch dimension
            out_norm = (out - out.mean(dim=0)) / out.std(dim=0)
            feat_norm = (feature - feature.mean(dim=0)) / feature.std(dim=0)
            # cross-correlation matrix
            c = torch.matmul(out_norm.T, out_norm) / batch_size
            cf = torch.matmul(feat_norm.T, feat_norm) / batch_size

            # loss
            if corr_neg_one_on_diag is False:
              on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
              on_diag_f = torch.diagonal(cf).add_(-1).pow_(2).sum()
            else:
              on_diag = torch.diagonal(c).add_(1).pow_(2).sum()
              on_diag_f = torch.diagonal(cf).add_(1).pow_(2).sum()
            if corr_neg_one is False:
                # the loss described in the original Barlow Twin's paper
                # encouraging off_diag to be zero
                off_diag = off_diagonal(c).pow_(2).sum()
                off_diag_f = off_diagonal(cf).pow_(2).sum()
            else:
                # inspired by HSIC
                # encouraging off_diag to be negative ones
                off_diag = off_diagonal(c).add_(1).pow_(2).sum()
                off_diag_f = off_diagonal(cf).add_(1).pow_(2).sum()

            on_diag_total += on_diag.item()
            off_diag_total += off_diag.item()
            on_diag_f_total += on_diag_f.item()
            off_diag_f_total += off_diag_f.item()
    
            test_bar.set_description('Test Epoch: [{}/{}] Acc@1:{:.2f}% Acc@5:{:.2f}%'
                                     .format(epoch, epochs, total_top1 / total_num * 100, total_top5 / total_num * 100))

    total_top1 = total_top1 / total_num
    total_top5 = total_top5 / total_num

    on_diag_total /= bt_cnt
    off_diag_total /= bt_cnt
    on_diag_f_total /= bt_cnt
    off_diag_f_total /= bt_cnt

    if USE_WANDB:
      wandb.log({
        'test_on_diag': on_diag_total, 
        'test_off_diag': off_diag_total,
        'test_on_diag_feat': on_diag_f_total, 
        'test_off_diag_feat': off_diag_f_total,
        'test_on_diag_avg': on_diag_total / args.feature_dim, 
        'test_off_diag_avg': off_diag_total / (args.feature_dim * (args.feature_dim-1)),
        'test_on_diag_feat_avg': on_diag_f_total / args.feature_dim, 
        'test_off_diag_feat_avg': off_diag_f_total / (args.feature_dim * (args.feature_dim-1)),
        'total_top1': total_top1,
        'total_top5': total_top5,
      })
 
    return total_top1 * 100, total_top5 * 100

def test_stats(net, data_loader, fSinVals='', save_feats=0, fsave_feats=''):
    net.eval()
    with torch.no_grad():
        on_diag_total, off_diag_total = 0, 0
        on_diag_f_total, off_diag_f_total = 0, 0
        bt_cnt = 0
        data_bar = tqdm(data_loader)
        ss_total, ssf_total = torch.zeros(args.feature_dim), torch.zeros(2048)
        z_outs, z_feats = [], []
        for data_tuple in data_bar:
            bt_cnt += 1

            (data, _), target = data_tuple
            data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)
            feature, out = net(data)

            # normalize the representations along the batch dimension
            o_std = out.std(dim=0)
            o_std[o_std==0] = 1
            out_norm = (out - out.mean(dim=0)) / o_std
            f_std = feature.std(dim=0)
            f_std[f_std==0] = 1
            feat_norm = (feature - feature.mean(dim=0)) / f_std
            # cross-correlation matrix
            c = torch.matmul(out_norm.T, out_norm) / batch_size
            cf = torch.matmul(feat_norm.T, feat_norm) / batch_size
            # check for the singular values
            _, ss, _ = torch.svd(c)
            _, ssf, _ = torch.svd(cf)
            ss_total += ss.detach().cpu()
            ssf_total += ssf.detach().cpu()

            # save features
            if save_feats:
              z_outs += out_norm.detach().cpu().numpy(),
              z_feats += feat_norm.detach().cpu().numpy(),

            # loss
            on_diag = torch.diagonal(c).pow_(2).sum()
            on_diag_f = torch.diagonal(cf).pow_(2).sum()
            if corr_neg_one is False:
                # the loss described in the original Barlow Twin's paper
                # encouraging off_diag to be zero
                off_diag = off_diagonal(c).pow_(2).sum()
                off_diag_f = off_diagonal(cf).pow_(2).sum()
            else:
                # inspired by HSIC
                # encouraging off_diag to be negative ones
                off_diag = off_diagonal(c).add_(1).pow_(2).sum()
                off_diag_f = off_diagonal(cf).add_(1).pow_(2).sum()

            on_diag_total += on_diag.item()
            off_diag_total += off_diag.item()
            on_diag_f_total += on_diag_f.item()
            off_diag_f_total += off_diag_f.item()
            if math.isnan(off_diag_f_total):
              print("NaN")
              pdb.set_trace()
    
    on_diag_total /= bt_cnt
    off_diag_total /= bt_cnt
    on_diag_f_total /= bt_cnt
    off_diag_f_total /= bt_cnt

    # print('test_on_diag:', on_diag_total)
    # print('test_off_diag:', off_diag_total)
    # print('test_on_diag_feat:', on_diag_f_total)
    # print('test_off_diag_feat:', off_diag_f_total)
    on_diag_avg = (on_diag_total / args.feature_dim)**0.5
    on_diag_avg = on_diag_avg**0.5
    print(f'feature_dim: {feature_dim} / lambda {lmbda}')
    print('\ntest_on_diag_avg:', on_diag_avg)
    print('test_off_diag_avg:', off_diag_total / (args.feature_dim * (args.feature_dim-1)))
    print('test_on_diag_feat_avg:', on_diag_f_total / args.feature_dim)
    print('test_off_diag_feat_avg:', off_diag_f_total / (args.feature_dim * (args.feature_dim-1)))

    # plot the singular values
    ss_total /= bt_cnt
    ssf_total /= bt_cnt
    # original values
    _, feat_dim = feature.shape
    out_dim = args.feature_dim
    fig, (ax1, ax2) = plt.subplots(2,1)
    fig.suptitle('Singular values')
    ax1.bar(range(out_dim), ss_total.numpy())
    ax1.set_ylabel(f'output (dim{out_dim})')
    ax2.bar(range(feat_dim), ssf_total.numpy())
    ax2.set_ylabel(f'features (dim{feat_dim})')
    plt.savefig(fSinVals)
    plt.clf()
    # log values
    _, feat_dim = feature.shape
    out_dim = args.feature_dim
    fig, (ax1, ax2) = plt.subplots(2,1)
    fig.suptitle('Log singular values')
    ax1.bar(range(out_dim), ss_total.log().numpy())
    ax1.set_ylabel(f'output (dim{out_dim})')
    ax2.bar(range(feat_dim), ssf_total.log().numpy())
    ax2.set_ylabel(f'features (dim{feat_dim})')
    plt.savefig(fSinVals.replace('.png', '_log.png'))
    plt.clf()

    if save_feats:
      print(f"Saving features to {fsave_feats}")
      hf = h5py.File(fsave_feats, 'w')
      z_outs = np.concatenate(z_outs)
      z_feats = np.concatenate(z_feats)
      hf.create_dataset('outs', data=z_outs)
      hf.create_dataset('feats', data=z_feats)
      hf.close()

    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train SimCLR')
    parser.add_argument('--dataset', default='cifar10', type=str, help='Dataset: cifar10 or tiny_imagenet or stl10')
    parser.add_argument('--feature_dim', default=128, type=int, help='Feature dim for latent vector')
    parser.add_argument('--temperature', default=0.5, type=float, help='Temperature used in softmax')
    parser.add_argument('--k', default=200, type=int, help='Top k most similar images used to predict the label')
    parser.add_argument('--batch_size', default=128, type=int, help='Number of images in each mini-batch')
    parser.add_argument('--epochs', default=1000, type=int, help='Number of sweeps over the dataset to train')
    parser.add_argument('--proj-head-type', default='2layer', choices=['none', 'linear', '2layer'])
    # for barlow twins
    
    parser.add_argument('--lmbda', default=0.005, type=float, help='Lambda that controls the on- and off-diagonal terms')
    # off diag entries: 0 or -1
    parser.add_argument('--corr_neg_one', dest='corr_neg_one', action='store_true')
    parser.add_argument('--corr_zero', dest='corr_neg_one', action='store_false')
    parser.set_defaults(corr_neg_one=False)
    # on diag entries: 1 or -1
    parser.add_argument('--corr_neg_one_on_diag', type=int, default=0, choices=[0,1],
                        help="Whether to force the on-diag entries to be -1.")

    # optimization
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--wd', default=1e-6, type=float)
    parser.add_argument('--loss-no-on-diag', default=0, type=int, choices=[0,1],
                        help="Whether to drop the loss term for on-diag entries.")
    parser.add_argument('--loss-no-off-diag', default=0, type=int, choices=[0,1],
                        help="Whether to drop the loss term for off-diag entries.")
    parser.add_argument('--symloss', dest='symloss', action='store_false',
                        help="Whether use feature-instance symmetric loss or not.") # if false, then use 1norm of Crosscovar - I, which is not symmetric
    parser.set_defaults(symloss=True)

    # logging
    parser.add_argument('--project', default='nonContrastive')
    parser.add_argument('--wb-name', default='default', type=str,
                        help="Run name for wandb.")
    parser.add_argument('--fSinVals', default='', type=str,
                        help="Filename (full path) for singular value plots on feat/out.")
    parser.add_argument('--save-feats', default=0, type=int,
                        help="Whether to save features (before and after proj head).")
    parser.add_argument('--fsave-feats', default='', type=str,
                        help="Full path to the file for saving features.")
    
    # testing
    parser.add_argument('--test-only', default=0, type=int, choices=[0, 1],
                        help="If test_only, then skip the training loop.")
    parser.add_argument('--load-ckpt', default=0, type=int, choices=[0,1],
                        help="Whether to load a ckpt.")
    parser.add_argument('--pretrained-path', default='', type=str,
                        help="Full path to a pretrained ckpt.")
    

    # args parse
    args = parser.parse_args()
    dataset = args.dataset
    feature_dim, temperature, k = args.feature_dim, args.temperature, args.k
    proj_head_type = args.proj_head_type
    batch_size, epochs = args.batch_size, args.epochs
    lr, wd = args.lr, args.wd
    loss_no_on_diag, loss_no_off_diag = args.loss_no_on_diag, args.loss_no_off_diag
    
    lmbda = args.lmbda
    corr_neg_one, corr_neg_one_on_diag = args.corr_neg_one, args.corr_neg_one_on_diag
    
    if USE_WANDB:
      if args.wb_name != 'default':
        wandb.init(project=args.project, name=args.wb_name, config=args)
      else:
        wandb.init(project=args.project, config=args)

    # data prepare
    DATA_ROOT = '/home/bingbin/datasets/'
    if dataset == 'cifar10':
        root = os.path.join(DATA_ROOT, 'cifar10_torch')
        train_data = torchvision.datasets.CIFAR10(root=root, train=True, \
                                                  transform=utils.CifarPairTransform(train_transform = True), download=True)
        memory_data = torchvision.datasets.CIFAR10(root=root, train=True, \
                                                  transform=utils.CifarPairTransform(train_transform = False), download=True)
        test_data = torchvision.datasets.CIFAR10(root=root, train=False, \
                                                  transform=utils.CifarPairTransform(train_transform = False), download=True)
    elif dataset == 'stl10':
        # TODO: update root
        root = DATA_ROOT
        train_data = torchvision.datasets.STL10(root=root, split="train+unlabeled", \
                                                  transform=utils.StlPairTransform(train_transform = True), download=True)
        memory_data = torchvision.datasets.STL10(root=root, split="train", \
                                                  transform=utils.StlPairTransform(train_transform = False), download=True)
        test_data = torchvision.datasets.STL10(root=root, split="test", \
                                                  transform=utils.StlPairTransform(train_transform = False), download=True)
    elif dataset == 'tiny_imagenet':
        # TODO: update root
        root = DATA_ROOT
        train_data = torchvision.datasets.ImageFolder('data/tiny-imagenet-200/train', \
                                                      utils.TinyImageNetPairTransform(train_transform = True))
        memory_data = torchvision.datasets.ImageFolder('data/tiny-imagenet-200/train', \
                                                      utils.TinyImageNetPairTransform(train_transform = False))
        test_data = torchvision.datasets.ImageFolder('data/tiny-imagenet-200/val', \
                                                      utils.TinyImageNetPairTransform(train_transform = False))
    
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=16, pin_memory=True,
                            drop_last=True)
    memory_loader = DataLoader(memory_data, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True)

    # model setup and optimizer config
    model = Model(feature_dim, proj_head_type, dataset).cuda()
    if args.load_ckpt and os.path.exists(args.pretrained_path):
      ckpt_dict = torch.load(args.pretrained_path, map_location='cpu')
      model.load_state_dict(ckpt_dict, strict=False)

    if USE_THOP:
      if dataset == 'cifar10':
          flops, params = profile(model, inputs=(torch.randn(1, 3, 32, 32).cuda(),))
      elif dataset == 'tiny_imagenet' or dataset == 'stl10':
          flops, params = profile(model, inputs=(torch.randn(1, 3, 64, 64).cuda(),))
  
      flops, params = clever_format([flops, params])
      print('# Model Params: {} FLOPs: {}'.format(params, flops))
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    
    NUM_CLS = len(memory_data.classes)

    # training loop
    results = {'train_loss': [], 'test_acc@1': [], 'test_acc@5': []}
    if corr_neg_one is True:
        corr_neg_one_str = 'neg_corr_'
    else:
        corr_neg_one_str = ''
    save_name_pre = '{}{}_{}_{}_{}'.format(corr_neg_one_str, lmbda, feature_dim, batch_size, dataset)
    
    if not os.path.exists('results'):
        os.mkdir('results')
    best_acc = 0.0

    if args.test_only:
      # plots
      fig_dir = os.path.dirname(args.fSinVals)
      os.makedirs(fig_dir, exist_ok=1)
      # features
      save_feats = args.save_feats
      fsave_feats = args.fsave_feats
      feat_dir = os.path.dirname(fsave_feats)
      os.makedirs(feat_dir, exist_ok=1)

      print("On test set")
      fSinVals_test = args.fSinVals + '_test.png'
      fsave_feats_test = args.fsave_feats + '_test.h5'
      test_stats(model, test_loader, fSinVals=fSinVals_test, save_feats=save_feats, fsave_feats=fsave_feats_test)
      exit()
      print("On train set")
      fSinVals_train = args.fSinVals + '_train.png'
      fsave_feats_train = args.fsave_feats + '_train.h5'
      test_stats(model, memory_loader, fSinVals=fSinVals_train, save_feats=save_feats, fsave_feats=fsave_feats_train)
      exit()

    test_acc_1, test_acc_5 = test(model, memory_loader, test_loader, epoch=-1)
    for epoch in range(1, epochs + 1):
        train_loss = train(model, train_loader, optimizer)
        if epoch % 5 == 0:
            results['train_loss'].append(train_loss)
            test_acc_1, test_acc_5 = test(model, memory_loader, test_loader, epoch=epoch)
            results['test_acc@1'].append(test_acc_1)
            results['test_acc@5'].append(test_acc_5)
            # save statistics
            data_frame = pd.DataFrame(data=results, index=range(5, epoch + 1, 5))
            data_frame.to_csv('results/{}_statistics.csv'.format(save_name_pre), index_label='epoch')
            if test_acc_1 > best_acc:
                best_acc = test_acc_1
                torch.save(model.state_dict(), 'results/{}_model.pth'.format(save_name_pre))
        if epoch % 50 == 0:
            torch.save(model.state_dict(), 'results/{}_model_{}.pth'.format(save_name_pre, epoch))
