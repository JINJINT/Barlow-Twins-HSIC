import argparse
import os

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

import utils
from model import Model

import torchvision

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True

import pdb

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

def plot(net, test_data_loader):
    net.eval()
    count = 0
    total = 0

    mean_feat_val = 0
    min_feat_val = 0
    max_feat_val = 0

    mean_out_val = 0
    min_out_val = 0
    max_out_val = 0

    with torch.no_grad():
        for data_tuple in test_data_loader:
            (data, _), target = data_tuple
            data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)
            feature, out = net(data)

            # normalize features
            o_std = out.std(dim=0)
            o_std[o_std==0] = 1
            out_norm = (out - out.mean(dim=0)) / o_std
            f_std = feature.std(dim=0)
            f_std[f_std==0] = 1
            feat_norm = (feature - feature.mean(dim=0)) / f_std

            _, F = feature.shape
            _, P = out.shape
            total += 1
            if total < 5:
                sim_matrix_feat = torch.mm(feat_norm, feat_norm.T) / F
                out_matrix_feat = torch.mm(out_norm, out_norm.T) / P

                dim_matrix_feat = torch.mm(feat_norm.T, feat_norm) / batch_size
                dim_matrix_out = torch.mm(out_norm.T, out_norm) / batch_size

                plt.clf()
                ax = sns.heatmap(sim_matrix_feat.cpu().numpy())
                plt.title(f"Inter Feature similarity (Encoder) {total}")
                plt.savefig(os.path.join(fig_dir, f"inter_enc_feat_sim{total}.png"), dpi=480)

                plt.clf()
                ax = sns.heatmap(out_matrix_feat.cpu().numpy())
                plt.title(f"Inter projection head feat similarity {total}")
                plt.savefig(os.path.join(fig_dir, f"inter_ph_out_sim{total}.png"), dpi=480)

                plt.clf()
                ax = sns.heatmap(dim_matrix_feat.cpu().numpy())
                plt.title(f"Inter feature dimension similarity (Encoder) {total}")
                plt.savefig(os.path.join(fig_dir, f"inter_dim_enc_feat_sim{total}.png"), dpi=480)

                plt.clf()
                ax = sns.heatmap(dim_matrix_out.cpu().numpy())
                plt.title(f"Inter feat dimension similarity (Projection head) {total}")
                plt.savefig(os.path.join(fig_dir, f"inter_dim_ph_out_sim{total}.png"), dpi=480)

                for i in range(5):
                    sim_matrix_feat = torch.mm(feature[i].view(F, 1), feature[i].view(1, F))
                    sim_matrix_out = torch.mm(out[i].view(P, 1), out[i].view(1, P))
                    plt.clf()
                    ax = sns.heatmap(sim_matrix_feat.cpu().numpy())
                    plt.title(f"feature correlation {i}")
                    plt.savefig(os.path.join(fig_dir, f"feat_correlation_{total}_{i}.png"), dpi=480)

                    plt.clf()
                    ax = sns.heatmap(sim_matrix_out.cpu().numpy())
                    plt.title(f"output correlation {i}")
                    plt.savefig(os.path.join(fig_dir, f"out_correlation_{total}_{i}.png"), dpi=480)

                    plt.clf()
                    plt.hist(sim_matrix_feat.cpu().flatten().cpu().numpy(), 10)
                    plt.title(f"feature correlation distribution {i}")
                    plt.savefig(os.path.join(fig_dir, f"feat_correlation_hist_{total}_{i}.png"), dpi=480)

                    plt.clf()
                    plt.hist(sim_matrix_out.cpu().flatten().cpu().numpy(), 10)
                    plt.title(f"output correlation distribution {i}")
                    plt.savefig(os.path.join(fig_dir, f"out_correlation_hist_{total}_{i}.png"), dpi=480)
                    count += 1
            
            mean_feat_val += feature.mean()
            min_feat_val += feature.min()
            max_feat_val += feature.max()

            mean_out_val += out.mean()
            min_out_val += out.min()
            max_out_val += out.max()

        print(f"Mean feature val {mean_feat_val/total}")
        print(f"Min feature val {min_feat_val/total}")
        print(f"Max feature val {max_feat_val/total}")

        print(f"Mean out val {mean_out_val/total}")
        print(f"Min out val {min_out_val/total}")
        print(f"Max out val {max_out_val/total}")



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train SimCLR')
    parser.add_argument('--dataset', default='cifar10', type=str, help='Dataset: cifar10 or tiny_imagenet or stl10')
    parser.add_argument('--feature_dim', default=128, type=int, help='Feature dim for latent vector')
    parser.add_argument('--temperature', default=0.5, type=float, help='Temperature used in softmax')
    parser.add_argument('--k', default=200, type=int, help='Top k most similar images used to predict the label')
    parser.add_argument('--batch_size', default=128, type=int, help='Number of images in each mini-batch')
    parser.add_argument('--num_workers', default=16, type=int, help='Number of workers to use in dataloader')
    parser.add_argument('--epochs', default=1000, type=int, help='Number of sweeps over the dataset to train')
    parser.add_argument('--proj-head-type', default='2layer', choices=['none', 'linear', '2layer'])
    parser.add_argument('--ckpt_path', type=str, required=True, help='path to checkpoint')
    # for barlow twins
    parser.add_argument('--lmbda', default=0.005, type=float, help='Lambda that controls the on- and off-diagonal terms')
    parser.add_argument('--corr_neg_one', dest='corr_neg_one', action='store_true')
    parser.add_argument('--corr_zero', dest='corr_neg_one', action='store_false')
    parser.set_defaults(corr_neg_one=False)

    parser.add_argument('--fig-dir', type=str, default='', help="Dir path in which the plots are saved.")

    # args parse
    args = parser.parse_args()
    dataset = args.dataset
    feature_dim, temperature, k = args.feature_dim, args.temperature, args.k
    proj_head_type = args.proj_head_type
    batch_size, epochs = args.batch_size, args.epochs
    
    fig_dir = args.fig_dir
    os.makedirs(fig_dir, exist_ok=1)
     
    lmbda = args.lmbda
    corr_neg_one = args.corr_neg_one

    # data prepare
    if dataset == 'cifar10':
        train_data = torchvision.datasets.CIFAR10(root='data', train=True, \
                                                  transform=utils.CifarPairTransform(train_transform = True), download=True)
        memory_data = torchvision.datasets.CIFAR10(root='data', train=True, \
                                                  transform=utils.CifarPairTransform(train_transform = False), download=True)
        test_data = torchvision.datasets.CIFAR10(root='data', train=False, \
                                                  transform=utils.CifarPairTransform(train_transform = False), download=True)

    elif dataset == 'stl10':
        train_data = torchvision.datasets.STL10(root='data', split="train+unlabeled", \
                                                  transform=utils.StlPairTransform(train_transform = True), download=True)
        memory_data = torchvision.datasets.STL10(root='data', split="train", \
                                                  transform=utils.StlPairTransform(train_transform = False), download=True)
        test_data = torchvision.datasets.STL10(root='data', split="test", \
                                                  transform=utils.StlPairTransform(train_transform = False), download=True)
    elif dataset == 'tiny_imagenet':
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
    model.load_state_dict(torch.load(args.ckpt_path), strict=False)

    plot(model, test_data_loader=test_loader)
