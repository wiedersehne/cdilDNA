import os
import sys
import math
import wandb
import torch
import logging
import argparse
import numpy as np
from datetime import datetime
from sklearn.metrics import roc_auc_score, average_precision_score

sys.path.append('../')
from cython_data import GenomeSequence, GenomicFeatures
from cython_file import BedFileSampler

from plants import seed_everything, plant_feature, plant_bed
from model_revolution import Revolution
from model_deeperdeepsea import DeeperDeepSEA


parser = argparse.ArgumentParser(description='experiment')
parser.add_argument('--pre_training', action='store_true')
parser.add_argument('--pre_froze', action='store_true')

parser.add_argument('--pre_two', action='store_true')
parser.add_argument('--pre_two_load', type=int, default=0)

parser.add_argument('--pre_plant', type=str, default='ar')  # bd mh sb si zm zs
parser.add_argument('--pre_len', type=int, default=1000)
parser.add_argument('--pre_mask', type=float, default=0.15)
parser.add_argument('--pre_bs', type=int, default=256)
parser.add_argument('--pre_load', type=int, default=49)
parser.add_argument('--pre_layer', type=int, default=9)
parser.add_argument('--pre_en_hide', type=int, default=128)
parser.add_argument('--pre_de_hide', type=int, default=32)

parser.add_argument('--seq_len', type=int, default=1000)
parser.add_argument('--n_train', type=float, default=1)

parser.add_argument('--model', type=str, default='revolution')
parser.add_argument('--model_ks', type=int, default=3)
parser.add_argument('--hide2', type=int, default=32)

parser.add_argument('--bs', type=int, default=256)
parser.add_argument('--epoch', type=int, default=50)
parser.add_argument('--seed', type=int, default=1)
args = parser.parse_args()

use_wandb = True

pre_training = args.pre_training
pre_froze = args.pre_froze
pre_two = args.pre_two
pre_two_load = args.pre_two_load

pre_plant = args.pre_plant
pre_len = args.pre_len
pre_mask = args.pre_mask
pre_bs = args.pre_bs
pre_load = args.pre_load
pre_layer = args.pre_layer
en_hide = args.pre_en_hide
de_hide = args.pre_de_hide

plant = pre_plant
seq_len = args.seq_len
n_train = args.n_train

model = args.model
model_ks = args.model_ks
hide2 = args.hide2

batch_size = args.bs
epoch = args.epoch
seed = args.seed

input_size = 4
seed_everything(seed)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
if pre_training:
    lr = 0.0001
elif pre_two:
    lr = 0.0001
else:
    lr = 0.001

fasta_path, bed_path, features_path, n_feature = plant_feature(plant)
sequence = GenomeSequence(fasta_path)
features = GenomicFeatures(bed_path, features_path)

train_path, val_path, test_path, num_train, num_eval = plant_bed(plant)
train_sampler = BedFileSampler(train_path, sequence, features)
val_sampler = BedFileSampler(val_path, sequence, features)
test_sampler = BedFileSampler(test_path, sequence, features)

add_len = int((seq_len - 1000) / 2)
train_samples = int(num_train/100*n_train)
pre_train_num = train_samples
layer = math.ceil(math.log(seq_len/2, 2))

pre_task = 'pre' + str(pre_mask) + '_L' + str(pre_len) + '_E' + str(en_hide) + 'D' + str(de_hide) + '_' + pre_plant + str(pre_train_num)
pre_log = 'K' + str(model_ks) + '_L' + str(pre_layer) + '_bs' + str(pre_bs)

task = 'train_' + plant + str(train_samples)
if model == 'deepsea':
    net = DeeperDeepSEA(seq_len, n_feature)
    para_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    log_name1 = 'deepsea'
else:
    net = Revolution(input_size, en_hide, hide2, n_feature, model_ks, layer, seq_len)
    para_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    log_name0 = 'revolution_L' + str(layer) + '_H' + str(en_hide) + 'H' + str(hide2)

    if pre_training:
        load_path = './pre/' + pre_task + '/' + pre_log + '_E' + str(pre_load) + '.pkl'
        load_pre = torch.load(load_path)

        net_dict = net.cdilNet.state_dict()
        load_pre = {k: v for k, v in load_pre.items() if k in net_dict}

        net_dict.update(load_pre)
        net.cdilNet.load_state_dict(net_dict)

        if pre_froze:
            tuning = '_pro'
            for i_conv in range(min(pre_layer, layer)):
                for p in net.cdilNet.conv_net[i_conv].parameters():
                    p.requires_grad = False
        else:
            tuning = '_ft'

        log_name1 = log_name0 + tuning + str(pre_train_num) + '_pre' + str(pre_mask) + '_L' + str(pre_len) + '_E' + str(pre_load)
    elif pre_two:
        load_path = './train/' + task + '/' + 'L' + str(seq_len) + log_name0 +\
                    '_pro' + str(pre_train_num) + '_pre' + str(pre_mask) + '_L' + str(pre_len) + '_E' + str(pre_load) + \
                    '_S' + str(seed) + '_bs' + str(batch_size) + '_' + str(pre_two_load) + '.pkl'

        net.load_state_dict({k.replace('module.', ''): v for k, v in torch.load(load_path).items()})
        log_name1 = log_name0 + '_two' + str(pre_train_num) + '_pre' + str(pre_mask) + '_L' + str(pre_len) + '_E' + str(pre_load) + '_P' + str(pre_two_load)
    else:
        log_name1 = log_name0 + '_nopre'
log_name = 'L' + str(seq_len) + log_name1 + '_S' + str(seed) + '_bs' + str(batch_size)

net = torch.nn.DataParallel(net)
net = net.to(device)
loss = torch.nn.BCELoss(reduction='sum')
optimizer = torch.optim.Adam(net.parameters(), lr=lr)

os.makedirs('./train/', exist_ok=True)
os.makedirs('./train/' + task, exist_ok=True)

log_file_name = './train/' + task + '/' + log_name + '.txt'
handlers = [logging.FileHandler(log_file_name), logging.StreamHandler()]
logging.basicConfig(level=logging.INFO, format='%(message)s', handlers=handlers)
loginf = logging.info

loginf(log_file_name)
loginf('n_parameters:{}'.format(para_num))
loginf('learning rate:{}'.format(lr))
loginf('n_train:{} \t n_eval:{}'.format(train_samples, num_eval))
if use_wandb:
    wandb.init(project=task + '_1_L1000', name=log_name, entity="leic-no")

train_iter = train_sampler.get_data_and_targets(batch_size, train_samples, add=add_len)
val_iter = val_sampler.get_data_and_targets(batch_size, num_eval, add=add_len)
test_iter = test_sampler.get_data_and_targets(batch_size, num_eval, add=add_len)


def net_eval_thresholds(eval_iter, t_v):
    eval_loss = 0
    eval_rocs = []
    eval_aps = []

    eval_targets_all = np.empty([0, n_feature])
    eval_preds_all = np.empty([0, n_feature])

    for (eval_sequences, eval_targets) in eval_iter:
        eval_sequences, eval_targets = torch.Tensor(eval_sequences), torch.Tensor(eval_targets)
        eval_sequences, eval_targets = eval_sequences.to(device), eval_targets.to(device)
        eval_preds = net(eval_sequences)
        eval_loss += loss(eval_preds, eval_targets).item()
        eval_targets_all = np.append(eval_targets_all, eval_targets.cpu().detach().numpy(), axis=0)
        eval_preds_all = np.append(eval_preds_all, eval_preds.cpu().detach().numpy(), axis=0)
    eval_loss_mean = eval_loss / num_eval

    for feature in range(n_feature):
        eval_rocs.append(roc_auc_score(eval_targets_all[:, feature], eval_preds_all[:, feature]))
        eval_aps.append(average_precision_score(eval_targets_all[:, feature], eval_preds_all[:, feature]))
    roc = np.average(eval_rocs)
    ap = np.average(eval_aps)
    eval_time = (datetime.now() - start_time).total_seconds()
    loginf('{} loss:{} -- roc:{} -- ap:{} -- time:{}'.format(t_v, eval_loss_mean, roc, ap, eval_time))
    return eval_loss_mean, roc


best_val = 0
start_time = datetime.now()

for e in range(epoch):
    train_loss = 0
    net.train()
    for (sequences, targets) in train_iter:
        optimizer.zero_grad()
        sequences, targets = torch.Tensor(sequences), torch.Tensor(targets)
        sequences, targets = sequences.to(device), targets.to(device)

        preds = net(sequences)
        batch_loss = loss(preds, targets)
        batch_loss.backward()
        optimizer.step()

        train_loss += batch_loss
    train_loss_mean = train_loss / train_samples
    train_time = (datetime.now() - start_time).total_seconds()
    loginf('EPOCH:{} -- Train loss:{} -- time:{}'.format(e, train_loss_mean, train_time))
    if e < 5 and pre_froze:
        torch.save(net.state_dict(), './train/' + task + '/' + log_name + '_' + str(e) + '.pkl')

    with torch.no_grad():
        net.eval()
        val_loss_mean, val_roc = net_eval_thresholds(val_iter, 'Val')
        test_loss_mean, test_roc = net_eval_thresholds(test_iter, 'Test')
        n = 50
        if val_roc >= best_val:
            torch.save(net.state_dict(), './train/' + task + '/' + log_name + '.pkl')
            best_val = val_roc
            best_test = test_roc
            n = 150
        loginf('='*n)

    if use_wandb:
        wandb.log({
            'train loss': train_loss_mean,
            'val loss': val_loss_mean,
            'test loss': test_loss_mean,
            'val roc': val_roc,
            'test roc': test_roc,
            })
    # if (e+1) % 20 == 0:
    #     loginf(best_test)
    #     loginf('=' * 200)

loginf(best_test)
