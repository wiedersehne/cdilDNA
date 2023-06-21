import os
import sys
import math
import torch
import logging
import argparse
from datetime import datetime

sys.path.append('../')
from cython_data import GenomeSequence, GenomicFeatures
from cython_file import BedFileSampler

from plants import seed_everything, plant_feature, plant_bed
from utils_pre import BatchMaking, pretrain_loss
from model_revolution import Model4Pretrain

parser = argparse.ArgumentParser(description='experiment')
parser.add_argument('--pre_plant', type=str, default='ar')  # ar bd mh sb si zm zs
parser.add_argument('--pre_len', type=int, default=1000)
parser.add_argument('--pre_n', type=float, default=1)

parser.add_argument('--pre_mask', type=float, default=0.15)
parser.add_argument('--pre_bs', type=int, default=256)
parser.add_argument('--pre_epoch', type=int, default=50)

parser.add_argument('--model_ks', type=int, default=3)
parser.add_argument('--en_hide', type=int, default=128)
parser.add_argument('--de_hide', type=int, default=32)
args = parser.parse_args()

pre_plant = args.pre_plant
pre_len = args.pre_len
pre_n = args.pre_n

pre_mask = args.pre_mask
pre_bs = args.pre_bs
pre_epoch = args.pre_epoch

model_ks = args.model_ks
en_hide = args.en_hide
de_hide = args.de_hide

input_size = 4
seed_everything(1)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

fasta_path, bed_path, features_path, n_feature = plant_feature(pre_plant)
sequence = GenomeSequence(fasta_path)
features = GenomicFeatures(bed_path, features_path)

train_path, val_path, test_path, num_train, num_eval = plant_bed(pre_plant)
train_sampler = BedFileSampler(train_path, sequence, features)
val_sampler = BedFileSampler(val_path, sequence, features)
test_sampler = BedFileSampler(test_path, sequence, features)

add_len = int((pre_len-1000)/2)
pre_train_num = int(num_train/100*pre_n)
pre_layer = math.ceil(math.log(pre_len/2, 2))

mask = BatchMaking(pre_mask)
pre_net = Model4Pretrain(input_size, en_hide, de_hide, model_ks, pre_layer)
pre_net = torch.nn.DataParallel(pre_net)
pre_net = pre_net.to(device)
pre_loss = torch.nn.BCELoss(reduction='sum')
pre_optimizer = torch.optim.Adam(pre_net.parameters())

pre_task = 'pre' + str(pre_mask) + '_L' + str(pre_len) + '_E' + str(en_hide) + 'D' + str(de_hide) + '_' + pre_plant + str(pre_train_num)
pre_log = 'K' + str(model_ks) + '_L' + str(pre_layer) + '_bs' + str(pre_bs)

os.makedirs('./pre/', exist_ok=True)
os.makedirs('./pre/' + pre_task, exist_ok=True)

log_file_name = './pre/' + pre_task + '/' + pre_log + '.txt'
handlers = [logging.FileHandler(log_file_name), logging.StreamHandler()]
logging.basicConfig(level=logging.INFO, format='%(message)s', handlers=handlers)
loginf = logging.info

pre_para = sum(p.numel() for p in pre_net.parameters() if p.requires_grad)
loginf(log_file_name)
loginf('n_parameters:{}'.format(pre_para))
loginf('n_train:{} \t n_eval:{}'.format(pre_train_num, num_eval))

train_iter = train_sampler.get_data_and_targets(pre_bs, pre_train_num, add=add_len)
val_iter = val_sampler.get_data_and_targets(pre_bs, num_eval, add=add_len)
test_iter = test_sampler.get_data_and_targets(pre_bs, num_eval, add=add_len)


def net_mask_eval(eval_iter, v_t):
    eval_loss = 0
    for (eval_batch_pretrain, _) in eval_iter:
        eval_batch_pretrain_copy = eval_batch_pretrain.copy()
        eval_batch_mask, eval_batch_vec = mask(eval_batch_pretrain_copy)

        eval_batch_pretrain, eval_batch_mask = torch.Tensor(eval_batch_pretrain), torch.Tensor(eval_batch_mask)
        eval_batch_pretrain, eval_batch_mask = eval_batch_pretrain.to(device), eval_batch_mask.to(device)

        eval_batch_mask_pred = pre_net(eval_batch_mask)
        eval_batch_loss = pretrain_loss(pre_loss, eval_batch_pretrain, eval_batch_mask_pred, eval_batch_vec)
        eval_loss += eval_batch_loss
    eval_loss_mean = eval_loss / num_eval
    eval_time = (datetime.now() - start_time).total_seconds()
    loginf('EPOCH:{} -- {} loss:{} -- time:{}'.format(e, v_t, eval_loss_mean, eval_time))


start_time = datetime.now()
for e in range(pre_epoch):
    train_loss = 0
    pre_net.train()
    for (batch_pretrain, _) in train_iter:
        pre_optimizer.zero_grad()
        batch_pretrain_copy = batch_pretrain.copy()
        batch_mask, batch_vec = mask(batch_pretrain_copy)

        batch_pretrain, batch_mask = torch.Tensor(batch_pretrain), torch.Tensor(batch_mask)
        batch_pretrain, batch_mask = batch_pretrain.to(device), batch_mask.to(device)

        batch_mask_pred = pre_net(batch_mask)
        batch_loss = pretrain_loss(pre_loss, batch_pretrain, batch_mask_pred, batch_vec)
        batch_loss.backward()
        pre_optimizer.step()

        train_loss += batch_loss
    train_loss_mean = train_loss / pre_train_num
    train_time = (datetime.now() - start_time).total_seconds()
    loginf('EPOCH:{} -- Train loss:{} -- time:{}'.format(e, train_loss_mean, train_time))

    if (e + 1) % 10 == 0:
        torch.save(pre_net.module.cdilNet.state_dict(), './pre/' + pre_task + '/' + pre_log + '_E' + str(e) + '.pkl')

    # with torch.no_grad():
    #     pre_net.eval()
    #     net_mask_eval(val_iter, 'Val')
    #     net_mask_eval(test_iter, 'Test')
    #     loginf('=' * 150)
