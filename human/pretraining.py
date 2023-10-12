from random import randint, shuffle
from random import random as rand
from utils import *
import pandas as pd
import torch.nn as nn
import sys
import yaml
import argparse
import cdil_models
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from datetime import datetime
import random
import copy
from pyfaidx import Fasta
from pyfaidx import Faidx
from multiprocessing import Pool
import multiprocessing as mp
from Bio import SeqIO
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from sklearn.metrics import roc_auc_score, average_precision_score
import wandb
from omegaconf import OmegaConf
from functools import lru_cache
from cdil_models import *
from torch.distributed import init_process_group, destroy_process_group
import pytorch_lightning as pl
import matplotlib.pyplot as plt
from torch import nn, optim
from transformers import get_cosine_schedule_with_warmup
# from pytorch_lightning.profiler import SimpleProfiler
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities.model_summary import ModelSummary
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, StochasticWeightAveraging, TQDMProgressBar
pl.seed_everything(42)

def pretrain_loss(loss, preds, labels, masks):
    masks_new = masks.repeat(5, 1, 1)#.reshape(preds.shape)
    masks_new = torch.reshape(masks_new, preds.shape)
    # # masks_new = torch.tensor(masks_new)
    # # print(masks_new.shape)

    # # labels_masked = labels*masks
    # # preds_masked = preds*masks_new
    _, predicted = preds.max(2)
    print(predicted[0][0:50])
    print(labels[0][0:50])

    # print(labels_masked.shape, preds_masked.shape)

    # preds_masked = preds_masked.to(torch.long)  # Convert to LongTensor

    loss_value = loss(
        (preds*masks_new).float().transpose(1, 2), (labels*masks).long()
    )
    non_zeros = len(masks[masks==1])
    loss_value = loss_value/non_zeros

    return loss_value


class DatasetCreator(Dataset):
    """
    Class to construct a dataset for training/inference
    """
    def __init__(self, genes, masked_genes, masks):
        self.genes = genes
        self.masked_genes = masked_genes
        self.masks = masks

    def __getitem__(self, index):
        return (self.genes[index], self.masked_genes[index], self.masks[index])

    def __len__(self):
        return len(self.genes)

# Model for Pretraining
class Model4Pretrain(nn.Module):
    "CDIL Model for Pretrain : Masked LM"
    def __init__(self, dim, hdim1, hdim2, kernel_size, n_layers, dropout):
        super().__init__()
        self.encoder = CDILLayer(dim, [hdim1]*n_layers, hdim1*2, kernel_size, dropout)
        self.hidden_list = [hdim2]*n_layers
        self.hidden_list[-1] = dim
        self.decoder = CDILLayer(hdim1, self.hidden_list, hdim2*2, kernel_size, dropout)
        # self.sigmoid = nn.Sigmoid()

    def forward(self, input_seq):
        input_seq = input_seq.float()
        # encoder
        h = torch.permute(input_seq, (0, 2, 1))
        h = self.encoder(h)
        # decoder
        h = self.decoder(h)
        h = torch.permute(h, (0, 2, 1))

        return h


class LightningWrapper(pl.LightningModule):
    def __init__(self, model, cfg, snapshot_path, train_set, val_set, loss):
        super().__init__()
        self.save_hyperparameters(cfg)
        self.model_config = self.hparams.CDIL
        self.batch_size = self.hparams.training.batch_size
        self.model = model(**self.model_config).apply(self._init_weights)
        self.save_every = self.hparams.training.save_every
        self.snapshot_path = snapshot_path
        self.train_set = train_set
        self.val_set = val_set
        self.loss = loss

        print(self.model)

    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        return self.model(x)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def training_step(self, batch, batch_idx):
        gene, masked_gene, masks = batch
        # print(gene.shape, masked_gene.shape, masks.shape)
        logits = torch.squeeze(self.model(masked_gene))
        # print(logits.shape)
        # predicted = torch.argmax(logits, dim=2)
        # predicted = torch.tensor(predicted, requires_grad=True)
        # print(predicted.shape, predicted[0][0:50], gene[0][0:50])
        # predicted = predicted[masks==1]
        # masked_gene = gene[masks==1]
        # print(predicted[0:10])
        # print(masked_gene[0:10])
        # import pdb; pdb.set_trace()
        loss = pretrain_loss(self.loss, logits, gene, masks)
        # print("loss_lm", loss_lm.shape)
        # non_zeros = len(masks[masks==1])
        # loss = (loss * masks).sum()/non_zeros
        self.log('train_loss', loss, sync_dist=True)
        if self.global_rank == 0 and self.global_step % self.save_every == 0:
            self._save_snapshot()
        return {"loss":loss}

    # def train_epoch_end(self, outputs):
    #     self._save_snapshot()

    def validation_step(self, batch, batch_idx):
        gene, masked_gene, masks = batch
        print(gene.shape, masked_gene.shape, masks.shape)
        # print(gene.shape)
        # print(masked_gene.shape)
        logits = torch.squeeze(self.model(masked_gene))
        print(logits.shape)
        # _, predicted = logits.max(2)
        # predicted = predicted[masks==1]
        # masked_gene = gene[masks==1]
        loss = pretrain_loss(self.loss, logits, gene, masks)
        # non_zeros = len(masks[masks==1])
        # loss = (loss * masks).sum()/non_zeros

        return {"loss":loss}

    def validation_epoch_end(self, outputs):
        val_loss = torch.stack([x["loss"] for x in outputs]).mean()
        self.log('val_loss', val_loss, sync_dist=True)
        # self._save_snapshot()

    def _save_snapshot(self):
        snapshot = {
            "MODEL_STATE": self.model.state_dict(),
            "EPOCHS_RUN": self.current_epoch ,
        }
        torch.save(snapshot, f"{self.snapshot_path}/model_{self.current_epoch}_8_16.pt")
        print(f"Epoch {self.current_epoch } | Training snapshot saved at {self.snapshot_path}")

    def _load_snapshot(self, snapshot_path):
        loc = f"cuda:0"
        snapshot = torch.load(snapshot_path, map_location=loc)
        self.model.load_state_dict(snapshot["MODEL_STATE"])
        self.epochs_run = snapshot["EPOCHS_RUN"]
        print(f"Resuming training from snapshot at Epoch {self.epochs_run}")

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_set,
            num_workers=1,
            pin_memory=True,
            shuffle=True,
            drop_last=True,
            batch_size=self.batch_size
            )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.val_set,
            num_workers=1,
            pin_memory=True,
            shuffle=False,
            drop_last=True,
            batch_size=self.batch_size
            )

    @lru_cache
    def total_steps(self):
        l = len(self.trainer._data_connector._train_dataloader_source.dataloader())
        print('Num devices', self.trainer.num_devices)
        max_epochs = self.trainer.max_epochs
        accum_batches = self.trainer.accumulate_grad_batches
        manual_total_steps = (l // accum_batches * max_epochs)/self.trainer.num_devices
        print('MANUAL Total steps', manual_total_steps)
        return manual_total_steps

    def configure_optimizers(self):
        optimizer = optim.AdamW(
            self.parameters(),
            lr=self.hparams.training.learning_rate,
            weight_decay=self.hparams.training.weight_decay
        )
        lr_scheduler = get_cosine_schedule_with_warmup(
                    optimizer,
                    num_warmup_steps=self.total_steps()*0.3,
                    num_training_steps=self.total_steps(),
                    num_cycles=self.hparams.training.n_cycles
        )
        return [optimizer], [{"scheduler": lr_scheduler, "interval": "step"}]


def pretrain_main(cfg):
    # 1. Load data
    genes_ve = torch.load(f"./data/pretrain/gene_ve_{cfg.Pretraining.training.max_len}.pt")
    masked_genes_ve = torch.load(f"./data/pretrain/masked_ve_{cfg.Pretraining.training.max_len}.pt")
    masks_ve = torch.load(f"./data/pretrain/mask_ve_{cfg.Pretraining.training.max_len}.pt")

    genes_hg38 = torch.load(f"./data/pretrain/gene_hg38_{cfg.Pretraining.training.max_len}_100k.pt")
    masked_genes_hg38 = torch.load(f"./data/pretrain/masked_hg38_{cfg.Pretraining.training.max_len}_100k.pt")
    masks_hg38 = torch.load(f"./data/pretrain/mask_hg38_{cfg.Pretraining.training.max_len}_100k.pt")

    genes_train = torch.cat((genes_ve, genes_hg38), 0)
    masked_genes_train = torch.cat((masked_genes_ve, masked_genes_hg38), 0)
    masks_train = torch.cat((masks_ve, masks_hg38), 0)

    genes_val = torch.load(f"./data/pretrain/gene_train_{cfg.Pretraining.training.max_len}.pt")
    masked_genes_val = torch.load(f"./data/pretrain/masked_train_{cfg.Pretraining.training.max_len}.pt")
    masks_val = torch.load(f"./data/pretrain/mask_train_{cfg.Pretraining.training.max_len}.pt")
    # 2. Generate data for pretraining
    print(genes_train.shape)
    train_set =   DatasetCreator(genes_train, masked_genes_train, masks_train)
    val_set =   DatasetCreator(genes_val, masked_genes_val, masks_val)

    ddp = DDPStrategy(process_group_backend="nccl", find_unused_parameters=True)
    # profiler = SimpleProfiler()
    snapshot_path = "./models/pretrain/"

    loss = nn.CrossEntropyLoss(reduce="sum")
    # loss = nn.MSELoss(reduce="sum")
    model = LightningWrapper(Model4Pretrain, cfg.Pretraining, snapshot_path, train_set, val_set, loss)
    summary = ModelSummary(model, max_depth=-1)
    wandb_logger = WandbLogger(dir="./wandb/", project="CDIL_VE_Pretrain", entity='tonyu', name=f'Pretraining_{cfg.Pretraining.training.max_len}')
    checkpoint_callback = ModelCheckpoint(monitor="val_loss", mode="min")

    lr_monitor = LearningRateMonitor(logging_interval='step')
    callbacks_for_trainer = [TQDMProgressBar(refresh_rate=10), lr_monitor, checkpoint_callback]
    # if cfg.Pretraining.training.patience != -1:
    #     early_stopping = EarlyStopping(monitor="val_loss", mode="min", min_delta=0., patience=cfg.Fine_tuning.training.patience)
    #     callbacks_for_trainer.append(early_stopping)
    # if cfg.Pretraining.training.swa_lrs != -1:
    #     swa = StochasticWeightAveraging(swa_lrs=1e-2)
    #     callbacks_for_trainer.append(swa)

    print(summary)
    trainer = pl.Trainer(
        check_val_every_n_epoch=1,
        enable_progress_bar=True,
        accelerator='gpu',
        strategy=ddp,
        devices=[0],
        max_epochs=cfg.Pretraining.training.n_epochs,
        gradient_clip_val=0.5,
        num_sanity_val_steps=0,
        # profiler=profiler,
        precision=16,
        logger=wandb_logger,
        callbacks=callbacks_for_trainer
    )
    trainer.fit(model)


if __name__ == "__main__":
    cfg = OmegaConf.load('./config/config.yaml')
    OmegaConf.set_struct(cfg, False)
    pretrain_main(cfg)
