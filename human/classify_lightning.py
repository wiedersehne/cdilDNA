import os
import wandb
import torch
import time
import numpy as np
import pandas as pd
from torch import nn, optim
from tqdm import tqdm
from omegaconf import OmegaConf
from functools import lru_cache
from torch.utils.data import DataLoader
from revolution_models import Classifier
# from CNN import Classifier
from data_utils import vcf_Dataset
from torchmetrics import AUROC
import pytorch_lightning as pl
import matplotlib.pyplot as plt
from transformers import get_cosine_schedule_with_warmup
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities.model_summary import ModelSummary
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, StochasticWeightAveraging, TQDMProgressBar
from x_formers import FormerClassifier
# from deepsea import Classifier
pl.seed_everything(42)


class LightningWrapper(pl.LightningModule):
    def __init__(self, model, cfg, snapshot_path, train_set, val_set, test_set, pretrained, loss, file_name):
        super().__init__()
        self.save_hyperparameters(cfg)
        self.model_config = self.hparams.Revolution
        self.batch_size = self.hparams.training.batch_size
        self.length = self.hparams.Revolution.max_len
        self.model = model(**self.model_config)#.apply(self._init_weights)
        self.save_every = self.hparams.training.save_every
        self.snapshot_path = snapshot_path
        self.train_set = train_set
        self.val_set = val_set
        self.loss = loss
        self.file_name = file_name
        self.train_auc = AUROC(task='binary',pos_label=1)
        self.val_auc = AUROC(task='binary',pos_label=1)

        print(self.model)

        if pretrained:
            pretrained_path = f'./pretrained_models/Best_Models/{self.file_name}'
            pretrained = torch.load(pretrained_path, map_location='cpu')
            pretrained = pretrained["MODEL_STATE"]

            # for k, v in pretrained.items():
            #     print("pretrained", k)

            from collections import OrderedDict
            new_state_dict = OrderedDict()

            for k, v in pretrained.items():
                if k.startswith('encoder') or k.startswith('embedding'):
                    new_state_dict[k] = v

            # print("pretrained", new_state_dict.keys())

            net_dict = self.model.state_dict()
            pretrained_cm = {k: v for k, v in new_state_dict.items() if k in net_dict}

            net_dict.update(pretrained_cm)
            self.model.load_state_dict(net_dict)
            for k, v in self.model.state_dict().items():
                print(k, v)
            print(self.file_name)
            print("*************pretrained model loaded***************")

        # if os.path.exists(snapshot_path):
        #     print("Loading snapshot")
        #     self._load_snapshot(snapshot_path)

    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        return self.model(x)

    def _init_weights(self, m):
        if isinstance(m, nn.Reear):
            nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def training_step(self, batch, batch_idx):
        # Record the start time
        start_time = time.time()

        # Record the GPU memory usage before forward pass (assuming you're using GPU)
        initial_memory = torch.cuda.memory_allocated()
        ref, alt, tissue, label = batch
        output = self.model(ref, alt, tissue).squeeze()

        # Record the end time
        end_time = time.time()

        # Calculate the time taken for the forward pass
        elapsed_time = end_time - start_time

        # Calculate the GPU memory used during the forward pass
        memory_used = torch.cuda.memory_allocated() - initial_memory

        # print(elapsed_time, memory_used)

        train_loss = self.loss(output, label)
        return {"loss":train_loss, "preds":output, "labels":label, "tissue":tissue}

    def validation_step(self, batch, batch_idx):
        ref, alt, tissue, label = batch
        output = self.model(ref, alt, tissue).squeeze()
        val_loss = self.loss(output, label)
        return {"loss":val_loss, "preds":output, "labels":label, "tissue":tissue}
    
    def test_step(self, batch, batch_idx):
        self.model.eval()
        ref, alt, tissue, label = batch
        output = self.model(ref, alt, tissue).squeeze()
        test_loss = self.loss(output, label)
        return {"loss":test_loss, "preds":output, "labels":label, "tissue":tissue}

    def training_step_end(self, outputs):
        train_preds = [[] for _ in range(self.model_config.output_size)]
        train_labels = [[] for _ in range(self.model_config.output_size)]
        train_loss = outputs["loss"]
        tissue = outputs["tissue"]
        label = outputs["labels"]
        output = outputs["preds"]

        for t, p, l in zip(tissue, output, label):
            t = t.to(torch.int8)
            train_preds[t.item()].append(p.item())
            train_labels[t.item()].append(l.item())
        train_rocs = []
        for i in range(self.model_config.output_size):
            if len(train_labels[i]) != 0 and sum(train_labels[i]) != len(train_labels[i]) and sum(train_labels[i]) != 0:
                rocauc = self.train_auc(torch.from_numpy(np.array(train_preds[i])), torch.from_numpy(np.array(train_labels[i])))
                # rocauc = roc_auc_score(train_labels[i], train_preds[i])
                train_rocs.append(rocauc)
        train_roc = np.average(train_rocs)
        self.log('train_roc', train_roc, sync_dist=True)
        self.log('train_loss', train_loss, sync_dist=True)

    def validation_epoch_end(self, outputs):
        val_preds = [[] for _ in range(self.model_config.output_size)]
        val_labels = [[] for _ in range(self.model_config.output_size)]
        val_loss = torch.stack([x["loss"] for x in outputs]).mean()
        tissue = torch.stack([x["tissue"] for x in outputs]).reshape((-1,))
        label = torch.stack([x["labels"] for x in outputs]).reshape((-1,))
        output = torch.stack([x["preds"] for x in outputs]).reshape((-1,))

        for t, p, l in zip(tissue, output, label):
            t = t.to(torch.int8)
            val_preds[t.item()].append(p.item())
            val_labels[t.item()].append(l.item())
        val_rocs = []
        for i in range(self.model_config.output_size):
            if len(val_labels[i]) != 0 and sum(val_labels[i]) != len(val_labels[i]) and sum(val_labels[i]) != 0:
                # print(val_labels[i], val_preds[i])
                rocauc = self.val_auc(torch.from_numpy(np.array(val_preds[i])), torch.from_numpy(np.array(val_labels[i])))
                # rocauc = roc_auc_score(val_labels[i], val_preds[i])
                # print(rocauc)
                val_rocs.append(rocauc)
        # print(val_rocs)
        val_roc = np.average(val_rocs)
        self.log("val_auroc", val_roc, sync_dist=True)
        self.log('val_loss', val_loss, sync_dist=True)

    def test_epoch_end(self, outputs):
        test_preds = [[] for _ in range(self.model_config.output_size)]
        test_labels = [[] for _ in range(self.model_config.output_size)]
        test_loss = torch.stack([x["loss"] for x in outputs]).mean()
        tissue = torch.stack([x["tissue"] for x in outputs]).reshape((-1,))
        label = torch.stack([x["labels"] for x in outputs]).reshape((-1,))
        output = torch.stack([x["preds"] for x in outputs]).reshape((-1,))

        for t, p, l in zip(tissue, output, label):
            t = t.to(torch.int8)
            test_preds[t.item()].append(p.item())
            test_labels[t.item()].append(l.item())
        test_rocs = []
        for i in range(self.model_config.output_size):
            if len(test_labels[i]) != 0 and sum(test_labels[i]) != len(test_labels[i]) and sum(test_labels[i]) != 0:
                # print(test_labels[i], test_preds[i])
                rocauc = self.test_auc(torch.from_numpy(np.array(test_preds[i])), torch.from_numpy(np.array(test_labels[i])))
                # rocauc = roc_auc_score(test_labels[i], test_preds[i])
                # print(rocauc)
                test_rocs.append(rocauc)
        test_roc = np.average(test_rocs)
        print(test_roc)
        self.log("test_auroc", test_roc, sync_dist=True)
        self.log('test_loss', test_loss, sync_dist=True)

    def _save_snapshot(self):
        snapshot = {
            "MODEL_STATE": self.model.state_dict(),
            "EPOCHS_RUN": self.current_epoch ,
        }
        torch.save(snapshot, self.snapshot_path)
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
    
    def test_dataloader(self):
        return DataLoader(
            dataset=self.test_set,
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
                    num_warmup_steps=int(self.total_steps()*0.3),
                    num_training_steps=self.total_steps(),
                    num_cycles=self.hparams.training.n_cycles
        )
        return [optimizer], [{"scheduler": lr_scheduler, "interval": "step"}]


def classify_main(cfg):
    pretrained = cfg.Fine_tuning.training.pretrained
    length = cfg.Fine_tuning.Revolution.max_len
    model_name = "cdil"

    loss = nn.BCEWithLogitsLoss()
    # print(x.shape)

    train_ref = torch.load(f"./data/ref_{length}_train.pt")
    train_alt = torch.load(f"./data/alt_{length}_train.pt")
    train_tissue = torch.load(f"./data/tissue_{length}_train.pt")
    train_label = torch.load(f"./data/label_{length}_train.pt")

    train_set =  vcf_Dataset(train_ref, train_alt, train_tissue, train_label)
    val_set = vcf_Dataset(torch.load(f"./data/ref_{length}_valid.pt"), torch.load(f"./data/alt_{length}_valid.pt"), torch.load(f"./data/tissue_{length}_valid.pt"), torch.load(f"./data/label_{length}_valid.pt"))
    test_set = vcf_Dataset(torch.load(f"./data/ref_{length}_test.pt"), torch.load(f"./data/alt_{length}_test.pt"), torch.load(f"./data/tissue_{length}_test.pt"), torch.load(f"./data/label_{length}_test.pt"))

    ddp = DDPStrategy(process_group_backend="nccl", find_unused_parameters=True)
    snapshot_path = "test.pt"
    file_name = "CM_5_1000_10_16.pt"

    model = LightningWrapper(Classifier, cfg.Fine_tuning, snapshot_path, train_set, val_set, test_set, pretrained, loss, file_name)
    summary = ModelSummary(model, max_depth=-1)


    # ------------
    # init trainer
    # ------------

    wandb_logger = WandbLogger(dir="./wandb/", project="VE", entity='tonyu', name=f'{model_name}_{length}_{pretrained}')
    checkpoint_callback = ModelCheckpoint(monitor="test_auroc", mode="max")


    print(len(train_set), len(val_set))

    lr_monitor = LearningRateMonitor(logging_interval='step')
    callbacks_for_trainer = [TQDMProgressBar(refresh_rate=10), lr_monitor, checkpoint_callback]
    if cfg.Fine_tuning.training.patience != -1:
        early_stopping = EarlyStopping(monitor="val_auroc", mode="max", min_delta=0., patience=cfg.Fine_tuning.training.patience)
        callbacks_for_trainer.append(early_stopping)
    if cfg.Fine_tuning.training.swa_lrs != -1:
        swa = StochasticWeightAveraging(swa_lrs=1e-2)
        callbacks_for_trainer.append(swa)

    print(summary)
    trainer = pl.Trainer(
        check_val_every_n_epoch=1,
        enable_progress_bar=True,
        accelerator='gpu',
        strategy=ddp,
        devices=[0,1,2],
        max_epochs=cfg.Fine_tuning.training.n_epochs,
        gradient_clip_val=0.5,
        num_sanity_val_steps=0,
        # profiler=profiler,
        precision=16,
        logger=wandb_logger
    )
    trainer.fit(model)


if __name__ == "__main__":
    cfg = OmegaConf.load('./config/config.yaml')
    OmegaConf.set_struct(cfg, False)
    classify_main(cfg)
