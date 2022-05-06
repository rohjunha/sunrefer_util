from argparse import Namespace

import pytorch_lightning as pl
import torch
from torch.nn.functional import smooth_l1_loss


class BoundingBoxRegressor(pl.LightningModule):
    def __init__(self, args: Namespace):
        pl.LightningModule.__init__(self)
        self.args = args
        self.model = []

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.params,
            lr=self.args.lr,
            weight_decay=self.args.weight_decay,
            amsgrad=self.args.amsgrad)

        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lambda x: (1 - x / (self.num_train_steps * self.args.epochs)) ** 0.9)
        return [optimizer], [lr_scheduler]

    def forward(self, batch, batch_idx):
        class_names, points, target_bboxs = batch
        output = self.model(points)
        return output, target_bboxs

    def training_step(self, batch, batch_idx):
        output, target_bboxs = self.forward(batch, batch_idx)
        loss = smooth_l1_loss(output, target_bboxs)
        self.log('train/loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        output, target_bboxs = self.forward(batch, batch_idx)
        loss = smooth_l1_loss(output, target_bboxs)
        self.log('val/loss', loss)
        return loss
