import os

import pandas as pd
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader

import datasets.v1
import directories
import wandb
from configs import settings
from models.v1 import NRMS

os.environ["WANDB_API_KEY"] = settings.wandb_api_key


PARALLEL_NUM = min(2, os.cpu_count())


def main():
    wandb.init(project="nrms")
    wandb_logger = WandbLogger()

    dataframe = pd.read_csv(directories.unittest_dataset_csv)
    dataset = datasets.v1.OheadlineDataset(
        dataframe,
        article_size=settings.article_size,
        sequence_size=settings.sequence_size,
        embed_dim=settings.embed_dim,
        K=settings.K,
    )

    train_size = int(0.94 * len(dataset))
    val_size = int(0.03 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size]
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=settings.batch_size,
        shuffle=True,
        num_workers=PARALLEL_NUM,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=settings.batch_size,
        shuffle=False,
        num_workers=PARALLEL_NUM,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=settings.batch_size,
        shuffle=False,
        num_workers=PARALLEL_NUM,
    )

    nrms = NRMS(
        embed_dim=settings.embed_dim,
        encoder_dim=settings.encoder_dim,
        num_heads_news_encoder=settings.num_heads_news_encoder,
        num_heads_user_encoder=settings.num_heads_user_encoder,
        lr=settings.lr,
        weight_decay=settings.weight_decay,
        dropout=settings.dropout,
    )

    # Define callbacks below
    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints",
        filename="{epoch}-{val_loss:.2f}",
        save_top_k=5,
        verbose=True,
        monitor="val_loss",
        mode="min",
    )

    # Define trainer and fit model
    trainer = pl.Trainer(
        max_epochs=1000,
        log_every_n_steps=1,
        logger=wandb_logger,
        callbacks=[checkpoint_callback],
    )
    trainer.fit(nrms, train_loader, val_loader)
    trainer.test(nrms, test_loader)
    wandb.finish()


if __name__ == "__main__":
    main()
