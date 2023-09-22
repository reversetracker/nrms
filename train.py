import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torch.utils.data import DataLoader, random_split

import datasets.v1
import directories
import wandb
from models.v1 import NRMS

PARALLEL_NUM = 2

BATCH_SIZE = 64  # users

ARTICLES = 64  # articles

SEQ_LENGTH = 20  # words number of each article

EMBED_SIZE = 768  # embedding size

NUM_HEADS = 8  # number of heads


def main():
    wandb_logger = WandbLogger()
    wandb.init(project="nrms")

    dataframe = pd.read_csv(directories.train_dataset_csv)
    dataset = datasets.v1.OheadlineDataset(dataframe)

    train_size = int(0.90 * len(dataset))
    val_size = int(0.05 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size]
    )

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=PARALLEL_NUM
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=PARALLEL_NUM
    )
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=PARALLEL_NUM
    )

    nrms = NRMS(EMBED_SIZE, NUM_HEADS)

    # Define callbacks below
    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints",
        filename="{epoch}-{val_loss:.2f}",
        save_top_k=-1,  # save all epochs
        verbose=True,
        monitor="val_loss",
        mode="min",
    )

    # Define trainer and fit model
    trainer = pl.Trainer(
        max_epochs=5,
        log_every_n_steps=5,
        logger=wandb_logger,
        callbacks=[checkpoint_callback],
    )
    trainer.fit(nrms, train_loader, val_loader)
    trainer.test(nrms, test_loader)
    wandb.finish()


if __name__ == "__main__":
    main()
