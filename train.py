import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torch.utils.data import DataLoader, random_split

import datasets.v1
import directories
import wandb
from models.v1 import NRMS

parallel_num = 4

batch_size = 64  # users
articles = 64  # articles
seq_length = 20  # words number of each article
embed_size = 768  # embedding size
num_heads = 8  # number of heads


def main():
    wandb_logger = WandbLogger()
    wandb.init(project="nrms")

    dataframe = pd.read_csv(directories.bq_results_csv)
    dataset = datasets.v1.OheadlineDataset(dataframe)

    train_size = int(0.90 * len(dataset))
    val_size = int(0.05 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=parallel_num
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=parallel_num
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=parallel_num
    )

    nrms = NRMS(embed_size, num_heads)

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
