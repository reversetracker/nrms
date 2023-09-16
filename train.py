import pandas as pd
import pytorch_lightning as pl
from lightning.pytorch.callbacks import EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader

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
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=parallel_num)
    nrms = NRMS(embed_size, num_heads)
    trainer = pl.Trainer(
        max_epochs=50,
        log_every_n_steps=1,
        logger=wandb_logger,
        callbacks=[
            EarlyStopping(monitor="avg_val_loss", patience=7),
        ],
    )
    trainer.fit(nrms, dataloader)

    checkpoint_path = "nrms_model.ckpt"
    trainer.save_checkpoint(checkpoint_path)


if __name__ == "__main__":
    main()
