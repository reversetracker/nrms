from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    batch_size: int = 64

    article_size: int = 32

    sequence_size: int = 20

    embed_dim: int = 768

    encoder_dim: int = 128

    num_heads_news_encoder: int = 16

    num_heads_user_encoder: int = 8

    lr: float = 2e-4

    weight_decay: float = 1e-5

    dropout: float = 0.2

    K: int = 4

    wandb_api_key: str = ""


settings = Settings()
