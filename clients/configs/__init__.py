from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    aws_region_name: str = "ap-northeast-2"


settings = Settings()
