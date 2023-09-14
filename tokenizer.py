import pandas as pd
import torch
from transformers import ElectraModel, ElectraTokenizer

TOKENIZER = ElectraTokenizer.from_pretrained("monologg/koelectra-base-v3-discriminator")

MODEL = ElectraModel.from_pretrained("monologg/koelectra-base-v3-discriminator")


def tokenize(self, texts: list[str]) -> dict:
    tokens = TOKENIZER(
        texts,
        return_tensors="pt",  # return pytorch tensor
        truncation=True,
        max_length=self.max_length,
        padding="max_length",
    )
    return tokens


def texts_to_embeddings(
    texts: list[str], max_length: int = 20
) -> tuple[torch.Tensor, torch.Tensor]:
    """Text 를 입력 받아서 임베딩 벡터로 변환."""
    tokens = TOKENIZER(
        texts,
        return_tensors="pt",  # return pytorch tensor
        truncation=True,
        max_length=max_length,
        padding="max_length",
    )
    with torch.no_grad():
        outputs = MODEL(**tokens)
    embeddings = outputs.last_hidden_state
    masks = tokens["attention_mask"]
    return embeddings, masks


def precompute_embeddings(
    df: pd.DataFrame,
    target_column: str,
    destination_column: str,
    mask_column: str = "mask",
) -> pd.DataFrame:
    """
    Precomputes embeddings for the target column and saves them in the embedding_column.

    Parameters:
    - df (pd.DataFrame): The dataframe containing the data.
    - target_column (str): The column of the dataframe for which embeddings are to be computed.
    - destination_column (str): The column where computed embeddings will be saved.

    Returns:
    - pd.DataFrame: DataFrame with the computed embeddings.
    """

    all_titles = df[target_column].unique().tolist()
    embeddings, masks = texts_to_embeddings(all_titles)
    embeddings, masks = embeddings.numpy(), masks.numpy()

    embedding_dict = {title: emb for title, emb in zip(all_titles, embeddings)}
    mask_dict = {title: mask for title, mask in zip(all_titles, masks)}

    df[destination_column] = df[target_column].map(embedding_dict)
    df[mask_column] = df[target_column].map(mask_dict)

    return df


if __name__ == "__main__":
    texts = ["이것은 테스트 문장 입니다.", "이것은 십새키 입니다."]
    embeddings, masks = texts_to_embeddings()
    print(embeddings.shape, masks.shape)
    # torch.Size([2, 20, 768]) torch.Size([2, 20])
