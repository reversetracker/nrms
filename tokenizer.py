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
        padding="max_length"
    )
    return tokens


def texts_to_embeddings(texts: list[str], max_length: int = 20) -> tuple[torch.Tensor, torch.Tensor]:
    """Text 를 입력 받아서 임베딩 벡터로 변환."""
    tokens = TOKENIZER(
        texts,
        return_tensors="pt",  # return pytorch tensor
        truncation=True,
        max_length=max_length,
        padding="max_length",
    )
    outputs = MODEL(**tokens)
    embeddings = outputs.last_hidden_state
    masks = tokens['attention_mask']
    return embeddings, masks


if __name__ == "__main__":
    embeddings, masks = texts_to_embeddings(
        [
            "이것은 테스트 문장 입니다.",
            "이것은 십새키 입니다.",
        ]
    )
    print(embeddings.shape, masks.shape)
    # torch.Size([2, 20, 768]) torch.Size([2, 20])
