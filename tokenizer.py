import random

from transformers import ElectraModel, ElectraTokenizer


TOKENIZER = ElectraTokenizer.from_pretrained("monologg/koelectra-base-v3-discriminator")

MODEL = ElectraModel.from_pretrained("monologg/koelectra-base-v3-discriminator")


def tokenize(text: str | None, sequence_size: int = 20) -> list[list[float]]:
    """한 기사의 타이틀의 기사 제목을 최대 20 개로 맞추며 부족 한 경우 패딩을 넣어 20 개를 채움."""

    if not text or isinstance(text, float):
        return [[0.001 for _ in range(128)] for _ in range(sequence_size)]

    tokens = [token for token in text.split(" ") if token]
    tokens = [[(random.randint(1, 100) / 100) for _ in range(128)] for _ in tokens]
    tokens = tokens[:sequence_size]
    paddings = [[0.001 for _ in range(128)] for _ in range(sequence_size - len(tokens))]
    return tokens + paddings


def texts_to_embeddings(texts: list[str], sequence_size: int = 20) -> list[list[float]]:
    """Text 를 입력 받아서 임베딩 벡터로 변환."""
    tokens = TOKENIZER(
        texts, return_tensors="pt", truncation=True, max_length=sequence_size, padding="max_length"
    )
    outputs = MODEL(**tokens)
    embeddings = outputs.last_hidden_state
    return embeddings


if __name__ == "__main__":
    embeddings = texts_to_embeddings(["안녕하세요", "반갑습니다."])
    print(embeddings.shape)