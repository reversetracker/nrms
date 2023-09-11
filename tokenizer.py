import random


def tokenize(text: str | None, sequence_size: int = 20) -> list[list[float]]:
    """한 기사의 타이틀의 기사 제목을 최대 20 개로 맞추며 부족 한 경우 패딩을 넣어 20 개를 채움."""

    if not text or isinstance(text, float):
        return [[0 for _ in range(128)] for _ in range(sequence_size)]

    tokens = [token for token in text.split(" ") if token]
    tokens = [[(random.randint(1, 100) / 10000) for _ in range(128)] for _ in tokens]
    tokens = tokens[:sequence_size]
    paddings = [[0 for _ in range(128)] for _ in range(sequence_size - len(tokens))]
    return tokens + paddings


if __name__ == "__main__":
    tokenized = tokenize("Hello, world! This is a test sentence.")
    print(tokenized)
