import cachetools
import torch
from transformers import ElectraModel, ElectraTokenizer

TOKENIZER = ElectraTokenizer.from_pretrained("monologg/koelectra-base-v3-discriminator")

MODEL = ElectraModel.from_pretrained("monologg/koelectra-base-v3-discriminator")

embeddings_cache = cachetools.LRUCache(maxsize=50000)

masks_cache = cachetools.LRUCache(maxsize=50000)


def complement(keys: list[str], cache: cachetools.LRUCache) -> list[str]:
    return [key for key in keys if key not in cache]


def tokenize(texts: list[str], max_length: int = 20) -> dict:
    return TOKENIZER(
        texts, return_tensors="pt", truncation=True, max_length=max_length, padding="max_length"
    )


def texts_to_embeddings(
    texts: list[str], max_length: int = 20
) -> tuple[torch.Tensor, torch.Tensor]:
    uncached_texts = complement(texts, embeddings_cache)

    if uncached_texts:
        tokens = tokenize(uncached_texts, max_length)
        with torch.no_grad():
            outputs = MODEL(**tokens)
        new_embeddings = outputs.last_hidden_state
        new_masks = tokens["attention_mask"]

        for text, emb, mask in zip(uncached_texts, new_embeddings, new_masks):
            embeddings_cache[text] = emb
            masks_cache[text] = mask
        return texts_to_embeddings(texts, max_length)

    text_embeddings = torch.stack([embeddings_cache[text] for text in texts])
    mask_embeddings = torch.stack([masks_cache[text] for text in texts])
    mask_embeddings = mask_embeddings.bool()
    mask_embeddings = ~mask_embeddings

    return text_embeddings, mask_embeddings


if __name__ == "__main__":
    texts = ["이것은 테스트 문장 입니다.", "이것은 십새키 입니다."]
    embeddings, masks = texts_to_embeddings(texts)
    print(embeddings.shape, masks.shape)
