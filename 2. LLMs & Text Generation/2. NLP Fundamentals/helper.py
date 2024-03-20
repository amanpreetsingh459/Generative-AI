from __future__ import annotations
from collections.abc import Sequence
from typing import Callable

import time
import math
import torch
import torch.nn as nn
from torch.utils.data import (
    Dataset,
)
from collections import Counter


class TokenMapping():
    def __init__(
        self,
        text_as_list: list[str],
        not_found_token: str = 'TOKEN_NOT_FOUND',
        not_found_id: int | None = None,
    ):
        self.counter = Counter(text_as_list)
        # Includes `not found token`
        self.n_tokens: int = len(self.counter) + 1
        # Token to ID mapping
        self._token2id: dict[str, int] = {
            token: idx
            for idx, (token, _) in enumerate(self.counter.items())
        }
        # Reverse mapping: ID to token mapping
        self._id2token: dict[int, str] = {
            idx: token
            for token, idx in self._token2id.items()
        }
        # Token representing not found 
        self._not_found_token = not_found_token
        # Not found ID defaults to next available number
        if not_found_id is None:
            self._not_found_id = max(self._token2id.values()) + 1
        else:
            self._not_found_id = not_found_id
    
    def encode(self, text_list: list[str]) -> list[int]:
        '''Encodes list of tokens (strings) into list of IDs (integers)'''
        encoded = [
            self.token2id(token)
            for token in text_list
        ]
        # Include the not found ID if it wasn't included yet
        if self._not_found_id not in encoded:
            encoded += [self._not_found_id]
        return encoded

    def token2id(self, token: str):
        '''Returns ID for given token (even if token not found)'''
        return self._token2id.get(token, self._not_found_id)
    
    def id2token(self, idx: int):
        '''Returns token for given ID or the not found token'''
        return self._id2token.get(idx, self._not_found_token)


class ShakespeareDataset(Dataset):
    def __init__(self, encoded_text: Sequence, sequence_length: int):
        self.encoded_text = encoded_text
        self.sequence_length = sequence_length

    def __len__(self):
        return len(self.encoded_text) - self.sequence_length

    def __getitem__(self, index):
        x = torch.tensor(
            self.encoded_text[index: (index+self.sequence_length)],
            dtype=torch.long,
        )
        # Target is shifted by one character/token
        y = torch.tensor(
            self.encoded_text[(index+1): (index+self.sequence_length+1)],
            dtype=torch.long,
        )
        return x, y


class ShakespeareModel(nn.Module):
    def __init__(self, n_tokens: int, embedding_dim: int, hidden_dim: int):
        super(ShakespeareModel, self).__init__()
        self.embedding = nn.Embedding(n_tokens, embedding_dim)
        self.rnn = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, n_tokens)
        
    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.rnn(x)
        x = self.fc(x)
        return x


def start_time() -> float:
    '''Returns a start time from right now'''
    return time.time()


def time_since(start: float) -> str:
    '''Return string of time since given time (float) in "{m:02}m {s:02.1f}s"'''
    s = time.time() - start
    m = math.floor(s / 60)
    s -= m * 60
    return f'{m:02}m {s:02.1f}s'


def build_model(
    n_tokens: int,
    embedding_dim: int = 16,
    hidden_dim: int = 32,
):
    '''Create basic RNN-based model (embeddings -> RNN -> Linear/Dense layer)'''
    return ShakespeareModel(n_tokens, embedding_dim, hidden_dim)


def tokens_to_id_tensor(
    tokens: list[str],
    token_id_mapping: dict[str, int],
) -> torch.Tensor:
    '''Create PyTorch Tensor from token to ID based on given mapping'''
    id_tensor = (
        torch.tensor(
            [token_id_mapping(token) for token in tokens],
            dtype=torch.long,
        )
        .unsqueeze(0)
    )
    return id_tensor


def tokenize_text_from_tokenizer(
    tokenizer,
    text: str,
) -> list[str]:
    '''Return list of token (strings) of text using given tokenizer'''
    max_seq_length = tokenizer.model_max_length
    # Chunk the string so tokenizer can take in full input
    chunks_generator = (
        text[i:i+max_seq_length]
        for i in range(0, len(text), max_seq_length)
    )
    # Special tokens to ignore
    ignore_tokens = (
        tokenizer.cls_token,
        tokenizer.sep_token,
    )
    # Get list of tokens (one chunk at a time)
    tokenized_text = [
        token
        for chunk in chunks_generator
        for token in tokenizer(chunk).tokens()
        if (
            token not in ignore_tokens
        )
    ]
    return tokenized_text


def encode_text(
    text: str,
    tokenize_func: Callable[[str], list[str]],
) -> tuple[list[int], TokenMapping]:
    '''Return encoded (IDs) list of text & TokenMapping via tokenize function'''
    tokenized_text: list[str] = tokenize_func(text)
    # Get object for translating tokens to IDs and back
    token_mapping = TokenMapping(tokenized_text)
    # Get your text encoded as list of IDs
    enocded_text: list[str] = token_mapping.encode(tokenized_text)
    return enocded_text, token_mapping


def encode_text_from_tokenizer(
    text: str,
    tokenizer,
) -> tuple[list[int], TokenMapping]:
    '''Return encoded (IDs) list of text & TokenMapping via given tokenizer'''
    # Using tokenizer, get text as list of tokens (strings)
    tokenized_text: list[str] = tokenize_text_from_tokenizer(
        tokenizer=tokenizer,
        text=text,
    )
    # Get object for translating tokens to IDs and back
    token_mapping = TokenMapping(tokenized_text)
    # Get your text encoded as list of IDs
    enocded_text: list[str] = token_mapping.encode(tokenized_text)
    return enocded_text, token_mapping


def next_token(
    tokenized_text: list[str],
    model,
    token_mapping: TokenMapping,
    temperature: float = 1.0,
    topk: int | None = None,
    device: str = 'cpu',
) -> str:
    '''Provide next token based on temperature and top-k (if given)'''
    # Set model into "evaluation mode" (deactivates things like Dropout layers)
    model.eval()
    input_tensor = tokens_to_id_tensor(
        tokens=tokenized_text,
        token_id_mapping=token_mapping.token2id,
    )

    with torch.no_grad():
        output = model(input_tensor.to(device))
        # Use temperature to change probabilities
        probabilities = nn.functional.softmax(
            output[0, -1] / temperature,
            dim=0,
        )
        # Sampling from probabilities
        sorted_ids = torch.argsort(probabilities, descending=True)
        # Top-k: Defaults to using all given characters
        if topk is None:
            sorted_ids_subset = sorted_ids
        else:
            sorted_ids_subset = sorted_ids[:topk]
        index_of_sorted = torch.multinomial(
            probabilities[sorted_ids_subset],
            1,
        ).item()
        next_char_idx = sorted_ids_subset[index_of_sorted].item()
        
        return token_mapping.id2token(next_char_idx)