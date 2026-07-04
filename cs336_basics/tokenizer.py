from __future__ import annotations
import pickle
import regex as re
from typing import Iterable, Iterator


class BPETokenizer:
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    def __init__(self, 
                 vocab: dict[int, bytes], 
                 merges: list[tuple[bytes, bytes]], 
                 special_tokens=None):
        self.vocab = vocab
        self.reverseVocab = {}
        self.merges = merges
        self.merge_rank = {pair: i for i, pair in enumerate(self.merges)}
        self.special_tokens = special_tokens
        for key, value in self.vocab.items():
            self.reverseVocab[value] = key

    @staticmethod
    def from_files(vocab_filepath, merges_filepath, special_tokens=None) -> BPETokenizer:
        with open(vocab_filepath, "rb") as f:
            vocab = pickle.load(f)

        with open(merges_filepath, "rb") as f:
            merges = pickle.load(f)

        return BPETokenizer(vocab, merges, special_tokens)

    def encode(self, text: str) -> list[int]:
        chunks = [text]
        encode_int = []
        if self.special_tokens is not None:
            sorted_tokens = sorted(self.special_tokens, key=len, reverse=True)
            pattern = f"({'|'.join(re.escape(t) for t in sorted_tokens)})"
            chunks = re.split(pattern, text)
        for chunk in chunks:
            if self.special_tokens is not None and chunk in self.special_tokens:
                encode_int.append(self._get_key_from_value(chunk.encode("utf-8")))
                continue
            uniChunks = chunk
            pretokenList = re.findall(self.PAT, uniChunks)
            for pretoken in pretokenList:
                keys = list(bytes([b]) for b in pretoken.encode("utf-8"))
                # for merge in self.merges:
                #     keys = self._merge(merge, keys)
                while True:
                    minRank = len(self.merges)
                    truePair = tuple()
                    for first, second in zip(keys[:-1], keys[1:]):
                        if self._get_merge_rank((first, second)) == -1:
                            continue
                        if self._get_merge_rank((first, second)) < minRank:
                            minRank = self._get_merge_rank((first, second))
                            truePair = (first, second)
                    if minRank != len(self.merges):
                        keys = self._merge(truePair, keys)
                    else:
                        break

                for key in keys:
                    encode_int.append(self._get_key_from_value(key)) 

        return encode_int
    
    def _get_merge_rank(self, pair: tuple[bytes, bytes]) -> int:
        if self.merge_rank.get(pair) is None:
            return -1
        return self.merge_rank[pair]
    
    def _merge(self, pair, key) -> list[bytes]:
        l = list(key)
        index = 0
        while index < len(l) - 1:
            first = l[index]
            second = l[index + 1]
            if (first, second) == pair:
                l[index] = first + second
                del(l[index + 1])
            
            index = index + 1
        return l 

    def _get_key_from_value(self, expectedValue) -> int:
        if self.reverseVocab.get(expectedValue) is not None:
            return self.reverseVocab[expectedValue]
        return 0


    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for chunk in iterable:
            for token_id in self.encode(chunk):
                yield token_id

    def decode(self, ids: list[int]) -> str:
        byte_parts = []
        for id in ids: 
            if self.vocab.get(id) is None:
                byte_parts.append(chr(65533).encode("utf-8"))
                continue
            else:
                byte_parts.append(self.vocab[id])
            
        return b"".join(byte_parts).decode("utf-8", errors="replace")