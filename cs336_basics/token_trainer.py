# token_trainer.py implements a TokenTrainer that trains a BPE tokenizer
import os
import regex as re
from collections import Counter
from pathlib import Path
from typing import BinaryIO
import multiprocessing as mp

class TokenTrainer: 
    # mini_chunk_size used under _find_chunk_boundaries, the real chunk reading size
    mini_chunk_size = 4096
    num_processes = 4
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    # init under test/adapters.py: run_train_bpe to build a trainer
    def __init__(self, input_path: str | os.PathLike, vocab_size: int, special_tokens: list[str]):
        self.input_path = Path(input_path)
        self.vocab_size = vocab_size
        self.special_tokens = special_tokens
        self.special_tokens_bytes = [t.encode("utf-8") for t in special_tokens]
        self.vocab = {}
        self.merges = []

    def train(self) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
        preToken = self._pretokenize_parallelized()
        self._merge(preToken)
        self._vocab()
        return self.vocab, self.merges

    def _pretokenize_parallelized(self) -> Counter:
        try:
            with open(self.input_path, "rb") as f:
                boundaries = self._find_chunk_boundaries(f, self.num_processes, self.special_tokens_bytes[0])
                chunks = [(start, end) for start, end in zip(boundaries[:-1], boundaries[1:])]
                with mp.Pool(processes=self.num_processes) as pool:
                    results = pool.starmap(self._pretokenize, chunks)
                
                total_counts = Counter()
                for result in results:
                    total_counts.update(result)
                
                return total_counts
                   
                    
        except FileNotFoundError:
            print(f"file in {self.input_path} not found")
            return Counter()

    def _merge(self, preTokenCount: Counter) -> Counter:
        neighCounts = self._check_neigh_around(preTokenCount)
        while True:
            # 2. check max in neighCounts
            maxKey = max(neighCounts, key=lambda k: (neighCounts[k], k))

            # 3. merge neighbour
            preTokenCount = self._merge_neigh_around(preTokenCount, maxKey, neighCounts)
            self.merges.append(maxKey)

            if len(self.merges) >= (self.vocab_size - 256 - len(self.special_tokens)):
                return neighCounts

    def _vocab(self):
        for index in range(256):
            self.vocab[index] = bytes([index])
        
        self.vocab[256] = self.special_tokens_bytes[0]
        index = 257
        for mergedPair in self.merges:
            self.vocab[index] = mergedPair[0] + mergedPair[1]
            index = index + 1

    def _merge_neigh_around(self, preTokenCount: Counter, maxKey: tuple[bytes, ...], neighCounts):
        newPreTokenCount = Counter()
        for key, value in preTokenCount.items():
            newKeyList = list(key)
            index = 0
            while index < len(newKeyList) - 1:
                first = newKeyList[index]
                second = newKeyList[index + 1]
                if maxKey == (first, second):
                    neighCounts[(first, second)] = neighCounts[(first, second)] - value
                    newKeyList[index] = first + second
                    if index > 0 :
                        leftFirst = newKeyList[index - 1]
                        neighCounts[(leftFirst, first)] = neighCounts[(leftFirst, first)] - value
                        if neighCounts.get((leftFirst, first + second)) is None:
                            neighCounts[(leftFirst, first + second)] = value
                        else:
                            neighCounts[(leftFirst, first + second)] = neighCounts[(leftFirst, first + second)] + value 
                    
                    if index < len(newKeyList) - 2:
                        rightSecond = newKeyList[index + 2]
                        neighCounts[(second, rightSecond)] = neighCounts[(second, rightSecond)] - value
                        if neighCounts.get((first + second, rightSecond)) is None:
                            neighCounts[(first + second, rightSecond)] = value
                        else:
                            neighCounts[(first + second, rightSecond)] = neighCounts[(first + second, rightSecond)] + value 
                    
                    del(newKeyList[index + 1])
                
                index = index + 1
            newPreTokenCount[tuple(newKeyList)] = value
        return newPreTokenCount


    def _check_neigh_around(self, preTokenCount: Counter):
        neighCounts = Counter()
        for key, value in preTokenCount.items():
            for first, second in zip(key[:-1], key[1:]):
                pair = (first, second)
                if neighCounts.get(pair) is None:
                    neighCounts[pair] = 1 * value
                else:
                    neighCounts[pair] = neighCounts[pair] + value
        
        return neighCounts

    # pretokenize: parallized
    def _pretokenize(self, start: int, end: int) -> dict[tuple[bytes, ...], int]:
        with open(self.input_path, "rb") as file:
            file.seek(start)
            chunk = file.read(end - start).decode("utf-8", errors="ignore")
            removeChunks = re.split("|".join(re.escape(token) for token in self.special_tokens), chunk)
            predict = Counter() 
            for uniChunks in removeChunks:
                pretokenList = re.findall(self.PAT, uniChunks)  
                for pretoken in pretokenList:
                    # strip first blankspace
                    # strippedToken = pretoken.lstrip()
                    strippedToken = pretoken
                    if not strippedToken:
                        continue

                    key = tuple(bytes([b]) for b in strippedToken.encode("utf-8"))
                    if predict.get(key) is None:
                        predict[key] = 1
                    else:
                        predict[key] = predict[key] + 1

        return predict

    # _find_chunk_boundaries derived from `pretokenization_example.py`
    def _find_chunk_boundaries(self, 
        file: BinaryIO,
        desired_num_chunks: int,
        split_special_token: bytes,
    ) -> list[int]:
        # 1. get basic information first
        assert isinstance(split_special_token, bytes), "special token as a bytestring"

        # 1.1 get file size: traditional C like function
        file.seek(0, os.SEEK_END)
        file_size = file.tell()
        file.seek(0)

        # 1.2 get chunk_size and guess the possible chunk init position 
        chunk_size = file_size // desired_num_chunks
        chunk_start = [index * chunk_size for index in range(0, desired_num_chunks + 1)]
        chunk_start[-1] = file_size

        # 2. update chunk start to avoid mid-docs
        for bi in range(1, len(chunk_start) - 1):
            initial_pos = chunk_start[bi]
            file.seek(initial_pos)
            while True:
                mini_chunk = file.read(self.mini_chunk_size)

                if mini_chunk == b"":
                    chunk_start[bi] = file_size
                    break

                # 2.1 find split special token in mini_chunk  
                found_at = mini_chunk.find(split_special_token)
                if found_at != -1:
                    # 2.2.1 found: update chunk_start
                    chunk_start[bi] = initial_pos + found_at
                    break

                # 2.2.2 not found: pass to next mini_chunk
                initial_pos = initial_pos + self.mini_chunk_size

        return sorted(set(chunk_start))

if __name__ == "__main__":
    trainTokenizer = TokenTrainer("../data/TinyStoriesV2-GPT4-valid.txt", 20000, ["<|endoftext|>"])
    trainTokenizer.train()
    