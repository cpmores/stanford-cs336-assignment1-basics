# CS336 Spring 2025 Assignment 1: Basics

For a full description of the assignment, see the assignment handout at
[cs336_assignment1_basics.pdf](./cs336_assignment1_basics.pdf)

If you see any issues with the assignment handout or code, please feel free to
raise a GitHub issue or open a pull request with a fix.

## Setup

### Environment
We manage our environments with `uv` to ensure reproducibility, portability, and ease of use.
Install `uv` [here](https://github.com/astral-sh/uv#installation) (recommended), or run `pip install uv`/`brew install uv`.
We recommend reading a bit about managing projects in `uv` [here](https://docs.astral.sh/uv/guides/projects/#managing-dependencies) (you will not regret it!).

You can now run any code in the repo using
```sh
uv run <python_file_path>
```
and the environment will be automatically solved and activated when necessary.

### Run unit tests


```sh
uv run pytest
```

Initially, all tests should fail with `NotImplementedError`s.
To connect your implementation to the tests, complete the
functions in [./tests/adapters.py](./tests/adapters.py).

### Download data
Download the TinyStories data and a subsample of OpenWebText

``` sh
mkdir -p data
cd data

wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt

wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_train.txt.gz
gunzip owt_train.txt.gz
wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_valid.txt.gz
gunzip owt_valid.txt.gz

cd ..
```

## Training

Run the full training pipeline (BPE tokenization + Transformer LM training):

```sh
uv run python cs336_basics/training_loop.py
```

The script handles:
1. BPE tokenizer training on TinyStories (vocab_size=10,000)
2. Tokenization of train/validation sets
3. Transformer LM training with the following hyperparameters:

| Parameter | Value |
|---|---|
| d_model | 512 |
| num_layers | 4 |
| num_heads | 16 |
| d_ff | 1,344 |
| context_length | 256 |
| vocab_size | 10,000 |
| Optimizer | AdamW (β₁=0.9, β₂=0.95, weight_decay=0.1) |
| LR schedule | Cosine with linear warmup (max_lr=3e-4, warmup=2000) |
| Training steps | 40,000 |
| Batch size | 32 |

Requires ~200MB disk for tokenized data, ~8GB GPU VRAM.  Training takes ~30-40 min on an RTX 4060.

## Results

After 40,000 steps (~327M tokens, 1 epoch on TinyStories):

- **Validation perplexity**: ~15-20

