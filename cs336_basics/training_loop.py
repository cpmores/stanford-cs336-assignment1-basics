import numpy as np
import torch
import time
import os
from cs336_basics.transformer_lm.token_trainer import TokenTrainer
from cs336_basics.transformer_lm.tokenizer import BPETokenizer, get_batch
from cs336_basics.training_utils.adamw_optimizer import AdamW
from cs336_basics.training_utils.adamw_optimizer import get_lr_cosine_schedule, gradient_clipping
from cs336_basics.training_utils.cross_entropy import CrossEntropy
from cs336_basics.training_utils.checkpoint import save_checkpoint, load_checkpoint
from cs336_basics.transformer_lm.transformer_lm import TransformerLM

ifTrainBPE = False
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

print("Starting BPE training loop...")
# ── train BPE ──────────────────────────────────────────────────────────────────────
start = time.time()
bpe_input_path = "data/TinyStoriesV2-GPT4-train.txt"
vocab_size = 10000
special_tokens = ["<|endoftext|>"]
bpeTrainer = TokenTrainer(bpe_input_path, vocab_size, special_tokens)
vocab = None
merges = None
if ifTrainBPE:
    vocab, merges = bpeTrainer.train()

print(f"BPE training: {time.time() - start:.1f}s")
# ── build tokenizer ──────────────────────────────────────────────────────────────────────
print("Building tokenizer...")
vocab_path = "cs336_basics/checkpoints/bpe_vocab"
merge_path = "cs336_basics/checkpoints/bpe_merge"
if vocab is None or merges is None:
    tokenizer = BPETokenizer.from_files(vocab_path, merge_path, special_tokens)
else:
    tokenizer = BPETokenizer(vocab, merges, special_tokens)

print("Tokenizer built successfully.")
# ── Tokenize file ──────────────────────────────────────────────────────────────────────
start = time.time()
print("Starting tokenization of training and validation files...")
train_path = "data/TinyStoriesV2-GPT4-train.txt"
valid_path = "data/TinyStoriesV2-GPT4-valid.txt"
train_encoded_path = "data/TinyStoriesV2-GPT4-train-tokenized.npy"
valid_encoded_path = "data/TinyStoriesV2-GPT4-valid-tokenized.npy"
train_ids = []
valid_ids = []
if os.path.exists(train_encoded_path) and os.path.exists(valid_encoded_path):
    print("Tokenized files already exist. Skipping tokenization.")
else:   
    with open(train_path, "r", encoding="utf-8") as f:
        train_ids = list(tokenizer.encode_iterable(f))
    np.save(train_encoded_path, np.array(train_ids, dtype=np.uint16))

    with open(valid_path, "r", encoding="utf-8") as f:
        valid_ids = list(tokenizer.encode_iterable(f))
    np.save(valid_encoded_path, np.array(valid_ids, dtype=np.uint16))

print(f"Tokenization completed: {time.time() - start:.1f}s")

# ── Transformer ──────────────────────────────────────────────────────────────────────
train_ids = np.load(train_encoded_path, mmap_mode="r")
valid_ids = np.load(valid_encoded_path, mmap_mode="r")

# ── Hyperparameters ──────────────────────────────────────────────────────────────────────
context_length = 256
d_model = 512
d_ff = 1344
rope_theta = 10000
num_layers = 4
num_heads = 16

num_steps = 40000
num_valid_batches = 100

# ── Training ──────────────────────────────────────────────────────────────────────
print("Initializing Transformer model...")
model = TransformerLM(vocab_size, context_length, num_layers)
model.init_layers(d_model, num_heads, d_ff, rope_theta)
model = model.to(device)

# Optimizer and loss function
lr = 1e-3
betas = (0.9, 0.95)
weight_decay = 0.1
max_lr = 3e-4
min_lr = 0
warmup_steps = 2000
cosine_cycle_iters =num_steps 

log_interval = 100
checkpoint_interval = 1000
optimizer = AdamW(model.parameters(), lr=lr, betas = betas, weight_decay=weight_decay)
loss_fn = CrossEntropy()
iteration = load_checkpoint("cs336_basics/checkpoints/checkpoint.pt", model, optimizer)
print(f"Resuming training from iteration {iteration}...")

# Training loop
print("Starting training loop...")
model.train()
for step in range(iteration, num_steps):
    start = time.time()
    x, y = get_batch(train_ids, batch_size=32, context_length=context_length, device=device)
    x, y = x.long(), y.long()
    logits = model.forward(x)
    loss = loss_fn.forward(logits.view(-1, vocab_size), y.view(-1))
    loss.backward()

    gradient_clipping(model.parameters(), max_l2_norm=1.0)
    
    optimizer.step()
    optimizer.zero_grad()

    lr = get_lr_cosine_schedule(step, max_lr, min_lr, warmup_steps, cosine_cycle_iters)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    
    if step % log_interval == 0:
        print(f"Step {step}, Loss: {loss.item()}, LR: {lr}")
        print(f"Time for step {step}: {time.time() - start:.2f}s")
    
    if step % checkpoint_interval == 0 and step > 0: 
        save_checkpoint(model, optimizer, step, f"cs336_basics/checkpoints/checkpoint.pt")


# ── Valid ──────────────────────────────────────────────────────────────────────
print("Starting validation...")
start = time.time()
model.eval()
total_loss = 0.0
total_tokens = 0
with torch.no_grad():
    for _ in range(num_valid_batches):
        x_val, y_val = get_batch(valid_ids, batch_size = 32, context_length=context_length, device=device)
        x_val, y_val = x_val.long(), y_val.long()
        val_logits = model.forward(x_val)
        val_loss = loss_fn.forward(val_logits.view(-1, vocab_size), y_val.view(-1))
        total_loss += val_loss.item() * (x_val.shape[0] * x_val.shape[1])
        total_tokens += x_val.shape[0] * x_val.shape[1]

avg_val_loss = total_loss / total_tokens
val_perplexity = np.exp(avg_val_loss)
print(f"Validation Loss: {avg_val_loss}, Validation Perplexity: {val_perplexity:.2f}")
print(f"Validation completed in {time.time() - start:.1f}s")