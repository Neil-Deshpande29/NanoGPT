# NanoGPT: A GPT-2-Style Transformer Built from Scratch

A from-scratch implementation of a decoder-only Transformer language model, following the architecture of OpenAI's GPT-2, written in PyTorch and trained on the complete works of Shakespeare (Tiny Shakespeare). The model learns to generate new text in Shakespeare's style, one character at a time.

Every component of the model is implemented by hand using basic PyTorch building blocks (`nn.Linear`, `nn.Embedding`, `nn.LayerNorm`): character-level tokenization, batching, masked self-attention, multi-head attention, the feed-forward network, residual connections, layer normalization, training, and sampling. No pretrained weights or high-level Transformer libraries are used.

## Overview

The repository contains two models. Each one builds on the previous:

| File | Model | Description |
|------|-------|-------------|
| [`bigram.py`](bigram.py) | Bigram language model | A baseline that predicts the next character from the current character alone, using a single embedding lookup table. |
| [`v2.py`](v2.py) | GPT language model | A GPT-2-style Transformer with ~10.8M parameters that attends over a 256-character context window. |

## Architecture

`v2.py` implements a GPT-2-style decoder-only Transformer:

- **Token and positional embeddings.** Learned embeddings for each of the 65 characters in the vocabulary and for each position in the context window, added together.
- **Masked (causal) self-attention.** Scaled dot-product attention, `softmax(QKᵀ / √d_k) V`, with a lower-triangular mask so that each position attends only to earlier positions.
- **Multi-head attention.** 6 parallel attention heads, concatenated and projected back to the embedding dimension.
- **Position-wise feed-forward network.** A two-layer MLP with a 4× hidden expansion and ReLU activation.
- **Pre-LayerNorm residual blocks.** As in GPT-2, layer normalization is applied *before* each sub-layer, and each sub-layer is wrapped in a residual connection: `x = x + SelfAttention(LN(x))`, then `x = x + FFN(LN(x))`.
- **Final LayerNorm and linear language-modeling head** that project to logits over the vocabulary.
- **Dropout** on attention weights, attention output, and feed-forward output for regularization.

### Hyperparameters

| Parameter | Value |
|-----------|-------|
| Embedding dimension (`n_embd`) | 384 |
| Transformer layers (`n_layer`) | 6 |
| Attention heads (`n_head`) | 6 (head size 64) |
| Context length (`block_size`) | 256 characters |
| Batch size | 64 |
| Dropout | 0.2 |
| Optimizer | AdamW, learning rate 3e-4 |
| Training iterations | 5,000 |
| Parameters | ~10.8M |

## Results

After 5,000 training iterations, the GPT model reaches the following loss (cross-entropy, final evaluation at step 4,500):

| Split | Loss |
|-------|------|
| Train | 1.1125 |
| Validation | 1.4869 |

For comparison, a uniform guess over the 65-character vocabulary gives a loss of ln(65) ≈ 4.17. The model produces text with Shakespearean structure: speaker names, line breaks, archaic diction, and mostly well-formed English words.

## Dataset

[Tiny Shakespeare](https://github.com/karpathy/char-rnn/tree/master/data/tinyshakespeare) (`input.txt`) is roughly 1.1 million characters taken from Shakespeare's plays. The text is tokenized at the character level, giving a vocabulary of 65 characters, and split 90/10 into training and validation sets.

## Getting Started

### Requirements

- Python 3.9+
- PyTorch 2.0+

```bash
pip install -r requirements.txt
```

### Training and Generation

```bash
# Baseline bigram model (trains in seconds)
python bigram.py

# GPT model
python v2.py
```

Each script trains the model, reports training and validation loss at regular intervals, and then samples 500 new characters from the trained model.

The scripts select the best available device automatically: CUDA, then Apple Silicon (MPS), then CPU. The full GPT model is meant to run on a GPU. On CPU, reduce `n_embd`, `n_layer`, `block_size`, and `max_iters` at the top of `v2.py` for a faster run.

## Project Structure

```
.
├── bigram.py          # Baseline bigram language model
├── v2.py              # GPT-2-style Transformer language model
├── input.txt          # Tiny Shakespeare dataset
├── requirements.txt
└── README.md
```

## Acknowledgements

This project was built while working through Andrej Karpathy's lecture [*Let's build GPT: from scratch, in code, spelled out*](https://www.youtube.com/watch?v=kCc8FmEb1nY). The architecture follows *Attention Is All You Need* (Vaswani et al., 2017) and OpenAI's GPT-2 (Radford et al., 2019).

## References

- Vaswani, A., et al. (2017). [Attention Is All You Need](https://arxiv.org/abs/1706.03762).
- Radford, A., et al. (2019). [Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf).
- He, K., et al. (2015). [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385).
- Ba, J. L., Kiros, J. R., & Hinton, G. E. (2016). [Layer Normalization](https://arxiv.org/abs/1607.06450).
