import torch
import torch.nn as nn
from torch.nn import functional as F

#hyperparameters
batch_size = 64
block_size = 256
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
eval_iters = 200
n_embd = 384
n_layer = 6
n_head = 6
dropout = 0.2

torch.manual_seed(1337)


#wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

print("length of dataset in characters: ", len(text))

print(text[:1000])

chars = sorted(list(set(text)))
vocab_size = len(chars)
print(''.join(chars))
print(vocab_size)

#tokenizing all the words
stoi ={ ch:i for i,ch in enumerate(chars)} #iterate thru chars and map each char to its index in chars
itos = {i:ch for i,ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])


data = torch.tensor(encode(text), dtype = torch.long)
n = int(0.9*len(data)) # training data is 90% of total
train_data = data[:n]
val_data = data[n:]


def get_batch(split):
  data = train_data if split == 'train' else val_data
  ix = torch.randint(len(data) - block_size, (batch_size,)) #batch_size number of random offsets selcted between 0 and data length - block_size
  x = torch.stack([data[i:i+block_size] for i in ix]) #for each offset in ix (size 4 rn), 1d tensor is made and stacked for the tokens for chars at index i to i+block_size
  y = torch.stack([data[i+1:i+block_size + 1] for i in ix]) #same as above but offset forward by 1 to be the targets
  x, y = x.to(device), y.to(device)
  return x, y

#estimate_loss
@torch.no_grad()
def estimate_loss():
  out = {}
  m.eval() #set model to evaluation mode
  for split in ['train', 'val']:
    losses = torch.zeros(eval_iters)
    for k in range(eval_iters):
      X, Y = get_batch(split)
      logits, loss = m(X,Y)
      losses[k] = loss.item()
    out[split] = losses.mean()
  m.train() #set model back to training mode
  return out

class Head(nn.Module):
    """ one head of self-attention """
    
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        
        self.dropout_layer = nn.Dropout(dropout)
    
    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)   # (B,T,C)
        q = self.query(x) # (B,T,C)
        # compute attention scores ("affinities")
        # scaled by 1/sqrt(head_size) so the dot products don't grow too large
        wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5 # (B,T,hs) @ (B,hs,T) ---> (B,T,T)
        wei = wei.masked_fill(self.tril[:T,:T] == 0, float('-inf')) # (B,T,T)
        wei = F.softmax(wei, dim=-1) # (B,T,T)
        wei = self.dropout_layer(wei) # (B,T,T)
        # perform the weighted aggregation of the values
        v = self.value(x) # (B,T,C)
        out = wei @ v # (B,T,T) @ (B,T,C) ---> (B,T,C)
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(num_heads * head_size, n_embd)
        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1) #concatenating over channel dimension
        return self.dropout_layer(self.proj(out))

class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4*n_embd),
            nn.ReLU(),
            nn.Linear(4*n_embd, n_embd),
            nn.Dropout(dropout)
        )
        
        
    def forward(self, x):
        return self.net(x)
    
    
class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd //n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        
    def forward(self, x):
        x = x+ self.sa(self.ln1(x)) #residual connections by doing x+
        x = x+self.ffwd(self.ln2(x))
        return x


class GPTLanguageModel(nn.Module):
  def __init__(self):
    super().__init__()
    self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
    self.position_embedding_table = nn.Embedding(block_size, n_embd) 
    self.blocks = nn.Sequential(*[Block(n_embd, n_head = n_head) for _ in range(n_layer)])
    self.ln_f = nn.LayerNorm(n_embd) #final layer norm
    self.lm_head = nn.Linear(n_embd, vocab_size)

  def forward(self, idx, targets=None):
    B, T = idx.shape
    
    tok_emb = self.token_embedding_table(idx) # Batch x Time x Channel (channel is the emb size so 32)
    pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # T x C
    x = tok_emb + pos_emb # B x T x C
    x = self.blocks(x)
    x = self.ln_f(x)
    logits = self.lm_head(x) # Batch x Time x Channel (channel is the vocab size so 65 chars)
    
    
    if targets is None:
      loss = None
    else:
      B, T, C = logits.shape
      logits = logits.view(B*T, C)
      targets = targets.view(B*T)
      loss = F.cross_entropy(logits,targets) #sees how well we are predicting the next character
    return logits, loss

  def generate(self,idx, max_new_tokens):
    #idx = b x t array of indices in current context
    for _ in range(max_new_tokens):
      #get prediction
      logits, loss = self(idx[:, -block_size:]) #only use last block_size tokens as context
      #focus only on last one
      logits = logits[:,-1, :] # becomes b x C
      #get probabilities
      probs = F.softmax(logits, dim = -1)
      #sample from probabilities
      idx_next = torch.multinomial(probs, num_samples = 1)
      #append sampled index to the running sequence
      idx = torch.cat((idx, idx_next), dim = 1)
    return idx

model = GPTLanguageModel()
print(sum(p.numel() for p in model.parameters())/1e6, 'M parameters')
m = model.to(device)


optimizer = torch.optim.AdamW(m.parameters(), lr = learning_rate)


for iter in range(max_iters):
  
  if iter % eval_interval == 0:
    losses = estimate_loss()
    print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
  
  xb,yb = get_batch('train')

  logits, loss = m(xb, yb)
  optimizer.zero_grad(set_to_none=True)
  loss.backward()
  optimizer.step()


context = torch.zeros((1,1), dtype = torch.long, device = device) #initial token
print(decode(m.generate(context,max_new_tokens = 500)[0].tolist()))


