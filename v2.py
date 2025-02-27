# using Bigram language model
import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters
batch_size = 64  # count of block processed parallely
block_size = 256  # size of each block
max_iters = 5000
eval_interval = 50
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2
# ---------------

torch.manual_seed(1337)

# !wget https://raw.githubusercontent.com/Samaksh22/nanoGPT/refs/heads/main/input.txt
with open("input.txt", 'r') as f:
    text = f.read()

# getting all the different types of characters present in the dataset
chars = sorted(list(set(text)))
vocab_size = len(chars)

# encoder and decoder
stoi = { ch:i for i, ch in enumerate(chars)}
itos = { i:ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join(itos[i] for i in l)

# encode entire dataset and store into a torch.Tensor
data = torch.tensor(encode(text), dtype=torch.long)

# train and validation split of 90%
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]




# getting a randomized batch each iteration
def get_batch(split):
    # selecting training or validation data
    data = train_data if split == 'train' else val_data

    indi = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in indi])
    y = torch.stack([data[i+1:i+block_size+1] for i in indi])
    x, y = x.to(device), y.to(device)
    return x, y



@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        
        # compute attention scores ("affenities")
        wei = q @ k.transpose(-2, -1) * C**-0.5
        wei = wei.masked_fill(self.tril[:T, :T]==0, float('-inf'))
        wei = F.softmax(wei, dim=1)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x)
        out = wei @ v
        return out

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        return out
    
    
class FeedForward(nn.Module):
    """ simple linear layer followed by non-linear layer """ 

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4*n_embd),
            nn.ReLU(),
            nn.Linear(4*n_embd, n_embd),
            nn.Dropout(dropout),
        )
        
    def forward(self, x):
        return self.net(x)


class LayerNorm:
    def __init__(self, dim, eps=1e-5, momentum=0.1):
        self.eps = eps
        self.gamma = torch.ones(dim)
        self.beta = torch.zeros(dim)
    
    def __call__(self, x):
        # calculate forward pass
        xmean = x.mean(1, keepdim=True)
        xvar = x.var(1, keepdim=True)
        xhat = (x - xmean) / torch.sqrt(xvar + self.beta)
        self.out = self.gamma * xhat + self.beta
    
    def parameters(self):
        return [self.gamma, self.alpha]
  
class Block(nn.Module):
    """ transformer block: communication followed by computation """
    
    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimensions, n_head: number of heads

        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.lnl = LayerNorm(n_embd)
    def forward(self, x):
        x = x + self.sa(x)
        x = x + self.ffwd(x)
        return x
        
# Modified bigram model
class LanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        
        # each token directly reads off the logits for the next token from the lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])       
       
        # self.sa_head = Head(n_embd)
        ## no need after introduction of blocks
        # self.sa_heads = MultiHeadAttention(4, n_embd//4) # i.e. 4 heads of 8-dimensional self-attention
        # self.ffwd = FeedForward(n_embd)
        
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size) # linear layer

    def forward(self, idx, targets=None):
        
        
        # idx and targets are both (B, T) tensors of integers
        B, T = idx.shape
        
        tok_embd = self.token_embedding_table(idx) # (B, T, C) batch by time by channel
        pos_embd = self.position_embedding_table(torch.arange(T, device=device)) # (T, C)
        x = tok_embd + pos_embd  # (B, T, C)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x) # (B, T, vocab_size)
        
        if targets is None:
            loss = None
        else:
            # reshaping
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last timestep
            logits = logits[:, -1, :] # becomes (B, C)
            
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx


model = LanguageModel()
model = model.to(device)


# creating a pytorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

# batch_size = 32
for iter in range(max_iters):

    # print(f"step {iter}")
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val_loss {losses['val']:.4f}")
    
    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()



# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print("Output:\n", decode(model.generate(context, max_new_tokens=500)[0].tolist()))

