# using Bigram language model
import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters
batch_size = 4  # count of block processed parallely
block_size = 8  # size of each block
max_iters = 3000
eval_interval = 300
learning_rate = 10e-2
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
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


# train_data[:block_size+1]

# x = train_data[:block_size]
# y = train_data[1:block_size+1]
# for t in range(block_size):
#     context = x[:t+1]
#     target = y[t]
#     print(f"When input is {context} the target is: {target}")

# this is done to achieve parallelism while training the transformers
# torch.manual_seed(1337)

# getting a randomized batch each iteration
def get_batch(split):
    # selecting training or validation data
    data = train_data if split == 'train' else val_data

    indi = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in indi])
    y = torch.stack([data[i+1:i+block_size+1] for i in indi])
    x, y = x.to(device), y.to(device)
    return x, y

# testing the function
# xb, yb = get_batch('train')
# print('inputs:')
# print(xb.shape)
# print(xb)
# print('targets:')
# print(yb.shape)
# print(yb)
# print('--------------')


# for b in range(batch_size):
#     for t in range(block_size):
#         context = xb[b, :t+1]
#         target = yb[b, t]
#         print(f"When input is {context.tolist()} the target: {target}")

# print(xb)


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


"""#Bigram Model
###It takes in previous token and produces next token based solely on that token.
###It is one of the simplest language models.
"""


class BiagramLangModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        # each token directly reads off the logits for the next token from the lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        # idx and targets are both (B, T) tensors of integers
        logits = self.token_embedding_table(idx) # (B, T, C) batch by token by channel

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
            # get the predictions

            ## The below method are not of much use in bigram as we could directly obtain the last token
            logits, loss = self(idx)
            # focus only on the last timestep
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities

            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx


model = BiagramLangModel(vocab_size)
m = model.to(device)

# print(decode(model.generate(idx = torch.zeros((1, 1), dtype=torch.long), max_new_tokens=100)[0].tolist()))

# creating a pytorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

# batch_size = 32
for iter in range(max_iters):

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

    # if (steps+1) % 1000 == 0:
    #     print(f"iteration: {steps+1} loss: {loss.item()}")


# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))

