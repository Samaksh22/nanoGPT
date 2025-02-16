# nanoGPT
Transformer Based language model.

## Dataset
[Tiny Shakespeare](https://github.com/karpathy/char-rnn/blob/master/data/tinyshakespeare/input.txt)
I have just copied the text from karpathy's repo.


## Bigram
Followed Karpathy's video and built a Bigram model.
From my understanding, it is a very simple model. 
The model just reads one token at a time and uses a probablity table (embedding table) to generate the next token.
This uses multinomial function on the current token's channels.
The model is simple and the only parameters to train is the embedding table.
