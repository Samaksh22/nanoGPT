# nanoGPT devlog
Transformer Based language model.

## Dataset
[Tiny Shakespeare](https://github.com/karpathy/char-rnn/blob/master/data/tinyshakespeare/input.txt)
I have just copied the text from karpathy's repo.


## Bigram
- Followed Karpathy's video and built a Bigram model.
- From my understanding, it is a very simple model. 
- The model just reads one token at a time and uses a probablity table (embedding table) to generate the next token.
- This uses `multinomial` function on the current token's channels.
- The model is simple and the only parameters to train is the embedding table.
- ~~Though I have followed Karpathy, I still have doubt as to what is actually run on the GPU. In the last few lines, we have declared the model but we have then moved it to GPU under the name 'm'. But in the optimizer, we have again used model.parameters. In the end, we have used 'm' to generate.~~ 
- All we actually do is move the model and parameters to `device` which is either `CPU` or `CUDA`. We must keep the name consistent at all the places and keep it as `model`.


## Attention Mechanism
The mechanism is quite complex. It took me several passes from that section of the video to understand the concepts. Still not fully grapsed but got the idea.
- I was stuck in setting the hyperparameters. For the expression `head_size = n_embd // n_head` , I had accidentaly set n_embd to 32 and n_head to 3. This caused the head_size to be 10 and later a matrix multiplication error `(32, 30) x (32, 32)`.
- I finally watched the full video and understood most of it. We have not built the encoder part as we just need to reproduce the input data.
- The final transformer model decoder block is in the `v2.py` file.
- Implemented single headed and multi headed attention. The next motivation is to also include encoder block.

This was a fun project and a great learning experience. Special thanks to Andrej Karpathy for providing these great learning resources for free.
