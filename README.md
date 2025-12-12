`TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 accelerate launch --config_file ./ddp.yaml main.py`

### Findings

- Dynamic chunker: absolute dog when encoder / decoder is not SSM based layer such as Mamba. Even after switching to Mamba2, it slows down training speed and convergence. I suppose one could argue that at larger sequence lengths it might help training speed and convergence because the heavy transformer layers do not run on entire sequence, but I have no idea if that is true.
- Differential Transformer: Again, it slows down training speed and convergence. And again, perhaps it makes needle in a haystack problems easier to solve at larger sequence lengths, but I have no idea if that is true.
- Energy Based Transformers: Surprisingly, convergence is about the same. But the extra refinement step adds a lot of compute, so overall training speed is worse, without apparently any benefit. Also, requires multiple backwards passes which doesn't work well with torch.compile
- Monarch Linear: Absolute dog, diverges almost immediately. Maybe I am not using it correctly, but honestly I can't tell if that is true or not because the documentation is so sparse.
- Isotropic activation functions: Basically, the idea is what if activation functions act on the entire vector by using its magnitude rather than elementwise. It supposedly helps by avoiding the inductive bias of elementwise activations. Does not converge, absolute dog.
- Elephant activation functions: Supposedly the sparser gradients helps with continual learning and avoiding catastrophic forgetting. Convergence is slightly worse, and training speed is slightly worse. Perhaps if the dataset was huge it would make a difference, but so far my model is large enough to overfit the dataset so I don't see any benefit here.
- Sequence packing: About 17% faster training speed, converge is the same since mathematically it is equivalent.
- RWKV7 instead of Mamba2: Had to halve the batch size to prevent OOM, and training speed still wasn't any faster. Absolute dog in that regard, I'm not waiting around to see if convergence is any different.