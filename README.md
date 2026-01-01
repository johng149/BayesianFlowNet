`TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 accelerate launch --config_file ./ddp.yaml main.py`

`FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 accelerate launch --config_file ddp_dynamo_bf16.yaml main.py`

### Findings

- Dynamic chunker: absolute dog when encoder / decoder is not SSM based layer such as Mamba. Even after switching to Mamba2, it slows down training speed and convergence. I suppose one could argue that at larger sequence lengths it might help training speed and convergence because the heavy transformer layers do not run on entire sequence, but I have no idea if that is true.
- Differential Transformer: Again, it slows down training speed and convergence. And again, perhaps it makes needle in a haystack problems easier to solve at larger sequence lengths, but I have no idea if that is true.
- Energy Based Transformers: Surprisingly, convergence is about the same. But the extra refinement step adds a lot of compute, so overall training speed is worse, without apparently any benefit. Also, requires multiple backwards passes which doesn't work well with torch.compile
- Monarch Linear: Absolute dog, diverges almost immediately. Maybe I am not using it correctly, but honestly I can't tell if that is true or not because the documentation is so sparse.
- Isotropic activation functions: Basically, the idea is what if activation functions act on the entire vector by using its magnitude rather than elementwise. It supposedly helps by avoiding the inductive bias of elementwise activations. Does not converge, absolute dog.
- Elephant activation functions: Supposedly the sparser gradients helps with continual learning and avoiding catastrophic forgetting. Convergence is slightly worse, and training speed is slightly worse. Perhaps if the dataset was huge it would make a difference, but so far my model is large enough to overfit the dataset so I don't see any benefit here.
- Sequence packing: About 17% faster training speed, converge is the same since mathematically it is equivalent.

Naively trying to implement inference time scaling with using the energy based model as a verifier did not work out well. Peformance is about the same but compute cost is higher. The way I tried to do it was to have one model forward pass to go from `x_0` to `x_1`, where `x_1` is the updated logits.

Then, using those `x_1` logits, sample `k` candidates from the distribution defined by `x_1` to produce indices for the sequences, then apply the normal noise adding step as if  those `k` sequences were the ground truth, then using the energy based model to score those `k` noisy logit sequences, and finally take the top scoring sequence as the new `x_1`. Seems with this method there is a train-test mismatch.