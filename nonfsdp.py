import torch
from src.datasets.discrete_synthetic.discrete_synthetic import DiscreteSyntheticDataset
from src.datasets.discrete_helper import collate_fn
from src.tokenizers.discrete_synthetic.discrete_synthetic_tokenizer import DiscreteSyntheticTokenizer
from src.nn.models.discrete_model import DiscreteModel
from src.training.training import train_discrete_model
from matplotlib import pyplot as plt
from src.inference.discrete_inference import dis_t, bayesian_inference
from accelerate import Accelerator
from src.training.checkpoint import save_checkpoint, load_checkpoint
from safetensors.torch import load_file

accelerator = Accelerator()
tokenizer = DiscreteSyntheticTokenizer()
max_seq_len = 32
train_ds = DiscreteSyntheticDataset(tokenizer, tokenized_length=max_seq_len)
train_dl = torch.utils.data.DataLoader(train_ds, batch_size=32, shuffle=True, collate_fn=collate_fn)

model = DiscreteModel(max_seq_len, tokenizer.vocab_size(), hidden_dim=64, num_heads=8)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)

model.load_state_dict(load_file("checkpoints/model.safetensors"))
# opt.load_state_dict(torch.load("checkpoints/optimizer.bin"))

model, opt, train_dl = accelerator.prepare(model, opt, train_dl)


accelerator.save_state("checkpoint")
