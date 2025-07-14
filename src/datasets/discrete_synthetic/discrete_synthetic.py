from torch.nn import functional as F
from torch.utils.data import Dataset
from src.tokenizers.discrete_synthetic.discrete_synthetic_tokenizer import DiscreteSyntheticTokenizer
import random

class DiscreteSyntheticDataset(Dataset):
    def __init__(self, tokenizer: DiscreteSyntheticTokenizer, length: int = 32, tokenized_length: int = 32, mini: int = 0, maxi: int = 100):
        self.length = length
        self.tokenized_length = tokenized_length
        self.tokenizer = tokenizer
        self.mini = mini
        self.maxi = maxi

    def generate_sequence(self):
        start = random.randint(self.mini, self.maxi - self.length)
        end = start + self.length
        acc = ""
        for i in range(start, end+1):
            for c in str(i):
                acc += " " + c
            acc += " ,"
        print(acc)
        tokenized = self.tokenizer.encode(acc)
        return tokenized[:self.tokenized_length]
    
    def __len__(self):
        return 10000
    
    def __getitem__(self, idx):
        return F.one_hot(self.generate_sequence(), num_classes=self.tokenizer.vocab_size())