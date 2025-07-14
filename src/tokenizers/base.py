from torch import Tensor

class TokenizerBase:
    def vocab_size(self) -> int:
        raise NotImplementedError("This method should be implemented by subclasses.")
    
    def encode(self, text: str) -> Tensor:
        raise NotImplementedError("This method should be implemented by subclasses.")
    
    def decode(self, tokens: Tensor) -> str:
        raise NotImplementedError("This method should be implemented by subclasses.")