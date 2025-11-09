import torch
from src.tokenizers.character_level.character_level import CharacterLevelTokenizer


def test_character_level_tokenizer_init():
    tokenizer = CharacterLevelTokenizer()
    assert tokenizer.unk_token == "<UNK>"
    assert isinstance(tokenizer.vocab, dict)
    assert isinstance(tokenizer.anti_vocab, dict)
    assert len(tokenizer.vocab) == len(tokenizer.anti_vocab)
    assert tokenizer.vocab["a"] == 0
    assert tokenizer.anti_vocab[0] == "a"
    assert tokenizer.vocab[tokenizer.unk_token] == len(tokenizer.vocab) - 1


def test_character_level_tokenizer_vocab_size():
    tokenizer = CharacterLevelTokenizer()
    assert tokenizer.vocab_size() == len(tokenizer.vocab)


def test_character_level_tokenizer_encode():
    tokenizer = CharacterLevelTokenizer()
    text = "Hello, world!"
    expected_ids = [
        tokenizer.vocab["h"],
        tokenizer.vocab["e"],
        tokenizer.vocab["l"],
        tokenizer.vocab["l"],
        tokenizer.vocab["o"],
        tokenizer.vocab[","],
        tokenizer.vocab[" "],
        tokenizer.vocab["w"],
        tokenizer.vocab["o"],
        tokenizer.vocab["r"],
        tokenizer.vocab["l"],
        tokenizer.vocab["d"],
        tokenizer.vocab[tokenizer.unk_token],  # for '!' which is in vocab
    ]
    # The tokenizer in the file has '!' in its vocab. Let's correct the test.
    expected_ids[-1] = tokenizer.vocab["!"]
    encoded = tokenizer.encode(text)
    assert torch.equal(encoded, torch.tensor(expected_ids, dtype=torch.long))

    # Test with unknown characters
    text_with_unk = "123"
    expected_unk_ids = [
        tokenizer.vocab[tokenizer.unk_token],
        tokenizer.vocab[tokenizer.unk_token],
        tokenizer.vocab[tokenizer.unk_token],
    ]
    encoded_unk = tokenizer.encode(text_with_unk)
    assert torch.equal(encoded_unk, torch.tensor(expected_unk_ids, dtype=torch.long))


def test_character_level_tokenizer_decode():
    tokenizer = CharacterLevelTokenizer()
    text = "test"
    encoded = tokenizer.encode(text)
    decoded = tokenizer.decode(encoded)
    assert decoded == text


def test_character_level_tokenizer_encode_decode_roundtrip():
    tokenizer = CharacterLevelTokenizer()
    text = "The quick brown fox jumps over the lazy dog.!?;"
    encoded = tokenizer.encode(text)
    decoded = tokenizer.decode(encoded)
    assert decoded == text.lower()


def test_character_level_tokenizer_decode_with_unk():
    tokenizer = CharacterLevelTokenizer()
    tokens = torch.tensor(
        [
            tokenizer.vocab["h"],
            tokenizer.vocab[tokenizer.unk_token],
            tokenizer.vocab["i"],
        ]
    )
    decoded = tokenizer.decode(tokens)
    assert decoded == "h<UNK>i"
