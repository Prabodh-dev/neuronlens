import pytest

from neuronlens.textdecode import _byte_decoder, token_pieces

FFFD = "�"


class ByteTok:
    """Offline stand-in for the GPT-2 tokenizer: byte-level vocab plus a few merges."""

    def __init__(self):
        enc = {b: c for c, b in _byte_decoder().items()}  # byte -> stand-in char
        self.vocab = [enc[b] for b in range(256)]
        self.vocab += ["They", "re", enc[0xE2] + enc[0x80]]  # "âĢ": first 2 bytes of U+2019
        self.vocab += ["<|endoftext|>"]
        self.index = {t: i for i, t in enumerate(self.vocab)}
        self.enc = enc

    def convert_ids_to_tokens(self, ids):
        return [self.vocab[i] for i in ids]

    def encode(self, text):
        # greedy, like GPT-2 splitting the apostrophe into 2 tokens: "They", "âĢ", "Ļ", "re"
        ids, i = [], 0
        data = text.encode("utf-8")
        while i < len(data):
            for piece in ("They", "re"):
                pb = piece.encode()
                if data[i:i + len(pb)] == pb:
                    ids.append(self.index[piece])
                    i += len(pb)
                    break
            else:
                if data[i:i + 2] == bytes([0xE2, 0x80]):
                    ids.append(self.index[self.enc[0xE2] + self.enc[0x80]])
                    i += 2
                else:
                    ids.append(data[i])
                    i += 1
        return ids


def test_curly_apostrophe_has_no_replacement_char():
    tok = ByteTok()
    text = "They’re"
    ids = tok.encode(text)
    assert len(ids) == 4  # They, âĢ, Ļ, re
    pieces = token_pieces(tok, ids)
    assert len(pieces) == len(ids)
    assert not any(FFFD in p for p in pieces)
    assert "".join(pieces) == text
    assert pieces == ["They", "", "’", "re"]


def test_pieces_join_equals_text_for_mixed_text():
    tok = ByteTok()
    text = "naïve café 日本語 😀 ok"
    ids = list(text.encode("utf-8"))  # every byte its own token: worst case
    pieces = token_pieces(tok, ids)
    assert len(pieces) == len(ids)
    assert "".join(pieces) == text and FFFD not in "".join(pieces)


def test_special_token_and_invalid_bytes():
    tok = ByteTok()
    eos = tok.index["<|endoftext|>"]
    assert token_pieces(tok, [eos]) == ["<|endoftext|>"]
    assert token_pieces(tok, []) == []
    assert FFFD in "".join(token_pieces(tok, [0xFF, ord("a")]))  # genuinely invalid
    assert FFFD in "".join(token_pieces(tok, [0xE2, 0x80]))      # truncated at the end


def test_window_slice_keeps_alignment():
    tok = ByteTok()
    ids = tok.encode("They’re")
    pieces = token_pieces(tok, ids)
    acts = [0.1, 0.2, 0.9, 0.3]
    assert len(pieces[1:]) == len(acts[1:])  # a window starting mid-character stays aligned


def real_tokenizer():
    """The cached GPT-2 tokenizer, or skip. (Newer transformers return an EMPTY tokenizer
    instead of raising when nothing is cached, so check that it can actually encode.)"""
    tr = pytest.importorskip("transformers")
    try:
        tok = tr.GPT2TokenizerFast.from_pretrained("gpt2", local_files_only=True)
        usable = len(tok.encode("They’re")) > 0
    except Exception:
        usable = False
    if not usable:
        pytest.skip("gpt2 tokenizer not cached")
    return tok


def test_real_gpt2_tokenizer():
    tok = real_tokenizer()
    text = "They’re here — “quoted” naïve 日本語 😀"
    ids = tok.encode(text)
    pieces = token_pieces(tok, ids)
    assert len(pieces) == len(ids)
    assert "".join(pieces) == text == tok.decode(ids)
    assert not any(FFFD in p for p in pieces)
