"""Byte-aware token rendering.

GPT-2 is a byte-level BPE: one character can be split across several tokens, and a single
token may hold only part of a character. tok.decode([id]) on such a token gives U+FFFD, so
decoding tokens one at a time garbles e.g. a curly apostrophe (3 UTF-8 bytes, 2 tokens).
"""
import codecs
from functools import lru_cache


@lru_cache(maxsize=None)
def _byte_decoder():
    """Inverse of GPT-2's bytes_to_unicode(): printable stand-in char -> raw byte."""
    keep = (list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1))
            + list(range(ord("®"), ord("ÿ") + 1)))
    chars, n = keep[:], 0
    for b in range(256):
        if b not in keep:
            keep.append(b)
            chars.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(keep, chars)}


def token_bytes(tok, ids):
    """Raw UTF-8 bytes of each token id (special tokens such as <|endoftext|> are their text)."""
    dec = _byte_decoder()
    out = []
    for s in tok.convert_ids_to_tokens([int(i) for i in ids]):
        try:
            out.append(bytes(dec[c] for c in s))
        except KeyError:  # special token: not byte-level encoded
            out.append(s.encode("utf-8"))
    return out


def token_pieces(tok, ids):
    """One display string per token id, same length and order as ids.

    The bytes of all tokens are decoded as one UTF-8 stream; each character is given to
    the token in which it ends. Earlier byte pieces of a split character get "" and the
    last piece gets the whole character, so "".join(pieces) equals the whole text decoded
    once. Only genuinely invalid UTF-8 is replaced with U+FFFD.
    """
    dec = codecs.getincrementaldecoder("utf-8")(errors="replace")
    pieces = [dec.decode(b) for b in token_bytes(tok, ids)]
    if pieces:  # a truncated character left at the very end
        pieces[-1] += dec.decode(b"", final=True)
    return pieces
