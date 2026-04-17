import pickle
from functools import lru_cache
from typing import Iterable, Iterator
import regex as re

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


class Tokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ) -> None:
        if special_tokens is None:
            special_tokens = []
        vocab_values = set(vocab.values())
        for special_token in special_tokens:
            byte_encoded_special_token = special_token.encode("utf-8")
            if byte_encoded_special_token not in vocab_values:
                vocab[len(vocab)] = byte_encoded_special_token
                vocab_values.add(byte_encoded_special_token)

        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens
        self._special_set = set(special_tokens)
        self._specials_longest_first = sorted(special_tokens, key=len, reverse=True)
        self._merge_rank = {pair: i for i, pair in enumerate(merges)}
        # Prefer the smallest token id when the same byte string appears more than once.
        self._bytes_to_id = {}
        for tid, piece in sorted(vocab.items()):
            if piece not in self._bytes_to_id:
                self._bytes_to_id[piece] = tid

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: list[str] | None = None,
    ) -> "Tokenizer":
        with open(vocab_filepath, "rb") as f:
            vocab = pickle.load(f)
        with open(merges_filepath, "rb") as f:
            merges = pickle.load(f)
        return cls(vocab, merges, special_tokens)

    @lru_cache(maxsize=65536)
    def _merge_bytes_within_pretoken(self, pretoken: bytes) -> tuple[int, ...]:
        """Apply merge list in training order: best-ranked adjacent pair first, repeat."""
        if not pretoken:
            return ()
        parts = [bytes([b]) for b in pretoken]
        merge_rank = self._merge_rank
        while len(parts) > 1:
            best_rank = float("inf")
            best_idx = -1
            for i in range(len(parts) - 1):
                rank = merge_rank.get((parts[i], parts[i + 1]), float("inf"))
                if rank < best_rank:
                    best_rank = rank
                    best_idx = i
            if best_idx == -1:
                break
            parts[best_idx : best_idx + 2] = [parts[best_idx] + parts[best_idx + 1]]
        return tuple(self._bytes_to_id[p] for p in parts)

    def _encode_segment(self, segment: str) -> list[int]:
        out = []
        for m in re.finditer(PAT, segment):
            out.extend(self._merge_bytes_within_pretoken(m.group(0).encode("utf-8")))
        return out

    def encode(self, text: str) -> list[int]:
        if not text:
            return []
        if not self.special_tokens:
            return self._encode_segment(text)

        # Longer specials first so `re.split` / regex alternation prefers combined tokens.
        pattern_tokens = sorted(self.special_tokens, key=len, reverse=True)
        pattern = "(" + "|".join(re.escape(t) for t in pattern_tokens) + ")"
        out = []
        for piece in re.split(pattern, text):
            if piece == "":
                continue
            if piece in self._special_set:
                out.append(self._bytes_to_id[piece.encode("utf-8")])
            else:
                out.extend(self._encode_segment(piece))
        return out

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for chunk in iterable:
            yield from self.encode(chunk)

    def decode(self, tokens: list[int]) -> str:
        raw = b"".join(self.vocab[i] for i in tokens)
        return raw.decode("utf-8", errors="replace")