import os
import regex as re
from collections import Counter, defaultdict

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def _adjust_pair_counts(
    pieces: list[bytes],
    freq: int,
    pair_counts: dict[tuple[bytes, bytes], int],
    delta: int,
) -> None:
    # Walk adjacent symbols, add (+1) or remove (-1) this word's weight from each pair's global count.
    for i in range(len(pieces) - 1):
        pair_counts[(pieces[i], pieces[i + 1])] += delta * freq


def _merge_adjacent(pieces: list[bytes], left: bytes, right: bytes) -> list[bytes]:
    # Replace every adjacent (left, right) with the concatenated token.
    out = []
    i = 0
    while i < len(pieces):
        if i + 1 < len(pieces) and pieces[i] == left and pieces[i + 1] == right:
            out.append(left + right)
            i += 2
        else:
            out.append(pieces[i])
            i += 1
    return out


def _best_pair(pair_counts: dict[tuple[bytes, bytes], int]) -> tuple[bytes, bytes] | None:
    # Highest count, ties -> lexicographically larger (first token, then second)
    best = None
    best_key = None
    for pair, cnt in pair_counts.items():
        if cnt <= 0:
            continue
        key = (cnt, pair[0], pair[1])
        if best_key is None or key > best_key:
            best_key = key
            best = pair
    return best


def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    Byte-level BPE: split on specials with `re`, pre-tokenize chunks with the given regex,
    merge most frequent adjacent byte-chunks inside each pre-token, then build vocab
    (specials, all 256 raw bytes, then one entry per merge).
    """
    with open(input_path, encoding="utf-8") as f:
        corpus = f.read()

    # Split on specials so pair counts never cross a metadata boundary.
    if special_tokens:
        chunks = re.split("|".join(re.escape(t) for t in special_tokens), corpus)
    else:
        chunks = [corpus]

    # How often each pre-token (UTF-8 bytes) appears; BPE statistics stay inside these spans.
    pretoken_counts = Counter()
    for chunk in chunks:
        for match in re.finditer(PAT, chunk):
            pretoken_counts[match.group(0).encode("utf-8")] += 1

    # Per distinct pre-token: current list of mergeable pieces (start as single bytes).
    splits = {word: [bytes([b]) for b in word] for word in pretoken_counts}
    # Global weighted counts of adjacent pairs (weight = pre-token frequency).
    pair_counts = defaultdict(int)
    for word, freq in pretoken_counts.items():
        _adjust_pair_counts(splits[word], freq, pair_counts, +1)

    # Each merge adds one vocabulary entry beyond the 256 raw bytes and specials.
    num_merges = max(0, vocab_size - 256 - len(special_tokens))
    merges = []

    for _ in range(num_merges):
        pair = _best_pair(pair_counts)
        if pair is None:
            break
        left, right = pair
        merges.append((left, right))

        # Incremental counts: only update words that contained this pair; strip old pairs, add new ones.
        for word, freq in pretoken_counts.items():
            pieces = splits[word]
            if not any(pieces[i] == left and pieces[i + 1] == right for i in range(len(pieces) - 1)):
                continue
            _adjust_pair_counts(pieces, freq, pair_counts, -1)
            new_pieces = _merge_adjacent(pieces, left, right)
            splits[word] = new_pieces
            _adjust_pair_counts(new_pieces, freq, pair_counts, +1)

    vocab: dict[int, bytes] = {}
    for b in range(256):
        vocab[b] = bytes([b])
    for special_token in special_tokens:
        vocab[len(vocab)] = special_token.encode("utf-8")
    for left, right in merges:
        vocab[len(vocab)] = left + right

    return vocab, merges
