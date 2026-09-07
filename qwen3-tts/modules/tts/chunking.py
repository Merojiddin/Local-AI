"""Long-text chunking: split text into synthesis-sized pieces on sentence
boundaries so a single over-long generation can't hit the token cap."""

from __future__ import annotations

import re


# --------------------------------------------------------------------------- #
# Long-text chunking
# --------------------------------------------------------------------------- #
# Each chunk is an independent generation, so its pace/intonation/loudness reset
# at the boundary — audible as the reading "restarting". The budget is therefore
# set as high as the hardware allows, so ordinary text (a full two-minute read
# included) stays a SINGLE coherent generation and this splitting never happens.
#
# What used to cap it was memory, not the model. The vocoder materialised the
# whole waveform in one pass, costing ~10 MB per generated token with nothing
# bounding it: measured on an M2/16 GB, 600 chars peaked at 16.3 GB and two
# minutes extrapolated to ~19 GB, past the machine's 17.2 GB — hence the old
# 200-char budget. Qwen now generates with stream=True (see engine.py), which
# leaves the talker's autoregressive loop unbroken across the whole text and
# only decodes the codec incrementally, carrying conv buffers and a KV cache
# across steps. That flattens peak memory to ~5 GB regardless of length, so the
# budget below is bounded by generation coherence rather than by RAM.
#
# The ceiling is now generation coherence, not RAM. This model degenerates into
# a repetition loop if pushed too far — a synthetic 800-char probe ran to the
# 4096-token cap and emitted 5.5 minutes of looping audio — so the budget stays
# near what has actually been measured clean: 625 chars of natural prose reached
# EOS normally at 1519 tokens, producing 121.5s of audio in one take (5.15 GB).
# Measured 5.14 chars/sec, so 700 chars is ~2.3 minutes: it keeps a two-minute
# read (~620 chars) in a single generation with a little margin, without
# extrapolating far past verified territory. Longer text still splits; the pieces
# are balanced and loudness-matched before joining (normalize_wav_loud).
MAX_CHUNK_CHARS = 700

# Fish S2 Pro is a different beast from the light 24 kHz Qwen models: it is a 4B
# model at 44.1 kHz whose vocoder (codec.decode) materialises a whole chunk's
# waveform in a SINGLE Metal buffer, growing linearly with the tokens generated.
# On a 16 GB Mac the largest single GPU buffer is ~8.9 GiB, and a normal-length
# chunk overflows it — the model dies with "[metal::malloc] Attempting to
# allocate … greater than the maximum allowed buffer size". That ceiling is a
# fixed device property: it cannot be raised with set_memory_limit /
# set_wired_limit (those bound the *total* working set, not one allocation), so
# the only cure is to decode less audio per call. Fish therefore uses a much
# smaller chunk and a hard token cap that keeps every decode well under the
# buffer limit while coexisting with the ~7.5 GB resident model.
FISH_MAX_CHUNK_CHARS = 60
FISH_MAX_TOKENS = 480

_SENT_END = "。！？!?；;…\n"
_SENT_SPLIT = re.compile(rf"[^{re.escape(_SENT_END)}]*[{re.escape(_SENT_END)}]?", re.UNICODE)
_CLAUSE_END = "，,、：:"
_CLAUSE_SPLIT = re.compile(rf"[^{re.escape(_CLAUSE_END)}]*[{re.escape(_CLAUSE_END)}]?", re.UNICODE)


def _hard_pieces(sentence: str, max_chars: int) -> list[str]:
    """Break an over-long sentence on clause punctuation, then on raw length."""
    pieces: list[str] = []
    for clause in _CLAUSE_SPLIT.findall(sentence):
        clause = clause.strip()
        if not clause:
            continue
        while len(clause) > max_chars:
            pieces.append(clause[:max_chars])
            clause = clause[max_chars:]
        if clause:
            pieces.append(clause)
    return pieces


def split_for_tts(text: str, max_chars: int = MAX_CHUNK_CHARS) -> list[str]:
    """Split text into the fewest synthesis chunks that each stay within
    max_chars, breaking on sentence boundaries and keeping the pieces roughly
    equal in size (so there is no stubby final chunk that reads oddly)."""
    text = (text or "").strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]

    # Sentence units, hard-splitting any single sentence longer than the budget.
    units: list[str] = []
    for s in (s.strip() for s in _SENT_SPLIT.findall(text)):
        if not s:
            continue
        units.extend(_hard_pieces(s, max_chars) if len(s) > max_chars else [s])

    # Aim for the fewest chunks possible, then even them out: packing to an even
    # target avoids a long chunk followed by a stubby one (e.g. 182 + 24).
    total = sum(len(u) for u in units)
    n_chunks = max(1, -(-total // max_chars))       # ceil(total / max_chars)
    target = -(-total // n_chunks)                   # ceil(total / n_chunks)

    chunks: list[str] = []
    cur = ""
    for u in units:
        # Flush before adding would overshoot the even target. target <= max_chars
        # and every unit <= max_chars, so no chunk can exceed max_chars.
        if cur and len(cur) + len(u) > target:
            chunks.append(cur)
            cur = ""
        cur += u
    if cur:
        chunks.append(cur)

    # Fold a stubby trailing chunk back into its neighbour when it still fits —
    # a lone 20-char tail generated on its own reads worse than a fuller chunk.
    # Pop first, then append to the new last chunk: writing `chunks[-2] +=
    # chunks.pop()` would resolve the -2 index *after* pop() shrinks the list and
    # corrupt the wrong chunk.
    while (
        len(chunks) >= 2
        and len(chunks[-1]) < target * 0.6
        and len(chunks[-2]) + len(chunks[-1]) <= max_chars
    ):
        tail = chunks.pop()
        chunks[-1] += tail
    return chunks


def chunk_max_tokens(chunk: str, fish: bool = False) -> int:
    """Per-chunk token cap: comfortably above what a chunk this long needs
    (~2.8 tokens/char observed) so generation always reaches EOS.

    Fish gets a hard, low ceiling instead: its 44.1 kHz vocoder decodes the whole
    chunk in one Metal buffer, so an over-long generation would blow past the
    ~8.9 GiB single-buffer limit. FISH_MAX_CHUNK_CHARS keeps chunks short enough
    that this cap is never actually reached mid-utterance (which would clip the
    tail to silence)."""
    if fish:
        return max(256, min(FISH_MAX_TOKENS, len(chunk) * 8 + 128))
    return max(512, min(4096, len(chunk) * 8 + 128))
