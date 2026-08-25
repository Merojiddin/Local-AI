"""Long-text chunking: split text into synthesis-sized pieces on sentence
boundaries so a single over-long generation can't hit the token cap."""

from __future__ import annotations

import re


# --------------------------------------------------------------------------- #
# Long-text chunking
# --------------------------------------------------------------------------- #
# The model generates a bounded number of tokens per call (~12.5 tokens/sec of
# audio). A long paragraph fed in one shot can exceed that budget and trail off
# into silence when the cap is hit mid-utterance, so long text is split on
# sentence boundaries and each piece is synthesized separately, then joined.
#
# Each chunk is an independent generation, so its pace/intonation/loudness reset
# at the boundary — audible as the reading "restarting". Two things keep that in
# check: (1) the budget is large enough that a normal paragraph stays a SINGLE
# coherent generation, and chunks are balanced to roughly equal size (no tiny
# tail pieces); (2) chunks are loudness-matched before joining (normalize_wav_loud).
MAX_CHUNK_CHARS = 200
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


def chunk_max_tokens(chunk: str) -> int:
    """Generous per-chunk token cap: comfortably above what a chunk this long
    needs (~2.8 tokens/char observed) so generation always reaches EOS."""
    return max(512, min(4096, len(chunk) * 8 + 128))
