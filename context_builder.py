from __future__ import annotations

import re
from dataclasses import dataclass, replace
from typing import Protocol, Sequence, TypedDict

import step1


class TokenCounter(Protocol):
    def __call__(self, text: str, /) -> int: ...


class ContextSourceDict(TypedDict):
    filename: str
    page_number: int
    char_start: int
    char_end: int
    ocr_used: bool
    chunk_ids: list[int]


class ContextPassageDict(TypedDict):
    rank: int
    text: str
    score: float
    truncated: bool
    sources: list[ContextSourceDict]


class ContextBundleDict(TypedDict):
    text: str
    token_count: int
    max_tokens: int
    passages: list[ContextPassageDict]


@dataclass(frozen=True)
class ContextSource:
    filename: str
    page_number: int
    char_start: int
    char_end: int
    ocr_used: bool
    chunk_ids: tuple[int, ...] = ()


@dataclass(frozen=True)
class ContextPassage:
    rank: int
    text: str
    score: float
    sources: tuple[ContextSource, ...]
    truncated: bool = False


@dataclass(frozen=True)
class ContextBundle:
    passages: tuple[ContextPassage, ...]
    text: str
    token_count: int
    max_tokens: int

    def to_dict(self) -> ContextBundleDict:
        return {
            "text": self.text,
            "token_count": self.token_count,
            "max_tokens": self.max_tokens,
            "passages": [
                {
                    "rank": passage.rank,
                    "text": passage.text,
                    "score": passage.score,
                    "truncated": passage.truncated,
                    "sources": [
                        {
                            "filename": source.filename,
                            "page_number": source.page_number,
                            "char_start": source.char_start,
                            "char_end": source.char_end,
                            "ocr_used": source.ocr_used,
                            "chunk_ids": list(source.chunk_ids),
                        }
                        for source in passage.sources
                    ],
                }
                for passage in self.passages
            ],
        }


@dataclass(frozen=True)
class _RankedSpan:
    rank: int
    text: str
    score: float
    source: ContextSource


def _merge_spans(
    first: _RankedSpan,
    second: _RankedSpan,
) -> _RankedSpan | None:
    first_source = first.source
    second_source = second.source
    if (
        first_source.filename != second_source.filename
        or first_source.page_number != second_source.page_number
        or second_source.char_start > first_source.char_end
    ):
        return None

    if (
        first_source.char_end - first_source.char_start != len(first.text)
        or second_source.char_end - second_source.char_start
        != len(second.text)
    ):
        return None

    overlap = first_source.char_end - second_source.char_start
    if second_source.char_end <= first_source.char_end:
        offset = second_source.char_start - first_source.char_start
        if first.text[offset:offset + len(second.text)] != second.text:
            return None
        text = first.text
        end = first_source.char_end
    else:
        if overlap > len(second.text):
            return None
        if overlap and first.text[-overlap:] != second.text[:overlap]:
            return None
        text = first.text + second.text[overlap:]
        end = second_source.char_end

    if first.rank <= second.rank:
        rank, score = first.rank, first.score
    else:
        rank, score = second.rank, second.score
    return _RankedSpan(
        rank=rank,
        text=text,
        score=score,
        source=replace(
            first_source,
            char_end=end,
            ocr_used=first_source.ocr_used or second_source.ocr_used,
            chunk_ids=tuple(
                dict.fromkeys(first_source.chunk_ids + second_source.chunk_ids)
            ),
        ),
    )


def _ranked_spans(
    results: Sequence[step1.SearchResult],
) -> list[ContextPassage]:
    pages: dict[tuple[str, int], list[_RankedSpan]] = {}
    for rank, result in enumerate(results, start=1):
        if not result.text.strip():
            continue
        source = ContextSource(
            filename=result.filename,
            page_number=result.page_number,
            char_start=result.char_start,
            char_end=result.char_end,
            ocr_used=result.ocr_used,
            chunk_ids=(result.chunk_id,),
        )
        pages.setdefault((result.filename, result.page_number), []).append(
            _RankedSpan(rank, result.text, result.score, source)
        )

    merged: list[_RankedSpan] = []
    for spans in pages.values():
        spans.sort(
            key=lambda span: (
                span.source.char_start,
                span.source.char_end,
                span.rank,
            )
        )
        current = spans[0]
        for span in spans[1:]:
            combined = _merge_spans(current, span)
            if combined is None:
                merged.append(current)
                current = span
            else:
                current = combined
        merged.append(current)
    merged.sort(key=lambda span: span.rank)

    unique: list[ContextPassage] = []
    text_indexes: dict[str, int] = {}
    for span in merged:
        normalized = re.sub(r"\s+", " ", span.text).strip().casefold()
        existing_index = text_indexes.get(normalized)
        if existing_index is None:
            text_indexes[normalized] = len(unique)
            unique.append(
                ContextPassage(
                    rank=span.rank,
                    text=span.text,
                    score=span.score,
                    sources=(span.source,),
                )
            )
            continue

        existing = unique[existing_index]
        sources = list(existing.sources)
        if span.source not in sources:
            sources.append(span.source)
        unique[existing_index] = replace(existing, sources=tuple(sources))
    return unique


def _format_passage(position: int, passage: ContextPassage) -> str:
    sources = "; ".join(
        f"{source.filename}, page {source.page_number} "
        f"(chars {source.char_start}-{source.char_end})"
        f"{', OCR' if source.ocr_used else ''}"
        for source in passage.sources
    )
    return f"[{position}] Source: {sources}\n{passage.text}"


def _render(passages: Sequence[ContextPassage]) -> str:
    return "\n\n".join(
        _format_passage(position, passage)
        for position, passage in enumerate(passages, start=1)
    )


def _count_tokens(token_counter: TokenCounter, text: str) -> int:
    count = token_counter(text)
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        raise ValueError("token_counter must return a nonnegative integer")
    return count


def _fit_prefix(
    passage: ContextPassage,
    selected: Sequence[ContextPassage],
    *,
    max_tokens: int,
    token_counter: TokenCounter,
) -> ContextPassage | None:
    low, high = 1, len(passage.text)
    best: ContextPassage | None = None
    while low <= high:
        length = (low + high) // 2
        text = passage.text[:length].rstrip()
        if not text:
            low = length + 1
            continue
        sources = tuple(
            replace(
                source,
                char_end=min(
                    source.char_end,
                    source.char_start + len(text),
                )
                if source.char_end - source.char_start == len(passage.text)
                else source.char_end,
            )
            for source in passage.sources
        )
        candidate = replace(
            passage,
            text=text,
            sources=sources,
            truncated=True,
        )
        rendered = _render([*selected, candidate])
        if _count_tokens(token_counter, rendered) <= max_tokens:
            best = candidate
            low = length + 1
        else:
            high = length - 1
    return best


def build_context(
    results: Sequence[step1.SearchResult],
    *,
    max_tokens: int,
    token_counter: TokenCounter,
) -> ContextBundle:
    if max_tokens <= 0:
        raise ValueError("max_tokens must be greater than zero")

    selected: list[ContextPassage] = []
    for passage in _ranked_spans(results):
        candidate = [*selected, passage]
        rendered = _render(candidate)
        if _count_tokens(token_counter, rendered) <= max_tokens:
            selected.append(passage)
            continue

        partial = _fit_prefix(
            passage,
            selected,
            max_tokens=max_tokens,
            token_counter=token_counter,
        )
        if partial is not None:
            selected.append(partial)

    context_text = _render(selected)
    token_count = _count_tokens(token_counter, context_text)
    if token_count > max_tokens:
        raise ValueError("Constructed context exceeds the token limit")
    return ContextBundle(
        passages=tuple(selected),
        text=context_text,
        token_count=token_count,
        max_tokens=max_tokens,
    )
