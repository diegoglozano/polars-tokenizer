"""Generate exact Gemma 3 count oracles with Google's pinned local tokenizer."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Final

from benchmarks._data import make_dataset
from benchmarks.gemini_local import artifact_metadata, load_google_modules

UNICODE_RANGES: Final = (
    (0x0000, 0x007F),
    (0x0080, 0x024F),
    (0x0300, 0x036F),
    (0x0370, 0x052F),
    (0x0600, 0x06FF),
    (0x0900, 0x097F),
    (0x3040, 0x30FF),
    (0x3400, 0x4DBF),
    (0x4E00, 0x9FFF),
    (0x1F300, 0x1FAFF),
    (0x200B, 0x200F),
)

EDGE_CASES: Final = (
    "",
    " ",
    "  ",
    " " * 31,
    " " * 257,
    " " * 4097,
    "\0",
    "null\0byte",
    "e\u0301",
    "é",
    "\r\n",
    "<start_of_turn>user",
    "abc 世界 🦀\r\n\0" * 512,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="gemini-2.5-flash")
    parser.add_argument("--dataset-rows", type=int, default=10_000)
    parser.add_argument("--unicode-rows", type=int, default=10_000)
    parser.add_argument("--max-unicode-scalars", type=int, default=160)
    parser.add_argument("--seed", type=int, default=20_260_922)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.dataset_rows < 0 or args.unicode_rows < 0:
        parser.error("row counts must not be negative")
    if args.max_unicode_scalars <= 0:
        parser.error("--max-unicode-scalars must be positive")
    return args


def random_unicode(rng: random.Random, max_scalars: int) -> str:
    characters = []
    for _ in range(rng.randrange(max_scalars)):
        start, end = rng.choice(UNICODE_RANGES)
        characters.append(chr(rng.randint(start, end)))
    return "".join(characters)


def build_corpus(args: argparse.Namespace) -> list[str]:
    values = make_dataset(
        args.dataset_rows,
        "short",
        1.0,
        args.seed,
        content="mixed",
        null_rate=0.0,
    )
    texts = [value for value in values if value is not None]
    rng = random.Random(args.seed)
    texts.extend(random_unicode(rng, args.max_unicode_scalars) for _ in range(args.unicode_rows))
    texts.extend(EDGE_CASES)
    return texts


def main() -> None:
    args = parse_args()
    texts = build_corpus(args)
    _, loader, _ = load_google_modules()
    tokenizer_name = str(loader.get_tokenizer_name(args.model))
    if tokenizer_name != "gemma3":
        raise ValueError(f"model resolved to {tokenizer_name!r}, not the Gemma 3 tokenizer")
    processor: Any = loader.get_sentencepiece(tokenizer_name)
    encoded = processor.encode(texts)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as output:
        for text, token_ids in zip(texts, encoded, strict=True):
            output.write(json.dumps((text, len(token_ids)), ensure_ascii=False) + "\n")

    metadata = {
        "artifact": artifact_metadata(loader, tokenizer_name),
        "bytes": sum(len(text.encode()) for text in texts),
        "cases": len(texts),
        "output": str(args.output),
        "tokens": sum(len(token_ids) for token_ids in encoded),
    }
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
