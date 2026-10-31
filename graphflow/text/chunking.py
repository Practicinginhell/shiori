'''Semantic Chunking Regex v1 — Python port.

Port of the "Semantic Chunking Regex v1" used in https://tokenize.jina.ai
(originally deployed as a JavaScript regex with the `gmu` flags). Splits text
into semantically meaningful chunks: headings, list items, block quotes, code
blocks, tables, sentences, quoted/bracketed spans, paragraphs, HTML tags,
LaTeX math, emoji, with a plain-line fallback.

Requires the third-party `regex` module (`pip install regex`) — the stdlib
`re` cannot compile this pattern (it needs variable-length lookbehind and
Unicode emoji properties like \\p{Extended_Pictographic}).

Ported for **result-parity with the JavaScript original**. The differences
between the engines are neutralised in the pattern itself:

  * ``\\w``/``\\d`` are Unicode-aware in Python but ASCII in JS — written out
    as ``[A-Za-z0-9_]`` / ``[0-9]`` (the exact JS sets).
  * ``.`` excludes only ``\\n`` in Python but also ``\\r\\u2028\\u2029`` in JS
    — written out as ``[^\\n\\r\\u2028\\u2029]`` (the exact JS set).
  * ``\\A`` / ``\\Z`` (absolute start/end of text) don't exist in JS — the JS
    runner translates them to ``(?<![\\s\\S])`` / ``(?![\\s\\S])``.
  * In VERBOSE mode a bare space is ignored, so the indented-code branch
    spells it ``[ ]{4}`` instead of ``\\x20{4}``.

Usage:
    from graphflow.text import chunk_text
    chunks = chunk_text(document)

    # or from the command line:
    python -m graphflow.text.chunking somefile.txt
'''

from __future__ import annotations

import sys

import regex

# The 14 alternatives, top to bottom — first match wins, scanning left to
# right (identical order-of-alternation semantics in Python and JS).
PATTERN = r'''
# 1. Headings (Setext-style, Markdown, and HTML-style)
(?:^(?:[#*=-]{1,6}|[A-Za-z0-9_][^\r\n]{0,199}\r?\n[-=]+|<h[1-6][^>]{0,99}>)[^\r\n]{1,200}(?:</h[1-6]>)?(?:\r?\n|$))
|
# 2. List items (bulleted, numbered, lettered, or task lists, including nested, up to three levels)
(?:(?:^|\r?\n)[ \t]{0,3}(?:[-*+•]|[0-9]{1,3}\.|[A-Za-z0-9_]\.|\[[ xX]\])[ \t]+[^\r\n]{1,200}
(?:(?:\r?\n[ \t]{2,5}(?:[-*+•]|[0-9]{1,3}\.|[A-Za-z0-9_]\.|\[[ xX]\])[ \t]+[^\r\n]{1,200}){0,5}
(?:\r?\n[ \t]{4,7}(?:[-*+•]|[0-9]{1,3}\.|[A-Za-z0-9_]\.|\[[ xX]\])[ \t]+[^\r\n]{1,200}){0,5})?)
|
# 3. Block quotes (including nested quotes and citations, up to three levels)
(?:(?:^>(?:>|\s{2,}){0,2}[^\r\n]{0,200}\r?\n?){1,10})
|
# 4. Code blocks (fenced, indented, or HTML pre/code tags)
(?:(?:^|\r?\n)(?:```|~~~)(?:[A-Za-z0-9_]{0,20})?\r?\n[\s\S]{0,1000}?(?:```|~~~)\r?\n?
|(?:(?:^|\r?\n)(?:[ ]{4}|\t)[^\r\n]{0,200}(?:\r?\n(?:[ ]{4}|\t)[^\r\n]{0,200}){0,20}\r?\n?)
|(?:<pre>(?:<code>)?[\s\S]{0,1000}?(?:</code>)?</pre>))
|
# 5. Tables (Markdown, grid tables, and HTML tables)
(?:(?:^|\r?\n)(?:\|[^\r\n]{0,200}\|(?:\r?\n\|[-:]{1,200}\|){0,1}(?:\r?\n\|[^\r\n]{0,200}\|){0,20}
|<table>[\s\S]{0,2000}?</table>))
|
# 6. Horizontal rules
(?:^(?:[-*_]){3,}\s*$|<hr\s*/?>)
|
# 7. Sentences or phrases ending with punctuation
(?:[^\r\n]{1,300}(?:[.!?…]|\.{3}|[…⁇-⁉]|[\p{Emoji_Presentation}\p{Extended_Pictographic}])(?:\s+|\r?\n|\Z))
|
# 8. Quoted text, parenthetical phrases, or bracketed content
(?:"""[^\r\n]{0,300}"""
|['"`‘’“”][^\r\n]{0,300}['"`‘’“”]
|\([^\r\n()]{0,200}(?:\([^\r\n()]{0,200}\)[^\r\n()]{0,200}){0,5}\)
|\[[^\r\n\[\]]{0,200}(?:\[[^\r\n\[\]]{0,200}\][^\r\n\[\]]{0,200}){0,5}\]
|\$[^\r\n$]{0,100}\$
|`[^`\r\n]{0,100}`)
|
# 9. Paragraphs
(?:(?<=\r?\n\r?\n|\A)(?:<p>)?(?:(?!\r?\n\r?\n|\Z)[^\n\r\u2028\u2029]){1,1000}(?:</p>)?(?=\r?\n\r?\n|\Z))
|
# 10. Standalone lines or phrases
(?:^(?:<[a-zA-Z][^>]{0,99}>)?[^\r\n]{1,200}(?:</[a-zA-Z]+>)?(?:\r?\n|$))
|
# 11. HTML-like tags and their content
(?:<[a-zA-Z][^>]{0,99}(?:>[\s\S]{0,1000}?</[a-zA-Z]+>|\s*/>))
|
# 12. LaTeX-style math expressions
(?:(?:\$\$[\s\S]{0,500}?\$\$)|(?:\$[^$\r\n]{0,100}\$))
|
# 13. Emoji and special characters
(?:[\p{Emoji_Presentation}\p{Extended_Pictographic}\p{So}☀-⛿\U0001F300-\U0001F6FF\U0001F700-\U0001F77F\U0001F780-\U0001F7FF\U0001F800-\U0001F8FF])
|
# 14. Fallback for any remaining content
(?:[^\r\n]{1,200})
'''

CHUNK_RE = regex.compile(PATTERN, regex.MULTILINE | regex.VERBOSE)


def chunk_text(text: str) -> list[str]:
    """Split ``text`` into semantic chunks (non-overlapping, in order)."""
    return [m.group(0) for m in CHUNK_RE.finditer(text)]


def main(argv: list[str]) -> int:
    text = open(argv[0], encoding="utf-8").read() if argv else sys.stdin.read()
    for i, chunk in enumerate(chunk_text(text), 1):
        print(f"--- chunk {i} ({len(chunk)} chars) ---")
        print(chunk)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
