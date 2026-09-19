"""Turn a document into a stream of characters the fly can actually be shown.

The fly has no eyes, no font renderer and no language model. It has 35
receptors, and the only way to put ink on them is `encoding.render`, which looks
a character up in a hand-drawn 5x7 bitmap font. So "can the fly read this
document" reduces to exactly one question:

    which characters in this file have a 5x7 glyph, and which do not?

For this project that answer is deliberately narrow:

    readable     A-Z  a-z  0-9  space
    not readable everything else - . , ! ? ; : ' " ( ) - / and every
                 accented letter, every quote mark, every dash, every newline

Digits are readable *as input and only as input*: FONT_DIGITS gives them a
bitmap on the receptor sheet, but CLASSES has no digit in it, so the fly can
never answer with one. That asymmetry is the project's negative control, and it
is why `classify` reports "digit" as its own case rather than lumping it with
the letters.

Nothing here is allowed to invent text. A PDF with no text layer produces no
text, an unreadable character stays unreadable, and neither is papered over with
a guess. `from_upload` returns the counts so the UI can say plainly how much of
the file the fly is actually able to be shown.

No third-party dependency is needed for text or EPUB (both are stdlib work).
PDF needs `pypdf` for the text layer; it is imported lazily so the rest of the
module keeps working without it.
"""

from __future__ import annotations

import base64
import binascii
import html
import io
import logging
import os
import re
import zipfile

# A document this large is a mistake, not a use case: the UI steps through the
# stream one character at a time, and the whole stream is sent to the browser in
# one JSON payload.
MAX_UPLOAD = 40 * 1024 * 1024      # raw bytes of the uploaded file
MAX_STREAM = 4 * 1024 * 1024       # characters after parsing
MAX_PAGES = 5000

_MB = 1024 * 1024

# The three buckets the UI and the counts are built from.
_READABLE = re.compile(r"[A-Za-z0-9 ]")

# Encoding candidates, most specific first. `latin-1` never fails, so it is the
# floor and there is no need for an errors= fallback below it.
_ENCODINGS = ("utf-8-sig", "utf-8", "cp1252", "latin-1")


def classify(ch: str) -> str:
    """One of 'letter', 'digit', 'space', 'unreadable'."""
    if ch == " ":
        return "space"
    if ch.isdigit():
        return "digit"
    if ch.isalpha():
        return "letter"
    return "unreadable"


def readable(ch: str) -> bool:
    """True if `render(ch)` would produce a bitmap rather than raise."""
    return bool(_READABLE.fullmatch(ch))


# --------------------------------------------------------------- containers

def decode_text(raw: bytes) -> str:
    for enc in _ENCODINGS:
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", "replace")


def _xhtml_to_text(blob: bytes) -> str:
    """Strip an EPUB content document down to its text."""
    s = decode_text(blob)
    # script/style/head can contain no visible text but plenty of angle brackets
    s = re.sub(r"<(script|style|head|title)\b.*?</\1\s*>", " ", s,
               flags=re.S | re.I)
    s = re.sub(r"<br\b[^>]*>", " ", s, flags=re.I)
    # a block boundary is a word boundary; the reading stream is one line, so
    # this only has to stop "end.Start" from becoming a single token
    s = re.sub(r"</(p|div|h[1-6]|li|tr|td|section|article|blockquote)\s*>", " ",
               s, flags=re.I)
    s = re.sub(r"<[^>]+>", " ", s)
    return html.unescape(s)


def read_plain(raw: bytes) -> list[tuple[str, str]]:
    return [("text", decode_text(raw))]


def read_pdf(raw: bytes) -> list[tuple[str, str]]:
    try:
        from pypdf import PdfReader
    except ImportError as exc:      # pragma: no cover - depends on the env
        raise RuntimeError(
            "pypdf is not installed, so PDF text cannot be extracted. "
            "Install it with:  "
            r".\.venv-flybrain\Scripts\python.exe -m pip install pypdf"
        ) from exc

    # pypdf logs recoverable damage ("EOF marker not found") straight to stderr
    # through its own logger. It is not an error we act on, and left alone it
    # interleaves with the server banner.
    logging.getLogger("pypdf").setLevel(logging.ERROR)

    try:
        reader = PdfReader(io.BytesIO(raw))
    except Exception as exc:        # noqa: BLE001 - pypdf raises many types
        raise RuntimeError(f"could not open the PDF: {exc}") from exc

    if reader.is_encrypted:
        try:
            if reader.decrypt("") == 0:
                raise RuntimeError("PDF is password-protected")
        except Exception as exc:    # noqa: BLE001
            raise RuntimeError("PDF is password-protected") from exc

    pages: list[tuple[str, str]] = []
    for i, page in enumerate(reader.pages):
        if i >= MAX_PAGES:
            break
        try:
            text = page.extract_text() or ""
        except Exception:           # noqa: BLE001 - one bad page is not fatal
            text = ""
        pages.append((f"page {i + 1}", text))
    return pages


def read_epub(raw: bytes) -> list[tuple[str, str]]:
    try:
        zf = zipfile.ZipFile(io.BytesIO(raw))
    except zipfile.BadZipFile as exc:
        raise RuntimeError("this EPUB is not a readable zip container") from exc

    with zf:
        names = {n.replace("\\", "/") for n in zf.namelist()}

        opf_path = None
        if "META-INF/container.xml" in names:
            container = decode_text(zf.read("META-INF/container.xml"))
            m = re.search(r'full-path\s*=\s*"([^"]+)"', container)
            if m:
                opf_path = m.group(1).replace("\\", "/")
        if not opf_path:
            candidates = [n for n in sorted(names) if n.lower().endswith(".opf")]
            if not candidates:
                raise RuntimeError("no OPF package found inside this EPUB")
            opf_path = candidates[0]

        opf = decode_text(zf.read(opf_path))
        base = os.path.dirname(opf_path)

        # Attribute order inside <item> is arbitrary, so id and href are pulled
        # out of each tag independently rather than matched as one pattern.
        manifest: dict[str, str] = {}
        for tag in re.findall(r"<item\b[^>]*>", opf):
            idm = re.search(r'\bid\s*=\s*"([^"]+)"', tag)
            href = re.search(r'\bhref\s*=\s*"([^"]+)"', tag)
            if idm and href:
                manifest[idm.group(1)] = href.group(1)

        spine = re.findall(r'<itemref\b[^>]*\bidref\s*=\s*"([^"]+)"', opf)
        hrefs = [manifest[i] for i in spine if i in manifest]
        if not hrefs:  # malformed spine: fall back to every manifest document
            hrefs = [h for h in manifest.values()
                     if h.lower().endswith((".xhtml", ".html", ".htm"))]

        pages: list[tuple[str, str]] = []
        for href in hrefs:
            if len(pages) >= MAX_PAGES:
                break
            rel = href.split("#", 1)[0]
            path = os.path.normpath(os.path.join(base, rel)).replace("\\", "/")
            if path not in names:
                continue
            pages.append((os.path.basename(path), _xhtml_to_text(zf.read(path))))
        return pages


_HANDLERS = {
    ".pdf": read_pdf,
    ".epub": read_epub,
}


# ------------------------------------------------------------------ assembly

def build(name: str, raw: bytes, fold: bool = False) -> dict:
    """Parse `raw` into a single reading stream plus its page map.

    `fold` replaces every unreadable character with a space, which is the only
    way to get a typical book to read as clean prose. With it off, the stream
    keeps the punctuation and the fly steps over it - which is the honest
    picture of what this model can do with a real book.
    """
    if len(raw) > MAX_UPLOAD:
        raise RuntimeError(
            f"{len(raw) / _MB:.1f} MB is over the {MAX_UPLOAD // _MB} MB limit"
        )
    if not raw:
        raise RuntimeError("the file is empty")

    ext = os.path.splitext(name or "")[1].lower()
    handler = _HANDLERS.get(ext, None)

    if handler is not None:
        pages = handler(raw)
        kind = ext.lstrip(".")
    elif ext in (".txt", ".text", ".md", ".markdown", ".csv", ".log", ".rst"):
        pages, kind = read_plain(raw), "text"
    else:
        # Sniff, because a browser will happily hand over a file with no
        # extension and guessing wrong should not look like a broken document.
        head = raw[:8]
        if head.startswith(b"%PDF-"):
            pages, kind = read_pdf(raw), "pdf"
        elif head.startswith(b"PK\x03\x04"):
            pages, kind = read_epub(raw), "epub"
        else:
            pages, kind = read_plain(raw), "text"

    stream = ""
    spans: list[dict] = []
    for label, text in pages:
        # Every whitespace run becomes one space. The stream is a single line so
        # that a character index is a character index; position in the document
        # is carried by the page map below, not by newlines in the text.
        clean = " ".join(text.split())
        if not clean:
            continue
        if stream and not stream.endswith(" "):
            stream += " "
        start = len(stream)
        stream += clean
        spans.append({"label": label, "start": start, "end": len(stream)})

    if not stream:
        raise RuntimeError(
            "no text came out of this file. A scanned PDF has a page image and "
            "no text layer, and pypdf cannot read pixels - run OCR on it first."
        )
    if len(stream) > MAX_STREAM:
        raise RuntimeError(
            f"the text is {len(stream) / _MB:.1f} M characters, over the "
            f"{MAX_STREAM // _MB} M limit"
        )

    counts = {"letter": 0, "digit": 0, "space": 0, "unreadable": 0}
    for ch in stream:
        counts[classify(ch)] += 1
    unreadable_chars = sorted({c for c in stream if not readable(c)})
    n_unreadable = counts["unreadable"]

    if fold and n_unreadable:
        stream = "".join(c if readable(c) else " " for c in stream)
        counts["space"] += n_unreadable
        counts["unreadable"] = 0

    return {
        "title": os.path.basename(name) if name else "document",
        "kind": kind,
        "stream": stream,
        "pages": spans,
        "chars": len(stream),
        "counts": counts,
        "readable": len(stream) - counts["unreadable"],
        "unreadable_chars": unreadable_chars[:48],
        "n_unreadable": counts["unreadable"],
        "folded": bool(fold and n_unreadable),
    }


def from_upload(name: str, b64: str, fold: bool = False) -> dict:
    """`build`, for the browser's base64 JSON POST."""
    # Reject on the encoded length first: decoding a 200 MB string only to then
    # reject it wastes a lot of memory for no reason. base64 is 4 bytes of
    # output per 3 bytes of input, plus padding.
    if len(b64) > (MAX_UPLOAD // 3 + 1) * 4 + 1024:
        raise RuntimeError(
            f"the upload is over the {MAX_UPLOAD // _MB} MB limit"
        )
    try:
        raw = base64.b64decode(b64, validate=False)
    except (binascii.Error, ValueError) as exc:
        raise RuntimeError(f"the upload is not valid base64: {exc}") from exc
    return build(name, raw, fold)
