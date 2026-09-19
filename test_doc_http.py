"""End-to-end check of the document path over HTTP against a live server.

Drives the same three calls the browser makes: parse, then present each
character in order, then report what the fly did with it. Prints the stream the
fly was actually shown, so "stepped over" and "shown but unanswerable" are
visibly different things rather than a claim in a comment.
"""

import base64
import io
import json
import urllib.error
import urllib.request
import zipfile

BASE = "http://127.0.0.1:8000"


def get(path):
    return json.load(urllib.request.urlopen(BASE + path, timeout=600))


def post(path, body):
    req = urllib.request.Request(
        BASE + path, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    try:
        return json.load(urllib.request.urlopen(req, timeout=600))
    except urllib.error.HTTPError as exc:
        # The server answers a bad document with 400 and a JSON reason; that is
        # a result to print, not a crash.
        return json.loads(exc.read() or b"{}")


# The fly's own glyph table, straight from the server: 37 entries - A-Z, 0-9 and
# space. Membership in this set is the only correct test for "can the fly be
# shown this character".
GLYPHS = set(get("/api/state")["glyphs"])


def tiny_epub() -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("mimetype", "application/epub+zip")
        z.writestr("META-INF/container.xml",
                   '<?xml version="1.0"?><container version="1.0" '
                   'xmlns="urn:oasis:names:tc:opendocument:xmlns:container">'
                   '<rootfiles><rootfile full-path="OEBPS/content.opf" '
                   'media-type="application/oebps-package+xml"/>'
                   '</rootfiles></container>')
        z.writestr("OEBPS/content.opf",
                   '<?xml version="1.0"?><package '
                   'xmlns="http://www.idpf.org/2007/opf" version="3.0">'
                   '<metadata xmlns:dc="http://purl.org/dc/elements/1.1/">'
                   '<dc:title>Tiny</dc:title></metadata><manifest>'
                   '<item id="c1" href="ch1.xhtml" '
                   'media-type="application/xhtml+xml"/></manifest>'
                   '<spine><itemref idref="c1"/></spine></package>')
        z.writestr("OEBPS/ch1.xhtml",
                   "<html><body><h1>Page One</h1>"
                   "<p>The quick brown fox jumps over the lazy dog. "
                   "It was 1998, and the price was 3.5%.</p></body></html>")
    return buf.getvalue()


def run(label, name, raw, fold=False, limit=None):
    print()
    print("=" * 68)
    print(label)
    print("=" * 68)
    d = post("/api/document", {
        "name": name, "data": base64.b64encode(raw).decode(), "fold": fold})

    if "error" in d:
        print("REFUSED:", d["error"])
        return

    print("title    :", d["title"], "(", d["kind"], ")")
    print("chars    :", d["chars"], " readable:", d["readable"])
    print("counts   :", d["counts"])
    print("pages    :", [(p["label"], p["start"], p["end"]) for p in d["pages"]])
    print("no glyph :", "".join(d["unreadable_chars"]))

    stream = d["stream"]
    n = min(len(stream), limit or len(stream))
    shown, skipped = [], []
    tally = {"hit": 0, "miss": 0, "unknown": 0, "skip": 0}

    for i in range(n):
        ch = stream[i]
        up = ch.upper()
        # The browser only POSTs a character if it is a key in the server's own
        # glyph table. That table is the ASCII set, so this must NOT be
        # `str.isalnum()`, which is true for 'e-acute' and would post a
        # character the fly has no bitmap for.
        if up not in GLYPHS:
            tally["skip"] += 1
            skipped.append(ch)
            continue
        r = post("/api/present", {"char": up, "learn": False})
        if r.get("known") is False:
            tally["unknown"] += 1
            shown.append(f"{up}->(no cell, said {r['predicted']!r})")
        elif r.get("correct"):
            tally["hit"] += 1
            shown.append(f"{up}->{r['predicted']}")
        else:
            tally["miss"] += 1
            shown.append(f"{up}->{r['predicted']}!")

    print()
    print(f"first {n} characters of the stream:")
    print("  ", repr(stream[:n]))
    print()
    print("what the fly did:")
    print("  ", " ".join(shown))
    print()
    print("stepped over (no glyph, never presented):", "".join(sorted(set(skipped))))
    print("tally:", tally)
    served = len(shown)
    if served:
        print(f"read correctly {tally['hit']}/{served} shown "
              f"= {100 * tally['hit'] / served:.1f}%")


run("plain text, with punctuation and accents",
    "sample.txt",
    b"Hello, world! It is 2026. Caf\xc3\xa9 na\xc3\xafve.\n")

run("the same text with unreadable characters folded to spaces",
    "sample.txt",
    b"Hello, world! It is 2026. Caf\xc3\xa9 na\xc3\xafve.\n", fold=True)

run("a real EPUB", "tiny.epub", tiny_epub())

run("a PDF that has no text layer",
    "scan.pdf", b"%PDF-1.4\n" + b"\x00" * 512)

run("an upload that is over the size limit",
    "big.txt", b"a" * (41 * 1024 * 1024), limit=1)
