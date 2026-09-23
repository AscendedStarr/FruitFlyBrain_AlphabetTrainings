"""Smoke test for flybrain.docread. Stdlib only except the PDF case."""

import base64
import io
import zipfile

# Repo root on sys.path, so this script still finds the package from scripts/.
import os as _os
import sys as _sys

_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

from flybrain import docread


def line(t):
    print()
    print("=" * 60)
    print(t)
    print("=" * 60)


def show(d):
    print("kind      :", d["kind"])
    print("chars     :", d["chars"], " readable:", d["readable"])
    print("counts    :", d["counts"])
    print("pages     :", len(d["pages"]))
    for p in d["pages"]:
        print("   %-14s [%d:%d]" % (p["label"], p["start"], p["end"]))
    print("unreadable:", "".join(d["unreadable_chars"]))
    print("stream    :", repr(d["stream"][:110]))


# ------------------------------------------------------------------ text
line("plain text")
raw = (b"Hello, world! It is 2026.\nNumbers like 42 and 7.\n\n"
       b"Accents: caf\xc3\xa9 na\xc3\xafve. Hex 0xDEAD.\n")
d = docread.build("sample.txt", raw)
show(d)
df = docread.build("sample.txt", raw, fold=True)
print("folded    :", df["counts"], repr(df["stream"][:110]))
print("folded n_unreadable:", df["n_unreadable"], "folded flag:", df["folded"])


# ------------------------------------------------------------------ epub
line("epub")
buf = io.BytesIO()
with zipfile.ZipFile(buf, "w") as z:
    z.writestr("mimetype", "application/epub+zip")
    z.writestr(
        "META-INF/container.xml",
        '<?xml version="1.0"?><container version="1.0" '
        'xmlns="urn:oasis:names:tc:opendocument:xmlns:container">'
        '<rootfiles><rootfile full-path="OEBPS/content.opf" '
        'media-type="application/oebps-package+xml"/></rootfiles></container>')
    # NOTE: id comes after href here on purpose - attribute order in a real OPF
    # is arbitrary and the parser must not depend on it.
    z.writestr(
        "OEBPS/content.opf",
        '<?xml version="1.0"?><package '
        'xmlns="http://www.idpf.org/2007/opf" version="3.0">'
        '<metadata xmlns:dc="http://purl.org/dc/elements/1.1/">'
        '<dc:title>A Tiny Book</dc:title></metadata><manifest>'
        '<item href="ch1.xhtml" media-type="application/xhtml+xml" id="c1"/>'
        '<item href="ch2.xhtml" id="c2" media-type="application/xhtml+xml"/>'
        '</manifest><spine><itemref idref="c1"/><itemref idref="c2"/></spine>'
        '</package>')
    z.writestr("OEBPS/ch1.xhtml",
               "<html><head><style>p{color:red}</style><title>x</title></head>"
               "<body><h1>Chapter One</h1>"
               "<p>It was a dark and stormy night; the rain fell.</p>"
               "<p>Prices rose 3.5% in 1998 &amp; never came back.</p>"
               "</body></html>")
    z.writestr("OEBPS/ch2.xhtml",
               "<html><body><h1>Chapter Two</h1>"
               "<p>The fly read one letter at a time.</p></body></html>")
e = docread.build("tiny.epub", buf.getvalue())
show(e)

line("base64 upload path (what the browser sends)")
rt = docread.from_upload("tiny.epub", base64.b64encode(buf.getvalue()).decode())
print("same chars:", rt["chars"] == e["chars"], rt["chars"])
print("same stream:", rt["stream"] == e["stream"])


# ------------------------------------------------------------------ failures
line("refusals")
for name, blob in [("broken.pdf", b"%PDF-1.7\nthis is not a pdf"),
                   ("empty.txt", b""),
                   ("scan.pdf", b"%PDF-1.4\n" + b"\x00" * 400)]:
    try:
        docread.build(name, blob)
        print("%-12s NOT REFUSED" % name)
    except RuntimeError as exc:
        print("%-12s refused: %s" % (name, str(exc)[:90]))
