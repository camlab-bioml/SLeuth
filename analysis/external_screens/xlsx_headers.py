#!/usr/bin/env python3
"""Print the first rows of every sheet of an .xlsx file using only the
standard library (zipfile + xml.etree iterparse), so no openpyxl is needed.

Usage: python -I xlsx_headers.py FILE.xlsx [N_ROWS]
"""
import re
import sys
import zipfile
import xml.etree.ElementTree as ET

NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
RNS = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"


def shared_strings(z):
    try:
        data = z.open("xl/sharedStrings.xml")
    except KeyError:
        return []
    out = []
    # iterparse to avoid loading everything; stop after many strings is fine
    for ev, el in ET.iterparse(data, events=("end",)):
        if el.tag == NS + "si":
            out.append("".join(t.text or "" for t in el.iter(NS + "t")))
            el.clear()
    return out


def col_index(ref):
    m = re.match(r"([A-Z]+)", ref)
    s = m.group(1)
    n = 0
    for ch in s:
        n = n * 26 + (ord(ch) - 64)
    return n - 1


def sheet_map(z):
    wb = ET.fromstring(z.read("xl/workbook.xml"))
    rels = ET.fromstring(z.read("xl/_rels/workbook.xml.rels"))
    rid2target = {r.get("Id"): r.get("Target") for r in rels}
    out = []
    for s in wb.iter(NS + "sheet"):
        t = rid2target[s.get(RNS + "id")]
        t = t.lstrip("/")
        if not t.startswith("xl/"):
            t = "xl/" + t
        out.append((s.get("name"), t))
    return out


def first_rows(z, path, sst, n_rows):
    rows = []
    with z.open(path) as fh:
        for ev, el in ET.iterparse(fh, events=("end",)):
            if el.tag == NS + "row":
                cells = {}
                for c in el.iter(NS + "c"):
                    v = c.find(NS + "v")
                    t = c.get("t")
                    if t == "s" and v is not None:
                        val = sst[int(v.text)]
                    elif t == "inlineStr":
                        val = "".join(x.text or "" for x in c.iter(NS + "t"))
                    else:
                        val = v.text if v is not None else None
                    cells[col_index(c.get("r"))] = val
                width = max(cells) + 1 if cells else 0
                rows.append([cells.get(i) for i in range(width)])
                el.clear()
                if len(rows) >= n_rows:
                    break
    return rows


def main():
    path = sys.argv[1]
    n_rows = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    with zipfile.ZipFile(path) as z:
        sst = shared_strings(z)
        for name, target in sheet_map(z):
            print(f"== sheet {name!r} ({target})")
            for i, r in enumerate(first_rows(z, target, sst, n_rows)):
                print(f"  row {i + 1}: {r}")


if __name__ == "__main__":
    main()
