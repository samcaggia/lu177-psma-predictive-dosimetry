#!/usr/bin/env python3
"""Lu-177 S-values of the ICRP 110 adult male from OpenDose (public form, CC BY-SA 4.0, research use), a few sources, 3 s between requests.

www.opendose.org redirects to opendose.polsl.pl. Writes data/processed/psma/mc_pilot/opendose_lu177_icrp110am.csv: S in mGy/(MBq s) as published.
"""
import http.cookiejar
import re
import time
import urllib.parse
import urllib.request
from html.parser import HTMLParser
from pathlib import Path

import pandas as pd

BASE = "https://opendose.polsl.pl/svalues/"
OUT = Path(__file__).resolve().parents[2] / "data/processed/psma/mc_pilot/opendose_lu177_icrp110am.csv"
PHANTOM = "2"
SOURCES = ["RKidney + LKidney", "Liver", "Spleen", "Salivary glands", "Urinary bladder contents", "Urinary bladder wall", "Small intestine wall",
           "Stomach wall", "Colon", "Total body except organ contents"]
HEADERS = {"User-Agent": "Mozilla/5.0"}


class Rows(HTMLParser):
    def __init__(self):
        super().__init__()
        self.rows, self.cell, self.row = [], None, None

    def handle_starttag(self, tag, attrs):
        if tag == "tr":
            self.row = []
        elif tag in ("td", "th"):
            self.cell = []

    def handle_endtag(self, tag):
        if tag in ("td", "th") and self.row is not None and self.cell is not None:
            self.row.append(" ".join("".join(self.cell).split()))
            self.cell = None
        elif tag == "tr" and self.row is not None:
            self.rows.append(self.row)
            self.row = None

    def handle_data(self, data):
        if self.cell is not None:
            self.cell.append(data)


def options(page, name):
    block = re.search(rf'<select name="{name}".*?</select>', page, flags=re.S).group(0)
    return {text.strip(): value for value, text in re.findall(r'<option value="(\d+)"[^>]*>([^<]+)<', block)}


def main():
    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar()))
    def get(url, data=None, referer=None):
        headers = dict(HEADERS, **({"Referer": referer} if referer else {}))
        return opener.open(urllib.request.Request(url, data=data, headers=headers), timeout=120).read().decode("utf-8", "replace")
    page = get(BASE)
    token = re.search(r'name="csrfmiddlewaretoken" value="([^"]+)"', page).group(1)
    regions = options(page, "selected_source_region")
    isotope = next(v for k, v in options(page, "selected_isotope").items() if k.startswith("Lu-177 "))
    frames = []
    for source in SOURCES:
        time.sleep(3)
        body = urllib.parse.urlencode(dict(csrfmiddlewaretoken=token, selected_phantom=PHANTOM, selected_source_region=regions[source], selected_isotope=isotope)).encode()
        parser = Rows()
        parser.feed(get(BASE + "select_pair/", body, BASE))
        rows = [r[-4:] for r in parser.rows if len(r) >= 4 and re.fullmatch(r"[\d.]+e[+-]\d+", r[-3])]
        frame = pd.DataFrame(rows, columns=["target", "s_mgy_per_mbq_s", "standard_error", "target_mass_g"])
        frame.insert(0, "source", source)
        frames.append(frame)
        print(source, len(frame), flush=True)
    table = pd.concat(frames, ignore_index=True)
    for c in table.columns[2:]:
        table[c] = table[c].astype(float)
    table.to_csv(OUT, index=False)
    print(OUT, len(table))


if __name__ == "__main__":
    main()
