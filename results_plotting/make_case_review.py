#!/usr/bin/env python3
"""Case review page for the discordant scans, made and used on Berzelius (open it in Firefox inside ThinLinc).

One row per discordant scan (discordant_by_confidence.py cases.csv, test and external set), confident ones first.
Per row: a link that opens the scan's case viewer (make_case_viewer.py) in a new tab, and the review questions as
drop-downs / text fields. The scan's numbers (model probability, SUVRs, amyloid, reader votes, ...) stay hidden until
'show numbers' is ticked for that row, so you can look first and judge blind.

Answers are kept in Firefox (local storage of the page, on Berzelius) as you type, and 'Save answers' writes them to a
CSV in Firefox's download folder on Berzelius (~/Downloads). 'Load answers' reads such a CSV back (e.g. on another
day, or after the page was rebuilt). --answers fills the page from a CSV when it is built (same column names, matched
on Set + ID), e.g. an earlier review exported from a spreadsheet.

Contains IDs and participant-level data: keep the page and the answer CSVs on Berzelius.

Usage (Berzelius login node, ai-pet):
    python results_plotting/make_case_review.py \\
        --cases "test=$RUN/figures/discordant_by_confidence/cases.csv" \\
                "AVID=$RUN/validation/figures/discordant_by_confidence/cases.csv" \\
        --viewers "test=$V/viewers_discordant_test" "AVID=$V/viewers_discordant_avid" \\
        [--answers earlier_review.csv] --out "$V/case_review.html"
"""
import argparse
import html
import json
from pathlib import Path

import numpy as np
import pandas as pd

QUESTIONS = [  # (column name, choices or None for free text)
    ("1 Where is the tau", None),
    ("1b Side (L / R / both)", ["L", "R", "both", "more L", "more R"]),
    ("2 Symmetric? (yes / no)", ["yes", "no"]),
    ("3 Off-target signal", ["yes", "slight", "no"]),
    ("4 Image quality / FOV", None),
    ("5 Would I call it positive? (yes / no / unsure)", ["yes", "no", "unsure"]),
    ("6 Occlusion evidence on the tau? (yes / partly / no)", ["yes", "partly", "no"]),
    ("Notes", None),
]
BAND_ORDER = {"Confident": 0, "Moderately confident": 1, "Uncertain": 2}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cases", nargs="+", required=True, help='"Set=cases.csv" per set')
    p.add_argument("--viewers", nargs="+", default=[], help='"Set=folder with <ID>_viewer.html" per set')
    p.add_argument("--answers", default=None, help="CSV with earlier answers (columns Set, ID + the question columns).")
    p.add_argument("--out", required=True)
    return p.parse_args()


def numbers(r) -> dict:
    """What is shown after 'show numbers' (descriptive, from cases.csv)."""
    ai = r.get("MetaTemporal_absAI", np.nan)
    li = ai * 50 if pd.notna(ai) and ai < 1 else ai  # older per-scan tables held |L - R| / mean = |LI| / 50
    fmt = lambda v, d=2: "" if pd.isna(v) else f"{v:.{d}f}"
    return {"Model P(positive)": fmt(r.get("prob"), 3), "Meta-temporal SUVR": fmt(r.get("MetaTemporal")),
            "Temporoparietal SUVR": fmt(r.get("TemporoParietal")), "Mesial temporal SUVR": fmt(r.get("MesialTemporal")),
            "Frontal SUVR": fmt(r.get("Frontal")), "|LI| % (meta-temporal)": fmt(li, 1),
            "Amyloid": {1: "positive", 0: "negative"}.get(r.get("amyloid_status"), ""),
            "Centiloid": fmt(r.get("CL"), 0), "Diagnosis": "" if pd.isna(r.get("dx")) else str(r.get("dx")),
            "Site": "" if pd.isna(r.get("site")) else str(r.get("site")),
            "Readers": "" if pd.isna(r.get("positive_votes")) else f"{int(r['positive_votes'])} of {int(r['n_readers'])} positive",
            "Reader grade": "" if pd.isna(r.get("read category")) else str(r.get("read category")),
            "% evidence meta-temporal": "" if pd.isna(r.get("evidence share Meta-temporal")) else
            f"{100 * r['evidence share Meta-temporal']:.0f}%"}


def main():
    args = parse_args()
    out = Path(args.out).resolve()
    viewers = {k: Path(v).resolve() for k, v in (x.split("=", 1) for x in args.viewers)}
    rows = []
    for spec in args.cases:
        st, f = spec.split("=", 1)
        c = pd.read_csv(f)
        c["Set"] = st
        rows.append(c)
    d = pd.concat(rows, ignore_index=True)
    d["_b"] = d["band"].map(BAND_ORDER).fillna(3)
    d["_s"] = (d["Set"] != d["Set"].iloc[0]).astype(int)
    d = d.sort_values(["_s", "_b", "direction", "prob"]).reset_index(drop=True)

    prior = {}
    if args.answers:
        a = pd.read_csv(args.answers, encoding="utf-8-sig").fillna("")
        for _, r in a.iterrows():
            prior[f"{r['Set']}|{int(r['ID'])}"] = {q: str(r[q]) for q, _ in QUESTIONS if q in a.columns and str(r[q])}

    cases, missing = [], 0
    for i, r in d.iterrows():
        f = viewers.get(r["Set"], Path("/nonexistent")) / f"{int(r['ID'])}_viewer.html"
        missing += not f.exists()
        cases.append({"n": i + 1, "set": r["Set"], "id": int(r["ID"]), "group": r["direction"], "band": r["band"],
                      "viewer": f.as_uri() if f.exists() else "", "numbers": numbers(r),
                      "prior": prior.get(f"{r['Set']}|{int(r['ID'])}", {})})
    page = TEMPLATE.replace("__CASES__", json.dumps(cases)).replace("__QUESTIONS__", json.dumps(QUESTIONS)) \
        .replace("__KEY__", html.escape(out.stem))
    out.write_text(page, encoding="utf-8")
    print(f"[save] {out}: {len(cases)} scans ({missing} without a viewer), {len(prior)} earlier answers loaded")
    print("Open it in Firefox inside ThinLinc:  firefox " + str(out) + " &")


TEMPLATE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Case review</title>
<style>
  body { font: 14px system-ui, sans-serif; margin: 20px; color: #111; }
  h2 { margin: 0 0 6px; color: #003466; }
  .top { display: flex; gap: 10px; align-items: center; margin-bottom: 12px; flex-wrap: wrap; }
  button, label.file { background: #2b6cb0; color: #fff; border: 0; border-radius: 4px; padding: 5px 12px; cursor: pointer; font: inherit; }
  label.file input { display: none; }
  #status { color: #666; }
  table { border-collapse: collapse; }
  th, td { padding: 4px 6px; border-bottom: 1px solid #ddd; text-align: left; vertical-align: top; }
  th { background: #003466; color: #fff; position: sticky; top: 0; font-weight: 600; font-size: 12px; }
  tr.Confident td { background: #fbecef; }
  td:nth-child(4), td:nth-child(5) { white-space: nowrap; }
  select, input[type=text] { font: inherit; font-size: 13px; }
  input[type=text] { width: 150px; }
  td.nums { font-size: 12px; color: #333; min-width: 220px; }
  td.nums div { white-space: nowrap; }
  a { color: #2b6cb0; font-weight: 600; }
  .miss { color: #999; }
</style></head><body>
<h2>Discordant case review</h2>
<div class="top">
  <button id="save">Save answers (CSV)</button>
  <label class="file">Load answers (CSV)<input type="file" id="load" accept=".csv"></label>
  <span id="status"></span>
</div>
<p style="color:#666;margin-top:0">Answers are kept in this browser as you type. Look at the scan first, then tick
'show numbers'. Contains IDs: keep on Berzelius.</p>
<table id="t"></table>
<script>
const cases = __CASES__, questions = __QUESTIONS__, KEY = "case_review:__KEY__";
let answers = {};
try { answers = JSON.parse(localStorage.getItem(KEY) || "{}"); } catch (e) {}
for (const c of cases) {  // answers given when the page was built fill only what is still empty
  const k = c.set + "|" + c.id; answers[k] = Object.assign({}, c.prior, answers[k] || {});
}
const store = () => { try { localStorage.setItem(KEY, JSON.stringify(answers)); } catch (e) {} };
store();
const esc = s => String(s).replace(/[&<>"]/g, ch => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;"}[ch]));
const t = document.getElementById("t");
t.innerHTML = "<tr><th>#</th><th>Set</th><th>ID</th><th>Group</th><th>Band</th><th>Viewer</th>" +
  questions.map(q => `<th>${esc(q[0])}</th>`).join("") + "<th>Numbers</th></tr>" +
  cases.map(c => {
    const k = c.set + "|" + c.id, a = answers[k] || {};
    const cells = questions.map(([q, ch]) => ch
      ? `<td><select data-k="${esc(k)}" data-q="${esc(q)}"><option></option>` +
        ch.map(o => `<option${a[q] === o ? " selected" : ""}>${esc(o)}</option>`).join("") + "</select></td>"
      : `<td><input type="text" data-k="${esc(k)}" data-q="${esc(q)}" value="${esc(a[q] || "")}"></td>`).join("");
    const nums = Object.entries(c.numbers).map(([n, v]) => `<div>${esc(n)}: <b>${esc(v)}</b></div>`).join("");
    return `<tr class="${esc(c.band.split(" ")[0])}"><td>${c.n}</td><td>${esc(c.set)}</td><td>${c.id}</td>` +
      `<td>${esc(c.group)}</td><td>${esc(c.band)}</td>` +
      `<td>${c.viewer ? `<a href="${esc(c.viewer)}" target="_blank">open</a>` : '<span class="miss">no viewer</span>'}</td>` +
      cells + `<td class="nums"><label><input type="checkbox" class="show"> show numbers</label>` +
      `<div class="n" hidden>${nums}</div></td></tr>`;
  }).join("");
t.addEventListener("input", e => {
  const el = e.target;
  if (el.classList.contains("show")) { el.closest("td").querySelector(".n").hidden = !el.checked; return; }
  if (!el.dataset.k) return;
  (answers[el.dataset.k] = answers[el.dataset.k] || {})[el.dataset.q] = el.value;
  store(); document.getElementById("status").textContent = "kept in this browser";
});
const csvCell = v => /[",\n]/.test(v) ? '"' + v.replace(/"/g, '""') + '"' : v;
document.getElementById("save").onclick = () => {
  const head = ["#", "Set", "ID", "Group", "Band", ...questions.map(q => q[0])];
  const lines = [head.map(csvCell).join(",")].concat(cases.map(c => {
    const a = answers[c.set + "|" + c.id] || {};
    return [String(c.n), c.set, String(c.id), c.group, c.band, ...questions.map(q => a[q[0]] || "")].map(csvCell).join(",");
  }));
  const blob = new Blob(["﻿" + lines.join("\n")], {type: "text/csv"});
  const a = document.createElement("a"); a.href = URL.createObjectURL(blob);
  a.download = "case_review_answers_" + new Date().toISOString().slice(0, 10) + ".csv"; a.click();
  document.getElementById("status").textContent = "saved to Firefox's download folder (on Berzelius)";
};
function parseCSV(text) {  // small CSV reader (quotes, commas, newlines inside quotes)
  const rows = []; let row = [], cell = "", q = false;
  for (let i = 0; i < text.length; i++) {
    const ch = text[i];
    if (q) { if (ch === '"') { if (text[i + 1] === '"') { cell += '"'; i++; } else q = false; } else cell += ch; }
    else if (ch === '"') q = true; else if (ch === ",") { row.push(cell); cell = ""; }
    else if (ch === "\n" || ch === "\r") { if (ch === "\r" && text[i + 1] === "\n") i++; row.push(cell); rows.push(row); row = []; cell = ""; }
    else cell += ch;
  }
  if (cell || row.length) { row.push(cell); rows.push(row); }
  return rows;
}
document.getElementById("load").onchange = e => {
  const f = e.target.files[0]; if (!f) return;
  f.text().then(text => {
    const rows = parseCSV(text.replace(/^﻿/, "")), head = rows[0], iS = head.indexOf("Set"), iI = head.indexOf("ID");
    let n = 0;
    for (const r of rows.slice(1)) {
      if (r.length < head.length) continue;
      const k = r[iS] + "|" + r[iI], a = answers[k] = answers[k] || {};
      questions.forEach(([q]) => { const j = head.indexOf(q); if (j >= 0 && r[j]) { a[q] = r[j]; n++; } });
    }
    store(); location.reload();
  });
};
</script></body></html>
"""

if __name__ == "__main__":
    main()
