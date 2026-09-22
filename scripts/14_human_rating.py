"""Step 14 - human rating of the generated reports (clarity / usefulness), blind and paired.

  make       python scripts/14_human_rating.py make [n_messages]        default 15 messages -> 2 x 15 = 30 items
             Takes messages that already have a report from the LLM writer AND from the template writer, shuffles the 2n texts and
             hides which writer produced which. Writes outputs/human_rating/rating_sheet.xlsx (give this to the rater) and
             outputs/human_rating/rating_key.csv (keep it away from the rater until the sheet is filled).
  summarize  python scripts/14_human_rating.py summarize <filled_sheet.xlsx>
             Joins the filled sheet with the key: mean ratings per writer, paired difference, share of "invented content" flags.

No API call is made: only stored reports are used.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import pandas as pd

from veremi_xai.config import class_names, load_config, path

OUT = Path(__file__).resolve().parents[1] / "outputs" / "human_rating"
LLM_WRITER = "groq_small"


def load(writer):
    recs = {}
    for line in open(path("reports") / f"reports_{writer}.jsonl", encoding="utf-8"):
        d = json.loads(line)
        recs[d["evidence"]["evidence_hash"]] = d
    return recs


def make(n):
    seed = load_config()["seed"]
    names = class_names()
    llm, tpl = load(LLM_WRITER), load("template")
    both = [h for h in llm if h in tpl and llm[h]["report"]["backend"] != "template"]        # a real LLM text, not a fallback
    rows = [{"hash": h, "true": int(llm[h]["true_class"]), "pred": llm[h]["evidence"]["prediction"]["class_name"],
             "level": llm[h]["evidence"]["prediction"]["confidence_level"]} for h in both]
    df = pd.DataFrame(rows)
    df["wrong"] = df.pred != df.true.map(names)
    rng = np.random.default_rng(seed)
    n_wrong = min(n // 3, int(df.wrong.sum()))
    pick = []
    for flag, k in ((True, n_wrong), (False, n - n_wrong)):                # a third misclassified messages, the rest correct; one per class first
        pool = df[df.wrong == flag].sample(frac=1, random_state=int(rng.integers(1 << 30)))
        pool = pd.concat([pool.drop_duplicates("true"), pool.loc[~pool.index.isin(pool.drop_duplicates("true").index)]])
        pick += list(pool.hash[:k])
    items = [(h, w, (llm if w == "llm" else tpl)[h]["report"]["text"]) for h in pick for w in ("llm", "template")]
    order = rng.permutation(len(items))
    sheet, key = [], []
    for i, j in enumerate(order, start=1):
        h, w, text = items[j]
        ev = llm[h]["evidence"]["prediction"]
        sheet.append({"item": i, "predicted_class": ev["class_name"], "confidence": f"{ev['confidence']:.4f} ({ev['confidence_level']})", "report": text,
                      "clarity_1to5": None, "usefulness_1to5": None, "anything_invented_or_unsupported_YN": None, "comment": None})
        key.append({"item": i, "writer": w, "evidence_hash": h, "true_class": names[int(llm[h]["true_class"])], "prediction_wrong": ev["class_name"] != names[int(llm[h]["true_class"])]})
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(key).to_csv(OUT / "rating_key.csv", index=False)
    write_xlsx(pd.DataFrame(sheet), OUT / "rating_sheet.xlsx")
    print(f"{len(sheet)} items ({n} messages x 2 writers, {n_wrong} of the messages are misclassified) -> {OUT}")


def write_xlsx(df, file):
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.worksheet.datavalidation import DataValidation

    wb = Workbook()
    ws = wb.active; ws.title = "instructions"
    text = ["How to rate (about 3 minutes per item)",
            "",
            "Each item is a report written for a security analyst about ONE vehicle message that a machine-learning model has classified.",
            "You do not need to judge whether the model's class is right; judge only the report text.",
            "Who or what wrote each text is hidden on purpose.",
            "",
            "clarity_1to5          1 = confusing, 5 = I understood everything on the first read",
            "usefulness_1to5       1 = I could not act on it, 5 = I know exactly what to check next",
            "anything_invented_or_unsupported_YN   Y if any statement looks made up or is not backed by the numbers in the report, else N",
            "comment               optional (what was unclear, what was missing)",
            "",
            "Use the sheet 'ratings'. Do not change the columns item / predicted_class / confidence / report."]
    for i, t in enumerate(text, start=1):
        ws.cell(i, 1, t)
    ws["A1"].font = Font(bold=True, size=13); ws.column_dimensions["A"].width = 130
    ws2 = wb.create_sheet("ratings")
    ws2.append(list(df.columns))
    for r in df.itertuples(index=False):
        ws2.append(list(r))
    for c, w in zip("ABCDEFGH", (6, 26, 20, 110, 14, 16, 22, 40)):
        ws2.column_dimensions[c].width = w
    for cell in ws2[1]:
        cell.font = Font(bold=True); cell.fill = PatternFill("solid", fgColor="DDEBF7"); cell.alignment = Alignment(wrap_text=True, vertical="center")
    for row in ws2.iter_rows(min_row=2):
        for cell in row:
            cell.alignment = Alignment(wrap_text=True, vertical="top")
        for cell in row[4:7]:
            cell.fill = PatternFill("solid", fgColor="FFF2CC")
    ws2.freeze_panes = "B2"
    n = len(df) + 1
    d15 = DataValidation(type="whole", operator="between", formula1=1, formula2=5, showErrorMessage=True, error="Enter a whole number from 1 to 5")
    dyn = DataValidation(type="list", formula1='"Y,N"', showErrorMessage=True, error="Enter Y or N")
    ws2.add_data_validation(d15); ws2.add_data_validation(dyn)
    d15.add(f"E2:F{n}"); dyn.add(f"G2:G{n}")
    wb.save(file)


def summarize(file):
    r = pd.read_excel(file, sheet_name="ratings")
    k = pd.read_csv(OUT / "rating_key.csv")
    d = r.merge(k, on="item")
    d = d.rename(columns={"anything_invented_or_unsupported_YN": "invented"})
    d["invented"] = d.invented.astype(str).str.strip().str.upper().eq("Y")
    print(f"rated items: {d.clarity_1to5.notna().sum()} of {len(d)}")
    g = d.groupby("writer").agg(items=("item", "count"), clarity=("clarity_1to5", "mean"), usefulness=("usefulness_1to5", "mean"), invented_flags=("invented", "sum")).round(2)
    print(g.to_string())
    p = d.pivot(index="evidence_hash", columns="writer", values=["clarity_1to5", "usefulness_1to5"])
    for m in ("clarity_1to5", "usefulness_1to5"):
        diff = (p[m]["llm"] - p[m]["template"]).dropna()
        print(f"{m}: mean paired difference (LLM - template) = {diff.mean():+.2f} over {len(diff)} messages; LLM better in {(diff > 0).sum()}, equal {(diff == 0).sum()}, worse {(diff < 0).sum()}")
    g.to_csv(OUT / "rating_summary.csv")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "make":
        make(int(sys.argv[2]) if len(sys.argv) > 2 else 15)
    elif len(sys.argv) > 2 and sys.argv[1] == "summarize":
        summarize(sys.argv[2])
    else:
        print(__doc__)
