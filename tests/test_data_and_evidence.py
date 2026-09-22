"""Tests for the split, the class mapping and the evidence package (no network needed).

The split test needs artifacts/messages.parquet and artifacts/sender_split.csv (created by scripts/01_prepare_data.py from the
dataset in data/) and is skipped when they are missing; the other two tests only need the files in models/.

Run: python tests/test_data_and_evidence.py
"""
import json
import sys
from pathlib import Path
from unittest import SkipTest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import pandas as pd

from veremi_xai.config import RAW_FEATURES, load_config, path
from veremi_xai.pipeline import Pipeline


def test_split_is_sender_disjoint_and_stratified():
    if not (path("parquet").exists() and path("split").exists()):
        raise SkipTest("artifacts/ is missing: place the dataset in data/ and run scripts/01_prepare_data.py")
    split = pd.read_csv(path("split"))
    msgs = pd.read_parquet(path("parquet"), columns=["sender", "class"])
    per_sender = msgs.groupby("sender")["class"].nunique()
    assert (per_sender == 1).all(), "a sender has more than one class"
    assert split.sender.is_unique and len(split) == msgs.sender.nunique() == 24663
    assert set(split.sender) == set(msgs.sender)
    frac = split.groupby(["class", "split"]).size().unstack(fill_value=0)
    assert (frac > 0).all().all(), "a class is missing in a partition"
    share = frac.div(frac.sum(axis=1), axis=0)
    assert (share["train"].sub(0.70).abs() < 0.02).all() and (share["test"].sub(0.15).abs() < 0.02).all()
    part = msgs.merge(split[["sender", "split"]], on="sender")
    assert part.groupby("sender")["split"].nunique().max() == 1      # every message of a sender is in exactly one partition


def test_class_mapping_is_the_verified_one():
    classes = load_config()["classes"]
    assert sorted(classes) == list(range(20))
    assert classes[13]["name"] == "DoS" and classes[9]["name"] == "Eventual stop" and classes[16]["name"] == "Grid Sybil"
    assert {k for k, v in classes.items() if v["group"] == "fault"} == {1, 2, 3, 4, 5, 6, 7, 8, 12}
    assert {k for k, v in classes.items() if v["group"] == "genuine"} == {0}


def test_evidence_schema_and_no_ground_truth():
    pipe = Pipeline.load()
    demo = pd.read_parquet(path("final") / "demo_messages.parquet")
    m = demo[demo.in_demo_pick & (demo["class"] == 5) & (demo.pred == 5)].iloc[10]
    cols = ["sendTime", "sender", "senderPseudo", "messageID", "class"] + RAW_FEATURES
    hist = demo.loc[demo.senderPseudo == m.senderPseudo, cols]
    ev = pipe.analyse(hist, m.messageID)["evidence"]
    assert set(ev) == {"schema_version", "message", "prediction", "shap", "transmitted_values", "model", "limitations", "evidence_hash"}
    assert set(ev["message"]) == {"messageID", "sendTime_s", "senderPseudo", "has_history"}       # no true sender id, no label
    keys = set()
    def walk(o):
        if isinstance(o, dict):
            for k, v in o.items():
                keys.add(k); walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)
    walk(ev)
    assert not keys & {"class", "label", "true_class", "sender", "ground_truth"}, keys      # the label / true sender id never enter the evidence
    assert ev["prediction"]["class_name"] == "Constant speed"
    assert len(ev["shap"]["top_contributions"]) == 6
    assert pipe.analyse(hist, m.messageID)["evidence"]["evidence_hash"] == ev["evidence_hash"]     # deterministic


if __name__ == "__main__":
    try:
        test_split_is_sender_disjoint_and_stratified()
    except SkipTest as e:
        print("SKIPPED (split test):", e)
    test_class_mapping_is_the_verified_one(); test_evidence_schema_and_no_ground_truth()
    print("all data/evidence tests passed")
