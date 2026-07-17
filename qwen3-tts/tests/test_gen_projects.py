"""Tests for modules/gen_projects.py — run with:
.venv/bin/python tests/test_gen_projects.py

Model-free: covers project CRUD, word-list parsing, queue persistence,
crash-safe results with duplicate handling, recovery and exports.
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from modules import storage  # noqa: E402

_TMP = Path(tempfile.mkdtemp(prefix="gen test "))  # space in path on purpose
storage.generator_dir = lambda: _TMP

from modules import gen_projects as gp  # noqa: E402

FAILURES = []


def check(name, cond, detail=""):
    if cond:
        print(f"  ok  {name}")
    else:
        print(f"FAIL  {name}  {detail}")
        FAILURES.append(name)


def test_project_crud():
    pid = gp.create_project("HSK Dictionary", "test project")
    check("create", gp.get_config(pid)["name"] == "HSK Dictionary")
    check("folders exist", (gp._proj_dir(pid) / "objects").is_dir()
          and (gp._proj_dir(pid) / "logs").is_dir())
    check("results starts as []", gp.load_results(pid) == [])

    gp.rename_project(pid, "HSK Dictionary v2")
    check("rename keeps folder", gp.get_config(pid)["name"] == "HSK Dictionary v2"
          and gp._proj_dir(pid).name == pid)

    cfg = gp.get_config(pid)
    cfg["temperature"] = 0.5
    cfg["collections"] = ["HSK word lists"]
    gp.save_config(pid, cfg)
    dup = gp.duplicate_project(pid, "HSK Copy")
    dcfg = gp.get_config(dup)
    check("duplicate copies settings", dcfg["temperature"] == 0.5
          and dcfg["collections"] == ["HSK word lists"])
    check("duplicate has fresh queue", gp.load_queue(dup) == [])

    exported = gp.export_config(pid)
    new_pid = gp.import_config(str(exported), "Imported HSK")
    check("export/import round-trip", gp.get_config(new_pid)["temperature"] == 0.5)

    msg = gp.delete_project(dup)
    check("delete", "Deleted" in msg and gp.get_config(dup) is None)
    return pid


def test_word_input(pid):
    rows = gp.rows_from_text("爱\n喜欢, 学习、工作；朋友\n\n  爱  ")
    words = [r["hanzi"] for r in rows]
    check("text parsing with CJK separators", words == ["爱", "喜欢", "学习", "工作", "朋友", "爱"])

    report = gp.analyze_rows(rows, existing_words=set())
    check("duplicate in input detected", report["duplicate_in_input"] == ["爱"])
    check("order preserved", [r["hanzi"] for r in report["valid"]] ==
          ["爱", "喜欢", "学习", "工作", "朋友"])

    with tempfile.TemporaryDirectory() as td:
        csvf = Path(td) / "words.csv"
        csvf.write_text("hanzi,pinyin,hsk_new,notes\n爱,ài,1,core word\n喜欢,xǐhuan,1,\n",
                        encoding="utf-8")
        rows = gp.rows_from_file(str(csvf))
        check("csv with columns", rows[0] == {"hanzi": "爱", "pinyin": "ài",
                                              "hsk_new": "1", "notes": "core word"})
        jsonf = Path(td) / "words.json"
        jsonf.write_text(json.dumps(["爱", {"hanzi": "喜欢", "topic": "feelings"}],
                                    ensure_ascii=False), encoding="utf-8")
        rows = gp.rows_from_file(str(jsonf))
        check("json list styles", rows == [{"hanzi": "爱"},
                                           {"hanzi": "喜欢", "topic": "feelings"}])
        try:
            gp.rows_from_file(str(Path(td) / "x.xlsx"))
            check("xlsx rejected clearly", False)
        except ValueError as exc:
            check("xlsx rejected clearly", "openpyxl" in str(exc))


def test_queue(pid):
    report = gp.enqueue(pid, gp.rows_from_text("爱\n喜欢\n学习"))
    check("enqueue", report["queued_total"] == 3)
    report = gp.enqueue(pid, gp.rows_from_text("爱\n工作"))
    check("queue dedupes vs queued", report["duplicate_existing"] == ["爱"]
          and report["queued_total"] == 4)

    q = gp.load_queue(pid)
    ids = [i["id"] for i in q]
    gp.reorder_queue(pid, list(reversed(ids)))
    check("reorder", [i["id"] for i in gp.load_queue(pid)] == list(reversed(ids)))
    gp.reorder_queue(pid, ids)

    removed = gp.remove_from_queue(pid, [ids[-1]])
    check("remove from queue", removed == 1 and gp.queue_counts(pid)["total"] == 3)

    q = gp.load_queue(pid)
    q[0]["status"] = "failed"
    gp.save_queue(pid, q)
    n = gp.reset_items(pid, statuses=("failed",))
    check("retry failed resets", n == 1 and gp.queue_counts(pid)["failed"] == 0)


def test_results(pid):
    obj1 = {"hanzi": "爱", "pinyin": "ài", "senses": [{"en": "to love"}]}
    meta = {"generated_by_model": "qwen3-4b", "prompt_version": 1,
            "source_collection_ids": ["HSK word lists"]}
    out = gp.upsert_result(pid, "爱", obj1, meta)
    check("append", out == "appended" and gp.load_results(pid) == [obj1])
    check("object file saved", (gp._proj_dir(pid) / "objects" / "爱.json").exists())
    check("results file always valid json",
          isinstance(json.loads(gp._pfile(pid, "results.json").read_text(encoding="utf-8")), list))

    # merge: missing fields filled, existing kept
    out = gp.upsert_result(pid, "爱", {"hanzi": "爱", "pinyin": "SHOULD-NOT-WIN",
                                       "hsk": 1}, meta, mode="merge")
    got = gp.load_results(pid)[0]
    check("merge keeps existing", out == "merged" and got["pinyin"] == "ài")
    check("merge fills missing", got["hsk"] == 1)

    out = gp.upsert_result(pid, "爱", {"hanzi": "爱", "pinyin": "x"}, meta, mode="skip")
    check("skip", out == "skipped" and gp.load_results(pid)[0]["pinyin"] == "ài")

    out = gp.upsert_result(pid, "爱", {"hanzi": "爱", "pinyin": "ĀI2"}, meta, mode="replace")
    check("replace", out == "replaced" and gp.load_results(pid)[0]["pinyin"] == "ĀI2")

    # approved entries are protected from replace
    gp.set_review_status(pid, "爱", "approved")
    out = gp.upsert_result(pid, "爱", {"hanzi": "爱", "pinyin": "no"}, meta, mode="replace")
    check("approved protected from replace", out == "review_candidate"
          and gp.load_results(pid)[0]["pinyin"] == "ĀI2")
    notes = gp.load_review_notes(pid, "爱")
    check("review note recorded", any(n["type"] == "duplicate" for n in notes))

    gp.upsert_result(pid, "喜欢", {"hanzi": "喜欢", "pinyin": "xǐhuan"}, meta)
    check("metadata outside objects",
          "created_at" not in gp.load_results(pid)[0]
          and gp.get_meta_for(pid, "爱").get("created_at"))

    # corruption recovery: backup, then object rebuild
    gp._pfile(pid, "results.json").write_text('[{"hanzi": "爱"', encoding="utf-8")
    recovered = gp.load_results(pid)
    check("auto-restore from backup", any(o.get("hanzi") == "爱" for o in recovered))

    gp._pfile(pid, "results.json").write_text("garbage", encoding="utf-8")
    # wipe backups so the object rebuild path is exercised
    for b in (gp._proj_dir(pid) / "backups").glob("*"):
        b.unlink()
    msg = gp.rebuild_results_from_objects(pid)
    words = {o["hanzi"] for o in gp.load_results(pid)}
    check("rebuild from object files", {"爱", "喜欢"} <= words, msg)

    report = gp.consistency_check(pid)
    check("consistency check runs", "results.json" in report)


def test_failures_sources_logs(pid):
    gp.add_failure(pid, "工作", ["invalid JSON: x"], "raw model text", attempts=3)
    check("failure saved", gp.load_failures(pid)[0]["hanzi"] == "工作")
    gp.clear_failure(pid, "工作")
    check("failure cleared", gp.load_failures(pid) == [])

    gp.save_sources_used(pid, "爱", [{"source": "hsk.txt", "collection": "HSK",
                                      "score": 0.9, "text": "爱 ài love"}])
    check("sources round-trip", gp.sources_for(pid, "爱")[0]["source"] == "hsk.txt")

    gp.log(pid, "word_done", "爱 completed", seconds=12.5)
    tail = gp.read_log_tail(pid)
    check("log tail", tail and tail[-1]["kind"] == "word_done")

    gp.save_progress(pid, {"current": "爱", "completed": 1})
    check("progress checkpoint", gp.load_progress(pid)["completed"] == 1)


def test_exports(pid):
    p_json = gp.export_results(pid, "json")
    p_jsonl = gp.export_results(pid, "jsonl")
    p_csv = gp.export_results(pid, "csv")
    p_zip = gp.export_results(pid, "zip")
    check("json export", json.loads(p_json.read_text(encoding="utf-8")))
    check("jsonl export lines", len(p_jsonl.read_text(encoding="utf-8").splitlines()) >= 2)
    csv_text = p_csv.read_text(encoding="utf-8-sig")
    check("csv export flattened", "hanzi" in csv_text and "爱" in csv_text)
    with zipfile.ZipFile(p_zip) as z:
        names = set(z.namelist())
    check("zip bundle contents", {"results.json", "config.json"} <= names, names)


if __name__ == "__main__":
    try:
        print("[project crud]")
        pid = test_project_crud()
        print("[word input]")
        test_word_input(pid)
        print("[queue]")
        test_queue(pid)
        print("[results + recovery]")
        test_results(pid)
        print("[failures/sources/logs]")
        test_failures_sources_logs(pid)
        print("[exports]")
        test_exports(pid)
    finally:
        shutil.rmtree(_TMP, ignore_errors=True)
    print()
    if FAILURES:
        print(f"{len(FAILURES)} FAILURES: {FAILURES}")
        sys.exit(1)
    print("All gen_projects tests passed.")
