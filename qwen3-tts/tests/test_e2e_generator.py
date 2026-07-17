"""End-to-end Generator test — run with: .venv/bin/python tests/test_e2e_generator.py

Part 1 (fast, no chat model): repair ladder, failure continuation and queue
mechanics with a stubbed generate_text.
Part 2 (real models): builds a source collection, then runs a REAL 5-word
queue through Qwen3 4B with pause/resume, skip, restart-resume, duplicate
handling and export checks. Takes several minutes on an M2.

All data lives in temp folders whose paths contain spaces.
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from modules import storage  # noqa: E402

_TMP = Path(tempfile.mkdtemp(prefix="e2e gen test "))  # spaces on purpose
(_TMP / "collections dir").mkdir()
(_TMP / "projects dir").mkdir()
storage.collections_dir = lambda: _TMP / "collections dir"
storage.generator_dir = lambda: _TMP / "projects dir"

from modules import collections_store as cs  # noqa: E402
from modules import gen_projects as gp  # noqa: E402
from modules import generator_engine as ge  # noqa: E402

FAILURES = []
WORDS = ["爱", "喜欢", "学习", "工作", "朋友"]


def check(name, cond, detail=""):
    if cond:
        print(f"  ok  {name}")
    else:
        print(f"FAIL  {name}  {detail}")
        FAILURES.append(name)


def results_file_is_valid(pid) -> bool:
    """The on-disk file itself must parse as a JSON array at any moment."""
    try:
        return isinstance(
            json.loads(gp._pfile(pid, "results.json").read_text(encoding="utf-8")), list)
    except (OSError, ValueError):
        return False


EXAMPLE = {"hanzi": "词", "pinyin": "cí", "hsk": 1, "meaning_en": "word",
           "meaning_vi": "từ", "example_zh": "这是一个词。"}
REQUIRED = ["hanzi", "pinyin", "meaning_en", "meaning_vi"]


def make_project(name, collections, **overrides) -> str:
    pid = gp.create_project(name, "e2e test")
    cfg = gp.get_config(pid)
    cfg.update(
        collections=collections,
        schema_mode="example", example_object=EXAMPLE, required_keys=REQUIRED,
        max_tokens=1024, temperature=0.2, use_rerank=False, chunks_per_word=6,
        max_retries=3, context_budget=16384,
    )
    cfg.update(overrides)
    gp.save_config(pid, cfg)
    return pid


# --------------------------------------------------------------------------- #
# Part 1 — stubbed model: repair ladder & failure continuation
# --------------------------------------------------------------------------- #
def test_repair_and_failures_stubbed():
    pid = make_project("Stub Project", [], only_sources=False,
                       allow_model_knowledge=True)
    real_generate = ge.generate_text

    def install(script):
        calls = {"n": 0}

        def fake(system, user, cfg, cancel_event=None, max_tokens_override=None):
            i = min(calls["n"], len(script) - 1)
            calls["n"] += 1
            return script[i], "stop"
        ge.generate_text = fake
        return calls

    try:
        good = json.dumps({"hanzi": "测试", "pinyin": "cèshì", "meaning_en": "test",
                           "meaning_vi": "thử"}, ensure_ascii=False)

        # trailing comma -> deterministic repair, no extra model calls
        calls = install(['{"hanzi": "测试", "pinyin": "cèshì", '
                         '"meaning_en": "test", "meaning_vi": "thử",}'])
        out = ge.process_word(pid, gp.get_config(pid), {"hanzi": "测试", "id": 1}, 1, 1)
        check("deterministic repair accepted", out["status"] == "completed"
              and calls["n"] == 1, out["errors"])

        # garbage first, valid on regenerate
        gp.load_results(pid)
        calls = install(["I cannot produce that.", good])
        out = ge.process_word(pid, gp.get_config(pid), {"hanzi": "测试", "id": 1}, 1, 1)
        check("regenerate after garbage", out["status"] == "completed"
              and calls["n"] == 2, (out["errors"], calls["n"]))

        # missing required field -> model repair call fixes it
        incomplete = json.dumps({"hanzi": "测试", "pinyin": "cèshì",
                                 "meaning_en": "test"}, ensure_ascii=False)
        calls = install([incomplete, good])
        out = ge.process_word(pid, gp.get_config(pid), {"hanzi": "测试", "id": 1}, 1, 1)
        check("model repair path", out["status"] == "completed" and calls["n"] == 2,
              (out.get("errors"), calls["n"]))

        # always garbage -> bounded retries, failure recorded, queue continues
        calls = install(["nonsense"])
        gp.enqueue(pid, [{"hanzi": "废话"}, {"hanzi": "继续"}], allow_duplicates=True)
        runner = ge.QueueRunner()
        runner.start(pid)
        for _ in range(200):
            if runner.snapshot()["status"] == "idle":
                break
            time.sleep(0.1)
        counts = gp.queue_counts(pid)
        check("bounded retries", calls["n"] <= 8, calls["n"])
        check("failures recorded, queue continued",
              counts["failed"] == 2 and counts["queued"] == 0, counts)
        fails = gp.load_failures(pid)
        check("failure has raw + errors", fails and fails[0]["raw_response"]
              and fails[0]["errors"], fails)

        # truncated output -> incomplete, marked failed with clear message
        calls = install(['{"hanzi": "测试", "pinyin": "cè'])
        out = ge.process_word(pid, gp.get_config(pid), {"hanzi": "测试", "id": 9}, 1, 1)
        check("truncation flagged incomplete", out["status"] == "failed"
              and any("cut off" in e for e in out["errors"]), out["errors"])
    finally:
        ge.generate_text = real_generate
    gp.delete_project(pid)


# --------------------------------------------------------------------------- #
# Part 2 — real models
# --------------------------------------------------------------------------- #
def build_test_collection():
    d = _TMP / "fixture files"
    d.mkdir(exist_ok=True)
    txt = d / "hsk words.txt"
    txt.write_text(
        "爱 ài — verb: to love; tình yêu / yêu. HSK level 1. 例句: 我爱我的家。\n"
        "喜欢 xǐhuan — verb: to like; thích. HSK level 1. 例句: 他喜欢喝茶。\n"
        "学习 xuéxí — verb: to study; học tập. HSK level 1. 例句: 我们在学校学习中文。\n"
        "工作 gōngzuò — verb/noun: to work, job; làm việc / công việc. HSK level 1. "
        "例句: 我爸爸在医院工作。\n"
        "朋友 péngyou — noun: friend; bạn bè. HSK level 1. 例句: 他是我最好的朋友。\n",
        encoding="utf-8")
    csvf = d / "hsk list.csv"
    csvf.write_text(
        "hanzi,pinyin,hsk,meaning_en,meaning_vi\n"
        "爱,ài,1,to love,yêu\n喜欢,xǐhuan,1,to like,thích\n学习,xuéxí,1,to study,học tập\n"
        "工作,gōngzuò,1,to work; job,làm việc\n朋友,péngyou,1,friend,bạn bè\n",
        encoding="utf-8")
    cs.create_collection("HSK sources", "e2e")
    cs.add_files("HSK sources", [str(txt), str(csvf)])
    print("  building index (loads Qwen3-Embedding 0.6B)…")
    cs.rebuild_index("HSK sources")
    check("collection indexed", cs.get_meta("HSK sources")["indexed"])

    res = cs.search(["HSK sources"], ["爱"], top_k=4, use_rerank=True, exact_word="爱")
    check("retrieval with reranker works", res and res[0]["exact"]
          and "rerank_score" in res[0], res[:1])


def test_real_queue():
    pid = make_project("HSK e2e", ["HSK sources"])
    rep = gp.enqueue(pid, [{"hanzi": w} for w in WORDS])
    check("5 words queued", rep["queued_total"] == 5)

    runner = ge.QueueRunner()
    print("  starting real 5-word queue (loads Qwen3 4B — takes a few minutes)…")
    msg = runner.start(pid)
    check("queue started", "started" in msg, msg)

    paused_ok = skipped = False
    t0 = time.time()
    valid_every_poll = True
    while time.time() - t0 < 900:
        snap = runner.snapshot()
        if not results_file_is_valid(pid):
            valid_every_poll = False
        counts = snap.get("counts", {})
        done = counts.get("completed", 0) + counts.get("failed", 0) + \
            counts.get("skipped", 0)
        if not paused_ok and counts.get("completed", 0) >= 1 and \
                snap["status"] == "running":
            runner.pause()
            for _ in range(600):
                st = runner.snapshot()["status"]
                if st in ("paused", "idle"):
                    break
                time.sleep(0.5)
            paused_ok = runner.snapshot()["status"] in ("paused", "idle")
            print(f"  paused after {counts.get('completed')} word(s) → resume")
            runner.resume()
        if not skipped and snap["status"] == "running" and done >= 2 and \
                snap.get("current_word"):
            runner.skip_current()
            skipped = True
            print(f"  skip requested during {snap.get('current_word')}")
        if snap["status"] == "idle":
            break
        time.sleep(1.0)

    counts = gp.queue_counts(pid)
    print(f"  queue finished: {counts}")
    check("pause/resume worked", paused_ok)
    check("results.json valid at every poll", valid_every_poll)
    check("no words lost", counts["total"] == 5 and counts["queued"] == 0, counts)

    if counts["skipped"] or counts["failed"]:
        n = gp.reset_items(pid, statuses=("skipped", "failed"))
        print(f"  reprocessing {n} skipped/failed item(s)…")
        runner2 = ge.QueueRunner()  # fresh runner = restart-resume from disk state
        runner2.start(pid)
        t0 = time.time()
        while time.time() - t0 < 600 and runner2.snapshot()["status"] != "idle":
            time.sleep(1.0)
        counts = gp.queue_counts(pid)
    check("restart-resume completes all 5", counts["completed"] == 5, counts)

    results = gp.load_results(pid)
    got = {o.get("hanzi") for o in results}
    check("5 objects in results.json", got == set(WORDS), got)
    check("required fields present", all(
        all(o.get(k) for k in REQUIRED) for o in results),
        [(o.get("hanzi"), [k for k in REQUIRED if not o.get(k)]) for o in results])

    for w in WORDS:
        check_f = (gp._proj_dir(pid) / "objects" / f"{w}.json").exists()
        if not check_f:
            check(f"object file for {w}", False)
            break
    else:
        check("object file per word", True)
    check("sources saved per word", all(gp.sources_for(pid, w) for w in WORDS))
    check("evidence provenance kept", all(
        s.get("collection") == "HSK sources" and s.get("source")
        for s in gp.sources_for(pid, WORDS[0])))
    check("raw response kept", "RAW MODEL RESPONSE" in gp.load_raw(pid, WORDS[0]))
    check("backups exist", list((gp._proj_dir(pid) / "backups").glob("results.*")))

    # duplicate handling on a real regenerate: merge must not create a 2nd 爱
    before = len(gp.load_results(pid))
    gp.enqueue(pid, [{"hanzi": "爱"}], allow_duplicates=True)
    runner3 = ge.QueueRunner()
    runner3.start(pid)
    t0 = time.time()
    while time.time() - t0 < 300 and runner3.snapshot()["status"] != "idle":
        time.sleep(1.0)
    results = gp.load_results(pid)
    check("duplicate merged, not appended",
          len(results) == before and
          len([o for o in results if o.get("hanzi") == "爱"]) == 1,
          len(results))

    # token ceiling 16,384 accepted by the real generate path
    cfg = gp.get_config(pid)
    cfg["max_tokens"] = 16384
    text, finish = ge.generate_text(
        "Reply with exactly one JSON object.",
        'Return exactly this JSON object: {"ok": true}', cfg)
    check("16384-token setting accepted", finish == "stop" and "ok" in text,
          (finish, text[:80]))

    # exports
    for fmt in ("json", "jsonl", "csv", "zip"):
        p = gp.export_results(pid, fmt)
        check(f"export {fmt}", p.exists() and p.stat().st_size > 0)

    print("  sample generated entry:")
    print("  " + json.dumps(results[0], ensure_ascii=False)[:300])


if __name__ == "__main__":
    try:
        print("[part 1 — stubbed model]")
        test_repair_and_failures_stubbed()
        print("[part 2 — real models]")
        build_test_collection()
        test_real_queue()
    finally:
        shutil.rmtree(_TMP, ignore_errors=True)
    print()
    if FAILURES:
        print(f"{len(FAILURES)} FAILURES: {FAILURES}")
        sys.exit(1)
    print("All end-to-end generator tests passed.")
