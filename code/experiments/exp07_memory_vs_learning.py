"""실험 07: 기억 트랙 4-1b — 언어 과제에서 "한 번 보고 기억" vs "반복 학습".

질문: 경험이 적을 때 해마형 기억(원샷)이 학습 모델보다 빨리 배우는가? 경험이 많을 때는 어디까지 가는가?

모델 (학습 데이터 n개, 테스트 5000개, 시드 5개)
  기억 (1회 통과, 역전파·반복 없음)
    mem2k   DG 2000뉴런, k=40, 시냅스 스케일링 판독
    mem8k   DG 8000뉴런, k=20, 시냅스 스케일링 판독
    mem8k_shuf  학습 라벨을 섞은 대조
  학습
    grid_1pass   BioGrid C4(공명+확률판독+허브+residual, 폭 4), 데이터 1회만 통과
    grid_full    같은 모델, 최대 15에폭 + 검증 조기 종료
    mlp64_1pass / mlp64_full / mlp16_full   역전파 MLP
    logreg_full  로지스틱 회귀
  참고
    ngram   n-gram 표 (완벽한 외우기 = 기억의 천장)
학습 모델의 조기 종료에는 검증 세트(5000개)를 쓴다 — 기억 모델은 검증 세트를 쓰지 않으므로 학습 모델에 유리한 조건이다.

실행: python3 experiments/exp07_memory_vs_learning.py
결과: results/exp07_memory_vs_learning.{json,md}
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import copy
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data.language import make_task  # noqa: E402
from src.baselines.models import MLP, accuracy, ngram_table_predict  # noqa: E402
from src.biogrid.screen import BioGridX  # noqa: E402
from src.memory.hippocampus import HippocampalMemory  # noqa: E402

SIZES = [50, 100, 300, 1000, 3000, 10000, 20000]
SEEDS = [0, 1, 2, 3, 4]
TASKS = ["content", "tag12"]
GRID_FEATS = {"resonance": {}, "prob_readout": {}, "hub": {}, "residual": {}}
GRID_LR = {"content": 3.0, "tag12": 1.0}           # 실험 05에서 C4 폭 4에 선택된 학습률
RULE = dict(output_signal="zone_target", hidden_credit="relay_w", gate="sensitivity")
MODELS = ["mem2k", "mem8k", "mem8k_shuf", "ngram", "logreg_full", "mlp16_full", "mlp64_1pass", "mlp64_full",
          "grid_1pass", "grid_full"]


def run_grid(d, seed, task, epochs):
    g = BioGridX(39, d["n_classes"], height=40, width=4, seed=seed, features=GRID_FEATS)
    rng = np.random.RandomState(seed + 100)
    best, best_acc, wait = None, -1, 0
    for _ in range(epochs):
        o = rng.permutation(len(d["y_train"]))
        for i in range(0, len(o), 32):
            idx = o[i:i + 32]
            g.train_step(d["X_train"][idx], d["y_train"][idx], GRID_LR[task], **RULE)
        va = accuracy(g.predict(d["X_val"]), d["y_val"])
        if va > best_acc + 1e-4:
            best, best_acc, wait = copy.deepcopy(g), va, 0
        else:
            wait += 1
            if wait >= 4:
                break
    return accuracy(best.predict(d["X_test"]), d["y_test"]), best.n_params()


def job(args):
    task, n, seed, model = args
    d = make_task(task, seed=seed, n_train=n)
    t0 = time.time()
    nc = d["n_classes"]
    if model.startswith("mem"):
        ndg, k = (2000, 40) if model == "mem2k" else (8000, 20)
        m = HippocampalMemory(39, nc, n_dg=ndg, k=k, seed=seed)
        y = np.random.RandomState(1000 + seed).permutation(d["y_train"]) if model.endswith("shuf") else d["y_train"]
        m.store(d["X_train"], y)
        acc, npar = accuracy(m.predict(d["X_test"]), d["y_test"]), m.n_params()
    elif model == "ngram":
        acc, npar = accuracy(ngram_table_predict(d["ctx_train"], d["y_train"], d["ctx_test"], nc), d["y_test"]), None
    elif model.startswith(("mlp", "logreg")):
        hid = {"logreg_full": 0, "mlp16_full": 16}.get(model, 64)
        m = MLP(39, hid, nc, seed)
        m.fit(d["X_train"], d["y_train"], d["X_val"], d["y_val"], epochs=1 if model.endswith("1pass") else 60)
        acc, npar = accuracy(m.predict(d["X_test"]), d["y_test"]), m.n_params()
    else:
        acc, npar = run_grid(d, seed, task, 1 if model == "grid_1pass" else 15)
    return {"task": task, "n": n, "seed": seed, "model": model, "test": acc, "n_params": npar,
            "sec": round(time.time() - t0, 1)}


def main():
    jobs = [(t, n, s, m) for t in TASKS for n in SIZES for s in SEEDS for m in MODELS]
    jobs.sort(key=lambda j: -("grid" in j[3]) * j[1])   # 오래 걸리는 것부터
    rows = []
    with Pool(4) as pool:
        for r in pool.imap_unordered(job, jobs, chunksize=1):
            rows.append(r)
            print(r["task"], r["n"], r["model"], r["seed"], round(r["test"] * 100, 2), f"{r['sec']}s", flush=True)
    lines = ["# 실험 07: 한 번 보고 기억 vs 반복 학습 (기억 트랙 4-1b)", "",
             "학습 데이터 n개에 따른 테스트 정확도 (시드 5개 평균). mem = 해마형 기억(원샷, 1회 통과).",
             "1pass = 데이터를 한 번만 봄, full = 반복 학습 + 검증 조기 종료.", ""]
    summary = {}
    for t in TASKS:
        lines += [f"## {t}", "", "| 모델 | 파라미터 | " + " | ".join(f"n={n}" for n in SIZES) + " |",
                  "|---|---|" + "---|" * len(SIZES)]
        for m in MODELS:
            cells, npar = [], None
            for n in SIZES:
                v = [r["test"] * 100 for r in rows if r["task"] == t and r["n"] == n and r["model"] == m]
                npar = next((r["n_params"] for r in rows if r["task"] == t and r["model"] == m), None)
                summary[f"{t}/{m}/{n}"] = [float(np.mean(v)), float(np.std(v))]
                cells.append(f"{np.mean(v):.1f}")
            lines.append(f"| {m} | {npar if npar is not None else '–'} | " + " | ".join(cells) + " |")
        lines.append("")
    (ROOT / "results" / "exp07_memory_vs_learning.json").write_text(json.dumps(
        {"sizes": SIZES, "seeds": SEEDS, "models": MODELS, "rows": rows, "summary": summary}, indent=2, ensure_ascii=False))
    (ROOT / "results" / "exp07_memory_vs_learning.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
