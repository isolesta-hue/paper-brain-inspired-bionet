"""실험 13: 격자 학습의 시드 간 불안정성 줄이기.

동기 (실험 09·11): wtag에서 같은 설정의 격자가 시드에 따라 42.1~45.2%로 흔들린다 (표준편차 1.1, MLP16은 0.7).
에폭마다 검증 정확도도 크게 오르내려서 조기 종료 시점이 들쭉날쭉하다 ([5, 3, 5, 6, 3]).

후보 (모두 C4 = 공명 + 확률 판독 + 허브 + residual, 폭 4, 높이 40 위에 하나씩)
  참고 자료에서 "실패하는 구성을 안정화"한 기록(△)이 있는 것
    homeostasis   N5  발화율이 목표(0.2)보다 높으면 문턱↑ (LayerNorm의 생물학 대응)
    gol_thresh    S4  이웃 발화가 많으면 문턱↓, 적으면 ↑
    v_share       S7  문턱을 위아래 이웃과 조금씩 평균
    lr_decay      M5  임계기 감쇠: 에폭마다 학습률 × 0.7 (발달기에는 크게, 이후 작게 배움)
  새 후보 (참고 자료에 없음 → REFERENCE에 추가)
    slow_avg      느린 시냅스: 빠른 가중치는 지금처럼 배우고, 판정에는 그 장기 평균(지수 평균, 0.999/배치)을 쓴다
                  (해마-피질의 "빠른 학습 / 느린 정착"을 시냅스 하나 안에서. 조기 종료도 평균 가중치로 판단)
과제: tag12 (n=20,000), wtag (n=200,000). 시드 5개, 최대 15에폭 + 검증 조기 종료(참을성 3), 판독은 구역 판독.
지표: 평균, 표준편차, 최저 시드, base와 같은 시드 짝 차이.

실행: python3 experiments/exp13_stability.py
결과: results/exp13_stability.{json,md}
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
from src.baselines.models import accuracy  # noqa: E402
from src.biogrid.screen import BioGridX  # noqa: E402
from experiments.exp07_memory_vs_learning import GRID_FEATS, RULE  # noqa: E402
from experiments.exp11_learned_readout import load, TASKS as ALL_TASKS  # noqa: E402

SEEDS = [0, 1, 2, 3, 4]
TASKS = {t: ALL_TASKS[t] for t in ["tag12", "wtag"]}
VARIANTS = ["base", "homeostasis", "gol_thresh", "v_share", "lr_decay", "slow_avg"]
EPOCHS, PATIENCE, BATCH, H = 15, 3, 32, 40
LR_DECAY, EMA = 0.7, 0.999
PARAMS = ["W_in", "W_loc", "V", "W_hub", "W_res"]


def averaged(g, ema):
    a = copy.deepcopy(g)
    for k, v in ema.items():
        setattr(a, k, v.copy())
    return a


def run(task, seed, variant):
    d = load(task, seed)
    feats = dict(GRID_FEATS)
    if variant in ("homeostasis", "gol_thresh", "v_share"):
        feats[variant] = {}
    g = BioGridX(d["X_train"].shape[1], d["n_classes"], height=H, width=4, seed=seed, features=feats)
    ema = {k: getattr(g, k).copy() for k in PARAMS if hasattr(g, k)} if variant == "slow_avg" else None
    rng = np.random.RandomState(seed + 100)
    lr = TASKS[task][1]
    best, best_acc, wait, ep_used, curve = None, -1, 0, 0, []
    for ep in range(EPOCHS):
        o = rng.permutation(len(d["y_train"]))
        for i in range(0, len(o), BATCH):
            idx = o[i:i + BATCH]
            g.train_step(d["X_train"][idx], d["y_train"][idx], lr, **RULE)
            if ema is not None:
                for k in ema:
                    ema[k] *= EMA
                    ema[k] += (1 - EMA) * getattr(g, k)
        if variant == "lr_decay":
            lr *= LR_DECAY
        m = averaged(g, ema) if ema is not None else g
        va = accuracy(m.predict(d["X_val"]), d["y_val"])
        curve.append(round(va * 100, 2))
        if va > best_acc + 1e-4:
            best, best_acc, wait, ep_used = copy.deepcopy(m), va, 0, ep + 1
        else:
            wait += 1
            if wait >= PATIENCE:
                break
    return {"test": accuracy(best.predict(d["X_test"]), d["y_test"]), "val": best_acc, "epochs": ep_used,
            "val_curve": curve}


def job(a):
    task, seed, variant = a
    t0 = time.time()
    r = run(task, seed, variant)
    r.update(task=task, seed=seed, variant=variant, sec=round(time.time() - t0, 1))
    print(task, seed, variant, round(r["test"] * 100, 2), r["val_curve"], f"{r['sec']}s", flush=True)
    return r


def main():
    jobs = sorted([(t, s, v) for t in TASKS for s in SEEDS for v in VARIANTS], key=lambda j: -(j[0] == "wtag"))
    with Pool(4) as pool:
        rows = list(pool.imap_unordered(job, jobs, chunksize=1))
    lines = ["# 실험 13: 격자 학습 안정화", "",
             "C4(폭 4, 높이 40) + 후보 하나씩, 구역 판독. 테스트 정확도, 시드 5개. 짝 차이 = 같은 시드 base 대비 평균 ± 표준오차.",
             f"lr_decay = 에폭마다 × {LR_DECAY}, slow_avg = 배치마다 지수 평균 {EMA}.", ""]
    summary = {}
    for t in TASKS:
        lines += [f"## {t}", "", "| 방식 | 평균 | 표준편차 | 최저 | 최고 | base 대비 | 에폭 |", "|---|---|---|---|---|---|---|"]
        get = lambda v: sorted([r for r in rows if r["task"] == t and r["variant"] == v], key=lambda r: r["seed"])
        b = np.array([r["test"] * 100 for r in get("base")])
        for v in VARIANTS:
            rr = get(v)
            a = np.array([r["test"] * 100 for r in rr])
            dd = a - b
            summary[f"{t}/{v}"] = {"mean": a.mean(), "std": a.std(), "min": a.min(), "diff": dd.mean(),
                                  "diff_se": dd.std(ddof=1) / np.sqrt(len(dd))}
            diff = "–" if v == "base" else f"{dd.mean():+.2f} ± {dd.std(ddof=1) / np.sqrt(len(dd)):.2f}"
            lines.append(f"| {v} | {a.mean():.2f} | {a.std():.2f} | {a.min():.2f} | {a.max():.2f} | {diff} | "
                         f"{[r['epochs'] for r in rr]} |")
        lines.append("")
    (ROOT / "results" / "exp13_stability.json").write_text(json.dumps(
        {"seeds": SEEDS, "variants": VARIANTS, "summary": summary, "rows": rows}, indent=2, ensure_ascii=False,
        default=float))
    (ROOT / "results" / "exp13_stability.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
