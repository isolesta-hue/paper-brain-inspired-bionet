"""실험 11: 학습되는 지역 판독 (REFERENCE R4) — 격자의 손실이 판독에서 생기는지 확인하고 고친다.

동기 (실험 09 후 진단, wtag 높이 40):
  학습된 격자의 각 열 발화 확률 위에 선형 탐침(로지스틱)을 붙이면 46.0% (마지막 열 45.7%) ≈ MLP16 46.2%.
  격자 자체 판독(행 구역 합, R1)은 44.6%. 무작위 격자 + 탐침은 마지막 열 24.2% (기준선).
  → 격자 내부는 MLP 수준 표현을 배웠고, 고정된 구역 합 판독이 손실을 낸다 (R1의 알려진 문제: 클래스별 기본값·가중치 없음).

판독 뉴런 (클래스마다 하나)
  마지막 열의 발화 확률 p(H개) → 판독 뉴런 z = W_out·p + b → softmax
  학습: 델타 규칙 ΔW_out = lr · (정답 one-hot − softmax) ⊗ p  (판독 뉴런이 아는 값만 쓰는 지역 규칙, 역전파 없음)
  격자 내부 학습은 지금까지와 같다 (구역 목표 + 바톤 터치 + 민감도 문). 판독은 격자로 신호를 보내지 않는다.

비교 (같은 시드·데이터·에폭)
  grid          격자 C4 + 구역 판독 (지금까지)
  grid+R4       같은 격자 + 학습되는 판독 (격자와 함께 온라인 학습)
  random+R4     학습하지 않는 무작위 격자 + 같은 판독 (reservoir 대조: 판독만으로 얼마나 나오는가)
  shuf+R4       grid+R4를 학습·검증 라벨을 섞어서 (셔플 대조)
과제: content, tag12 (n=20,000, 폭 4, 높이 40), wtag (n=200,000, 높이 40). 최대 15에폭 + 검증 조기 종료(참을성 3).
격자 학습률은 이전에 고른 값 (content 3.0, tag12 1.0, wtag 1.0). 판독 학습률은 {0.3, 1.0, 3.0} 중 (사전 점검: 0.1 이하는 15에폭 안에 거의 못 배움) 시드 0 검증으로 선택.
목표치: MLP64 content 69.59 / tag12 40.14, wtag MLP16 46.20 / MLP64 46.70 (실험 09).

실행: python3 experiments/exp11_learned_readout.py
결과: results/exp11_learned_readout.{json,md}
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
from src.data.language import make_task, make_word_task  # noqa: E402
from src.baselines.models import accuracy  # noqa: E402
from src.biogrid.screen import BioGridX  # noqa: E402
from experiments.exp07_memory_vs_learning import GRID_FEATS, RULE  # noqa: E402

SEEDS = [0, 1, 2, 3, 4]
TASKS = {"content": (20000, 3.0), "tag12": (20000, 1.0), "wtag": (200000, 1.0)}
OUT_LRS = [0.3, 1.0, 3.0]
VARIANTS = ["grid", "grid+R4", "random+R4", "shuf+R4"]
EPOCHS, PATIENCE, BATCH, H = 15, 3, 32, 40


class DeltaReadout:
    """클래스별 판독 뉴런. 입력 = 마지막 열 발화 확률, softmax 델타 규칙 (지역 규칙)."""

    def __init__(self, n_feat, n_classes):
        self.W = np.zeros((n_classes, n_feat))
        self.b = np.zeros(n_classes)

    def probs(self, F):
        z = F @ self.W.T + self.b
        z -= z.max(1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(1, keepdims=True)

    def update(self, F, y, lr):
        err = -self.probs(F)
        err[np.arange(len(y)), y] += 1.0
        self.W += lr * err.T @ F / len(y)
        self.b += lr * err.mean(0)

    def n_params(self):
        return self.W.size + self.b.size


def load(task, seed):
    n = TASKS[task][0]
    return make_word_task(n_train=n, seed=seed) if task == "wtag" else make_task(task, seed=seed, n_train=n)


def last_p(g, X, n_votes=1):
    return np.mean([g.forward(X)[1][-1][1] for _ in range(n_votes)], 0)


def evaluate(g, ro, X, y):
    if ro is None:
        return accuracy(g.predict(X), y)
    return accuracy(ro.probs(last_p(g, X, 9)).argmax(1), y)


def run(task, seed, variant, out_lr):
    d = load(task, seed)
    if variant.startswith("shuf"):
        r = np.random.RandomState(1000 + seed)
        d["y_train"], d["y_val"] = r.permutation(d["y_train"]), r.permutation(d["y_val"])
    g = BioGridX(d["X_train"].shape[1], d["n_classes"], height=H, width=4, seed=seed, features=GRID_FEATS)
    ro = None if variant == "grid" else DeltaReadout(H, d["n_classes"])
    train_grid = not variant.startswith("random")
    rng = np.random.RandomState(seed + 100)
    best, best_acc, wait, ep_used = None, -1, 0, 0
    for ep in range(EPOCHS):
        o = rng.permutation(len(d["y_train"]))
        for i in range(0, len(o), BATCH):
            idx = o[i:i + BATCH]
            X, y = d["X_train"][idx], d["y_train"][idx]
            if train_grid:
                g.train_step(X, y, TASKS[task][1], **RULE)
            if ro is not None:
                ro.update(last_p(g, X), y, out_lr)
        va = evaluate(g, ro, d["X_val"], d["y_val"])
        if va > best_acc + 1e-4:
            best, best_acc, wait, ep_used = (copy.deepcopy(g), copy.deepcopy(ro)), va, 0, ep + 1
        else:
            wait += 1
            if wait >= PATIENCE:
                break
    g, ro = best
    npar = g.n_params() + (ro.n_params() if ro is not None else 0)
    return {"test": evaluate(g, ro, d["X_test"], d["y_test"]), "val": best_acc, "epochs": ep_used, "n_params": npar}


def job(a):
    task, seed, variant, out_lr = a
    t0 = time.time()
    r = run(task, seed, variant, out_lr)
    r.update(task=task, seed=seed, variant=variant, out_lr=out_lr, sec=round(time.time() - t0, 1))
    print(task, seed, variant, out_lr, round(r["test"] * 100, 2), f"val {r['val'] * 100:.2f}", f"{r['sec']}s", flush=True)
    return r


def main():
    rows = []
    order = lambda j: -(j[0] == "wtag")
    with Pool(4) as pool:
        sel = [(t, 0, "grid+R4", lr) for t in TASKS for lr in OUT_LRS]
        for r in pool.imap_unordered(job, sorted(sel, key=order), chunksize=1):
            rows.append(r)
        best = {t: max((r for r in rows if r["task"] == t), key=lambda r: r["val"])["out_lr"] for t in TASKS}
        print("chosen readout lr", best, flush=True)
        jobs = [(t, s, v, best[t]) for t in TASKS for s in SEEDS for v in VARIANTS
                if not (v == "grid+R4" and s == 0)]
        for r in pool.imap_unordered(job, sorted(jobs, key=order), chunksize=1):
            rows.append(r)
    final = [r for r in rows if r["variant"] == "grid" or r["out_lr"] == best[r["task"]]]
    target = {"content": [("MLP64", 69.59), ("로지스틱", 67.16)], "tag12": [("MLP64", 40.14), ("로지스틱", 38.55)],
              "wtag": [("MLP16", 46.20), ("MLP64", 46.70), ("로지스틱", 44.52)]}
    lines = ["# 실험 11: 학습되는 지역 판독 (R4)", "",
             "격자 C4 (폭 4, 높이 40) + 클래스별 판독 뉴런(델타 규칙). 테스트 정확도, 시드 5개 평균 ± 표준편차.",
             f"판독 학습률 (시드 0 검증으로 선택): {best}", ""]
    summary = {}
    for t in TASKS:
        lines += [f"## {t}", "", "| 방식 | 정확도 | 파라미터 | 에폭 |", "|---|---|---|---|"]
        for v in VARIANTS:
            rr = sorted([r for r in final if r["task"] == t and r["variant"] == v], key=lambda r: r["seed"])
            acc = [r["test"] * 100 for r in rr]
            summary[f"{t}/{v}"] = [float(np.mean(acc)), float(np.std(acc))]
            lines.append(f"| {v} | {np.mean(acc):.2f} ± {np.std(acc):.2f} | {rr[0]['n_params']:,} | "
                         f"{[r['epochs'] for r in rr]} |")
        a = np.array([r["test"] * 100 for r in sorted([r for r in final if r["task"] == t and r["variant"] == "grid+R4"],
                                                      key=lambda r: r["seed"])])
        b = np.array([r["test"] * 100 for r in sorted([r for r in final if r["task"] == t and r["variant"] == "grid"],
                                                      key=lambda r: r["seed"])])
        dd = a - b
        lines += ["", f"grid+R4 − grid (같은 시드 짝): {dd.mean():+.2f} ± {dd.std(ddof=1) / np.sqrt(len(dd)):.2f} (표준오차)",
                  "목표치: " + ", ".join(f"{n} {v:.2f}" for n, v in target[t]), ""]
    (ROOT / "results" / "exp11_learned_readout.json").write_text(json.dumps(
        {"seeds": SEEDS, "tasks": TASKS, "best_out_lr": best, "summary": summary, "rows": rows}, indent=2,
        ensure_ascii=False))
    (ROOT / "results" / "exp11_learned_readout.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
