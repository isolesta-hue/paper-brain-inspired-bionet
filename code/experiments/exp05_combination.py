"""실험 05: 확정된 기능의 누적 조합 (2-B 공명 + 판독 개선).

조합 사다리 (하나씩 더함):
  C0 base               A3 (구역 오차 + 바톤 터치 · 민감도)
  C1 +resonance          공명 (L15)
  C2 +prob_readout       확률 판독 (N11)
  C3 +hub                허브 열 (S15)
  C4 +residual           원본 보존 경로
- 학습률: 조합·과제·폭마다 {0.3, 1, 3} 중 시드 0 검증으로 선택
- 평가: 새 시드 20~24 (실험 03·04와 겹치지 않음), 셔플 대조(폭 8, 학습·검증 라벨 모두 셔플)
- 목표치: content 로지스틱 67.16 / MLP 69.59, tag12 로지스틱 38.55 / MLP 40.14

실행: python3 experiments/exp05_combination.py
결과: results/exp05_combination.{json,md}
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
from experiments.exp03_feature_screening import RULE_A, CONFIG as C3, data  # noqa: E402
from src.baselines.models import accuracy  # noqa: E402
from src.biogrid.screen import BioGridX  # noqa: E402

COMBOS = {
    "C0_base": [],
    "C1_res": ["resonance"],
    "C2_res_prob": ["resonance", "prob_readout"],
    "C3_res_prob_hub": ["resonance", "prob_readout", "hub"],
    "C4_res_prob_hub_resid": ["resonance", "prob_readout", "hub", "residual"],
}
CONFIG = {"tasks": ["content", "tag12"], "widths": [4, 8], "seeds": [20, 21, 22, 23, 24],
          "lr_grid": [0.3, 1.0, 3.0], "shuffle_width": 8}
TARGETS = {"content": (51.90, 67.16, 69.59), "tag12": (23.94, 38.55, 40.14)}


def train(task, width, combo, lr, seed, shuffle=False):
    d = data(task, seed)
    y_tr, y_va = d["y_train"], d["y_val"]
    if shuffle:
        rs = np.random.RandomState(1000 + seed)
        y_tr, y_va = rs.permutation(y_tr), rs.permutation(y_va)
    feats = {k: {} for k in COMBOS[combo]}
    g = BioGridX(d["X_train"].shape[1], d["n_classes"], height=C3["height"], width=width, seed=seed, features=feats)
    order_rng = np.random.RandomState(seed + 100)
    best_acc, best, wait = -1.0, None, 0
    for _ in range(C3["epochs"]):
        o = order_rng.permutation(len(y_tr))
        for i in range(0, len(o), C3["batch"]):
            idx = o[i:i + C3["batch"]]
            g.train_step(d["X_train"][idx], y_tr[idx], lr, **RULE_A)
        if not (np.all(np.isfinite(g.W_loc)) and np.all(np.isfinite(g.W_in))):
            return {"invalid": True, "val": 0.0, "test": 0.0}
        va = accuracy(g.predict(d["X_val"], C3["n_votes"]), y_va)
        if va > best_acc + 1e-4:
            best_acc, best, wait = va, copy.deepcopy(g), 0
        else:
            wait += 1
            if wait >= C3["patience"]:
                break
    return {"val": best_acc, "test": accuracy(best.predict(d["X_test"], C3["n_votes"]), d["y_test"]),
            "n_params": best.n_params()}


def job(a):
    task, width, combo, lr, seed, shuffle = a
    t0 = time.time()
    r = train(task, width, combo, lr, seed, shuffle)
    r.update(task=task, width=width, combo=combo, lr=lr, seed=seed, shuffle=shuffle, sec=round(time.time() - t0, 1))
    return r


def main():
    cfg = [(t, w, c) for t in CONFIG["tasks"] for w in CONFIG["widths"] for c in COMBOS]
    with Pool(4) as pool:
        lr_rows = pool.map(job, [(t, w, c, lr, 0, False) for (t, w, c) in cfg for lr in CONFIG["lr_grid"]], chunksize=1)
        best = {}
        for r in lr_rows:
            k = (r["task"], r["width"], r["combo"])
            if k not in best or r["val"] > best[k][1]:
                best[k] = (r["lr"], r["val"])
        print("학습률 선택 완료", flush=True)
        jobs = [(t, w, c, best[(t, w, c)][0], s, False) for (t, w, c) in cfg for s in CONFIG["seeds"]]
        jobs += [(t, CONFIG["shuffle_width"], c, best[(t, CONFIG["shuffle_width"], c)][0], s, True)
                 for t in CONFIG["tasks"] for c in COMBOS for s in CONFIG["seeds"]]
        rows = []
        for r in pool.imap_unordered(job, jobs, chunksize=1):
            rows.append(r)
            print(r["task"], r["width"], r["combo"], r["seed"], "shuf" if r["shuffle"] else "", round(r["test"] * 100, 2), flush=True)

    def vals(t, w, c, sh):
        return np.array([r["test"] * 100 for r in sorted(rows, key=lambda r: r["seed"])
                         if r["task"] == t and r["width"] == w and r["combo"] == c and r["shuffle"] == sh])

    lines = ["# 실험 05: 확정 기능의 누적 조합", "",
             "새 시드 20~24, 테스트 정확도 평균 ± 표준편차. 학습률은 칸마다 시드 0 검증으로 선택.", ""]
    summary = {}
    for t in CONFIG["tasks"]:
        base, logreg, mlp = TARGETS[t]
        lines += [f"## {t}", "", f"목표치: 기준선 {base} / 로지스틱 {logreg} / **MLP64 {mlp}**", "",
                  "| 조합 | 파라미터(폭8) | 폭 4 | 폭 8 | 셔플(폭 8) | lr (폭4/폭8) |", "|---|---|---|---|---|---|"]
        for c in COMBOS:
            cells = []
            for w in CONFIG["widths"]:
                v = vals(t, w, c, False)
                summary[f"{t}/W{w}/{c}"] = [float(v.mean()), float(v.std())]
                cells.append(f"{v.mean():.2f} ± {v.std():.2f}")
            sh = vals(t, CONFIG["shuffle_width"], c, True)
            summary[f"{t}/W8/{c}/shuffle"] = float(sh.mean())
            npar = next(r["n_params"] for r in rows if r["task"] == t and r["width"] == 8 and r["combo"] == c and not r["shuffle"])
            lines.append(f"| {c} | {npar} | " + " | ".join(cells) + f" | {sh.mean():.2f} | "
                         f"{best[(t, 4, c)][0]} / {best[(t, 8, c)][0]} |")
        lines.append("")
    (ROOT / "results" / "exp05_combination.json").write_text(json.dumps(
        {"config": CONFIG, "combos": COMBOS, "lr": {f"{k[0]}/W{k[1]}/{k[2]}": v for k, v in best.items()},
         "rows": rows, "lr_rows": lr_rows, "summary": summary}, indent=2, ensure_ascii=False, default=float))
    (ROOT / "results" / "exp05_combination.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
