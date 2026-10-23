"""실험 04: 실험 03(1차 스크리닝)에서 효과가 보인 기능의 2차 확인.

- 1차와 겹치지 않는 새 시드 10~14 (같은 시드로 고르고 확인하면 선택 편향)
- 학습률은 실험 03에서 고른 값을 그대로 사용 (새로 고르지 않음)
- 셔플 대조: 폭 8, 학습·검증 라벨 모두 셔플
- 판정: 베이스 대비 Δ와, 시드별 짝지은 차이(같은 시드끼리 기능 − 베이스)의 평균 ± 표준오차

실행: python3 experiments/exp04_feature_confirm.py
결과: results/exp04_feature_confirm.{json,md}
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
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
import copy  # noqa: E402

FEATURES = ["base", "resonance", "prob_readout", "hub", "lif", "homeostasis", "residual",
            "fatigue", "gol_thresh", "topk", "v_share", "mean_inhib"]
CONFIG = {"tasks": ["content", "tag12"], "widths": [4, 8], "seeds": [10, 11, 12, 13, 14],
          "shuffle_width": 8}
LR = json.loads((ROOT / "results" / "exp03_feature_screening.json").read_text())["lr_choice"]


def train(task, width, feature, lr, seed, shuffle):
    d = data(task, seed)
    y_tr, y_va = d["y_train"], d["y_val"]
    if shuffle:
        rs = np.random.RandomState(1000 + seed)
        y_tr, y_va = rs.permutation(y_tr), rs.permutation(y_va)
    feats = {} if feature == "base" else {feature: {}}
    g = BioGridX(d["X_train"].shape[1], d["n_classes"], height=C3["height"], width=width, seed=seed, features=feats)
    order_rng = np.random.RandomState(seed + 100)
    best_acc, best, wait = -1.0, None, 0
    for _ in range(C3["epochs"]):
        o = order_rng.permutation(len(y_tr))
        for i in range(0, len(o), C3["batch"]):
            idx = o[i:i + C3["batch"]]
            g.train_step(d["X_train"][idx], y_tr[idx], lr, **RULE_A)
        if not (np.all(np.isfinite(g.W_loc)) and np.all(np.isfinite(g.W_in))):
            return {"invalid": True, "test": 0.0}
        va = accuracy(g.predict(d["X_val"], C3["n_votes"]), y_va)
        if va > best_acc + 1e-4:
            best_acc, best, wait = va, copy.deepcopy(g), 0
        else:
            wait += 1
            if wait >= C3["patience"]:
                break
    return {"test": accuracy(best.predict(d["X_test"], C3["n_votes"]), d["y_test"]), "val": best_acc}


def job(args):
    task, width, feature, seed, shuffle = args
    lr = LR[f"{task}/W{width}/A/{feature}"]["lr"]
    t0 = time.time()
    r = train(task, width, feature, lr, seed, shuffle)
    r.update(task=task, width=width, feature=feature, seed=seed, shuffle=shuffle, lr=lr, sec=round(time.time() - t0, 1))
    return r


def main():
    jobs = [(t, w, f, s, False) for t in CONFIG["tasks"] for w in CONFIG["widths"] for f in FEATURES for s in CONFIG["seeds"]]
    jobs += [(t, CONFIG["shuffle_width"], f, s, True) for t in CONFIG["tasks"] for f in FEATURES for s in CONFIG["seeds"]]
    rows = []
    with Pool(4) as pool:
        for r in pool.imap_unordered(job, jobs, chunksize=1):
            rows.append(r)
            print(r["task"], r["width"], r["feature"], r["seed"], "shuf" if r["shuffle"] else "", round(r["test"] * 100, 2), flush=True)

    def get(t, w, f, sh):
        return {r["seed"]: r["test"] * 100 for r in rows if r["task"] == t and r["width"] == w and r["feature"] == f and r["shuffle"] == sh}

    summary, lines = {}, ["# 실험 04: 기능 효과 2차 확인 (새 시드 10~14)", "",
                          "Δ = 같은 시드끼리 (기능 − 베이스)의 평균 ± 표준오차(%p). 표준오차의 2배보다 |Δ|가 크면 ✔ (대략 95% 수준).",
                          "셔플 = 폭 8에서 학습·검증 라벨을 모두 섞었을 때의 테스트 정확도 (기준선 근처 또는 그 아래면 정상).", ""]
    for t in CONFIG["tasks"]:
        lines += [f"## {t}", "", "| 기능 | 폭 4 정확도 | 폭 4 Δ | 폭 8 정확도 | 폭 8 Δ | 셔플(폭 8) |", "|---|---|---|---|---|---|"]
        for f in FEATURES:
            cells = []
            for w in CONFIG["widths"]:
                a, b = get(t, w, f, False), get(t, w, "base", False)
                vals = np.array([a[s] for s in CONFIG["seeds"]])
                diff = np.array([a[s] - b[s] for s in CONFIG["seeds"]])
                m, se = diff.mean(), diff.std(ddof=1) / np.sqrt(len(diff))
                mark = "" if f == "base" else (" ✔" if abs(m) > 2 * se else "")
                summary[f"{t}/W{w}/{f}"] = {"mean": vals.mean(), "std": vals.std(), "delta": m, "se": se}
                cells += [f"{vals.mean():.2f} ± {vals.std():.2f}", "–" if f == "base" else f"{m:+.2f} ± {se:.2f}{mark}"]
            sh = np.array(list(get(t, CONFIG["shuffle_width"], f, True).values()))
            summary[f"{t}/W8/{f}/shuffle"] = float(sh.mean())
            lines.append(f"| {f} | " + " | ".join(cells) + f" | {sh.mean():.2f} |")
        lines.append("")
    (ROOT / "results" / "exp04_feature_confirm.json").write_text(json.dumps(
        {"config": CONFIG, "features": FEATURES, "rows": rows, "summary": summary}, indent=2, ensure_ascii=False, default=float))
    (ROOT / "results" / "exp04_feature_confirm.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
