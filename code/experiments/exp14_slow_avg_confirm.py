"""실험 14: 느린 시냅스 평균(slow_avg) 확인 — 새 시드, 같은 시드 MLP, 긴 학습, 수면 판독 결합.

실험 13(시드 0~4)에서 slow_avg가 wtag +1.86 ± 0.25 (46.04%, MLP16 46.20 근접). 좋은 수치이므로 다시 확인한다.
  - 새 시드 10~14 (실험 13과 겹치지 않음). 설정은 실험 13에서 고정한 그대로 (EMA 0.999, 학습률 이전 값)
  - 같은 시드·같은 데이터의 MLP 4/16/64·로지스틱을 함께 돌려 짝 비교
  - slow_avg가 에폭 상한(15)에 닿았으므로 30에폭(참을성 4)도 측정
  - 수면 판독(실험 12)을 slow_avg 격자 위에 결합. 판독 학습률은 실험 12에서 과제별로 고른 값
  - 셔플 대조: slow_avg30을 학습·검증 라벨 모두 섞어서

변형
  grid          C4 폭 4 높이 40, 15에폭 (지금까지의 기준)
  slow15        + 느린 시냅스 평균, 15에폭
  slow30        + 느린 시냅스 평균, 30에폭 (참을성 4)
  slow30+sleepR slow30 격자를 고정하고 판독 뉴런 학습
  slow30_shuf   셔플 대조
  mlp0 / mlp4 / mlp16 / mlp64   역전파 MLP (Adam, 검증 조기 종료)

실행: python3 experiments/exp14_slow_avg_confirm.py
결과: results/exp14_slow_avg_confirm.{json,md}
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
from src.baselines.models import MLP, accuracy  # noqa: E402
from src.biogrid.screen import BioGridX  # noqa: E402
from experiments.exp07_memory_vs_learning import GRID_FEATS, RULE  # noqa: E402
from experiments.exp11_learned_readout import load, TASKS  # noqa: E402
from experiments.exp12_sleep_readout import feats, sleep_readout  # noqa: E402
from experiments.exp13_stability import averaged, PARAMS, EMA  # noqa: E402

SEEDS = [10, 11, 12, 13, 14]
SLEEP_LR = {"content": 3.0, "tag12": 0.3, "wtag": 0.3}      # 실험 12에서 시드 0 검증으로 고른 값
GRID_VARIANTS = ["grid", "slow15", "slow30", "slow30_shuf"]
MLPS = [0, 4, 16, 64]
BATCH, H = 32, 40


def train(d, task, seed, slow, epochs, patience):
    g = BioGridX(d["X_train"].shape[1], d["n_classes"], height=H, width=4, seed=seed, features=GRID_FEATS)
    ema = {k: getattr(g, k).copy() for k in PARAMS if hasattr(g, k)} if slow else None
    rng = np.random.RandomState(seed + 100)
    best, best_acc, wait, ep_used = None, -1, 0, 0
    for ep in range(epochs):
        o = rng.permutation(len(d["y_train"]))
        for i in range(0, len(o), BATCH):
            idx = o[i:i + BATCH]
            g.train_step(d["X_train"][idx], d["y_train"][idx], TASKS[task][1], **RULE)
            if ema is not None:
                for k in ema:
                    ema[k] *= EMA
                    ema[k] += (1 - EMA) * getattr(g, k)
        m = averaged(g, ema) if ema is not None else g
        va = accuracy(m.predict(d["X_val"]), d["y_val"])
        if va > best_acc + 1e-4:
            best, best_acc, wait, ep_used = copy.deepcopy(m), va, 0, ep + 1
        else:
            wait += 1
            if wait >= patience:
                break
    return best, ep_used


def job(a):
    task, seed, variant = a
    t0 = time.time()
    d = load(task, seed)
    out = {"task": task, "seed": seed}
    if variant.startswith("mlp"):
        h = int(variant[3:])
        m = MLP(d["X_train"].shape[1], h, d["n_classes"], seed)
        m.fit(d["X_train"], d["y_train"], d["X_val"], d["y_val"])
        out.update({variant: accuracy(m.predict(d["X_test"]), d["y_test"]), variant + "_params": m.n_params()})
    else:
        if variant.endswith("shuf"):
            r = np.random.RandomState(1000 + seed)
            d["y_train"], d["y_val"] = r.permutation(d["y_train"]), r.permutation(d["y_val"])
        slow = variant != "grid"
        epochs, patience = (30, 4) if variant.startswith("slow30") else (15, 3)
        g, ep = train(d, task, seed, slow, epochs, patience)
        out.update({variant: accuracy(g.predict(d["X_test"]), d["y_test"]), variant + "_epochs": ep,
                    variant + "_params": g.n_params()})
        if variant == "slow30":
            F, Fv, Ft = feats(g, d["X_train"]), feats(g, d["X_val"]), feats(g, d["X_test"])
            va, pred, sep = sleep_readout(F, d["y_train"], Fv, d["y_val"], Ft, "sleepR", SLEEP_LR[task],
                                          g.zone_of_row, d["n_classes"], seed)
            out["slow30+sleepR"] = accuracy(pred, d["y_test"])
            out["slow30+sleepR_params"] = g.n_params() + H * d["n_classes"] + d["n_classes"]
    out["variant"] = variant
    out["sec"] = round(time.time() - t0, 1)
    print(task, seed, {k: (round(v * 100, 2) if isinstance(v, float) and k != "sec" else v) for k, v in out.items()
                       if k not in ("task", "seed")}, flush=True)
    return out


def main():
    jobs = [(t, s, v) for t in TASKS for s in SEEDS for v in GRID_VARIANTS + [f"mlp{h}" for h in MLPS]]
    jobs.sort(key=lambda j: -((j[0] == "wtag") * 10 + j[2].startswith("slow30")))
    with Pool(4) as pool:
        rows = list(pool.imap_unordered(job, jobs, chunksize=1))
    names = ["grid", "slow15", "slow30", "slow30+sleepR", "slow30_shuf"] + [f"mlp{h}" for h in MLPS]

    def vec(t, k):
        v = {}
        for r in rows:
            if r["task"] == t and k in r:
                v[r["seed"]] = r[k] * 100
        return np.array([v[s] for s in SEEDS])

    def par(t, k):
        return next(r[k + "_params"] for r in rows if r["task"] == t and k + "_params" in r)

    lines = ["# 실험 14: 느린 시냅스 평균 확인 (새 시드 10~14)", "",
             "테스트 정확도, 시드 5개 평균 ± 표준편차. 짝 차이 = 같은 시드끼리의 차이 평균 ± 표준오차.", ""]
    summary = {}
    for t in TASKS:
        lines += [f"## {t}", "", "| 모델 | 파라미터 | 정확도 | 시드별 |", "|---|---|---|---|"]
        for k in names:
            v = vec(t, k)
            summary[f"{t}/{k}"] = [float(v.mean()), float(v.std())]
            lines.append(f"| {k} | {par(t, k):,} | {v.mean():.2f} ± {v.std():.2f} | {np.round(v, 2).tolist()} |")
        lines += ["", "| 짝 비교 | 차이 |", "|---|---|"]
        for a, b in [("slow15", "grid"), ("slow30", "grid"), ("slow30+sleepR", "slow30"), ("slow30+sleepR", "mlp16"),
                     ("slow30+sleepR", "mlp64"), ("slow30", "mlp16"), ("slow30", "mlp0")]:
            dd = vec(t, a) - vec(t, b)
            se = dd.std(ddof=1) / np.sqrt(len(dd))
            summary[f"{t}/{a}-{b}"] = [float(dd.mean()), float(se)]
            lines.append(f"| {a} − {b} | {dd.mean():+.2f} ± {se:.2f} |")
        ep = [r["slow30_epochs"] for r in sorted(rows, key=lambda r: r["seed"]) if r["task"] == t and "slow30_epochs" in r
              and r["variant"] == "slow30"]
        lines += ["", f"slow30 사용 에폭: {ep}", ""]
    (ROOT / "results" / "exp14_slow_avg_confirm.json").write_text(json.dumps(
        {"seeds": SEEDS, "summary": summary, "rows": rows}, indent=2, ensure_ascii=False, default=float))
    (ROOT / "results" / "exp14_slow_avg_confirm.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
