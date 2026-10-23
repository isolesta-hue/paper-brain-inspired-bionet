"""실험 15: 이웃 연결 넓히기 — 격자 뉴런이 왼쪽 열에서 받는 이웃 수 3 → 5 → 7 (사용자 질문, 2026-10-02).

REFERENCE 확인: 8방향 연결은 BioBrain V1.0(16번 대화)에서 썼지만 그때 실패 원인은 데이터 정답이 난수였던 버그 → 공정한 시험은 없었다.
지금 격자(C4 + A3 + slow_avg)는 각 뉴런이 왼쪽 열의 (r−1, r, r+1)에서만 받음 (허브 열은 전체). fan k = 위아래 k//2칸까지.
  처음 가중치 표준편차 × sqrt(3/k) (처음 입력 크기를 3이웃과 같게). 그 밖의 설정은 실험 14의 slow30 그대로 (EMA 0.999, 30에폭 참을성 4, 과제별 학습률).
  구현 확인: fan 5에서 바깥 두 이웃 가중치를 0으로 두면 첫 학습 단계 후 모든 가중치가 3이웃 격자와 소수점까지 같음.
A. BioNet 본래 과제 (content / tag12 / wtag, 시드 10~14 = 실험 14와 같은 시드·데이터, 테스트): fan3(= 실험 14 slow30 재현) / fan5 / fan7 / fan7 셔플 대조
   기준: 실험 14의 같은 시드 MLP16·MLP64
B. 글 세상 생애 (NH 7 고정 몸, 피질만 fan 바꿈, 30시드): 바깥 시험 움직임 끝, 뒤집힌 방, 옛 방
비교 (미리 고정): fan5 − fan3, fan7 − fan3 (A: 짝 차이 평균 ± 표준오차, B: 짝지은 부호 검정)
실행: python3 experiments/exp15_fan.py
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import copy
import json
import pickle
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.baselines.models import accuracy  # noqa: E402
from src.biogrid.screen import BioGridX  # noqa: E402
from experiments.exp07_memory_vs_learning import GRID_FEATS, RULE  # noqa: E402
from experiments.exp11_learned_readout import load, TASKS  # noqa: E402
from experiments.exp13_stability import averaged, PARAMS, EMA  # noqa: E402
from new_horizons.run_content import sign_p  # noqa: E402

SEEDS = [10, 11, 12, 13, 14]
FANS = [3, 5, 7]
BATCH, H = 32, 40
CACHE = ROOT / "results" / "nh_cache"


def feats_of(k):
    return {**GRID_FEATS, "fan": {"k": k}} if k != 3 else GRID_FEATS


def train(d, task, seed, k, epochs=30, patience=4):
    g = BioGridX(d["X_train"].shape[1], d["n_classes"], height=H, width=4, seed=seed, features=feats_of(k))
    ema = {kk: getattr(g, kk).copy() for kk in PARAMS if hasattr(g, kk)}
    rng = np.random.RandomState(seed + 100)
    best, best_acc, wait = None, -1, 0
    for ep in range(epochs):
        o = rng.permutation(len(d["y_train"]))
        for i in range(0, len(o), BATCH):
            idx = o[i:i + BATCH]
            g.train_step(d["X_train"][idx], d["y_train"][idx], TASKS[task][1], **RULE)
            for kk in ema:
                ema[kk] *= EMA
                ema[kk] += (1 - EMA) * getattr(g, kk)
        m = averaged(g, ema)
        va = accuracy(m.predict(d["X_val"]), d["y_val"])
        if va > best_acc + 1e-4:
            best, best_acc, wait = copy.deepcopy(m), va, 0
        else:
            wait += 1
            if wait >= patience:
                break
    return best


def job_a(a):
    task, seed, k, shuf = a
    d = load(task, seed)
    if shuf:
        r = np.random.RandomState(1000 + seed)
        d["y_train"], d["y_val"] = r.permutation(d["y_train"]), r.permutation(d["y_val"])
    g = train(d, task, seed, k)
    acc = accuracy(g.predict(d["X_test"]), d["y_test"]) * 100
    print(task, seed, k, shuf, round(acc, 2), flush=True)
    return a, (acc, g.n_params())


def job_b(a):
    k, seed = a
    f = CACHE / f"exp15_world_fan{k}_s{seed}.pkl"
    if f.exists():
        return a, pickle.loads(f.read_bytes())
    import new_horizons.nh05_worldmodel as W5
    import new_horizons.nh07_method as M
    from new_horizons.body import SLOW_PARAMS

    class Cortex(W5.WorldCortex):
        def __init__(self, senses, seed_, lr, horizon=1.0):
            self.s, self.lr, self.horizon = senses, lr, horizon
            self.g = BioGridX(3 * senses.F.shape[1], len(senses.to_cells), height=W5.H, width=4, seed=seed_, features=feats_of(k))
            self.ema = {kk: getattr(self.g, kk).copy() for kk in SLOW_PARAMS if hasattr(self.g, kk)}
    M.WorldCortex = Cortex
    r = M.life("fixed", seed)
    f.write_bytes(pickle.dumps(r))
    return a, r


def main():
    jobs = [(t, s, k, False) for t in TASKS for s in SEEDS for k in FANS] + [(t, s, 7, True) for t in TASKS for s in SEEDS]
    jobs.sort(key=lambda j: -(j[0] == "wtag"))
    with Pool(4) as p:
        A = dict(p.map(job_a, jobs, chunksize=1))
        B = dict(p.map(job_b, [(k, s) for s in range(30) for k in FANS]))
    ref = json.loads((ROOT / "results" / "exp14_slow_avg_confirm.json").read_text())["rows"]
    refv = lambda t, key: np.array([next(r[key] for r in ref if r["task"] == t and r["seed"] == s and key in r) for s in SEEDS]) * 100
    lines = ["# 실험 15: 이웃 연결 넓히기 (fan 3 / 5 / 7)", "", "## A. BioNet 본래 과제 (시드 10~14, 테스트 %, 평균 ± 표준편차)", "",
             "| 과제 | fan3 (= 실험 14 slow30) | fan5 | fan7 | fan7 셔플 | MLP16 | MLP64 | fan5 − fan3 | fan7 − fan3 |", "|---|---|---|---|---|---|---|---|---|"]
    summary = {}
    for t in TASKS:
        v = {k: np.array([A[(t, s, k, False)][0] for s in SEEDS]) for k in FANS}
        sh = np.array([A[(t, s, 7, True)][0] for s in SEEDS])
        m16, m64 = refv(t, "mlp16"), refv(t, "mlp64")
        se = lambda dd: dd.std(ddof=1) / np.sqrt(len(dd))
        d5, d7 = v[5] - v[3], v[7] - v[3]
        lines.append(f"| {t} | {v[3].mean():.2f} ± {v[3].std():.2f} | {v[5].mean():.2f} ± {v[5].std():.2f} | {v[7].mean():.2f} ± {v[7].std():.2f} | "
                     f"{sh.mean():.2f} | {m16.mean():.2f} | {m64.mean():.2f} | {d5.mean():+.2f} ± {se(d5):.2f} | {d7.mean():+.2f} ± {se(d7):.2f} |")
        reg = np.abs(v[3] - refv(t, "slow30")).max()
        summary[t] = {"fan3": v[3].tolist(), "fan5": v[5].tolist(), "fan7": v[7].tolist(), "fan7_shuf": sh.tolist(), "regression_maxdiff": float(reg)}
    lines += ["", "회귀 확인 (fan3 − 실험 14 slow30, 같은 시드 최대 차이): " + ", ".join(f"{t} {summary[t]['regression_maxdiff']:.2g}" for t in TASKS),
              "파라미터 (wtag): " + ", ".join(f"fan{k} {A[('wtag', SEEDS[0], k, False)][1]:,}" for k in FANS), ""]
    C = lambda k, key: np.array([B[(k, s)][-1][key] for s in range(30)], dtype=float)

    def t_(d):
        d = d[~np.isnan(d)]
        pos, neg = int((d > 1e-12).sum()), int((d < -1e-12).sum())
        return f"{d.mean():+.3f} ({pos}:{neg}, p={sign_p(pos, neg):.3g})"
    lines += ["## B. 글 세상 생애 (NH 7 고정 몸, 30시드, 바깥 시험 끝)", "", "| fan | 움직임 끝 | 뒤집힌 방 | 옛 방 |", "|---|---|---|---|"]
    for k in FANS:
        lines.append(f"| {k} | {np.nanmean(C(k, 'motion')):.3f} | {np.nanmean(C(k, 'flip')):.3f} | {np.nanmean(C(k, 'keep')):.3f} |")
    for k in (5, 7):
        lines.append(f"- fan{k} − fan3: 끝 {t_(C(k, 'motion') - C(3, 'motion'))}, 뒤집힌 방 {t_(C(k, 'flip') - C(3, 'flip'))}, 옛 방 {t_(C(k, 'keep') - C(3, 'keep'))}")
    (ROOT / "results" / "exp15_fan.json").write_text(json.dumps(summary, indent=2))
    (ROOT / "results" / "exp15_fan.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
