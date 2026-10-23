"""실험 06: 기억 트랙 4-1 — 지형을 연상 기억으로 (REFERENCE U1).

질문
  1. 저장한 패턴을 절반짜리 단서로 되살릴 수 있는가 (패턴 완성, 원샷 저장)
  2. 몇 개까지 저장되는가 (용량), 연결 방식·연결 수에 따라 어떻게 달라지는가
  3. GeoBrain식 공간 지형(F1)은 기억으로 쓰일 수 있는가

설정
  - 격자 20×20 = 400 뉴런, 패턴 희소도 f = 0.1 (40개 활성), 각 패턴은 한 번만 보고 저장(원샷)
  - 단서: 활성 비트의 50%만 남기고 + 엉뚱한 비트 4개를 켬 (단서 자체의 겹침 = 0.5)
  - 지표: 회상 겹침 (원래 활성 비트를 되찾은 비율). 단서보다 낮으면 기억이 오히려 방해한 것
  - 용량: 평균 겹침 ≥ 0.9를 유지하는 최대 패턴 수
  - 시드 5개
공간 지형은 자기 구조(윗줄 20칸 주입 → 아랫줄 판독)에 맞춰 패턴 길이 20, 활성 4개로 따로 측정.

실행: python3 experiments/exp06_memory_capacity.py
결과: results/exp06_memory_capacity.{json,md}
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.memory.assoc import (StateLandscapeMemory, SpatialTerrainMemory, make_patterns,  # noqa: E402
                              make_cue, overlap)

H = W = 20
N, F = H * W, 0.1
P_LIST = [5, 10, 20, 40, 60, 80, 120, 160]
SEEDS = [0, 1, 2, 3, 4]
NETS = [("full", 0), ("local8", 0), ("smallworld", 8), ("random", 8), ("smallworld", 40), ("random", 40),
        ("smallworld", 100), ("random", 100)]
N_TEST = 30


def run_state(kind, n_long, P, seed):
    rng = np.random.RandomState(1000 * seed + P)
    m = StateLandscapeMemory(H, W, kind, n_long=n_long, f=F, seed=seed)
    X = make_patterns(P, N, F, rng)
    for x in X:
        m.store(x)
    idx = rng.choice(P, min(P, N_TEST), replace=False)
    rec, cue = [], []
    for i in idx:
        c = make_cue(X[i], 0.5, 4, rng)
        rec.append(overlap(m.recall(c), X[i])); cue.append(overlap(c, X[i]))
    return float(np.mean(rec)), float(np.mean(cue)), m.n_connections() / N


def run_spatial(P, seed):
    rng = np.random.RandomState(1000 * seed + P + 7)
    t = SpatialTerrainMemory(20, 20, seed=seed)
    X = make_patterns(P, 20, 0.2, rng)
    for x in X:
        t.store(x)
    idx = rng.choice(P, min(P, N_TEST), replace=False)
    rec, cue = [], []
    for i in idx:
        c = make_cue(X[i], 0.5, 0, rng)
        rec.append(overlap(t.recall(c, 4), X[i])); cue.append(overlap(c, X[i]))
    return float(np.mean(rec)), float(np.mean(cue))


def capacity(curve):
    ok = [P for P, v in zip(P_LIST, curve) if v >= 0.9]
    return max(ok) if ok else 0


def main():
    results = {}
    for kind, nl in NETS:
        name = f"{kind}+{nl}" if kind in ("smallworld", "random") else kind
        curves, conn = [], None
        for s in SEEDS:
            row = []
            for P in P_LIST:
                r, c, conn = run_state(kind, nl, P, s)
                row.append(r)
            curves.append(row)
        curves = np.array(curves)
        results[name] = {"conn_per_neuron": conn, "mean": curves.mean(0).tolist(), "std": curves.std(0).tolist(),
                         "capacity_each_seed": [capacity(c) for c in curves]}
        print(name, round(conn, 1), np.round(curves.mean(0), 3).tolist(), flush=True)
    sp = np.array([[run_spatial(P, s)[0] for P in [2, 5, 10, 20]] for s in SEEDS])
    results["spatial_terrain"] = {"P": [2, 5, 10, 20], "mean": sp.mean(0).tolist(), "std": sp.std(0).tolist()}
    print("spatial", np.round(sp.mean(0), 3).tolist())

    lines = ["# 실험 06: 지형을 연상 기억으로 (기억 트랙 4-1)", "",
             "400뉴런 격자, 희소 패턴(40개 활성) 원샷 저장, 단서 = 활성 비트 50% + 엉뚱한 비트 4개 (단서 겹침 0.50).",
             "값 = 회상 겹침 (원래 활성 비트를 되찾은 비율, 시드 5개 평균). **0.50보다 낮으면 기억이 단서를 망친 것.**",
             "용량 = 겹침 ≥ 0.9를 유지한 최대 저장 개수 (시드별).", "",
             "## 상태 지형 (홉필드형)", "",
             "| 연결 방식 | 뉴런당 연결 | " + " | ".join(f"P={P}" for P in P_LIST) + " | 용량 (시드별) |",
             "|---|---|" + "---|" * len(P_LIST) + "---|"]
    for name, r in results.items():
        if name == "spatial_terrain":
            continue
        lines.append(f"| {name} | {r['conn_per_neuron']:.1f} | " + " | ".join(f"{v:.2f}" for v in r["mean"])
                     + f" | {r['capacity_each_seed']} |")
    sp = results["spatial_terrain"]
    lines += ["", "## 공간 지형 (GeoBrain F1을 기억에 사용)", "",
              "패턴 길이 20 (윗줄), 활성 4개, 단서 = 활성 2개 (겹침 0.50). 아랫줄 에너지 상위 4칸을 회상으로 본다.", "",
              "| 저장 개수 | " + " | ".join(str(P) for P in sp["P"]) + " |", "|---|" + "---|" * len(sp["P"]),
              "| 회상 겹침 | " + " | ".join(f"{v:.2f}" for v in sp["mean"]) + " |", ""]
    (ROOT / "results" / "exp06_memory_capacity.json").write_text(json.dumps(
        {"P_LIST": P_LIST, "seeds": SEEDS, "results": results}, indent=2, ensure_ascii=False))
    (ROOT / "results" / "exp06_memory_capacity.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
