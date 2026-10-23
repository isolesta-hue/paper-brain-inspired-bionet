"""실험 08: 기억 트랙 4-2 — 격자(피질) + 기억(해마) + 수면 재생 (REFERENCE U2·U3, L14, R7).

상황: 데이터가 한 번씩만 흘러가는 학습 (각 경험을 한 번만 봄). 학습 데이터 n개, 32개씩 도착.
  - 격자: BioGrid C4 (공명 + 확률 판독 + 허브 + residual, 폭 4), 새 경험으로 한 번 갱신
  - 기억: 해마형 기억 (DG 8000, k=20, 원샷 헤비안). 경험 색인도 보관 (일화 저장소)
  - 수면: SLEEP_EVERY개 경험마다 잠을 자며, 지금까지의 경험 중 무작위로 (재생 배수 × SLEEP_EVERY)개를 다시 틀어 격자 학습
      replay_true  : 저장된 실제 정답으로 재생
      replay_mem   : 기억이 내놓는 답(회상)으로 가르침 — 해마가 피질을 가르치는 공고화
  - 이중 판정 (M-Path R7): 최종 점수 = w·격자 확률 + (1−w)·기억 확률, w는 검증 세트로만 선택 {0, .25, .5, .75, 1}

비교
  grid            격자만, 1회 통과
  mem             기억만
  grid+replay1    격자 + 수면 재생(실제 정답), 재생 배수 1
  grid+replay4    재생 배수 4
  grid+memreplay4 재생 배수 4, 기억의 답으로 가르침
  dual            grid + mem 이중 판정
  dual+replay4    grid+replay4 + mem 이중 판정
  (참고) grid_full: 같은 데이터를 15에폭 반복 + 조기 종료 = 실험 07 값

실행: python3 experiments/exp08_cls_replay.py
결과: results/exp08_cls_replay.{json,md}
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
from src.data.language import make_task  # noqa: E402
from src.baselines.models import accuracy  # noqa: E402
from src.biogrid.screen import BioGridX  # noqa: E402
from src.memory.hippocampus import HippocampalMemory  # noqa: E402
from experiments.exp07_memory_vs_learning import GRID_FEATS, GRID_LR, RULE  # noqa: E402

SIZES = [1000, 3000, 20000]
SEEDS = [0, 1, 2, 3, 4]
TASKS = ["content", "tag12"]
SLEEP_EVERY = 500
BATCH = 32
VARIANTS = {"grid": (0, None), "grid+replay1": (1, "true"), "grid+replay4": (4, "true"), "grid+memreplay4": (4, "mem")}


def grid_probs(g, X, n_votes=9):
    """격자의 클래스별 확률: 확률 판독(구역별 발화 확률 합)을 미세 스텝 여러 번 평균 후 정규화."""
    tot = np.zeros((len(X), g.n_classes))
    for _ in range(n_votes):
        _, rec = g.forward(X)
        p = rec[-1][1]
        for k in range(g.n_classes):
            tot[:, k] += p[:, g.zone_of_row == k].sum(1)
    return tot / tot.sum(1, keepdims=True)


def mem_probs(m, X):
    A = m.A / np.maximum(1.0, m.A.sum(0, keepdims=True))
    s = np.vstack([m.code(X[i:i + 2000]) @ A.T for i in range(0, len(X), 2000)])
    return s / np.maximum(1e-9, s.sum(1, keepdims=True))


def run(task, n, seed):
    d = make_task(task, seed=seed, n_train=n)
    lr = GRID_LR[task]
    order = np.random.RandomState(seed + 100).permutation(n)
    out = {}
    mem = HippocampalMemory(39, d["n_classes"], seed=seed)
    mem.store(d["X_train"][order], d["y_train"][order])        # 기억은 흐름 그대로 한 번씩 저장
    pm_val, pm_test = mem_probs(mem, d["X_val"]), mem_probs(mem, d["X_test"])
    out["mem"] = accuracy(pm_test.argmax(1), d["y_test"])
    for name, (ratio, kind) in VARIANTS.items():
        g = BioGridX(39, d["n_classes"], height=40, width=4, seed=seed, features=GRID_FEATS)
        rrng = np.random.RandomState(seed + 7)
        mem_online = HippocampalMemory(39, d["n_classes"], seed=seed) if kind == "mem" else None
        seen, updates = 0, 0
        for i in range(0, n, BATCH):
            idx = order[i:i + BATCH]
            g.train_step(d["X_train"][idx], d["y_train"][idx], lr, **RULE); updates += len(idx)
            if mem_online is not None:
                mem_online.store(d["X_train"][idx], d["y_train"][idx])
            seen = i + len(idx)
            if ratio and (seen % SLEEP_EVERY < BATCH or seen == n):     # 수면
                past = order[:seen]
                rep = past[rrng.randint(0, seen, ratio * SLEEP_EVERY)]
                for j in range(0, len(rep), BATCH):
                    r = rep[j:j + BATCH]
                    y = d["y_train"][r] if kind == "true" else mem_probs(mem_online, d["X_train"][r]).argmax(1)
                    g.train_step(d["X_train"][r], y, lr, **RULE); updates += len(r)
        pg_val, pg_test = grid_probs(g, d["X_val"]), grid_probs(g, d["X_test"])
        out[name] = accuracy(pg_test.argmax(1), d["y_test"])
        out[name + "_updates"] = updates
        if name in ("grid", "grid+replay4"):
            best_w, best_v = 1.0, -1
            for w in [0.0, 0.25, 0.5, 0.75, 1.0]:
                v = accuracy((w * pg_val + (1 - w) * pm_val).argmax(1), d["y_val"])
                if v > best_v:
                    best_w, best_v = w, v
            key = "dual" if name == "grid" else "dual+replay4"
            out[key] = accuracy((best_w * pg_test + (1 - best_w) * pm_test).argmax(1), d["y_test"])
            out[key + "_w"] = best_w
    return out


def job(a):
    task, n, seed = a
    t0 = time.time()
    r = run(task, n, seed)
    r.update(task=task, n=n, seed=seed, sec=round(time.time() - t0, 1))
    return r


def main():
    jobs = sorted([(t, n, s) for t in TASKS for n in SIZES for s in SEEDS], key=lambda j: -j[1])
    rows = []
    with Pool(4) as pool:
        for r in pool.imap_unordered(job, jobs, chunksize=1):
            rows.append(r)
            print(r["task"], r["n"], r["seed"], {k: round(v * 100, 2) for k, v in r.items()
                                                   if isinstance(v, float) and k not in ("sec",) and not k.endswith("_w")}, flush=True)
    names = ["grid", "mem", "grid+replay1", "grid+replay4", "grid+memreplay4", "dual", "dual+replay4"]
    ref = json.loads((ROOT / "results" / "exp07_memory_vs_learning.json").read_text())["summary"]
    lines = ["# 실험 08: 격자 + 기억 + 수면 재생 (기억 트랙 4-2)", "",
             "데이터를 한 번씩만 보는 흐름 학습. 테스트 정확도, 시드 5개 평균 ± 표준편차.",
             f"수면: {SLEEP_EVERY}개 경험마다 과거 경험을 재생. 이중 판정 비율 w는 검증 세트로만 선택.",
             "참고: grid_full / mlp64_full = 같은 데이터를 여러 에폭 반복 학습 (실험 07, 시드 0~4).", ""]
    summary = {}
    for t in TASKS:
        lines += [f"## {t}", "", "| 방식 | " + " | ".join(f"n={n}" for n in SIZES) + " |", "|---|" + "---|" * len(SIZES)]
        for nm in names:
            cells = []
            for n in SIZES:
                v = [r[nm] * 100 for r in rows if r["task"] == t and r["n"] == n]
                summary[f"{t}/{nm}/{n}"] = [float(np.mean(v)), float(np.std(v))]
                cells.append(f"{np.mean(v):.2f} ± {np.std(v):.2f}")
            lines.append(f"| {nm} | " + " | ".join(cells) + " |")
        for refm in ["grid_full", "mlp64_full"]:
            lines.append(f"| (참고) {refm} | " + " | ".join(f"{ref[f'{t}/{refm}/{n}'][0]:.2f}" for n in SIZES) + " |")
        ws = {n: [r["dual_w"] for r in rows if r["task"] == t and r["n"] == n] for n in SIZES}
        lines += ["", f"이중 판정에서 고른 격자 비중 w (시드별): {ws}", ""]
    (ROOT / "results" / "exp08_cls_replay.json").write_text(json.dumps(
        {"sizes": SIZES, "seeds": SEEDS, "sleep_every": SLEEP_EVERY, "rows": rows, "summary": summary}, indent=2, ensure_ascii=False))
    (ROOT / "results" / "exp08_cls_replay.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
