"""실험 10: 기억 모델 압축 — 해마형 기억의 저장량을 얼마나 줄일 수 있는가 (PLAN "다음 목표 2").

기억 = DG 희소 부호(고정 무작위 투영 + 상위 k) + 클래스 × DG 저장 행렬 A (원샷 헤비안 카운트).
판독은 시냅스 스케일링 Â[y, j] = A[y, j] / Σ_y A[y, j]. 투영 P는 시드로 다시 만들 수 있으므로 저장량에 넣지 않는다
(시드 하나 = 32비트). 압축은 모두 "다 저장한 뒤" Â에 적용한다 (저장 중 압축은 다음 단계).

압축 방법 (저장 비트 수 = 실제로 남겨야 하는 정보량)
  full        Â를 32비트 실수로 그대로                       C·N·32
  q8/q4/q2    Â를 [0,1] 구간 균등 양자화 8/4/2비트             C·N·b
  bin_chance  1비트: Â[y,j] > 1/C 이면 1 (우연보다 자주 나온 클래스) C·N·1
  willshaw    1비트: A[y,j] > 0 이면 1 (고전 Willshaw 기억)     C·N·1
  top1        뉴런마다 가장 자주 나온 클래스 하나만            N·⌈log2 C⌉
  top1_q4     top1 + 그 비율을 4비트로                          N·(⌈log2 C⌉+4)
  lowrank{r}  Â ≈ U·V (SVD 랭크 r), 32비트                      r·(C+N)·32
  prune{p}    사전 분포와 가장 다른 상위 p% 뉴런만 q4로 남기고 나머지는 투표하지 않음 (모두 지워진 입력만 사전 분포) (색인·사전 비용 포함)
  + DG 크기 N × 활성 수 k 곡선 (full / q4 / top1)

과제: tag12 (n=20,000, 입력 39) 와 wtag (n=200,000, 입력 1252, 실험 09 과제). 테스트 정확도, 시드 5개.
비교 기준: MLP64 (tag12 40.14, 파라미터 3,340개 ≈ 106,880비트), 격자 C4 폭 4 (3,720개 ≈ 119,040비트, 실험 08 공고화 38.77).

실행: python3 experiments/exp10_memory_compression.py
결과: results/exp10_memory_compression.{json,md}
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import json
import math
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data.language import make_task, make_word_task  # noqa: E402
from src.baselines.models import accuracy  # noqa: E402
from src.memory.hippocampus import HippocampalMemory  # noqa: E402

SEEDS = [0, 1, 2, 3, 4]
TASKS = {"tag12": dict(n=20000), "wtag": dict(n=200000)}
NK = [(500, 5), (500, 20), (1000, 10), (1000, 20), (2000, 10), (2000, 20), (2000, 40), (4000, 20), (8000, 10),
      (8000, 20), (8000, 40)]
MAIN = (8000, 20)


def load(task, seed):
    if task == "tag12":
        return make_task("tag12", seed=seed, n_train=TASKS[task]["n"])
    return make_word_task(n_train=TASKS[task]["n"], seed=seed)


def quant(M, b):
    L = 2 ** b - 1
    return np.round(np.clip(M, 0, 1) * L) / L


def compress(A, method):
    """A: 원래 카운트 (C, N). 반환 (판독 행렬, 저장 비트)."""
    C, N = A.shape
    tot = A.sum(0, keepdims=True)
    Ah = A / np.maximum(1.0, tot)
    lc = math.ceil(math.log2(C))
    if method == "full":
        return Ah, C * N * 32
    if method in ("q8", "q4", "q2"):
        b = int(method[1:])
        return quant(Ah, b), C * N * b
    if method == "bin_chance":
        return (Ah > 1.0 / C).astype(float), C * N
    if method == "willshaw":
        return (A > 0).astype(float), C * N
    if method.startswith("top1"):
        M = np.zeros_like(Ah)
        j = np.arange(N)
        top = Ah.argmax(0)
        M[top, j] = 1.0 if method == "top1" else quant(Ah[top, j], 4)
        M[:, tot[0] == 0] = 0
        return M, N * (lc + (4 if method == "top1_q4" else 0))
    if method.startswith("lowrank"):
        r = int(method[7:])
        U, S, Vt = np.linalg.svd(Ah, full_matrices=False)
        return (U[:, :r] * S[:r]) @ Vt[:r], r * (C + N) * 32
    if method.startswith("prune"):
        p = int(method[5:]) / 100
        prior = A.sum(1) / max(1.0, A.sum())
        info = np.abs(Ah - prior[:, None]).sum(0) * tot[0]   # 사전 분포와 다른 정도 × 경험 수 (한두 번 켜진 뉴런은 우연)
        keep = np.argsort(-info)[:max(1, int(N * p))]
        M = np.repeat(prior[:, None] * 1e-6, N, 1)   # 지운 뉴런은 투표하지 않음 (켜진 뉴런이 모두 지워졌을 때만 사전 분포로 결정)
        M[:, keep] = quant(Ah[:, keep], 4)
        idx_bits = min(N, len(keep) * math.ceil(math.log2(N)))   # 비트맵과 색인 목록 중 작은 쪽
        return M, len(keep) * C * 4 + idx_bits + C * 32
    raise ValueError(method)


METHODS = ["full", "q8", "q4", "q2", "bin_chance", "willshaw", "top1", "top1_q4", "lowrank2", "lowrank4", "lowrank8",
           "prune50", "prune25", "prune10"]


def scores(m, M, X):
    return np.vstack([m.code(X[i:i + 500]) @ M.T for i in range(0, len(X), 500)])


def job(a):
    task, seed, n_dg, k, methods = a
    t0 = time.time()
    d = load(task, seed)
    m = HippocampalMemory(d["X_train"].shape[1], d["n_classes"], n_dg=n_dg, k=k, seed=seed)
    m.store(d["X_train"], d["y_train"], chunk=500)
    out = []
    used = float((m.A.sum(0) > 0).mean())
    for meth in methods:
        M, bits = compress(m.A, meth)
        s = scores(m, M, d["X_test"])
        pred = (s + np.random.RandomState(0).rand(*s.shape) * 1e-9).argmax(1)
        out.append({"task": task, "seed": seed, "n_dg": n_dg, "k": k, "method": meth, "bits": bits,
                    "test": accuracy(pred, d["y_test"]), "used_frac": used})
    print(task, seed, n_dg, k, {o["method"]: round(o["test"] * 100, 2) for o in out}, f"{time.time() - t0:.0f}s",
          flush=True)
    return out


def main():
    jobs = [(t, s, MAIN[0], MAIN[1], METHODS) for t in TASKS for s in SEEDS]
    jobs += [(t, s, n, k, ["full", "q4", "top1"]) for t in TASKS for s in SEEDS for n, k in NK if (n, k) != MAIN]
    jobs.sort(key=lambda j: -(j[0] == "wtag") * j[2])
    rows = []
    with Pool(4) as pool:
        for r in pool.imap_unordered(job, jobs, chunksize=1):
            rows += r

    def agg(sel):
        v = [r["test"] * 100 for r in rows if sel(r)]
        b = [r["bits"] for r in rows if sel(r)]
        return float(np.mean(v)), float(np.std(v)), int(np.mean(b))

    summary = {}
    lines = ["# 실험 10: 기억 모델 압축", "",
             "해마형 기억(원샷 저장)을 저장한 뒤 저장 행렬을 압축하고 테스트 정확도를 잰다. 시드 5개 평균 ± 표준편차.",
             "저장량 = 남겨야 하는 비트 수 (투영 행렬은 시드로 재생성 가능하므로 제외). 1 KB = 8,192비트.",
             "비교: MLP64 = 3,340개 × 32비트 = 106,880비트 (tag12 40.14) / 격자 C4 = 3,720개 × 32비트 = 119,040비트.", ""]
    for t in TASKS:
        lines += [f"## {t} — DG {MAIN[0]}, k={MAIN[1]}", "", "| 방법 | 저장 비트 | 정확도 | full 대비 크기 |",
                  "|---|---|---|---|"]
        fb = agg(lambda r: r["task"] == t and r["n_dg"] == MAIN[0] and r["k"] == MAIN[1] and r["method"] == "full")[2]
        for meth in METHODS:
            mu, sd, b = agg(lambda r, meth=meth: r["task"] == t and (r["n_dg"], r["k"]) == MAIN and r["method"] == meth)
            summary[f"{t}/{meth}"] = [mu, sd, b]
            lines.append(f"| {meth} | {b:,} | {mu:.2f} ± {sd:.2f} | {b / fb:.3f} |")
        lines += ["", f"### {t} — DG 크기 N × 활성 수 k", "", "| N | k | full | q4 | top1 | top1 비트 | 사용된 뉴런 비율 |",
                  "|---|---|---|---|---|---|---|"]
        for n, k in NK:
            cells = []
            for meth in ["full", "q4", "top1"]:
                mu, sd, b = agg(lambda r, meth=meth, n=n, k=k: r["task"] == t and r["n_dg"] == n and r["k"] == k
                                and r["method"] == meth)
                summary[f"{t}/N{n}k{k}/{meth}"] = [mu, sd, b]
                cells.append(f"{mu:.2f}")
            used = np.mean([r["used_frac"] for r in rows if r["task"] == t and r["n_dg"] == n and r["k"] == k])
            lines.append(f"| {n} | {k} | " + " | ".join(cells) + f" | {summary[f'{t}/N{n}k{k}/top1'][2]:,} | {used:.3f} |")
        lines.append("")
    (ROOT / "results" / "exp10_memory_compression.json").write_text(json.dumps(
        {"seeds": SEEDS, "tasks": TASKS, "main": MAIN, "summary": summary, "rows": rows}, indent=2, ensure_ascii=False))
    (ROOT / "results" / "exp10_memory_compression.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
