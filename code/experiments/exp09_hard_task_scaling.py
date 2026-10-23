"""실험 09: 효율 단계 — 어려운 과제(wtag)에서 크기별 MLP vs 격자 vs 기억 (사용자 효율 원칙, REFERENCE E3·E7).

과제 wtag: 직전 4개 위치의 [단어(빈도 상위 300) | 품사] → 다음 품사 12종. 입력 1252차원, 학습 200,000개.
  기존 tag12(입력 39차원)는 n-gram 표가 천장(39.6%)이라 모델 크기가 의미가 없었다. 여기서는 단어 조합이 많아
  표로 외울 수 없고, 사전 점검에서 MLP가 47~48%까지 올라간다.

질문
  1. 같은 파라미터 수에서 격자가 MLP보다 정확한가
  2. 같은 정확도에 파라미터가 더 적은가
  3. 격자는 크기를 키우면 실제로 좋아지는가 (사전 점검: 높이 80이 40보다 나빴음 → 학습률 탓인지 확인)

모델
  mlp{h}   역전파 MLP, 은닉 h ∈ {0(로지스틱), 4, 8, 16, 32, 64, 128, 256}, Adam, 검증 조기 종료
  grid{H}  BioGrid C4(공명+확률판독+허브+residual), 폭 4, 높이 H ∈ {20, 40, 80, 160},
           최대 15에폭 + 검증 조기 종료(참을성 3). 학습률은 크기마다 {0.1, 0.3, 1.0} 중 시드 0 검증으로 선택
  mem{N}   해마형 기억(원샷, 1회 통과), DG N ∈ {2000, 8000, 32000}, k=20
  대조     mlp64_shuf, grid40_shuf (학습·검증 라벨 모두 섞음), 찍기 기준선, n-gram 표(품사 4개 / 마지막 단어)

연산량 (예시 하나 추론, 곱셈-덧셈 횟수)
  dense  = 입력을 밀집 벡터로 계산할 때
  sparse = 입력의 0이 아닌 칸(평균 약 7개)만 계산할 때 — 입력층 비용이 크게 줄어든다
  격자는 예측 때 9번 투표(n_votes=9)하므로 ×9를 포함한다. 공명 T=5는 같은 입력 전류에서 표본만 여러 번 뽑으므로 곱셈이 늘지 않는다.

실행: python3 experiments/exp09_hard_task_scaling.py
결과: results/exp09_hard_task_scaling.{json,md}
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
from src.data.language import make_word_task  # noqa: E402
from src.baselines.models import MLP, accuracy, majority_predict, ngram_table_predict  # noqa: E402
from src.biogrid.screen import BioGridX  # noqa: E402
from src.memory.hippocampus import HippocampalMemory  # noqa: E402
from experiments.exp07_memory_vs_learning import GRID_FEATS, RULE  # noqa: E402

N_TRAIN = 200000
SEEDS = [0, 1, 2, 3, 4]
MLP_H = [0, 4, 8, 16, 32, 64, 128, 256]
GRID_H = [20, 40, 80, 160]
GRID_LRS = [0.1, 0.3, 1.0]
MEM_N = [2000, 8000, 32000]
EPOCHS, PATIENCE, BATCH = 15, 3, 32

_DATA = {}


def data(seed):
    if seed not in _DATA:
        _DATA.clear()
        _DATA[seed] = make_word_task(n_train=N_TRAIN, seed=seed)
    return _DATA[seed]


def shuffled(d, seed):
    d = dict(d)
    r = np.random.RandomState(1000 + seed)
    d["y_train"], d["y_val"] = r.permutation(d["y_train"]), r.permutation(d["y_val"])
    return d


def run_grid(d, H, lr, seed):
    g = BioGridX(d["X_train"].shape[1], d["n_classes"], height=H, width=4, seed=seed, features=GRID_FEATS)
    rng = np.random.RandomState(seed + 100)
    best, best_acc, wait, ep_used = None, -1, 0, 0
    for ep in range(EPOCHS):
        o = rng.permutation(len(d["y_train"]))
        for i in range(0, len(o), BATCH):
            idx = o[i:i + BATCH]
            g.train_step(d["X_train"][idx], d["y_train"][idx], lr, **RULE)
        va = accuracy(g.predict(d["X_val"]), d["y_val"])
        if va > best_acc + 1e-4:
            best, best_acc, wait, ep_used = copy.deepcopy(g), va, 0, ep + 1
        else:
            wait += 1
            if wait >= PATIENCE:
                break
    return best, best_acc, ep_used


def macs(model, n_in, nnz, C=12):
    """예시 하나 추론의 곱셈-덧셈 횟수 (dense, sparse)."""
    kind, size = model
    if kind == "mlp":
        if size == 0:
            return n_in * C, int(round(nnz * C))
        return n_in * size + size * C, int(round(nnz * size + size * C))
    if kind == "grid":
        H, W = size, 4
        hub = W // 2
        rest = sum(H * H if c == hub else 3 * H for c in range(1, W)) + H * (W - 2)   # 지역/허브 + residual
        return 9 * (H * n_in + rest), int(round(9 * (H * nnz + rest)))
    if kind == "mem":
        return size * n_in + 20 * C, int(round(size * nnz + 20 * C))
    return None, None


def job(a):
    kind, size, seed, extra = a
    t0 = time.time()
    d = data(seed)
    n_in, C = d["X_train"].shape[1], d["n_classes"]
    out = {"kind": kind, "size": size, "seed": seed, "extra": extra}
    if kind == "mlp":
        dd = shuffled(d, seed) if extra == "shuf" else d
        m = MLP(n_in, size, C, seed)
        m.fit(dd["X_train"], dd["y_train"], dd["X_val"], dd["y_val"])
        out.update(test=accuracy(m.predict(d["X_test"]), d["y_test"]), n_params=m.n_params())
    elif kind == "grid":
        lr = extra if isinstance(extra, float) else extra[1]
        dd = shuffled(d, seed) if isinstance(extra, list) and extra[0] == "shuf" else d
        g, va, ep = run_grid(dd, size, lr, seed)
        out.update(test=accuracy(g.predict(d["X_test"]), d["y_test"]), val=va, epochs=ep, lr=lr,
                   n_params=g.n_params())
    elif kind == "mem":
        m = HippocampalMemory(n_in, C, n_dg=size, k=20, seed=seed)
        m.store(d["X_train"], d["y_train"], chunk=500)
        out.update(test=accuracy(m.predict(d["X_test"], chunk=500), d["y_test"]), n_params=m.n_params())
    elif kind == "base":
        cx = d["context"]
        out["majority"] = accuracy(majority_predict(d["y_train"], len(d["y_test"])), d["y_test"])
        out["ngram_tags"] = accuracy(ngram_table_predict(d["ctx_train"][:, cx:], d["y_train"], d["ctx_test"][:, cx:], C),
                                     d["y_test"])
        out["ngram_word"] = accuracy(ngram_table_predict(d["ctx_train"][:, [cx - 1]], d["y_train"],
                                                         d["ctx_test"][:, [cx - 1]], C), d["y_test"])
        out["nnz"] = float(d["X_test"].sum(1).mean())
        out["test"] = out["majority"]
    out["sec"] = round(time.time() - t0, 1)
    return out


def log(r):
    print(r["kind"], r["size"], r["seed"], r["extra"], round(r["test"] * 100, 2), f"{r['sec']}s", flush=True)


def main():
    rows = []
    with Pool(4) as pool:
        # 1단계: 격자 학습률 선택 (시드 0 검증)
        for r in pool.imap_unordered(job, [("grid", H, 0, lr) for H in GRID_H[::-1] for lr in GRID_LRS], chunksize=1):
            log(r); rows.append(r)
        best_lr = {H: max((r for r in rows if r["size"] == H), key=lambda r: r["val"])["lr"] for H in GRID_H}
        print("chosen lr", best_lr, flush=True)
        jobs = [("grid", H, s, best_lr[H]) for H in GRID_H[::-1] for s in SEEDS if s != 0]
        jobs += [("grid", 40, s, ["shuf", best_lr[40]]) for s in SEEDS]
        jobs += [("mlp", h, s, None) for h in MLP_H[::-1] for s in SEEDS]
        jobs += [("mlp", 64, s, "shuf") for s in SEEDS]
        jobs += [("mem", n, s, None) for n in MEM_N[::-1] for s in SEEDS]
        jobs += [("base", 0, s, None) for s in SEEDS]
        for r in pool.imap_unordered(job, jobs, chunksize=1):
            log(r); rows.append(r)
    # 격자 시드 0은 선택된 학습률의 1단계 결과를 그대로 쓴다
    final = [r for r in rows if not (r["kind"] == "grid" and r["seed"] == 0 and not isinstance(r["extra"], list)
                                     and r["lr"] != best_lr[r["size"]])]
    nnz = float(np.mean([r["nnz"] for r in final if r["kind"] == "base"]))
    n_in = 4 * (300 + 13)

    def stats(sel):
        v = [r["test"] * 100 for r in final if sel(r)]
        return float(np.mean(v)), float(np.std(v)), len(v)

    table = []
    for kind, sizes in [("mlp", MLP_H), ("grid", GRID_H), ("mem", MEM_N)]:
        for s in sizes:
            sel = lambda r, k=kind, s=s: r["kind"] == k and r["size"] == s and r["extra"] not in ("shuf",) \
                and not isinstance(r["extra"], list)
            m, sd, n = stats(sel)
            npar = next(r["n_params"] for r in final if sel(r))
            dm, sm = macs((kind, s), n_in, nnz)
            extra = ""
            if kind == "grid":
                extra = f"lr {best_lr[s]}, 에폭 " + str([r["epochs"] for r in final if sel(r)])
            name = {"mlp": "로지스틱" if s == 0 else f"MLP {s}", "grid": f"격자 높이 {s}", "mem": f"기억 DG {s}"}[kind]
            table.append(dict(name=name, kind=kind, size=s, mean=m, std=sd, n=n, n_params=npar, macs_dense=dm,
                              macs_sparse=sm, note=extra))
    ctrl = {"mlp64_shuf": stats(lambda r: r["kind"] == "mlp" and r["extra"] == "shuf"),
            "grid40_shuf": stats(lambda r: r["kind"] == "grid" and isinstance(r["extra"], list))}
    base = {k: stats(lambda r, k=k: False) for k in []}
    for k in ["majority", "ngram_tags", "ngram_word"]:
        v = [r[k] * 100 for r in final if r["kind"] == "base"]
        base[k] = (float(np.mean(v)), float(np.std(v)), len(v))

    lines = ["# 실험 09: 어려운 과제(wtag)에서 크기별 비교 — MLP vs 격자 vs 기억", "",
             f"과제: 직전 4개 위치의 [단어(상위 300) | 품사] → 다음 품사 12종. 입력 {n_in}차원 (평균 {nnz:.1f}칸만 1), "
             f"학습 {N_TRAIN:,} / 검증 5,000 / 테스트 5,000. 테스트 정확도, 시드 5개 평균 ± 표준편차.",
             "연산량 = 예시 하나 추론의 곱셈-덧셈 수. sparse는 입력의 0이 아닌 칸만 계산한 값. 격자는 9번 투표 포함.", "",
             "| 모델 | 파라미터 | 정확도 | 연산량 dense | 연산량 sparse | 비고 |", "|---|---|---|---|---|---|"]
    for t in table:
        lines.append(f"| {t['name']} | {t['n_params']:,} | {t['mean']:.2f} ± {t['std']:.2f} | {t['macs_dense']:,} | "
                     f"{t['macs_sparse']:,} | {t['note']} |")
    lines += ["", "## 기준선과 대조", "", "| 항목 | 정확도 |", "|---|---|"]
    for k, (m, sd, _) in {**base, **ctrl}.items():
        lines.append(f"| {k} | {m:.2f} ± {sd:.2f} |")
    lines += ["", f"격자 학습률 선택 (시드 0 검증): {best_lr}", ""]
    (ROOT / "results" / "exp09_hard_task_scaling.json").write_text(json.dumps(
        {"n_train": N_TRAIN, "seeds": SEEDS, "best_lr": {str(k): v for k, v in best_lr.items()}, "nnz": nnz,
         "table": table, "controls": ctrl, "baselines": base, "rows": rows}, indent=2, ensure_ascii=False))
    (ROOT / "results" / "exp09_hard_task_scaling.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
