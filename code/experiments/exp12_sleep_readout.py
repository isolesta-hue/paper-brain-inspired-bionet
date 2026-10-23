"""실험 12: 수면기 판독 정리 — 격자를 고정한 뒤 판독 뉴런만 따로 학습 (REFERENCE R4 + L14 수면기).

동기 (실험 11): 격자와 함께 온라인으로 배우는 판독은 효과가 거의 없었지만, 사전 점검에서 격자를 고정한 뒤 판독을
따로 학습하면 wtag +1.5%p. 판독이 계속 변하는 표현을 쫓아가지 못하는 것이 원인으로 보인다.

절차 (과제·시드마다)
  1. 깨어 있는 학습: 격자 C4(폭 4, 높이 40)를 지금까지와 같이 학습 (최대 15에폭, 구역 판독 검증 조기 종료)
  2. 수면: 격자를 고정. 학습 데이터를 다시 흘려 마지막 열 발화 확률(9번 평균)을 얻고, 판독 뉴런만 델타 규칙으로 학습
     ΔW_out = lr · (정답 one-hot − softmax) ⊗ p   (지역 규칙, 격자로는 신호를 보내지 않음)
     최대 20에폭, 판독 검증 정확도가 가장 높은 에폭을 쓴다.

판독 방식 (같은 격자 위에서 비교)
  zone        구역 합 판독 (지금까지)
  zone+bias   구역 합 + 클래스별 기본값만 학습 (클래스 수만큼의 값) — 이득 중 "기본값" 몫을 분리
  sleepR      판독 뉴런 전체 학습 (H × C + C)
대조
  random+sleepR  학습하지 않은 무작위 격자 + 수면 판독 (판독만으로 얼마나 나오는가)
  shuf+sleepR    학습·검증 라벨을 모두 섞고 전체 절차 반복

판독 학습률은 {0.3, 1.0, 3.0} 중 시드 0 검증으로 과제별 선택 (zone+bias와 sleepR 각각).
목표치: content MLP64 69.59 / tag12 MLP64 40.14 / wtag MLP16 46.20, MLP64 46.70 (실험 09).

실행: python3 experiments/exp12_sleep_readout.py
결과: results/exp12_sleep_readout.{json,md}
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
from experiments.exp11_learned_readout import DeltaReadout, load, TASKS  # noqa: E402

SEEDS = [0, 1, 2, 3, 4]
OUT_LRS = [0.3, 1.0, 3.0]
EPOCHS, PATIENCE, BATCH, H = 15, 3, 32, 40
SLEEP_EPOCHS = 20
READOUTS = ["zone+bias", "sleepR"]


def train_grid(d, task, seed):
    g = BioGridX(d["X_train"].shape[1], d["n_classes"], height=H, width=4, seed=seed, features=GRID_FEATS)
    rng = np.random.RandomState(seed + 100)
    best, best_acc, wait, ep_used = copy.deepcopy(g), -1, 0, 0
    for ep in range(EPOCHS):
        o = rng.permutation(len(d["y_train"]))
        for i in range(0, len(o), BATCH):
            idx = o[i:i + BATCH]
            g.train_step(d["X_train"][idx], d["y_train"][idx], TASKS[task][1], **RULE)
        va = accuracy(g.predict(d["X_val"]), d["y_val"])
        if va > best_acc + 1e-4:
            best, best_acc, wait, ep_used = copy.deepcopy(g), va, 0, ep + 1
        else:
            wait += 1
            if wait >= PATIENCE:
                break
    return best, ep_used


def feats(g, X, n_votes=9, chunk=5000):
    out = []
    for i in range(0, len(X), chunk):
        out.append(np.mean([g.forward(X[i:i + chunk])[1][-1][1] for _ in range(n_votes)], 0))
    return np.vstack(out)


def sleep_readout(F, y, Fv, yv, Ft, kind, lr, zone_of_row, n_classes, seed):
    ro = DeltaReadout(F.shape[1], n_classes)
    if kind == "zone+bias":
        ro.W = (zone_of_row[None, :] == np.arange(n_classes)[:, None]).astype(float)
    rng = np.random.RandomState(seed + 200)
    best = (-1, None, 0)
    for ep in range(SLEEP_EPOCHS):
        o = rng.permutation(len(y))
        for i in range(0, len(o), BATCH):
            idx = o[i:i + BATCH]
            W0 = ro.W.copy()
            ro.update(F[idx], y[idx], lr)
            if kind == "zone+bias":
                ro.W = W0                      # 기본값(b)만 학습
        va = accuracy(ro.probs(Fv).argmax(1), yv)
        if va > best[0]:
            best = (va, ro.probs(Ft).argmax(1), ep + 1)
    return best


def job(a):
    task, seed, variant, lrs = a       # lrs: {readout kind: [lr, ...]}
    t0 = time.time()
    d = load(task, seed)
    if variant == "shuf":
        r = np.random.RandomState(1000 + seed)
        d["y_train"], d["y_val"] = r.permutation(d["y_train"]), r.permutation(d["y_val"])
    if variant == "random":
        g = BioGridX(d["X_train"].shape[1], d["n_classes"], height=H, width=4, seed=seed, features=GRID_FEATS)
        ep = 0
    else:
        g, ep = train_grid(d, task, seed)
    out = {"task": task, "seed": seed, "variant": variant, "grid_epochs": ep, "n_params_grid": g.n_params(),
           "zone": accuracy(g.predict(d["X_test"]), d["y_test"])}
    F, Fv, Ft = feats(g, d["X_train"]), feats(g, d["X_val"]), feats(g, d["X_test"])
    for kind, lr_list in lrs.items():
        for lr in lr_list:
            va, pred, sep = sleep_readout(F, d["y_train"], Fv, d["y_val"], Ft, kind, lr, g.zone_of_row,
                                          d["n_classes"], seed)
            out[f"{kind}@{lr}"] = {"test": accuracy(pred, d["y_test"]), "val": va, "sleep_epochs": sep}
    out["sec"] = round(time.time() - t0, 1)
    print(task, seed, variant, "zone", round(out["zone"] * 100, 2),
          {k: round(v["test"] * 100, 2) for k, v in out.items() if "@" in k}, f"{out['sec']}s", flush=True)
    return out


def main():
    rows = []
    order = lambda j: -(j[0] == "wtag")
    all_lrs = {k: OUT_LRS for k in READOUTS}
    with Pool(4) as pool:
        first = [(t, 0, "grid", all_lrs) for t in TASKS]
        for r in pool.imap_unordered(job, sorted(first, key=order), chunksize=1):
            rows.append(r)
        best = {t: {k: max(OUT_LRS, key=lambda lr: next(r for r in rows if r["task"] == t)[f"{k}@{lr}"]["val"])
                    for k in READOUTS} for t in TASKS}
        print("chosen readout lr", best, flush=True)
        jobs = [(t, s, "grid", all_lrs) for t in TASKS for s in SEEDS if s != 0]
        jobs += [(t, s, v, {"sleepR": [best[t]["sleepR"]]}) for t in TASKS for s in SEEDS for v in ["random", "shuf"]]
        for r in pool.imap_unordered(job, sorted(jobs, key=order), chunksize=1):
            rows.append(r)

    def acc(t, v, key):
        rr = sorted([r for r in rows if r["task"] == t and r["variant"] == v], key=lambda r: r["seed"])
        return np.array([(r[key] if key == "zone" else r[key]["test"]) * 100 for r in rr])

    target = {"content": "MLP64 69.59, 로지스틱 67.16", "tag12": "MLP64 40.14, 로지스틱 38.55",
              "wtag": "MLP16 46.20, MLP64 46.70, 로지스틱 44.52"}
    lines = ["# 실험 12: 수면기 판독 정리", "",
             "깨어 있는 동안 격자 C4(폭 4, 높이 40)를 학습 → 수면 때 격자를 고정하고 판독 뉴런만 델타 규칙으로 학습.",
             "테스트 정확도, 시드 5개 평균 ± 표준편차. 차이는 같은 시드끼리 짝지은 평균 ± 표준오차.",
             f"판독 학습률 (시드 0 검증으로 선택): {best}", ""]
    summary = {}
    for t in TASKS:
        z = acc(t, "grid", "zone")
        zb = acc(t, "grid", f"zone+bias@{best[t]['zone+bias']}")
        sr = acc(t, "grid", f"sleepR@{best[t]['sleepR']}")
        rnd = acc(t, "random", f"sleepR@{best[t]['sleepR']}")
        sh = acc(t, "shuf", f"sleepR@{best[t]['sleepR']}")
        se = lambda x: x.std(ddof=1) / np.sqrt(len(x))
        summary[t] = {"zone": [z.mean(), z.std()], "zone+bias": [zb.mean(), zb.std()], "sleepR": [sr.mean(), sr.std()],
                      "random+sleepR": [rnd.mean(), rnd.std()], "shuf+sleepR": [sh.mean(), sh.std()],
                      "sleepR-zone": [(sr - z).mean(), se(sr - z)], "zone+bias-zone": [(zb - z).mean(), se(zb - z)]}
        lines += [f"## {t}", "", "| 판독 | 정확도 | 구역 판독 대비 |", "|---|---|---|",
                  f"| zone (지금까지) | {z.mean():.2f} ± {z.std():.2f} | – |",
                  f"| zone+bias | {zb.mean():.2f} ± {zb.std():.2f} | {(zb - z).mean():+.2f} ± {se(zb - z):.2f} |",
                  f"| **sleepR** | **{sr.mean():.2f} ± {sr.std():.2f}** | {(sr - z).mean():+.2f} ± {se(sr - z):.2f} |",
                  f"| random+sleepR (대조) | {rnd.mean():.2f} ± {rnd.std():.2f} | |",
                  f"| shuf+sleepR (대조) | {sh.mean():.2f} ± {sh.std():.2f} | |",
                  "", f"시드별 sleepR: {np.round(sr, 2).tolist()}  /  목표치: {target[t]}", ""]
    (ROOT / "results" / "exp12_sleep_readout.json").write_text(json.dumps(
        {"seeds": SEEDS, "best_lr": best, "summary": summary, "rows": rows}, indent=2, ensure_ascii=False,
        default=float))
    (ROOT / "results" / "exp12_sleep_readout.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
