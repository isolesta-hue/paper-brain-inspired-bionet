"""실험 03: 참고 자료(REFERENCE.md) 메커니즘 1차 스크리닝.

방법
  - 베이스: 그룹 A = A3(출력 구역 오차 + 바톤 터치 · 민감도), 폭 4·8
            그룹 B = A0(전역 도파민), 폭 1·2  — 학습 규칙 변형(L2, L4, L13)만
  - 베이스에 기능 하나만 켠다 (기본 설정값은 원본 코드의 값)
  - 학습률은 기능·과제·폭마다 {0.3, 1.0, 3.0} 중 시드 0 검증 정확도로 선택 (테스트는 안 봄)
  - 시드 3개 (1차 스크리닝). 효과가 보이는 기능만 2차에서 시드 5개 + 셔플 대조로 확정한다.
  - 판정: 베이스 대비 차이 Δ와, 시드 간 표준편차로 만든 대략적 오차 범위를 같이 기록

실행: python3 experiments/exp03_feature_screening.py  (4개 프로세스 병렬)
결과: results/exp03_feature_screening.{json,md}
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
from src.data.language import make_task  # noqa: E402
from src.baselines.models import accuracy  # noqa: E402
from src.biogrid.screen import BioGridX, DEFAULTS  # noqa: E402

GROUP_B = ["hebb_pm1", "dopa_asym", "conn_tag"]
GROUP_A = [k for k in DEFAULTS if k not in GROUP_B]
CONFIG = {
    "tasks": ["content", "tag12"],
    "widths_A": [4, 8],
    "widths_B": [1, 2],
    "height": 40,
    "seeds": [0, 1, 2],
    "lr_grid": [0.3, 1.0, 3.0],
    "epochs": 15,
    "patience": 4,
    "batch": 32,
    "n_votes": 9,
    "data": {"context": 3, "n_train": 20000, "n_val": 5000, "n_test": 5000},
}
RULE_A = dict(output_signal="zone_target", hidden_credit="relay_w", gate="sensitivity")
RULE_B = dict(output_signal="global", hidden_credit="global")
_DATA = {}


def data(task, seed):
    key = (task, seed)
    if key not in _DATA:
        _DATA[key] = make_task(task, seed=seed, **CONFIG["data"])
    return _DATA[key]


def train(task, width, feature, lr, seed, rule):
    d = data(task, seed)
    feats = {} if feature == "base" else {feature: {}}
    g = BioGridX(d["X_train"].shape[1], d["n_classes"], height=CONFIG["height"], width=width, seed=seed, features=feats)
    order_rng = np.random.RandomState(seed + 100)
    best_acc, best, wait, curve = -1.0, None, 0, []
    for _ in range(CONFIG["epochs"]):
        o = order_rng.permutation(len(d["y_train"]))
        for i in range(0, len(o), CONFIG["batch"]):
            idx = o[i:i + CONFIG["batch"]]
            g.train_step(d["X_train"][idx], d["y_train"][idx], lr, **rule)
        if not (np.all(np.isfinite(g.W_loc)) and np.all(np.isfinite(g.W_in)) and np.all(np.isfinite(g.V))):
            return {"invalid": True, "val": 0.0, "curve": curve}
        va = accuracy(g.predict(d["X_val"], CONFIG["n_votes"]), d["y_val"])
        curve.append(round(va * 100, 2))
        if va > best_acc + 1e-4:
            best_acc, best, wait = va, copy.deepcopy(g), 0
        else:
            wait += 1
            if wait >= CONFIG["patience"]:
                break
    out = {"val": best_acc, "curve": curve, "n_params": best.n_params(), "replaced": best.replaced,
           "alive": float(best.alive.mean())}
    out["test"] = accuracy(best.predict(d["X_test"], CONFIG["n_votes"]), d["y_test"])
    _, rec = best.forward(d["X_val"][:1000])
    out["firing"] = [round(float(r[2].mean()), 3) for r in rec]
    return out


def job(args):
    task, width, feature, lr, seed, group = args
    t0 = time.time()
    res = train(task, width, feature, lr, seed, RULE_A if group == "A" else RULE_B)
    res.update({"task": task, "width": width, "feature": feature, "lr": lr, "seed": seed, "group": group,
                "sec": round(time.time() - t0, 1)})
    return res


def configs():
    for task in CONFIG["tasks"]:
        for w in CONFIG["widths_A"]:
            for feat in ["base"] + GROUP_A:
                yield task, w, feat, "A"
        for w in CONFIG["widths_B"]:
            for feat in ["base"] + GROUP_B:
                yield task, w, feat, "B"


def run():
    cfgs = list(configs())
    with Pool(4) as pool:
        # 1) 학습률 선택: 시드 0, 검증 정확도
        lr_jobs = [(t, w, f, lr, 0, g) for (t, w, f, g) in cfgs for lr in CONFIG["lr_grid"]]
        lr_res = pool.map(job, lr_jobs, chunksize=1)
        best_lr = {}
        for r in lr_res:
            k = (r["task"], r["width"], r["feature"], r["group"])
            if k not in best_lr or r["val"] > best_lr[k][1]:
                best_lr[k] = (r["lr"], r["val"])
        print("학습률 선택 완료", flush=True)
        # 2) 선택된 학습률로 시드 3개
        seed_jobs = [(t, w, f, best_lr[(t, w, f, g)][0], s, g) for (t, w, f, g) in cfgs for s in CONFIG["seeds"]]
        rows = []
        for r in pool.imap_unordered(job, seed_jobs, chunksize=1):
            rows.append(r)
            print(r["task"], r["width"], r["group"], r["feature"], "seed", r["seed"], "lr", r["lr"],
                  "test", round(r.get("test", 0) * 100, 2), f"{r['sec']}s", flush=True)
    return rows, {f"{k[0]}/W{k[1]}/{k[3]}/{k[2]}": {"lr": v[0], "val": round(v[1] * 100, 2)} for k, v in best_lr.items()}, lr_res


def write_report(rows, lr_choice, lr_res):
    out = ROOT / "results"
    summ = {}
    for r in rows:
        k = (r["task"], r["width"], r["group"], r["feature"])
        summ.setdefault(k, []).append(r)
    table = {}
    for k, rs in summ.items():
        tests = [x.get("test", 0.0) for x in rs]
        table[k] = {"mean": float(np.mean(tests)) * 100, "std": float(np.std(tests)) * 100,
                    "invalid": any(x.get("invalid") for x in rs), "n_params": rs[0].get("n_params"),
                    "min_firing": float(np.mean([min(x.get("firing", [0])) for x in rs])),
                    "replaced": float(np.mean([x.get("replaced", 0) for x in rs])),
                    "alive": float(np.mean([x.get("alive", 1) for x in rs]))}
    (out / "exp03_feature_screening.json").write_text(json.dumps(
        {"config": CONFIG, "lr_choice": lr_choice, "rows": rows, "lr_runs": lr_res,
         "summary": {"/".join(map(str, k)): v for k, v in table.items()}}, indent=2, ensure_ascii=False))

    lines = ["# 실험 03: 참고 자료 메커니즘 1차 스크리닝", "",
             "베이스에 기능 하나만 켰을 때의 테스트 정확도 (시드 3개 평균 ± 표준편차). Δ = 기능 − 베이스 (%p).",
             "그룹 A 베이스 = A3(구역 오차 + 바톤 터치·민감도), 그룹 B 베이스 = A0(전역 도파민).",
             "학습률은 칸마다 {0.3, 1, 3} 중 시드 0 검증으로 선택. 시드 3개라 |Δ| < 약 1%p는 노이즈 범위로 본다.", ""]
    for group, widths, feats in [("A", CONFIG["widths_A"], GROUP_A), ("B", CONFIG["widths_B"], GROUP_B)]:
        lines += [f"## 그룹 {group}", "",
                  "| 기능 | " + " | ".join(f"{t} 폭{w}" for t in CONFIG["tasks"] for w in widths) + " |",
                  "|---|" + "---|" * (len(CONFIG["tasks"]) * len(widths))]
        for feat in ["base"] + feats:
            cells = []
            for t in CONFIG["tasks"]:
                for w in widths:
                    e = table[(t, w, group, feat)]
                    b = table[(t, w, group, "base")]
                    if feat == "base":
                        cells.append(f"**{e['mean']:.2f}** ± {e['std']:.2f}")
                    else:
                        d = e["mean"] - b["mean"]
                        flag = " ⚠NaN" if e["invalid"] else ""
                        cells.append(f"{e['mean']:.2f} ({d:+.2f}){flag}")
            lines.append(f"| {feat} | " + " | ".join(cells) + " |")
        lines.append("")
    (out / "exp03_feature_screening.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    rows, lr_choice, lr_res = run()
    write_report(rows, lr_choice, lr_res)
