"""실험 01: BioGrid 최소 코어 + 3-요소 규칙(전역 도파민 × 자격 흔적).

질문
  1. 역전파 없이 전역 보상 하나로 격자가 무언가를 배우는가? (합격선 1: 기준선 + 5%p, 셔플 대조는 기준선)
  2. 격자 깊이(폭 W)가 늘면 어떻게 되는가?

비교 (같은 데이터·분할·시드)
  - learn        : 3-요소 규칙으로 학습, 구역 판독, 확률적 발화 9회 다수결로 테스트
  - learn_shuf   : learn과 같으나 학습·검증 라벨을 모두 셔플 (테스트 라벨은 원본)
  - fixed_zone   : 학습 없음. 무작위 초기 격자 + 구역 판독
  - fixed_linear : 학습 없음. 무작위 초기 격자의 모든 뉴런 발화율에 역전파 선형 판독 (reservoir 대조)
학습률은 시드 0의 검증 정확도로만 고른다 (테스트는 보지 않음).

실행: python3 experiments/exp01_biogrid_reinforce.py
결과: results/exp01_biogrid_reinforce.{json,md}
"""
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data.language import make_task  # noqa: E402
from src.baselines.models import MLP, accuracy, majority_predict  # noqa: E402
from src.biogrid.core import BioGrid  # noqa: E402

CONFIG = {
    "tasks": ["content", "tag12"],
    "widths": [1, 2, 4, 8],
    "height": 40,
    "seeds": [0, 1, 2, 3, 4],
    "lr_grid": [0.1, 0.3, 1.0],
    "epochs": 15,
    "patience": 4,
    "batch": 32,
    "n_votes": 9,
    "data": {"context": 3, "n_train": 20000, "n_val": 5000, "n_test": 5000},
}
TARGETS = {"content": {"majority": 51.90, "logreg": 67.16, "mlp64": 69.59},
           "tag12": {"majority": 23.94, "logreg": 38.55, "mlp64": 40.14}}


def train_grid(d, width, lr, seed, y_train, y_val):
    g = BioGrid(d["X_train"].shape[1], d["n_classes"], height=CONFIG["height"], width=width, seed=seed)
    order_rng = np.random.RandomState(seed + 100)
    best_acc, best, wait, curve = -1.0, None, 0, []
    for ep in range(CONFIG["epochs"]):
        o = order_rng.permutation(len(y_train))
        for i in range(0, len(o), CONFIG["batch"]):
            idx = o[i:i + CONFIG["batch"]]
            g.train_step(d["X_train"][idx], y_train[idx], lr)
        va = accuracy(g.predict(d["X_val"], CONFIG["n_votes"]), y_val)
        curve.append(round(va * 100, 2))
        if va > best_acc + 1e-4:
            best_acc, best, wait = va, copy.deepcopy(g), 0
        else:
            wait += 1
            if wait >= CONFIG["patience"]:
                break
    return best, best_acc, curve


def pick_lr(task, width):
    d = make_task(task, seed=0, **CONFIG["data"])
    scores = {lr: train_grid(d, width, lr, 0, d["y_train"], d["y_val"])[1] for lr in CONFIG["lr_grid"]}
    return max(scores, key=scores.get), scores


def run():
    rows, lr_choice = [], {}
    for task in CONFIG["tasks"]:
        for width in CONFIG["widths"]:
            lr, scores = pick_lr(task, width)
            lr_choice[f"{task}/W{width}"] = {"lr": lr, "val_by_lr": {str(k): round(v * 100, 2) for k, v in scores.items()}}
            print(f"[{task} W={width}] lr 선택 {lr} (검증 {scores})", flush=True)
            for seed in CONFIG["seeds"]:
                t0 = time.time()
                d = make_task(task, seed=seed, **CONFIG["data"])
                yt = d["y_test"]
                res = {"task": task, "width": width, "seed": seed, "lr": lr}
                res["majority"] = accuracy(majority_predict(d["y_train"], len(yt)), yt)

                g, _, curve = train_grid(d, width, lr, seed, d["y_train"], d["y_val"])
                res["learn"] = accuracy(g.predict(d["X_test"], CONFIG["n_votes"]), yt)
                res["learn_silent"] = g.last_silent
                res["learn_val_curve"] = curve
                res["n_params"] = g.n_params()

                rs = np.random.RandomState(1000 + seed)
                ys, yvs = rs.permutation(d["y_train"]), rs.permutation(d["y_val"])
                gs, _, _ = train_grid(d, width, lr, seed, ys, yvs)
                res["learn_shuf"] = accuracy(gs.predict(d["X_test"], CONFIG["n_votes"]), yt)

                g0 = BioGrid(d["X_train"].shape[1], d["n_classes"], height=CONFIG["height"], width=width, seed=seed)
                res["fixed_zone"] = accuracy(g0.predict(d["X_test"], CONFIG["n_votes"]), yt)
                F_tr, F_va, F_te = g0.features(d["X_train"]), g0.features(d["X_val"]), g0.features(d["X_test"])
                m = MLP(F_tr.shape[1], 0, d["n_classes"], seed)
                m.fit(F_tr, d["y_train"], F_va, d["y_val"])
                res["fixed_linear"] = accuracy(m.predict(F_te), yt)
                res["sec"] = round(time.time() - t0, 1)
                rows.append(res)
                print({k: (round(v * 100, 2) if isinstance(v, float) and k not in ("lr", "sec", "learn_silent") else v)
                       for k, v in res.items() if k != "learn_val_curve"}, flush=True)
    return rows, lr_choice


def write_report(rows, lr_choice):
    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    keys = ["majority", "learn", "learn_shuf", "fixed_zone", "fixed_linear"]
    summary = {}
    for task in CONFIG["tasks"]:
        for w in CONFIG["widths"]:
            r = [x for x in rows if x["task"] == task and x["width"] == w]
            summary[f"{task}/W{w}"] = {k: [round(float(np.mean([x[k] for x in r])) * 100, 2),
                                           round(float(np.std([x[k] for x in r])) * 100, 2)] for k in keys}
            summary[f"{task}/W{w}"]["n_params"] = r[0]["n_params"]
            summary[f"{task}/W{w}"]["silent"] = round(float(np.mean([x["learn_silent"] for x in r])), 3)
    (out / "exp01_biogrid_reinforce.json").write_text(json.dumps(
        {"config": CONFIG, "lr_choice": lr_choice, "rows": rows, "summary": summary}, indent=2, ensure_ascii=False))

    lines = ["# 실험 01: BioGrid 최소 코어 + 3-요소 규칙 (전역 도파민 × 자격 흔적)", "",
             f"격자 높이 {CONFIG['height']}, 폭 W. 시드 {len(CONFIG['seeds'])}개 평균 ± 표준편차, 테스트 {CONFIG['data']['n_test']}개. "
             "학습률은 시드 0 검증 정확도로 선택.", ""]
    for task in CONFIG["tasks"]:
        t = TARGETS[task]
        lines += [f"## {task}", "",
                  f"목표치: 기준선 {t['majority']} / 로지스틱 {t['logreg']} / MLP64 {t['mlp64']}", "",
                  "| 폭 W | 파라미터 | 학습(3-요소) | 셔플 대조 | 무작위 고정 + 구역 판독 | 무작위 고정 + 선형 판독(역전파) | lr |",
                  "|---|---|---|---|---|---|---|"]
        for w in CONFIG["widths"]:
            s = summary[f"{task}/W{w}"]
            f = lambda k: f"{s[k][0]:.2f} ± {s[k][1]:.2f}"  # noqa: E731
            lines.append(f"| {w} | {s['n_params']} | {f('learn')} | {f('learn_shuf')} | {f('fixed_zone')} | "
                         f"{f('fixed_linear')} | {lr_choice[f'{task}/W{w}']['lr']} |")
        lines.append("")
    (out / "exp01_biogrid_reinforce.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    rows, lr_choice = run()
    write_report(rows, lr_choice)
