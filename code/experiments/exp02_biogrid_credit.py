"""실험 02: 2단계 — credit assignment (선택적 보상).

1단계에서 깊은 격자가 실패한 원인 중 "모든 뉴런이 같은 도파민 숫자 하나를 받는다"를 다룬다.
학습 신호를 사다리처럼 하나씩 바꿔 가며 비교한다 (같은 격자·데이터·시드).

  A0 global        : 1단계와 동일 (전역 도파민 × 자격 흔적)
  A1 zone          : 출력 뉴런만 방향 있는 오차(정답 구역 1, 나머지 0 − p). 은닉은 전역 도파민   [설계서 c-7, V3.1 클래스 구역 태그]
  A2 relay_fired   : A1 + 오차를 오른쪽→왼쪽 이웃 바톤 터치, 실제 발화한 뉴런만 받음                [좌표 ID 역추적 / "흐른 라인에만"]
  A3 relay_sens    : A1 + 바톤 터치, 문턱 근처 민감도 p(1-p)만큼 받음                              [설계서 c-9 뒷뉴런 피드백]
  A4 relay_fa_sens : A3와 같으나 전달 세기가 고정 무작위 (순방향 가중치를 몰라도 되는 버전)

A2·A3는 순방향 가중치로 오차를 전달하므로 이웃 간 국소 역전파에 해당한다(전역 역전파는 아님, 설계서 c-9가 허용한 범위).
A4는 그 가중치 정보조차 쓰지 않는다.

실행: python3 experiments/exp02_biogrid_credit.py
결과: results/exp02_biogrid_credit.{json,md}
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
from src.baselines.models import accuracy  # noqa: E402
from src.biogrid.core import BioGrid  # noqa: E402

VARIANTS = {
    "A0_global": dict(output_signal="global", hidden_credit="global"),
    "A1_zone": dict(output_signal="zone_target", hidden_credit="global"),
    "A2_relay_fired": dict(output_signal="zone_target", hidden_credit="relay_w", gate="fired"),
    "A3_relay_sens": dict(output_signal="zone_target", hidden_credit="relay_w", gate="sensitivity"),
    "A4_relay_fa_sens": dict(output_signal="zone_target", hidden_credit="relay_fa", gate="sensitivity"),
}
CONFIG = {
    "tasks": ["content", "tag12"],
    "widths": [2, 4, 8],
    "height": 40,
    "seeds": [0, 1, 2, 3, 4],
    "lr_grid": [0.1, 0.3, 1.0],
    "epochs": 15,
    "patience": 4,
    "batch": 32,
    "n_votes": 9,
    "shuffle_variants": ["A3_relay_sens", "A4_relay_fa_sens"],
    "data": {"context": 3, "n_train": 20000, "n_val": 5000, "n_test": 5000},
}
TARGETS = {"content": {"majority": 51.90, "logreg": 67.16, "mlp64": 69.59},
           "tag12": {"majority": 23.94, "logreg": 38.55, "mlp64": 40.14}}


def train_grid(d, width, lr, seed, y_train, y_val, kw):
    g = BioGrid(d["X_train"].shape[1], d["n_classes"], height=CONFIG["height"], width=width, seed=seed)
    order_rng = np.random.RandomState(seed + 100)
    best_acc, best, wait, curve = -1.0, None, 0, []
    for _ in range(CONFIG["epochs"]):
        o = order_rng.permutation(len(y_train))
        for i in range(0, len(o), CONFIG["batch"]):
            idx = o[i:i + CONFIG["batch"]]
            g.train_step(d["X_train"][idx], y_train[idx], lr, **kw)
        if not np.all(np.isfinite(g.W_loc)) or not np.all(np.isfinite(g.W_in)):
            raise FloatingPointError("NaN/inf 가중치 — 이 실행은 무효")
        va = accuracy(g.predict(d["X_val"], CONFIG["n_votes"]), y_val)
        curve.append(round(va * 100, 2))
        if va > best_acc + 1e-4:
            best_acc, best, wait = va, copy.deepcopy(g), 0
        else:
            wait += 1
            if wait >= CONFIG["patience"]:
                break
    return best, best_acc, curve


def firing_by_column(g, X):
    _, rec = g.forward(X)
    return [round(float(r[2].mean()), 3) for r in rec]


def run():
    rows, lr_choice = [], {}
    for task in CONFIG["tasks"]:
        d0 = make_task(task, seed=0, **CONFIG["data"])
        for width in CONFIG["widths"]:
            for vname, kw in VARIANTS.items():
                scores = {lr: train_grid(d0, width, lr, 0, d0["y_train"], d0["y_val"], kw)[1] for lr in CONFIG["lr_grid"]}
                lr = max(scores, key=scores.get)
                lr_choice[f"{task}/W{width}/{vname}"] = {"lr": lr, "val_by_lr": {str(k): round(v * 100, 2) for k, v in scores.items()}}
                for seed in CONFIG["seeds"]:
                    t0 = time.time()
                    d = make_task(task, seed=seed, **CONFIG["data"])
                    g, _, curve = train_grid(d, width, lr, seed, d["y_train"], d["y_val"], kw)
                    res = {"task": task, "width": width, "variant": vname, "seed": seed, "lr": lr,
                           "test": accuracy(g.predict(d["X_test"], CONFIG["n_votes"]), d["y_test"]),
                           "silent": g.last_silent, "firing": firing_by_column(g, d["X_val"][:1000]),
                           "val_curve": curve}
                    if vname in CONFIG["shuffle_variants"]:
                        rs = np.random.RandomState(1000 + seed)
                        gs, _, _ = train_grid(d, width, lr, seed, rs.permutation(d["y_train"]), rs.permutation(d["y_val"]), kw)
                        res["test_shuf"] = accuracy(gs.predict(d["X_test"], CONFIG["n_votes"]), d["y_test"])
                    res["sec"] = round(time.time() - t0, 1)
                    rows.append(res)
                    print(task, width, vname, seed, "lr", lr, "test", round(res["test"] * 100, 2),
                          "shuf", round(res.get("test_shuf", float("nan")) * 100, 2), "firing", res["firing"], flush=True)
    return rows, lr_choice


def write_report(rows, lr_choice):
    out = ROOT / "results"
    summary = {}
    for task in CONFIG["tasks"]:
        for w in CONFIG["widths"]:
            for v in VARIANTS:
                r = [x for x in rows if x["task"] == task and x["width"] == w and x["variant"] == v]
                e = {"test": [round(float(np.mean([x["test"] for x in r])) * 100, 2),
                              round(float(np.std([x["test"] for x in r])) * 100, 2)],
                     "min_col_firing": round(float(np.mean([min(x["firing"]) for x in r])), 3)}
                if "test_shuf" in r[0]:
                    e["shuf"] = [round(float(np.mean([x["test_shuf"] for x in r])) * 100, 2),
                                 round(float(np.std([x["test_shuf"] for x in r])) * 100, 2)]
                summary[f"{task}/W{w}/{v}"] = e
    (out / "exp02_biogrid_credit.json").write_text(json.dumps(
        {"config": CONFIG, "variants": VARIANTS, "lr_choice": lr_choice, "rows": rows, "summary": summary},
        indent=2, ensure_ascii=False))
    lines = ["# 실험 02: credit assignment — 선택적 보상 사다리", "",
             f"격자 높이 {CONFIG['height']}. 시드 {len(CONFIG['seeds'])}개 평균 ± 표준편차, 테스트 {CONFIG['data']['n_test']}개. "
             "학습률은 변형·폭마다 시드 0 검증으로 선택. 셔플 대조는 학습·검증 라벨 모두 셔플.",
             "'최소 열 발화율' = 열별 평균 발화율 중 가장 낮은 값 (0에 가까우면 그 열이 죽은 것).", ""]
    names = {"A0_global": "A0 전역 도파민 (1단계)", "A1_zone": "A1 출력 방향 오차",
             "A2_relay_fired": "A2 바톤 터치 · 발화한 뉴런만", "A3_relay_sens": "A3 바톤 터치 · 민감도",
             "A4_relay_fa_sens": "A4 바톤 터치 · 무작위 전달 · 민감도"}
    for task in CONFIG["tasks"]:
        t = TARGETS[task]
        lines += [f"## {task}", "", f"목표치: 기준선 {t['majority']} / 로지스틱 {t['logreg']} / MLP64 {t['mlp64']}", "",
                  "| 학습 신호 | " + " | ".join(f"폭 {w}" for w in CONFIG["widths"]) + " | 셔플 대조 (폭 8) | 최소 열 발화율 (폭 8) |",
                  "|---|" + "---|" * (len(CONFIG["widths"]) + 2)]
        for v, label in names.items():
            cells = []
            for w in CONFIG["widths"]:
                s = summary[f"{task}/W{w}/{v}"]["test"]
                cells.append(f"{s[0]:.2f} ± {s[1]:.2f}")
            s8 = summary[f"{task}/W{CONFIG['widths'][-1]}/{v}"]
            shuf = f"{s8['shuf'][0]:.2f} ± {s8['shuf'][1]:.2f}" if "shuf" in s8 else "–"
            lines.append(f"| {label} | " + " | ".join(cells) + f" | {shuf} | {s8['min_col_firing']} |")
        lines.append("")
    (out / "exp02_biogrid_credit.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    rows, lr_choice = run()
    write_report(rows, lr_choice)
