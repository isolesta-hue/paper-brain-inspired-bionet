"""실험 00: 언어 과제의 기준선과 MLP 목표치 측정.

목적
  - 과제가 학습 가능한지 확인 (MLP가 찍기 기준선을 확실히 넘는가)
  - 앞으로 만들 모델이 도달해야 할 목표치(MLP)와 천장(n-gram 표)을 정한다
  - 정답 셔플 대조: 학습 라벨을 섞으면 MLP도 기준선으로 떨어져야 정상

실행: python3 experiments/exp00_language_baselines.py
결과: results/exp00_language_baselines.json, .md
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data.language import make_task  # noqa: E402
from src.baselines.models import MLP, accuracy, majority_predict, ngram_table_predict  # noqa: E402

CONFIG = {
    "tasks": ["content", "tag12"],
    "context": 3,
    "n_train": 20000,
    "n_val": 5000,
    "n_test": 5000,
    "seeds": [0, 1, 2, 3, 4],
    "mlp_hidden": [16, 64],
}


def run():
    rows = []
    for task in CONFIG["tasks"]:
        for seed in CONFIG["seeds"]:
            d = make_task(task, CONFIG["context"], CONFIG["n_train"], CONFIG["n_val"], CONFIG["n_test"], seed)
            nc, n_in = d["n_classes"], d["X_train"].shape[1]
            yt = d["y_test"]
            res = {"task": task, "seed": seed}
            res["majority"] = accuracy(majority_predict(d["y_train"], len(yt)), yt)
            res["ngram_table"] = accuracy(ngram_table_predict(d["ctx_train"], d["y_train"], d["ctx_test"], nc), yt)

            m = MLP(n_in, 0, nc, seed)
            m.fit(d["X_train"], d["y_train"], d["X_val"], d["y_val"])
            res["logreg"] = accuracy(m.predict(d["X_test"]), yt)

            for h in CONFIG["mlp_hidden"]:
                t0 = time.time()
                m = MLP(n_in, h, nc, seed)
                m.fit(d["X_train"], d["y_train"], d["X_val"], d["y_val"])
                res[f"mlp{h}"] = accuracy(m.predict(d["X_test"]), yt)
                res[f"mlp{h}_params"] = m.n_params()
                res[f"mlp{h}_sec"] = round(time.time() - t0, 2)

            # 정답 셔플 대조: 학습·검증 라벨을 모두 섞는다 (테스트 라벨만 원본).
            # 검증 라벨을 원본으로 두면 조기 종료가 "원본 정답에 우연히 맞는 시점"을 골라
            # 셔플 모델도 기준선보다 높게 나온다 (첫 실행에서 56.5% 관측 — GA 선택 편향과 같은 현상).
            rs = np.random.RandomState(1000 + seed)
            y_shuf, yv_shuf = rs.permutation(d["y_train"]), rs.permutation(d["y_val"])
            m = MLP(n_in, 64, nc, seed)
            m.fit(d["X_train"], y_shuf, d["X_val"], yv_shuf)
            res["mlp64_shuffled"] = accuracy(m.predict(d["X_test"]), yt)
            rows.append(res)
            print(res, flush=True)
    return rows


def summarize(rows):
    keys = ["majority", "ngram_table", "logreg", "mlp16", "mlp64", "mlp64_shuffled"]
    summary = {}
    for task in CONFIG["tasks"]:
        r = [x for x in rows if x["task"] == task]
        summary[task] = {k: (float(np.mean([x[k] for x in r])), float(np.std([x[k] for x in r]))) for k in keys}
        summary[task]["mlp16_params"] = r[0]["mlp16_params"]
        summary[task]["mlp64_params"] = r[0]["mlp64_params"]
    return summary


def write_report(rows, summary):
    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    (out / "exp00_language_baselines.json").write_text(
        json.dumps({"config": CONFIG, "rows": rows, "summary": summary}, indent=2, ensure_ascii=False))
    n = CONFIG["n_test"]
    lines = ["# 실험 00: 언어 과제 기준선과 MLP 목표치", "",
             f"Brown 코퍼스 품사 시퀀스, 직전 {CONFIG['context']}개 품사 → 다음 단어. "
             f"train {CONFIG['n_train']} / val {CONFIG['n_val']} / test {n}, 시드 {len(CONFIG['seeds'])}개 평균 ± 표준편차 (테스트 정확도).",
             f"테스트 {n}개 기준 표준오차는 약 ±0.7%p.", ""]
    names = {"majority": "찍기 기준선 (최빈 클래스)", "ngram_table": "n-gram 표 (천장 근사)", "logreg": "로지스틱 회귀",
             "mlp16": "MLP (은닉 16)", "mlp64": "MLP (은닉 64)", "mlp64_shuffled": "MLP 64 + 정답 셔플 (대조, train·val 모두 셔플)"}
    for task, s in summary.items():
        title = "내용어/기능어 이진 분류" if task == "content" else "다음 품사 12종 분류"
        lines += [f"## {task} — {title}", "", "| 모델 | 테스트 정확도 |", "|---|---|"]
        for k, label in names.items():
            mu, sd = s[k]
            lines.append(f"| {label} | {mu * 100:.2f}% ± {sd * 100:.2f} |")
        lines += ["", f"파라미터 수: MLP16 = {s['mlp16_params']}, MLP64 = {s['mlp64_params']}", ""]
    (out / "exp00_language_baselines.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    rows = run()
    write_report(rows, summarize(rows))
