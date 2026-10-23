"""실험 16: 참고 자료 기능 다시 시험 — 지금의 뉴호라이즌스 몸에서 하나씩 빼고 넣기 (사용자 질문 2026-10-06: 다중 호르몬, 생명게임 응용, 허브 뉴런이 정말 의미 있었나).

예전 시험(exp03~05)은 "분류 정확도"에서, 작은 격자 하나로 했다. 이번엔 지금의 몸(해마 + 규칙 세트 + 지형 선택기 + 7이웃 피질 + 숨은 열 항상성, 고정 설정)의
글자 그림 세상 생애에서 바깥 시험 끝으로 잰다. 기능의 설정값은 screen.py의 DEFAULTS(참고 자료 원본의 값) 그대로 — 고치지 않음.
  빼기 (지금 몸에 있는 것): 허브 S15, residual, 공명 L15
  넣기: 생명게임 S4(이웃 수로 문턱) · S5 화학(이웃이 강하면 도파민 문턱↓/침묵하면 억제) · S3(고립·과밀 발화 거르기) · S2(뉴런 사멸),
        S13 성상교세포, S7 문턱 공유, N4 피로, N6 장기+단기 가중치, S9 평균 억제, N8 역할 5종, M6 체력·교체, L14 수면기(태그 모아 반영), L13 연결 태그, S10 40Hz, M9 동적 연결
  (N7 LIF는 공명과 함께 켜면 계산에 쓰이지 않아 뺌.) 다중 호르몬은 이미 방법 층의 손잡이 세 개(도파민 = 학습률, 세로토닌 = 지평, 아세틸콜린 = 재생)로 들어 있음 (NH 7·27).
비교 (미리 고정): 변형 − 기본 몸, 끝, 30시드 짝지은 부호 검정. 비교가 18개라 p < 0.005를 "판정"으로 (여럿 비교 보정의 대략). 좋은 쪽으로 p < 0.05인 것은 100시드로 다시.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import new_horizons.nh09_context as C9  # noqa: E402
from new_horizons.run_content import sign_p  # noqa: E402

BASE = "fixed+ctx+verify+select+fan7+homeo"
VARIANTS = [("no:hub", "허브 빼기 (S15)"), ("no:residual", "residual 빼기"), ("no:resonance", "공명 빼기 (L15)"),
            ("feat:gol_thresh", "생명게임 문턱 (S4)"), ("feat:chem", "이웃 화학: 도파민·억제 (S5)"), ("feat:gol_filter", "생명게임 거르기 (S3)"),
            ("feat:gol_death", "생명게임 사멸 (S2)"), ("feat:astrocyte", "성상교세포 (S13)"), ("feat:v_share", "문턱 공유 (S7)"),
            ("feat:fatigue", "피로 (N4)"), ("feat:long_short", "장기+단기 가중치 (N6)"), ("feat:mean_inhib", "평균 억제 GABA (S9)"),
            ("feat:roles", "역할 5종 (N8)"), ("feat:vitality", "체력·교체 (M6)"), ("feat:sleep", "수면기 (L14)"),
            ("feat:conn_tag", "연결 태그 (L13)"), ("feat:wave40", "40Hz (S10)"), ("feat:dyn_conn", "동적 연결 (M9)")]


def t(d):
    d = d[~np.isnan(d)]
    pos, neg = int((d > 1e-12).sum()), int((d < -1e-12).sum())
    return d.mean(), pos, neg, sign_p(pos, neg)


def main(n, only=None):
    seeds = list(range(n))
    vs = [v for v in VARIANTS if only is None or v[0] in only]
    modes = [BASE] + [BASE + "+" + v for v, _ in vs]
    with Pool(4) as p:
        R = dict(p.map(C9.job, [(m, s) for s in seeds for m in modes], chunksize=1))
    E = lambda m, k="motion": np.array([R[(m, s)][-1][k] for s in seeds], dtype=float)
    base = E(BASE)
    lines = [f"# 실험 16: 참고 자료 기능 다시 시험 — 지금 몸에서 빼고 넣기 ({n}시드)", "", f"기본 몸 끝 = {base.mean():.3f}", "",
             "| 변형 | 끝 | 차이 (나음:못함) | p | 뒤집힌 방 끝 차이 |", "|---|---|---|---|---|"]
    for v, lab in vs:
        m = BASE + "+" + v
        d, pos, neg, p = t(E(m) - base)
        fl = np.nanmean(E(m, "flip") - E(BASE, "flip"))
        mark = " **판정**" if p < 0.005 else (" (경향)" if p < 0.05 else "")
        lines.append(f"| {lab} | {E(m).mean():.3f} | {d:+.3f} ({pos}:{neg}) | {p:.2g}{mark} | {fl:+.3f} |")
    (ROOT / "results" / f"exp16_reference_screen{'' if only is None else '_confirm'}.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    main(n, sys.argv[2].split(",") if len(sys.argv) > 2 else None)
