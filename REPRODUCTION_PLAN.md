# 재현 계획 (Reproduction plan)

원본(`new-transformer-model-research`, 브랜치 `claude/great-einstein-ln7j5x`, 읽기만)의 실험 파일 → 결과 파일. 공개 사본에는 **마지막 격자 + 기억 + 시험 틀**만
옮긴다(옛 버전·대화 기록은 요약만).

| 글의 절 | 실험 | 실행 파일 | 결과 |
|---|---|---|---|
| 2 시험 틀 | E 00 | `experiments/exp00_language_baselines.py` | `results/exp00_language_baselines.md` |
| 3 격자: 전역 보상 | E 01 | `experiments/exp01_biogrid_reinforce.py` | `results/exp01_*.md` |
| 3 격자: 바톤 터치 | E 02 | `experiments/exp02_biogrid_credit.py` | `results/exp02_*.md` |
| 3 장치 고르기 | E 03–05 | `exp03_feature_screening.py`, `exp04_feature_confirm.py`, `exp05_combination.py` | `results/exp03~05_*.md` |
| 4 기억 | E 06–08, E 10 | `exp06_memory_capacity.py`, `exp07_memory_vs_learning.py`, `exp08_cls_replay.py`, `exp10_memory_compression.py` | `results/exp06~08, 10_*.md` |
| 5 비용 | E 09, E 11–12, E 15 | `exp09_hard_task_scaling.py`, `exp11_learned_readout.py`, `exp12_sleep_readout.py`, `exp15_fan.py` | `results/exp09, 11, 12, 15_*.md` |
| 3 느린 평균(대표 결과) | E 13–14 | `exp13_stability.py`, `exp14_slow_avg_confirm.py` | `results/exp13~14_*.md` |

## 할 일
- [ ] 결과 `.json`에서 글의 표 1~4를 다시 만드는 스크립트 하나(새 실행 없이)
- [ ] 대표 결과(E 14, 시드 10~14)를 공개 사본에서 처음부터 다시 돌려 같은 숫자 확인
- [ ] 의존성 고정(`requirements.txt`), CPU 기준 실행 시간
- [ ] 데이터 출처·라이선스 표기(`data/`의 언어 자료)
- [ ] `archive/`, `docs/HISTORY_*`는 공개 사본에 넣지 않고 글 6절의 요약으로만
