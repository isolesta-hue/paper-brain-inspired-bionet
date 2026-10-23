# BatonNet — 뇌 원리를 본뜬 지역 학습 신경망

**상태: 첫 논문과 함께 공개** (저장소 주인 결정 2026-10-07 "Helix와 BatonNet을 함께", 2026-10-23 논문과 함께 공개로 전환). 논문 *Learning from Its Own Input: A Small Self-Tuning Learner, Its Infant-Inspired Logic, and a Sealed Exam*은 짝 저장소 `isolesta-hue/paper-human-learning-lineages`(Zenodo 보관)에 있고, 이 저장소는 그 논문의 **Helix Blue**가 몸으로 쓰는 **BatonNet**의 글과 코드다. (English summary: `README_EN.md`)

대상 연구: 전역 역전파 없이 뉴런(격자)의 지역 규칙만으로 학습이 되는가 — **BatonNet**(원본 이름 바이오넷: 격자 피질 + "바톤 터치" 지역 신용 할당 +
해마형 기억 + 수면 재생)과 그 이전의 긴 시행착오(v1~v10, BioBrain, GeoBrain, 여러 AI가 만든 코드의 비교와 셔플 대조).
원본: 비공개 작업 저장소 `new-transformer-model-research`(비공개로 남음, 브랜치 `claude/great-einstein-ln7j5x`) — 글의 "E nn"은 그 저장소의 `results/expNN_*.md`, `code/`는 그 저장소에서 복사만 한 것(원본 커밋 `code/SOURCE_COMMIT.txt`).

## 이 글의 기여 (2026-10-06 주인 결정)
- **언어 과제에서의 성과가 기여다**: 지역 규칙만 쓰는 격자가 세 언어 과제에서 같은 시드의 MLP16과 같은 정확도(내용어 69.38 vs 69.26, 다음 품사 39.89 vs 39.96,
  낱말+품사 45.95 vs 46.06; E 14 · 15), 셔플 대조는 기준선으로 떨어짐(내용어 51.33 · 낱말+품사 23.68) — "학습한 척"이 아님을 확인.
- 해마형 기억: 한 번 보고 MLP·격자보다 6~12점 높음(E 07); 재생으로 기억이 격자를 가르쳐, 한 번씩만 본 격자가 여러 번 학습한 격자에 다가감(E 08);
  기억 행마다 가장 큰 칸 하나만 남기면 저장량 1/96에 정확도는 같음(39.08 vs 39.22, E 10).
- MNIST 같은 비언어 과제는 이 글에 필요 없음. 효율 우위는 주장하지 않음(격자는 MLP16의 2.6~5.5배 파라미터).
- 더 나은 성능은 규칙을 다듬기보다 **그 위에 구조를 쌓아서** — BatonNet을 몸으로 쓰는 **Helix Blue**(원본 뉴호라이즌스). 첫 논문에서 Helix Blue는 공용 봉인 시험 v1(30생애)의
  열 기준 중 셋(C0 요소 기억 · C2 새 낱말 · C9 공간 말)이 채점기 통과선 위였고, 셋 모두 설계한 부품 하나씩에 기댐 — 배운 낱말과 설계한 부품이 함께 풂(논문 §7).

## 파일
| 파일 | 언어 | 내용 |
|---|---|---|
| `README_EN.md` | 영어 | 연구 요약(한 쪽) |
| `PAPER_OUTLINE.md` | 영어 | 논문 뼈대 v0.2, 숫자마다 원본 실험 번호 |
| `TECH_REPORT_DRAFT.md` | 영어 | 기술 보고서 초안 v0.1 |
| `RELATED_WORK.md` | 영어 | 관련 연구 |
| `REPRODUCTION_PLAN.md` | 한국어 | 실험 → 실행 파일 표, 재현·공개 사본 할 일 |
| `NAMES.md` | 한국어 | 이름 결정 기록 |
| `code/` | 영어 | 공개용 코드 사본(격자·해마형 기억·시험 틀·실험 00~16과 결과), 원본에서 복사만 — `code/README.md` |

## 라이선스 · 만든 방식
코드 MIT, 글 CC BY 4.0(주인 결정 2026-10-23). 저자는 저장소 주인 한 사람(isolesta-hue, Independent Researcher) — 목표와 결정을 맡음; 아이디어 토론은 ChatGPT · Gemini(무료판), 옛 버전의 구현은 주로 Gemini · 검증은 Claude, 다시 짓기와 BatonNet은 Claude(Anthropic, Claude Code)가 씀 — "모든 주장은 셔플 대조를 통과해야 한다"는 규칙 아래(자세히: 보고서의 AI 사용 표기).

## 원칙
숫자는 원본 결과 파일(`results/expNN_*.md`)에서만, 출처를 달아 옮긴다. 실패의 역사(셔플 대조로 "학습 0"을 확인한 단계들)도 쓴다 — 이 연구의
가장 큰 교훈이 거기 있다. 원본 저장소는 읽기만 하고, 원본의 이름·기록은 바꾸지 않는다.
