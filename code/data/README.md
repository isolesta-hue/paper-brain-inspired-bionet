# data

- `brown_tags.npz` — Brown 코퍼스(NLTK 배포본)의 universal tagset 품사 id 시퀀스와 문장 시작 위치.
  단어 자체는 들어 있지 않다. `src/data/language.py`가 이 캐시만으로 동작하므로 nltk 코퍼스 다운로드가 필요 없다.
  다시 만들려면 NLTK `brown`, `universal_tagset` 데이터를 받은 뒤 이 파일을 지우고 `python3 src/data/language.py` 실행.
- `brown_words.npz` — 같은 순서의 단어 id 시퀀스(소문자, 빈도 상위 5,000개 순위, 그 밖은 5000 = OOV)와 어휘 목록.
  `make_word_task`(wtag 과제, 실험 09~)가 쓴다. 다시 만들려면 이 파일을 지우고 NLTK `brown` 데이터가 있는 상태에서 `make_word_task()` 호출.
