# BatonNet — code (release copy)

A copy of the final lattice, the hippocampus-like memory and the test harness from the source research repository
(read-only source; commit in `SOURCE_COMMIT.txt`). Source names are kept inside the code ("biogrid", "BioNet");
the public name is **BatonNet**. Older versions (v1–v10, BioBrain, GeoBrain) and conversation logs are not included —
the report's "lessons from the long search" section summarises them.

## Layout
| path | what |
|---|---|
| `src/biogrid/core.py`, `src/biogrid/screen.py` | the lattice: left-to-right flow, baton-touch credit assignment, the screened mechanisms (C4), slow synaptic average |
| `src/memory/hippocampus.py`, `src/memory/assoc.py` | sparse dentate-gyrus code with one-shot Hebbian storage; associative memory |
| `src/baselines/models.py` | logistic regression and MLPs of several sizes |
| `src/data/language.py` | the three language tasks (content, tag12, wtag) from the Brown corpus |
| `experiments/expNN_*.py` | one script per experiment; `results/expNN_*.md / .json` are their outputs |
| `tests/test_screen_equivalence.py` | regression check: the fast screening engine equals the reference lattice |

## Data
`data/README.md` explains the two cached files (`brown_tags.npz`, `brown_words.npz`). They are **not** copied here
(their licence is not yet checked); rebuild them with NLTK (`brown`, `universal_tagset`):
`pip install -r requirements.txt`, download the NLTK data, then `python3 src/data/language.py`.

## Reproducing the headline table (E 14 / E 15)
`python3 experiments/exp14_slow_avg_confirm.py` (seeds 10–14), `python3 experiments/exp15_fan.py`.
Each script prints and writes its table to `results/`.
