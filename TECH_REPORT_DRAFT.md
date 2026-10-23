# BatonNet: a lattice that learns language tasks with local rules only, verified by shuffle controls

*Technical report — draft v0.1 (work in progress). "E nn" = source result `results/expNN_*.md`; "H" = the project's history notes.
Every number must be re-checked against the regeneration script before release (REPRODUCTION_PLAN.md).*

## Abstract
We build a network of simple model neurons that learns with local rules only and check, with label-shuffle controls and
MLP/logistic references, that it really learns. BatonNet is a 2-D lattice with left-to-right flow in which each column
passes a sensitivity signal to the column before it ("baton touch"), plus resonance, probabilistic readout, hubs, residual
paths and a slow synaptic average used for judging. On three language tasks it matches a same-seed MLP with 16 hidden
units (69.38 vs 69.26; 39.89 vs 39.96; 45.95 vs 46.06) while its shuffle control falls to the baseline. A hippocampus-like
memory adds one-shot learning (+6–12 points after one pass), replay in which the memory teaches the lattice, and 96×
compression of the stored memory without loss. We claim no efficiency advantage; the lattice uses 2.6–5.5× the parameters
of the MLP it matches. The study also documents a long series of earlier versions that looked successful but learned
nothing under a shuffle control.

## 1. Introduction
Backpropagation needs a global backward pass; brains are thought to rely on local signals. Many local-rule proposals exist
(feedback alignment, three-factor rules, forward-forward). This project began as an amateur search with several AI coding
assistants, and its first lesson was negative: without controls, apparent learning was an artefact (H: stock direction 53.5%
vs a 53.95% baseline; word-list tasks at the majority-class rate). The contribution here is a local-rule learner that
survives honest controls on language tasks, and a memory system around it.

## 2. Harness (E 00)
Three tasks from a tagged text corpus: **content** (is the next word a content word; baseline 51.9, logistic 67.16, MLP64
69.59), **tag12** (next part-of-speech tag among 12; baseline 23.94, logistic 38.55, MLP64 40.14) and **wtag** (previous four
positions' word and tag → next tag; 1,252 sparse inputs; logistic 44.52, MLP64 46.70). References: majority, n-gram,
logistic regression, MLPs with 0–256 hidden units; label-shuffle control on every new model; five seeds; learning rates
chosen on validation of seed 0 only.

## 3. The lattice
- **Global reward is not enough (E 01–02).** Dopamine × eligibility trace learns at width 2 (66.96) but stays at the baseline
  at widths 4–8 (51.6 / 51.9).
- **Baton touch (E 02).** Passing a sensitivity signal column to column (A3) keeps 68.1–68.4 at widths 2–8 on content; its
  shuffle control stays at 51.7. A weight-free random-feedback variant (A4) is close (65.5–68.0).
- **Screening (E 03–05).** 26 mechanisms screened, then confirmed on new seeds; many early "wins" did not replicate; the
  survivors combine into C4.
- **Slow synaptic average (E 13–14).** Judging with a slow average of the weights: content 69.38 ± 0.32 (MLP16 69.26 ± 0.68),
  shuffle 51.33.
- **Neighbour fan (E 15).** Wider fan leaves task accuracy unchanged (−0.36 to +0.16) — but see §6.

## 4. Memory and sleep
- **One-shot memory (E 07).** Sparse dentate-gyrus-like code with one-shot Hebbian storage. After one pass over 1,000
  examples: content 65.5 vs lattice 56.5 and one-pass MLP64 59.3; tag12 34.6 vs lattice 22.6.
- **Replay as teaching (E 08).** In a stream where each example is seen once, replay in which the memory teaches the lattice
  brings it close to multi-epoch training (content n=20,000: dual + replay 69.22 vs multi-epoch lattice 69.52 and MLP64
  69.59); consolidation moves the memory's knowledge into a lattice of 3,720 parameters (the tag12 memory has 96,000).
- **Compression (E 10).** Keeping one top cell per memory row stores 32,000 bits instead of 3,072,000 (96×) at 39.08 vs 39.22;
  low-rank compression loses more.

## 5. Cost, stated plainly (E 09, E 11–12)
On wtag, the lattice reaches 43.3–44.8 across sizes (25k–228k parameters) while MLPs reach 44.1–46.7 (5k–324k), and memory
alone 29.5–39.0. Learned and sleep-time readouts keep accuracy (content 69.54) — the readout is not the bottleneck. These
tasks are nearly linear (logistic is close to MLP), so they cannot show an efficiency advantage, and we do not claim one.

## 6. From rule to structure
Tuning the local rule further gave small or no gains; adding structure did. In the self-improving learner that uses
BatonNet as its body (Helix Blue, companion write-up), the same fan change that is neutral on the tasks raises the learner's
end score by +0.071 (29:1, 30 seeds; E 15 B), and the learner meets its self-improvement criterion (NH 27). The owner's
decision (2026-10-06): the language results are this paper's contribution; further improvement continues as structure in
Helix Blue rather than as more rule tuning or new task families.

## 7. Lessons from the long search (H)
What fooled us: overflow and NaN masked as learning, scoring bugs, efficiency metrics that rewarded doing nothing. What
helped: a harness before models, shuffle controls on every claim, one AI designing and another verifying, and structural
constraints instead of penalty terms.

## Authorship and AI-use statement
**Author:** the repository owner (sole human author; responsible for all content and decisions, including the design
specification and the label-shuffle control).
AI systems are not listed as authors (they cannot take responsibility for the work, and most venues and arXiv do not allow
it); their roles are disclosed here. Sources: the owner's uploaded conversation logs (`archive/conversations/`, summarised
in `docs/HISTORY_*.md` and `docs/ASSESSMENT.md`) and the commit trailers.
- **Concept phase — idea discussion:** Gemini (Google, free tier): additive rather than multiplicative computation, spike
  timing (STDP), chemical signals as a global context, receptors, Game-of-Life rules as brakes and sparsity. ChatGPT
  (OpenAI, free tier): splitting one weight into role-specific state variables, a staged feature-ablation design, and
  research-method advice ("do not mistake a good result for its cause"; "smarter connections, fewer connections").
- **Early versions (v1–v10, BioBrain, GeoBrain):** Gemini — main implementer (vectorised engines); Claude Sonnet 5
  (Anthropic, chat) — repairs, verification and the fairness harness (v6.8), "structural constraints instead of penalties"
  (v8.0); ChatGPT — concept and experiment design (dynamic connectivity, neuron roles). AI interpretations were often
  over-confident before verification; the decisive check — the label-shuffle control — came from the owner (§7).
- **Rebuild and BatonNet (harness E 00, experiments E 01–E 15), and Helix Blue:** Claude Opus 5.5 (Anthropic), via Claude Code —
  designed and wrote the code, ran the experiments and drafted this report, under the owner's direction.
