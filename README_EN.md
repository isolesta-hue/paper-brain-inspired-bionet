# BatonNet — a brain-inspired lattice that learns language tasks with local rules only

**Status: released together with the first paper** (the owner's decisions: publish Helix and BatonNet together, 2026-10-07; make both public with the paper, 2026-10-23). The paper, *Learning from Its Own Input: A Small Self-Tuning Learner, Its Infant-Inspired Logic, and a Sealed Exam*, is in the companion repository `isolesta-hue/paper-human-learning-lineages` (archived on Zenodo); this repository holds BatonNet, the body its learner **Helix Blue** is built on.
Source: the private working repository `new-transformer-model-research` (it stays private); "E nn" refers to its `results/expNN_*.md`, and `code/` is a copy (source commit in `code/SOURCE_COMMIT.txt`).

## Question
Can a network of simple model neurons learn with **local rules only** — no global backpropagation — and can we prove it
learned, rather than fooling ourselves?

## Answer so far
- **Yes, on three language tasks, at MLP accuracy.** BatonNet — a 2-D lattice with left-to-right flow and *baton-touch*
  credit assignment (each column passes a sensitivity signal to the column before it) — matches a same-seed MLP with 16
  hidden units: content-word 69.38 vs 69.26, next-tag 39.89 vs 39.96, word+tag 45.95 vs 46.06 (E 14–15).
  The label-shuffle control falls to the baseline (content 51.33, word+tag 23.68): the lattice is not "learning" leaked structure.
- **Global reward alone fails in depth.** Dopamine × eligibility learns only in shallow lattices; at width 4–8 it stays at
  the baseline, while baton-touch keeps 68 points (E 01–02).
- **Memory and sleep.** A hippocampus-like memory (sparse dentate code, one-shot Hebbian storage) beats the MLP and the
  lattice by 6–12 points after one pass over the data (E 07); replay in which the memory teaches the lattice lets a lattice
  that saw each example once approach one trained for many epochs (E 08); keeping one top cell per memory row stores 96×
  less with the same accuracy (39.08 vs 39.22, E 10).
- **What we do not claim.** No efficiency advantage: the lattice uses 2.6–5.5× the parameters of the MLP16 it matches, and
  the tasks are nearly linear. The contribution is the local-rule learner verified by honest controls; further improvement
  is pursued by building structure on top: the learner Helix Blue (companion paper). On the shared sealed exam v1 (30 lives) Helix Blue
  cleared the grader's pass line on three of ten criteria (C0 element memory, C2 a word heard once, C9 spatial words), each leaning on
  one designed part — learned words and designed parts solve them together (paper, section 7).

## The long road (why controls come first)
Earlier versions (v1–v10, BioBrain, GeoBrain; code written with several AI assistants) looked successful until a label
shuffle showed they learned nothing (e.g. stock direction 53.5% vs a 53.94% always-up baseline). The study was rebuilt around a harness
first: tasks where an MLP clearly beats the baseline, shuffle controls, logistic and MLP references of several sizes, seeds.

## Files
`PAPER_OUTLINE.md`, `TECH_REPORT_DRAFT.md`, `RELATED_WORK.md` (English); `REPRODUCTION_PLAN.md`, `NAMES.md` (Korean).

## Licence and how the work was done
Code: MIT; text: CC BY 4.0 (the owner's decision, 2026-10-23).
Author: isolesta-hue (Independent Researcher), the repository owner — a non-specialist who set the goals and made the decisions.
Ideas were discussed with ChatGPT and Gemini (free tiers); early versions were implemented mainly by Gemini and verified by Claude;
the rebuild and BatonNet were written by Claude (Anthropic) through Claude Code — under the rule that every claim must survive a
shuffle control (details: the report's AI-use statement).
