# Paper outline (draft v0.2) — BatonNet: learning with local rules only

Public name decided 2026-10-06 (internal name: BioNet; see NAMES.md). Source files keep the internal name.

Numbers cite the source results: "E nn" = `results/expNN_*.md`, "H" = docs/HISTORY_*.md / ASSESSMENT.md, "NH n" = Helix Blue finding n.

## Title candidates
1. *Local rules, honest controls: a brain-inspired lattice that learns language tasks without global backpropagation*
2. *From shuffle controls to a working local-learning lattice: lessons from a long amateur search*
3. *Cortex, hippocampus and sleep in a small lattice network: one-shot memory, replay as teaching, consolidation as compression*

## Abstract (draft)
Can a network of simple model neurons learn with local rules only — no global backpropagation? After a long series of
versions that looked successful but learned nothing once a label-shuffle control was applied (H: stock prediction 53.5% vs
53.95% baseline; word-list tasks at the majority-class rate), we rebuilt the study around a strict harness: tasks where a
tuned MLP clearly beats the baseline, label-shuffle controls, logistic and MLP references of several sizes, and seeds.
The resulting **BatonNet** — a 2-D lattice with left-to-right flow, *baton-touch* local credit assignment (each column passes a
sensitivity signal to the previous one; E 02), resonance, probabilistic readout, hubs and residual paths (E 03–05), and a
slow synaptic average used for judging (E 13–14) — matches a same-seed MLP with 16 hidden units on three language tasks
(content-word 69.38 vs 69.26; next-tag 39.89 vs 39.96; word+tag 45.95 vs 46.06; E 14–15), while the shuffle control falls to
the baseline (51.33 / 23.68).
A hippocampus-like memory (sparse DG code, one-shot Hebbian storage) beats MLP and lattice by 6–12 points after one
exposure (E 07); **replay in which the memory teaches the lattice** lets a lattice that saw each example once match one
trained for many epochs (E 08), and **consolidation is compression**: the lattice keeps the memory's accuracy with
3,720 vs 96,000 parameters (E 08); storing one class per memory neuron loses nothing (96× compression, E 10).
The contribution we claim is this: **a local-rule lattice that reaches MLP accuracy on language tasks, verified by shuffle
controls**, plus a memory system whose replay teaches and compresses. Limits we report plainly: the lattice needs
2.6–5.5× the parameters of the MLP16 it matches, and these tasks are nearly linear (logistic regression is close); we make no
efficiency claim. Further improvement is pursued not by tuning the local rule but by building structure on top of it
(Helix Blue: the same lattice as the body of a self-improving learner, companion write-up).

## 1. Introduction — the question and why controls matter (the shuffle story)
## 2. A harness first (E 00): tasks, baselines, MLP sizes, shuffle control, seeds
## 3. The lattice
- E 01: global dopamine × eligibility trace learns only in shallow lattices (≈ logistic); deep → baseline.
- E 02: baton-touch / sensitivity (A3) — first local rule to exceed logistic in deep lattices; weight-free variant A4.
- E 03–05: screening 26 mechanisms → confirmation on new seeds → combination C4 (many early "wins" did not replicate).
- E 13–14: slow synaptic average for judging.
## 4. Memory and sleep
- E 06: state-space (Hopfield-like) landscape works as memory; spatial landscape (GeoBrain) does not.
- E 07: hippocampus-like memory, one-shot advantage; E 08: replay as teaching, consolidation as compression; E 10: compression.
## 5. Cost, stated plainly (E 09, E 11–12, E 15)
- E 09 (word+tag, sizes): lattice 43–45 vs MLP 44–47 across sizes; memory alone 30–39 — no efficiency advantage.
- E 11–12: a learned local readout and a sleep-time readout keep accuracy (content 69.54) — the readout is not the bottleneck.
- E 15: wider neighbour fan does not change task accuracy (≤ ±0.4) but helps the body inside a learner's life (Helix Blue,
  end +0.071, 29:1) — structure matters more than the rule's details.
## 6. Lessons from the long search (H): what fooled us (overflow, NaN, scoring bugs, fake efficiency metrics),
   multi-AI coding (one designs, one verifies), structural constraints instead of penalties.
## 7. Outlook: structure on top of the local rule — BatonNet as the body of Helix Blue (companion write-up: world model first,
   context rule sets, hippocampus + cortex; the self-improvement criterion met, NH 27).

## Figures / tables
1. Table: tasks × (baseline, logistic, MLP0/4/16/64, BatonNet, BatonNet-shuffle) with params (E 00, E 14).
2. Lattice depth vs accuracy, global reward vs baton-touch (E 01–02).
3. One-shot memory vs MLP by data size (E 07); replay teaching curves (E 08).
4. Compression curve: stored bits vs accuracy (E 10).

## Needed before release
- Decided by the owner (2026-10-06): the language-task results are the contribution; no non-language task (e.g. MNIST) is
  required for this write-up. Further improvement continues as structure in Helix Blue.
- Clean public code: the final lattice + memory + harness only (not the whole archive).
