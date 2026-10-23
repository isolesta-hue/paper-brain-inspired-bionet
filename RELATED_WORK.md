# Related work (draft)
- **Local / biologically plausible learning**: three-factor rules and eligibility traces (Gerstner et al. 2018), feedback
  alignment (Lillicrap et al. 2016), direct feedback alignment (Nøkland 2016), target propagation, forward-forward
  (Hinton 2022), equilibrium propagation (Scellier & Bengio 2017). BatonNet's baton-touch passes a sensitivity signal column to
  column — compare with these.
- **Spiking / neuromorphic nets**: STDP, LIF neurons; surrogate-gradient SNNs.
- **Associative memory**: Hopfield (1982), modern Hopfield (Ramsauer et al. 2020), sparse distributed memory (Kanerva),
  Willshaw nets — BatonNet's DG code + one-shot Hebbian storage.
- **Complementary learning systems and replay**: McClelland et al. 1995; generative replay in continual learning.
- **Controls**: label-shuffle / permutation tests as a guard against leakage and self-deception.
