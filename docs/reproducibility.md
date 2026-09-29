# Reproducibility

- Implementation details of the code that differ from the wording of the paper: [paper_notes.md](paper_notes.md).
- How to reproduce each table and figure: [reproducing.md](reproducing.md).
- Retraining does not give bit-identical weights (GPU nondeterminism, bf16 autocast, `torch.compile`, gradient
  fits with Adam on GPU). Reproduction is claimed from the released artefacts (weights, cached latents) or
  within statistical tolerance.
