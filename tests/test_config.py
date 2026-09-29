def test_prior_configs_only_read_latents_that_fit_betas_writes():
    """Every latents path in the prior configs must be a file `mrextrap fit-betas` produces."""
    import re
    from pathlib import Path

    root = Path(__file__).resolve().parents[1] / "configs"
    for name in ("unet_prior.yaml", "diffusion_prior.yaml"):
        text = (root / name).read_text()
        paths = re.findall(r"latents_path_\w+:\s*(\S+)", text)
        assert len(paths) == 3, name
        for p in paths:
            assert re.search(r"/(train|val|test)_latent_betas_l1\.pt$", p), (name, p)
