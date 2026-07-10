def build_experiment_config_names(
    n_comps_min: int,
    n_comps_max: int,
):
    """Generate experiment configuration names."""

    names = [
        "acoustic",
        "acoustic_balanced_weights",
        "spectral_paper",
        "combined_paper",
    ]

    for n in range(n_comps_min, n_comps_max + 1):

        names.append(f"spectral_{n:02d}")
        names.append(f"combined_{n:02d}")

        names.append(f"sep_{n:02d}_raw")
        names.append(f"sep_{n:02d}_a01")
        names.append(f"sep_{n:02d}_a02")

    return names