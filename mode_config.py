import math
import yaml
import numpy as np


def load_config(path):
    with open(path, "r") as stream:
        return yaml.safe_load(stream)


def normalize_allocations(weights):
    total = float(sum(weights.values()))
    if total <= 0.0:
        raise ValueError("allocation weights must have positive sum")
    return {mode: float(value) / total for mode, value in weights.items() if float(value) > 0.0}


def default_mam4_allocations():
    return {
        "SO4": {"a1": 0.90, "a2": 0.10, "a3": 0.0, "a4": 0.0},
        "OCPHILIC": {"a1": 1.0},
        "BCPHILIC": {"a1": 1.0},
        "OCPHOBIC": {"a4": 1.0},
        "BCPHOBIC": {"a4": 1.0},
        "NO3": {"a1": 0.70, "a3": 0.30},
        "POM": {"a1": 0.80, "a2": 0.20},
        "SOA": {"a1": 0.80, "a2": 0.20},
    }


def map_mam4_to_mam3(allocations):
    mapped = {}
    for species, weights in allocations.items():
        next_weights = {}
        for mode, value in weights.items():
            target = "a1" if mode == "a4" else mode
            next_weights[target] = next_weights.get(target, 0.0) + float(value)
        mapped[species] = normalize_allocations(next_weights)
    return mapped


def _lognormal_pdf(radius_um, median_um, sigma_g):
    radius = np.asarray(radius_um, dtype=np.float64)
    radius = np.clip(radius, 1.0e-12, None)
    log_sigma = math.log(float(sigma_g))
    prefactor = 1.0 / (radius * log_sigma * math.sqrt(2.0 * math.pi))
    exponent = -((np.log(radius) - math.log(float(median_um))) ** 2) / (2.0 * log_sigma ** 2)
    return prefactor * np.exp(exponent)


def allocate_size_bins_to_modes(bin_radii_um, mode_specs):
    allocation = []
    for radius in np.asarray(bin_radii_um, dtype=np.float64):
        weights = {}
        for mode, spec in mode_specs.items():
            weights[mode] = float(_lognormal_pdf(radius, spec["dry_radius_um"], spec["sigma_g"]))
        allocation.append(normalize_allocations(weights))
    return allocation


def resolved_allocations(config, scheme):
    scheme_info = config["Schemes"][scheme]
    allocations = {
        species: normalize_allocations(weights)
        for species, weights in scheme_info.get("allocations", {}).items()
    }
    mode_specs = scheme_info["modes"]
    for group in scheme_info.get("size_bins", {}).values():
        generated = allocate_size_bins_to_modes(group["radii_um"], mode_specs)
        for species, weights in zip(group["species"], generated):
            allocations[species] = weights
    return allocations
