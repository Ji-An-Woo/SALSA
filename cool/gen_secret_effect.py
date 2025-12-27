#!/usr/bin/env python3
import argparse
import os
import pickle

import numpy as np

from data import Data
from reduction import make_n_reduced_samples, reduce_with_BKZ, reduce_with_flatter


def _make_secret(
    secret_dim,
    hamming_weight,
    rng,
    n_brute_force=None,
    hamming_weight_in_brute_force=None,
):
    if hamming_weight_in_brute_force is None:
        vec = np.zeros(secret_dim, dtype=np.int64)
        idx = rng.choice(secret_dim, size=hamming_weight, replace=False)
        vec[idx] = 1
        return vec
    part1 = np.zeros(n_brute_force, dtype=np.int64)
    idx1 = rng.choice(n_brute_force, size=hamming_weight_in_brute_force, replace=False)
    part1[idx1] = 1
    part2_len = secret_dim - n_brute_force
    part2_hw = hamming_weight - hamming_weight_in_brute_force
    part2 = np.zeros(part2_len, dtype=np.int64)
    if part2_hw > 0:
        idx2 = rng.choice(part2_len, size=part2_hw, replace=False)
        part2[idx2] = 1
    return np.concatenate([part1, part2])


def _resolve_q(args):
    if args.Q and args.Q > 0:
        return int(args.Q)
    return int(2 ** args.log2q)


def _resolve_seed(value, default):
    if value is None or int(value) < 0:
        return int(default)
    return int(value)


def _save_bundle(secret_dir, hamming, seed, data):
    bundle_path = os.path.join(secret_dir, f"bundle_{hamming}_{seed}.npz")
    np.savez(
        bundle_path,
        RA=data["RA"],
        RB=data["RB"],
        origA=data["origA"],
        origB=data["origB"],
        secret=data["secret"],
        noise=data["noise"],
        Q=np.int64(data["Q"]),
    )
    return bundle_path


def _write_dataset(base_dir, secret_subdir, hamming, seed, payload, params, save_bundle):
    os.makedirs(base_dir, exist_ok=True)
    np.save(os.path.join(base_dir, "orig_A.npy"), payload["origA"])
    np.save(os.path.join(base_dir, "reduced_A.npy"), payload["RA"])

    secret_dir = os.path.join(base_dir, secret_subdir)
    os.makedirs(secret_dir, exist_ok=True)
    np.save(os.path.join(secret_dir, f"secret_{hamming}_{seed}.npy"), payload["secret"])
    np.save(os.path.join(secret_dir, f"b_{hamming}_{seed}.npy"), payload["RB"])
    np.save(os.path.join(secret_dir, f"orig_b_{hamming}_{seed}.npy"), payload["origB"])
    if payload.get("secret_raw") is not None:
        np.save(
            os.path.join(secret_dir, f"secret_raw_{hamming}_{seed}.npy"),
            payload["secret_raw"],
        )
    if payload.get("noise") is not None:
        np.save(os.path.join(secret_dir, "noise.npy"), payload["noise"])
    with open(os.path.join(secret_dir, "params.pkl"), "wb") as f:
        pickle.dump(params, f)
    bundle_path = None
    if save_bundle:
        bundle_path = _save_bundle(secret_dir, hamming, seed, payload)
    return secret_dir, bundle_path


def _make_variant(
    origA,
    Rs,
    subsets,
    noise,
    secret_raw,
    Q,
    use_hallucination,
    hallucination_k_seed,
    hallucination_k_bits,
    hallucination_degrees,
    hallucination_coeff_choices,
):
    secret = secret_raw.copy()
    hallucination_k = None
    hallucination_coeffs = None
    if use_hallucination:
        degrees = [d for d in Data._parse_int_list(hallucination_degrees) if d > 0]
        if not degrees:
            degrees = [1]
        coeff_choices = Data._parse_int_list(hallucination_coeff_choices)
        if not coeff_choices:
            coeff_choices = [-1, 1]
        hallucination_k, k_bytes = Data._make_hallucination_k(
            hallucination_k_bits, hallucination_k_seed
        )
        secret, hallucination_coeffs = Data._maclaurin_obfuscate(
            secret, Q, k_bytes, degrees, coeff_choices
        )

    origB = (origA @ secret + noise) % Q
    RA, RB = Data._make_RAs_RBs(origA, origB, Rs, subsets, Q)
    return {
        "RA": RA,
        "RB": RB,
        "origB": origB,
        "secret": secret,
        "secret_raw": secret_raw if use_hallucination else None,
        "hallucination_k": hallucination_k,
        "hallucination_coeffs": hallucination_coeffs,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate plain + hallucinated datasets with fixed A/R/noise.",
    )
    parser.add_argument("--N", type=int, default=256, help="secret dimension (n)")
    parser.add_argument("--log2q", type=int, default=12)
    parser.add_argument("--Q", type=int, default=0, help="override modulus q")
    parser.add_argument("--n_sniffs", type=int, default=5000)
    parser.add_argument("--how_many", type=int, default=20000)
    parser.add_argument("--lll_penalty", type=int, default=10)
    parser.add_argument("--reduction_chunk_size", type=int, default=0)
    parser.add_argument("--n_workers", type=int, default=1)
    parser.add_argument("--reduction", choices=["flatter", "bkz"], default="bkz")
    parser.add_argument("--hamming", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0, help="base seed for A/reduction")
    parser.add_argument("--secret_seed", type=int, default=-1)
    parser.add_argument("--noise_seed", type=int, default=-1)
    parser.add_argument("--noise_variance", type=float, default=3.2)
    parser.add_argument("--n_brute_force", type=int, default=None)
    parser.add_argument("--hamming_weight_in_brute_force", type=int, default=None)
    parser.add_argument("--savepath", type=str, default="data")
    parser.add_argument("--plain_tag", type=str, default="plain")
    parser.add_argument("--hallucination_tag", type=str, default="hallucination")
    parser.add_argument("--secret_dir", type=str, default="secret")
    parser.add_argument("--hallucination_secret_dir", type=str, default="secret_hallucination")
    parser.add_argument("--hallucination_k_seed", type=int, default=-1)
    parser.add_argument("--hallucination_k_bits", type=int, default=128)
    parser.add_argument("--hallucination_degrees", type=str, default="1,3,5")
    parser.add_argument("--hallucination_coeff_choices", type=str, default="-1,1")
    parser.add_argument("--save_reduction", action="store_true")
    parser.add_argument("--save_bundle", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    Q = _resolve_q(args)
    reduction_chunk_size = args.reduction_chunk_size or args.N
    secret_seed = _resolve_seed(args.secret_seed, args.seed)
    noise_seed = _resolve_seed(args.noise_seed, args.seed)

    rng_A = np.random.default_rng(args.seed)
    origA = rng_A.integers(0, Q, size=(args.n_sniffs, args.N), dtype=np.int64) - Q // 2

    np.random.seed(args.seed)
    reduction_fn = reduce_with_flatter if args.reduction == "flatter" else reduce_with_BKZ
    Rs, subsets = make_n_reduced_samples(
        origA,
        how_many=args.how_many,
        Q=Q,
        reduction_chunk_size=reduction_chunk_size,
        lll_penalty=args.lll_penalty,
        n_workers=args.n_workers,
        reduction_fn=reduction_fn,
    )

    rng_secret = np.random.default_rng(secret_seed)
    secret_raw = _make_secret(
        args.N,
        args.hamming,
        rng_secret,
        n_brute_force=args.n_brute_force,
        hamming_weight_in_brute_force=args.hamming_weight_in_brute_force,
    )

    rng_noise = np.random.default_rng(noise_seed)
    noise = rng_noise.normal(0, args.noise_variance, size=args.n_sniffs).round(0)

    plain = _make_variant(
        origA=origA,
        Rs=Rs,
        subsets=subsets,
        noise=noise,
        secret_raw=secret_raw,
        Q=Q,
        use_hallucination=False,
        hallucination_k_seed=args.hallucination_k_seed,
        hallucination_k_bits=args.hallucination_k_bits,
        hallucination_degrees=args.hallucination_degrees,
        hallucination_coeff_choices=args.hallucination_coeff_choices,
    )
    hallucinated = _make_variant(
        origA=origA,
        Rs=Rs,
        subsets=subsets,
        noise=noise,
        secret_raw=secret_raw,
        Q=Q,
        use_hallucination=True,
        hallucination_k_seed=args.hallucination_k_seed,
        hallucination_k_bits=args.hallucination_k_bits,
        hallucination_degrees=args.hallucination_degrees,
        hallucination_coeff_choices=args.hallucination_coeff_choices,
    )

    if not np.array_equal(plain["RA"], hallucinated["RA"]):
        raise RuntimeError("RA mismatch; base A/R should be fixed.")

    base_name = f"{args.N}_{Q}_{args.n_sniffs}_{args.lll_penalty}"
    plain_base = os.path.join(args.savepath, args.plain_tag, base_name)
    halluc_base = os.path.join(args.savepath, args.hallucination_tag, base_name)

    base_params = {
        "Q": Q,
        "n_sniffs": args.n_sniffs,
        "secret_dim": args.N,
        "hamming_weight": args.hamming,
        "sigma": args.noise_variance,
        "n_brute_force": args.n_brute_force,
        "hamming_in_brute_force": args.hamming_weight_in_brute_force,
        "N": args.N,
        "log2q": int(np.log2(Q)),
        "seed": args.seed,
        "secret_seed": secret_seed,
        "noise_seed": noise_seed,
        "reduction": args.reduction,
        "lll_penalty": args.lll_penalty,
        "reduction_chunk_size": reduction_chunk_size,
        "n_workers": args.n_workers,
        "how_many": args.how_many,
        "hallucination_k_bits": args.hallucination_k_bits,
        "hallucination_degrees": args.hallucination_degrees,
        "hallucination_coeff_choices": args.hallucination_coeff_choices,
    }

    plain_payload = {
        "origA": origA,
        "RA": plain["RA"],
        "RB": plain["RB"],
        "origB": plain["origB"],
        "secret": plain["secret"],
        "secret_raw": None,
        "noise": noise,
        "Q": Q,
    }
    hall_payload = {
        "origA": origA,
        "RA": hallucinated["RA"],
        "RB": hallucinated["RB"],
        "origB": hallucinated["origB"],
        "secret": hallucinated["secret"],
        "secret_raw": secret_raw,
        "noise": noise,
        "Q": Q,
    }

    plain_params = dict(base_params)
    plain_params.update(
        {
            "use_hallucination": False,
            "hallucination_k_seed": -1,
            "hallucination_k": None,
            "hallucination_coeffs": None,
            "secret_raw": None,
        }
    )
    hall_params = dict(base_params)
    hall_params.update(
        {
            "use_hallucination": True,
            "hallucination_k_seed": args.hallucination_k_seed,
            "hallucination_k": hallucinated["hallucination_k"],
            "hallucination_coeffs": hallucinated["hallucination_coeffs"],
            "secret_raw": secret_raw.tolist(),
        }
    )

    if args.save_reduction:
        np.save(os.path.join(plain_base, "Rs.npy"), Rs)
        np.save(os.path.join(plain_base, "subsets.npy"), subsets)
        np.save(os.path.join(halluc_base, "Rs.npy"), Rs)
        np.save(os.path.join(halluc_base, "subsets.npy"), subsets)

    plain_secret_dir, plain_bundle = _write_dataset(
        plain_base,
        args.secret_dir,
        args.hamming,
        args.seed,
        plain_payload,
        plain_params,
        args.save_bundle,
    )
    hall_secret_dir, hall_bundle = _write_dataset(
        halluc_base,
        args.hallucination_secret_dir,
        args.hamming,
        args.seed,
        hall_payload,
        hall_params,
        args.save_bundle,
    )

    print("Plain dataset:", plain_base)
    print("Plain secret dir:", plain_secret_dir)
    if plain_bundle:
        print("Plain bundle:", plain_bundle)
    print("Hallucination dataset:", halluc_base)
    print("Hallucination secret dir:", hall_secret_dir)
    if hall_bundle:
        print("Hallucination bundle:", hall_bundle)


if __name__ == "__main__":
    main()
