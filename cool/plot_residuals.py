#!/usr/bin/env python3
import argparse
import os
import pickle

import numpy as np


def _load_params(secret_dir):
    params_path = os.path.join(secret_dir, "params.pkl")
    if not os.path.exists(params_path):
        return {}
    with open(params_path, "rb") as f:
        params = pickle.load(f)
    if isinstance(params, dict):
        return params
    return vars(params)


def _resolve_q(params, q_override):
    if q_override and q_override > 0:
        return int(q_override)
    q_val = params.get("Q")
    if q_val is None:
        raise ValueError("Q not found in params.pkl; pass --Q")
    return int(q_val)


def _load_arrays(secret_dir, hamming, seed, use_orig):
    base_dir = os.path.dirname(secret_dir.rstrip("/"))
    if use_orig:
        ra_path = os.path.join(base_dir, "orig_A.npy")
        rb_path = os.path.join(secret_dir, f"orig_b_{hamming}_{seed}.npy")
    else:
        ra_path = os.path.join(base_dir, "reduced_A.npy")
        rb_path = os.path.join(secret_dir, f"b_{hamming}_{seed}.npy")
    secret_path = os.path.join(secret_dir, f"secret_{hamming}_{seed}.npy")
    if not os.path.exists(ra_path):
        raise FileNotFoundError(ra_path)
    if not os.path.exists(rb_path):
        raise FileNotFoundError(rb_path)
    if not os.path.exists(secret_path):
        raise FileNotFoundError(secret_path)
    RA = np.load(ra_path)
    RB = np.load(rb_path)
    secret = np.load(secret_path)
    return base_dir, RA, RB, secret


def _make_prefix_only_candidate(secret, nu):
    n_dim = secret.shape[0]
    nu = max(0, min(int(nu), n_dim))
    cand = np.zeros_like(secret)
    cand[:nu] = secret[:nu]
    return cand


def _load_candidate_secret(
    secret_dir, hamming, seed, candidate, candidate_path, secret, nu
):
    if candidate == "partial":
        return None, None
    if candidate == "prefix_only":
        return _make_prefix_only_candidate(secret, nu), None
    if candidate == "secret_raw":
        raw_path = os.path.join(secret_dir, f"secret_raw_{hamming}_{seed}.npy")
        if not os.path.exists(raw_path):
            raise FileNotFoundError(raw_path)
        return np.load(raw_path), raw_path
    if candidate == "custom":
        if not candidate_path:
            raise ValueError("candidate_path required when candidate=custom")
        if not os.path.exists(candidate_path):
            raise FileNotFoundError(candidate_path)
        return np.load(candidate_path), candidate_path
    raise ValueError(f"unknown candidate type: {candidate}")


def _is_binary(vec):
    return np.all((vec == 0) | (vec == 1))


def _make_partial_candidate(secret, nu, q, rng):
    n_dim = secret.shape[0]
    nu = max(0, min(int(nu), n_dim))
    cand = secret.copy()
    if _is_binary(secret):
        total_ones = int(secret.sum())
        prefix_ones = int(secret[:nu].sum())
        remain_ones = max(total_ones - prefix_ones, 0)
        tail = np.array(
            [1] * remain_ones + [0] * (n_dim - nu - remain_ones),
            dtype=secret.dtype,
        )
        rng.shuffle(tail)
        cand[nu:] = tail
    else:
        cand[nu:] = rng.integers(0, q, size=n_dim - nu, dtype=secret.dtype)
    return cand


def _make_random_candidate(secret, q, rng):
    if _is_binary(secret):
        return rng.permutation(secret)
    return rng.integers(0, q, size=secret.shape[0], dtype=secret.dtype)


def _residuals(RA, RB, vec, q, centered, normalize):
    vals = (RA @ vec - RB) % q
    if centered:
        vals = ((vals + q // 2) % q) - q // 2
    if normalize:
        return vals / q
    return vals


def _sample_residuals(resid, n_samples, rng):
    if n_samples is None or n_samples <= 0:
        return resid
    m = resid.shape[0]
    if n_samples <= m:
        idx = rng.choice(m, size=n_samples, replace=False)
    else:
        idx = rng.integers(0, m, size=n_samples)
    return resid[idx]


def plot_residual_histogram(
    RA,
    RB,
    secret,
    q,
    nu,
    candidate_secret,
    candidate_label,
    n_samples,
    seed,
    centered,
    normalize,
    bins,
    title,
    ax,
):
    rng = np.random.default_rng(seed)
    true_resid = _sample_residuals(_residuals(RA, RB, secret, q, centered, normalize), n_samples, rng)
    if candidate_secret is None:
        candidate_secret = _make_partial_candidate(secret, nu, q, rng)
        candidate_label = candidate_label or f"candidate secret (nu={nu})"
    else:
        candidate_label = candidate_label or "candidate secret"
    candidate_resid = _sample_residuals(
        _residuals(RA, RB, candidate_secret, q, centered, normalize), n_samples, rng
    )
    random_secret = _make_random_candidate(secret, q, rng)
    random_resid = _sample_residuals(_residuals(RA, RB, random_secret, q, centered, normalize), n_samples, rng)

    uniform_scale = 1.0 if normalize else float(q)
    uniform_std = uniform_scale / np.sqrt(12)

    def ratio_label(name, resid):
        ratio = float(np.std(resid) / uniform_std)
        return f"{name}, std/uniform = {ratio:.2f}"

    ax.hist(true_resid, bins=bins, alpha=0.5, label=ratio_label("true secret", true_resid))
    ax.hist(
        candidate_resid,
        bins=bins,
        alpha=0.5,
        label=ratio_label(candidate_label, candidate_resid),
    )
    ax.hist(
        random_resid,
        bins=bins,
        alpha=0.5,
        label=ratio_label("fully random secret", random_resid),
    )
    if normalize:
        ax.set_xlabel(r"$(A \cdot s - b)$ (mod q) / q")
    else:
        ax.set_xlabel(r"$(A \cdot s - b)$ (mod q)")
    ax.set_ylabel("Count")
    if title:
        ax.set_title(title)
    ax.legend()

    prefix_len = min(int(nu), secret.shape[0])
    meta = {
        "binary_secret": bool(_is_binary(secret)),
        "nu": int(nu),
        "prefix_matches": int(
            np.sum(secret[:prefix_len] == candidate_secret[:prefix_len])
        ),
        "true_hamming": int(secret.sum()) if _is_binary(secret) else None,
        "candidate_hamming": int(candidate_secret.sum()) if _is_binary(secret) else None,
        "random_hamming": int(random_secret.sum()) if _is_binary(secret) else None,
    }
    return meta


def _plot_one(secret_dir, tag, args, plt):
    params = _load_params(secret_dir)
    q = _resolve_q(params, args.Q)
    base_dir, RA, RB, secret = _load_arrays(
        secret_dir, args.hamming, args.seed, args.use_orig
    )
    candidate_secret, candidate_path = _load_candidate_secret(
        secret_dir,
        args.hamming,
        args.seed,
        args.candidate,
        args.candidate_path,
        secret,
        args.nu,
    )
    candidate_label = None
    if args.candidate == "secret_raw":
        candidate_label = "raw secret"
    elif args.candidate == "custom":
        candidate_label = "custom secret"
    elif args.candidate == "prefix_only":
        candidate_label = f"candidate secret (nu={args.nu}, rest 0)"
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    title = args.title or f"{tag} (n={secret.shape[0]}, log2q={int(np.log2(q))})"
    meta = plot_residual_histogram(
        RA=RA,
        RB=RB,
        secret=secret,
        q=q,
        nu=args.nu,
        candidate_secret=candidate_secret,
        candidate_label=candidate_label,
        n_samples=args.n_samples,
        seed=args.rand_seed,
        centered=not args.no_centered,
        normalize=not args.no_normalize,
        bins=args.bins,
        title=title,
        ax=ax,
    )
    out_tag = args.tag or tag
    outfile = args.outfile
    if not outfile:
        outfile = os.path.join(
            secret_dir,
            f"residual_{out_tag}_h{args.hamming}_s{args.seed}.png",
        )
    fig.tight_layout()
    fig.savefig(outfile, dpi=200)
    if args.show:
        plt.show()
    plt.close(fig)

    print("Output:", outfile)
    print("Base dir:", base_dir)
    print("Secret dir:", secret_dir)
    print("A source:", "orig_A.npy" if args.use_orig else "reduced_A.npy")
    print("Candidate:", args.candidate)
    if candidate_path:
        print("Candidate path:", candidate_path)
    print("Centered:", not args.no_centered, "Normalize:", not args.no_normalize)
    print("Binary secret:", meta["binary_secret"], "nu:", meta["nu"])
    print("Prefix matches:", meta["prefix_matches"])
    if meta["true_hamming"] is not None:
        print(
            "Hamming (true/candidate/random):",
            meta["true_hamming"],
            meta["candidate_hamming"],
            meta["random_hamming"],
        )
    print()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot residual histograms with plain/obfuscated outputs separated.",
    )
    parser.add_argument("--secret_path", type=str, default="", help="path to secret dir")
    parser.add_argument(
        "--plain_secret_path",
        type=str,
        default="",
        help="path to plain secret dir",
    )
    parser.add_argument(
        "--hallucination_secret_path",
        type=str,
        default="",
        help="path to hallucinated secret dir",
    )
    parser.add_argument("--hamming", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--nu", type=int, default=75)
    parser.add_argument("--n_samples", type=int, default=4000000)
    parser.add_argument("--bins", type=int, default=200)
    parser.add_argument("--Q", type=int, default=0, help="override modulus q")
    parser.add_argument("--use_orig", action="store_true")
    parser.add_argument("--no_centered", action="store_true")
    parser.add_argument("--no_normalize", action="store_true")
    parser.add_argument(
        "--candidate",
        choices=["partial", "prefix_only", "secret_raw", "custom"],
        default="partial",
        help="candidate secret type",
    )
    parser.add_argument(
        "--candidate_path",
        type=str,
        default="",
        help="path to custom candidate secret .npy",
    )
    parser.add_argument("--rand_seed", type=int, default=0)
    parser.add_argument("--title", type=str, default="")
    parser.add_argument("--tag", type=str, default="")
    parser.add_argument("--outfile", type=str, default="")
    parser.add_argument("--show", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.show:
        import matplotlib
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tasks = []
    if args.secret_path:
        tasks.append(("custom", args.secret_path))
    if args.plain_secret_path:
        tasks.append(("plain", args.plain_secret_path))
    if args.hallucination_secret_path:
        tasks.append(("hallucination", args.hallucination_secret_path))
    if not tasks:
        raise SystemExit("Provide --secret_path or --plain_secret_path/--hallucination_secret_path")

    if len(tasks) > 1 and args.outfile:
        print("Warning: --outfile ignored for multiple outputs")

    for tag, secret_dir in tasks:
        _plot_one(secret_dir, tag, args, plt)


if __name__ == "__main__":
    main()
