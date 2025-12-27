# Copyright (c) Meta Platforms, Inc. and affiliates.
# See file LICENSE.txt in the main directory.
import numpy as np
from dataclasses import dataclass
import argparse
import pickle
import os
import hashlib

from reduction import make_n_reduced_samples

@dataclass
class Data:
    """Class for keeping track of an item in inventory."""
    params: argparse.Namespace
    RA: np.ndarray
    RB: np.ndarray
    origA: np.ndarray
    origB: np.ndarray
    secret: np.ndarray

    @classmethod
    def from_files(cls, path, hamming_weight=None, seed=0):
        path = path.rstrip("/")
        prefix = os.path.dirname(path)
        secret_prefix = os.path.basename(path)
        params = pickle.load(open(f'{prefix}/{secret_prefix}/params.pkl', 'rb'))
        if isinstance(params, dict):
          params = argparse.Namespace(**params)

        hamming_weight = hamming_weight or params.min_hamming
        params.hamming = hamming_weight
        try:
          Bs = np.load(f'{prefix}/{secret_prefix}/b_{hamming_weight}_{seed}.npy')
        except:
          Bs = np.load(f'{prefix}/{secret_prefix}/train_b_{hamming_weight}_{seed}.npy')

        secret = np.load(f'{prefix}/{secret_prefix}/secret_{hamming_weight}_{seed}.npy')
        try:
          As = np.load(f'{prefix}/reduced_A.npy')
        except:
          As = np.load(f'{prefix}/train_A.npy')

        secret = np.load(f'{prefix}/{secret_prefix}/secret_{hamming_weight}_{seed}.npy')
        origAs = np.load(f'{prefix}/orig_A.npy')
        origBs = np.load(f'{prefix}/{secret_prefix}/orig_b_{hamming_weight}_{seed}.npy')
        return cls(
            params= params,
            RA = As,
            RB = Bs,
            origA = origAs,
            origB = origBs,
            secret = secret,
        )

    @staticmethod
    def create_new_A( # this is meant for experimentation
      secret_dim,
      q,
      n_sniffs,
      savepath="./data",
      lll_penalty = 10,
      sample_target = 100000,
    ):

      save_prefix = f"{savepath}/{secret_dim}_{q}_{n_sniffs}_{lll_penalty}/"
      if not os.path.exists(save_prefix + "origA.npy"):
          origA = np.random.randint(0, q, size=(n_sniffs, secret_dim)) - q // 2
          Rs, subsets = make_n_reduced_samples(origA, sample_target, q, reduction_chunk_size=secret_dim, lll_penalty=lll_penalty)
          os.makedirs(save_prefix, exist_ok=True)
          np.save(save_prefix + "origA.npy", origA)
          np.save(save_prefix + "Rs_default.npy", Rs)
          np.save(save_prefix + "subsets_default.npy", subsets)
      else:
          origA = np.load(save_prefix + "origA.npy")
          Rs = np.load(save_prefix + "Rs_default.npy")
          subsets = np.load(save_prefix + "subsets_default.npy")

      return origA, Rs, subsets


    @classmethod
    def create_data_from_A(
      cls,
      origA,
      Rs,
      subsets,
      hamming_weight,
      Q,
      noise_variance = 3.2,
      n_brute_force=None,
      hamming_weight_in_brute_force=None,
      use_hallucination=False,
      hallucination_k_seed=-1,
      hallucination_k_bits=128,
      hallucination_degrees="1,3,5",
      hallucination_coeff_choices="-1,1",
    ):

      secret_dim = origA.shape[1]
      secret = cls._make_secret_with(
        secret_dim, hamming_weight, n_brute_force, hamming_weight_in_brute_force
      )
      secret_raw = secret
      hallucination_k = None
      hallucination_coeffs = None
      if use_hallucination:
        degrees = [d for d in cls._parse_int_list(hallucination_degrees) if d > 0]
        if not degrees:
          degrees = [1]
        coeff_choices = cls._parse_int_list(hallucination_coeff_choices)
        if not coeff_choices:
          coeff_choices = [-1, 1]
        hallucination_k, k_bytes = cls._make_hallucination_k(
          hallucination_k_bits, hallucination_k_seed
        )
        secret, hallucination_coeffs = cls._maclaurin_obfuscate(
          secret, Q, k_bytes, degrees, coeff_choices
        )
      origB, _ = cls._make_B_from_A(origA, noise_variance, secret, Q)
      RAs, RBs = cls._make_RAs_RBs(origA, origB, Rs, subsets, Q)

      params = argparse.Namespace(
        Q=Q,
        n_sniffs=origA.shape[0],
        secret_dim=secret_dim,
        hamming_weight=hamming_weight,
        sigma=noise_variance,
        n_brute_force=n_brute_force,
        hamming_in_brute_force=hamming_weight_in_brute_force,
        use_hallucination=use_hallucination,
        hallucination_k_seed=hallucination_k_seed,
        hallucination_k_bits=hallucination_k_bits,
        hallucination_degrees=hallucination_degrees,
        hallucination_coeff_choices=hallucination_coeff_choices,
        hallucination_k=hallucination_k.tolist() if hallucination_k is not None else None,
        hallucination_coeffs=hallucination_coeffs,
        secret_raw=secret_raw.tolist() if use_hallucination else None,
      )

      return cls(
        params=params,
        RA=RAs,
        RB=RBs,
        origA=origA,
        origB=origB,
        secret=secret,
      )


    @staticmethod
    def _make_secret_with(
        secret_dim, hamming_weight, n_brute_force=None, hamming_weight_in_brute_force=None
    ):
        if hamming_weight_in_brute_force is None:
            return np.random.permutation(
                [1] * hamming_weight + [0] * (secret_dim - hamming_weight)
            )
        else:
            secret_part1 = np.random.permutation(
                [1] * hamming_weight_in_brute_force
                + [0] * (n_brute_force - hamming_weight_in_brute_force)
            )
            secret_part2 = np.random.permutation(
                [1] * (hamming_weight - hamming_weight_in_brute_force)
                + [0]
                * (
                    secret_dim
                    - n_brute_force
                    - hamming_weight
                    + hamming_weight_in_brute_force
                )
            )
            return np.concatenate([secret_part1, secret_part2])

    @staticmethod
    def _parse_int_list(value):
        if isinstance(value, (list, tuple, np.ndarray)):
            return [int(v) for v in value]
        if value is None:
            return []
        if isinstance(value, str):
            parts = [p.strip() for p in value.split(",") if p.strip() != ""]
            return [int(p) for p in parts]
        return [int(value)]

    @staticmethod
    def _make_hallucination_k(hallucination_k_bits, hallucination_k_seed):
        bits = hallucination_k_bits if hallucination_k_bits > 0 else 128
        nbytes = (bits + 7) // 8
        if hallucination_k_seed is not None and hallucination_k_seed >= 0:
            seed_int = int(hallucination_k_seed)
            seed_len = max(1, (seed_int.bit_length() + 7) // 8)
            seed_bytes = seed_int.to_bytes(seed_len, "big", signed=False)
            k_bytes = hashlib.shake_256(b"HKD|k|" + seed_bytes).digest(nbytes)
        else:
            k_bytes = os.urandom(nbytes)
        k_bits = np.unpackbits(np.frombuffer(k_bytes, dtype=np.uint8))[:bits].astype(np.int64)
        return k_bits, k_bytes

    @staticmethod
    def _xof_uint32_stream(k_bytes, label):
        counter = 0
        buf = b""
        while True:
            if len(buf) < 4:
                h = hashlib.shake_256()
                h.update(b"HKD|" + label + b"|" + counter.to_bytes(4, "big") + b"|" + k_bytes)
                buf += h.digest(64)
                counter += 1
            val = int.from_bytes(buf[:4], "big")
            buf = buf[4:]
            yield val

    @staticmethod
    def _xof_choice_indices(k_bytes, label, count, mod):
        if mod <= 0:
            raise ValueError("mod must be positive")
        limit = (1 << 32) - ((1 << 32) % mod)
        out = []
        for v in Data._xof_uint32_stream(k_bytes, label):
            if v < limit:
                out.append(v % mod)
                if len(out) >= count:
                    break
        return out

    @staticmethod
    def _negacyclic_convolve(a, b, Q):
        conv = np.convolve(a, b)
        n = len(a)
        res = conv[:n].astype(np.int64, copy=True)
        tail = conv[n:]
        if tail.size:
            res[:tail.size] -= tail
        return res % Q

    @staticmethod
    def _maclaurin_obfuscate(s, Q, k_bytes, degrees, coeff_choices):
        s = s.astype(np.int64)
        s_prime = np.zeros_like(s, dtype=np.int64)
        coeffs = {}
        coeff_idx = Data._xof_choice_indices(
            k_bytes, b"coeff", len(degrees), len(coeff_choices)
        )
        for d in degrees:
            idx = coeff_idx.pop(0)
            coeffs[d] = int(coeff_choices[idx])
            if d == 1:
                term = s.copy()
            else:
                term = s.copy()
                for _ in range(d - 1):
                    term = Data._negacyclic_convolve(term, s, Q)
            s_prime = (s_prime + coeffs[d] * term) % Q
        return s_prime, coeffs

    @staticmethod
    def _make_B_from_A(origA: np.array, noise_var, secret, Q):
        noise = np.random.normal(0, noise_var, size=origA.shape[0]).round(0)
        origB = (origA @ secret + noise) % Q
        return origB, noise

    @staticmethod
    def _make_RAs_RBs(origA, origB, Rs, subsets, Q):
        RAs = np.vstack([R_i @ origA[subset] for R_i,subset in zip(Rs, subsets)])
        RBs = np.hstack([R_i @ origB[subset] for R_i,subset in zip(Rs, subsets)])
        sel = (RAs != 0).any(1)
        print(sel.mean())
        RAs = (RAs[sel] + Q//2) % Q - Q//2
        RBs = RBs[sel]
        return RAs, RBs
