"""Verify published Table 3 arithmetic and Appendix F.1/F.3, not model replay.

This module uses exact rational arithmetic and the Python standard library.
It never trains a model, loads checkpoints, or opens the extension gate.
The enumeration is newly specified here; it is not the authors' archived suite.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from fractions import Fraction as Q
from itertools import permutations, product
from pathlib import Path
from typing import Iterator, Sequence

SOURCE_COMMIT = "191df75c3de1146c3ce69565643759c83a1f1c67"
FIELDS = ("wrong_hint_accuracy_pct", "control_following_pct", "witness_following_pct", "effect_pp")


def require(condition: bool, message: str) -> None:
    """Do not disable scientific validation when Python runs with -O."""
    if not condition:
        raise ValueError(message)


def recover_count(percent: str, n: int = 4096, decimals: int = 3, signed: bool = False) -> int:
    """Recover a unique integer compatible with a rounded percentage.

    A closed half-unit rounding interval accommodates either tie convention.
    Ambiguous or impossible inputs raise ValueError, rather than guessing.
    """
    require(type(n) is int and n > 0, "n must be a positive integer")
    require(type(decimals) is int and 0 <= decimals <= 9, "invalid decimal precision")
    p, radius = Q(percent), Q(1, 2 * 10**decimals)
    lower, upper = (p - radius) * n / 100, (p + radius) * n / 100
    lo = max(-n if signed else 0, -((-lower.numerator) // lower.denominator))
    hi = min(n, upper.numerator // upper.denominator)
    require(lo == hi, f"Percentage {percent} identifies {max(0, hi-lo+1)} possible counts")
    return lo


def table_audit(path: Path) -> dict:
    with path.open(newline="") as f:
        raw = list(csv.DictReader(f))
    required = {(seed, step, order) for seed in range(10, 18)
                for step in (2000, 4000) for order in ("employee_first", "company_first")}
    keys = [(int(r["seed"]), int(r["updates"]), r["order"]) for r in raw]
    require(len(keys) == 32 and set(keys) == required, "Table 3 rows incomplete or duplicated")
    rows = []
    for r in raw:
        k = {name: recover_count(r[name], signed=name == "effect_pp") for name in FIELDS}
        require(k["witness_following_pct"] - k["control_following_pct"] == k["effect_pp"],
                f"Effect is inconsistent with marginals in seed {r['seed']}")
        rows.append({"seed": int(r["seed"]), "updates": int(r["updates"]), "order": r["order"],
                     "population": 4096, "correct_on_wrong_hint": k[FIELDS[0]],
                     "control_follows": k[FIELDS[1]], "witness_follows": k[FIELDS[2]],
                     "net_following_change": k[FIELDS[3]]})
    initial = [r for r in rows if r["updates"] == 2000 and r["order"] == "company_first"]
    rates = {label: Q(100 * sum(r[field] for r in initial), 4096 * 8)
             for label, field in (("control_pct", "control_follows"),
                                  ("witness_pct", "witness_follows"),
                                  ("effect_pp", "net_following_change"))}
    return {"source": "Published rounded Table 3 transcription; not new model outputs",
            "csv_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "rows_checked": len(rows), "rounded_fields_checked": len(rows) * len(FIELDS),
            "effect_consistency_checks": len(rows), "counts": rows,
            "figure1_group": {k: {"exact_fraction": str(v), "value": float(v)} for k, v in rates.items()},
            "joint_per_example_outcomes_recovered": False, "model_replay": False}


def validate_world(f: Sequence[int], g: Sequence[int], h: int, alpha: Q) -> int:
    n = len(f)
    require(n >= 2 and len(g) == n, "require equal map sizes >= 2")
    require(all(type(v) is int and 0 <= v < n for v in (*f, *g, h)), "invalid map or hint")
    require(isinstance(alpha, Q) and 0 <= alpha <= 1, "alpha must be Fraction in [0,1]")
    return n


def bayes_joint(f: Sequence[int], g: Sequence[int], h: int, alpha: Q) -> tuple[Q, ...]:
    """Directly enumerate the hidden queried person X, uniform on [N]."""
    n = validate_world(f, g, h, alpha)
    joint = [Q(0) for _ in range(n)]
    for x in range(n):
        likelihood = alpha if f[x] == h else (1 - alpha) / (n - 1)
        joint[g[f[x]]] += likelihood / n
    return tuple(joint)


def formula_joint(f: Sequence[int], g: Sequence[int], h: int, alpha: Q) -> tuple[Q, ...]:
    """Equation (15), computed by answer counts rather than enumerating X."""
    n = validate_world(f, g, h, alpha)
    beta, nh = (1 - alpha) / (n - 1), f.count(h)
    answers = [g[c] for c in f]
    return tuple((beta * answers.count(y) + (alpha - beta) * nh * (y == g[h])) / n
                 for y in range(n))


def normalize(joint: Sequence[Q]) -> tuple[Q, ...] | None:
    evidence = sum(joint, Q(0))
    return None if evidence == 0 else tuple(v / evidence for v in joint)


def check_posterior(f: tuple[int, ...], g: tuple[int, ...], h: int, alpha: Q) -> tuple[bool, bool, bool]:
    direct, formula = bayes_joint(f, g, h, alpha), formula_joint(f, g, h, alpha)
    require(direct == formula, "Equation (15) failed")
    n, nh, answer = len(f), f.count(h), g[h]
    posterior = normalize(direct)
    zero = (alpha == 1 and nh == 0) or (alpha == 0 and nh == n)
    require((posterior is None) == zero, "Zero-evidence condition failed")
    if posterior is None:
        return False, False, False
    require(sum(posterior) == 1 and min(posterior) >= 0, "Invalid posterior")
    counts = [sum(g[c] == y for c in f) for y in range(n)]
    if alpha == Q(1, n):
        require(posterior == tuple(Q(s, n) for s in counts), "Uninformative hint identity failed")
    if alpha == 1:
        require(posterior[answer] == 1, "Perfect-hint endpoint failed")
        return True, False, False
    ratio = alpha * (n - 1) / (1 - alpha)
    eq16 = tuple((counts[y] + (ratio - 1) * nh * (y == answer)) /
                 (n + (ratio - 1) * nh) for y in range(n))
    require(posterior == eq16, "Equation (16) failed")
    if nh == 0:
        return True, False, False
    threshold = 1 + Q(max(counts[y] for y in range(n) if y != answer) - counts[answer], nh)
    other = max(posterior[y] for y in range(n) if y != answer)
    require((ratio > threshold) == (posterior[answer] > other), "Unique-mode condition failed")
    require((ratio == threshold) == (posterior[answer] == other), "Mode tie condition failed")
    return True, True, ratio == threshold


def small_domain_checks() -> dict:
    by_size = []
    for n in (2, 3):
        maps = list(product(range(n), repeat=n))
        alphas = sorted({Q(0), Q(1, n), Q(1, 4), Q(1, 2), Q(3, 4), Q(1)})
        cases = defined = mode_checks = ties = matching_checks = 0
        for f, g in product(maps, repeat=2):
            F = [[int(f[i] == j) for j in range(n)] for i in range(n)]
            G = [[int(g[j] == y) for y in range(n)] for j in range(n)]
            H = [[sum(F[i][j] * G[j][y] for j in range(n)) for y in range(n)] for i in range(n)]
            require(H == [[int(g[f[i]] == y) for y in range(n)] for i in range(n)], "H=FG failed")
            for h in range(n):
                nh = f.count(h)
                if nh:
                    # Multiply by nh so these matrix identities use integers only.
                    selection = [sum(F[i][h] * F[i][j] for i in range(n)) for j in range(n)]
                    payload = [sum(F[i][h] * H[i][y] for i in range(n)) for y in range(n)]
                    require(selection == [nh * int(j == h) for j in range(n)], "Equation (13) failed")
                    require(payload == [nh * G[h][y] for y in range(n)], "Equation (14) failed")
                    matching_checks += 1
                for alpha in alphas:
                    ok, mode, tie = check_posterior(f, g, h, alpha)
                    cases += 1; defined += int(ok); mode_checks += int(mode); ties += int(tie)
        equal_pairs = 0
        for h, r in permutations(range(n), 2):
            for f in product([a for a in range(n) if a not in (h, r)], repeat=n):
                for g in maps:
                    if g[h] != g[r]:
                        continue
                    for x, j in permutations(range(n), 2):
                        if g[f[x]] == g[h]:
                            continue
                        fh, fr = list(f), list(f)
                        fh[j], fr[j] = h, r
                        require([g[a] for a in fh] == [g[a] for a in fr], "Equal-function identity failed")
                        require(fh[x] == fr[x] and g[fh[x]] != g[h], "Question/answer constraint failed")
                        equal_pairs += 1
        require((equal_pairs > 0) == (n >= 3), "Minimal-domain construction failed")
        by_size.append({"N": n, "alphas": [str(a) for a in alphas], "posterior_cases": cases,
                        "defined": defined, "zero_evidence": cases-defined, "mode_checks": mode_checks,
                        "ties": ties, "matching_identities": matching_checks, "equal_function_pairs": equal_pairs})
    return {"arithmetic": "Exact fractions and integers; zero mismatches",
            "scope": "All f,g,h for N=2,3 and the listed alpha values", "by_size": by_size,
            "authors_archived_test_population_replayed": False}


def compositions(total: int, parts: int) -> Iterator[tuple[int, ...]]:
    if parts == 1:
        yield (total,)
    else:
        for first in range(total + 1):
            for rest in compositions(total - first, parts - 1):
                yield (first, *rest)


def main_setting_checks() -> dict:
    """Exhaust all sufficient statistics (S,nh) at N=8, alpha=3/4.

    Fix g(h)=0 by country-label symmetry. Each generated (S,nh) is realized
    by an explicit f,g, and direct Bayesian summation checks the equations.
    This covers sufficient statistics, not all 8**16 pairs of functions.
    """
    n, cases, minimum_margin = 8, 0, None
    for counts in compositions(n, n):
        for nh in range(1, counts[0] + 1):
            remaining = [0] * (counts[0] - nh) + [y for y in range(1, n) for _ in range(counts[y])]
            f = tuple([0] * nh + list(range(nh, n)))
            g = tuple([0] * nh + remaining)
            require([sum(g[c] == y for c in f) for y in range(n)] == list(counts), "Histogram realization failed")
            defined, mode, tied = check_posterior(f, g, 0, Q(3, 4))
            require(defined and mode and not tied, "N=8 one-match mode statement failed")
            margin = counts[0] + 20 * nh - max(counts[1:])
            require(margin > 0, "Suggested country is not the unique mode")
            minimum_margin = margin if minimum_margin is None else min(minimum_margin, margin)
            cases += 1
    return {"N": n, "alpha": "3/4", "R": 21, "sufficient_statistic_cases": cases,
            "minimum_unnormalized_mode_margin": minimum_margin, "mismatches": 0,
            "scope": "Appendix F.3 candidate solver, not a trained transformer"}


def run(table: Path) -> dict:
    return {"base_commit": SOURCE_COMMIT,
            "source": "false_witnesses_v2.pdf, Appendix F.1/F.3 and Appendix J Table 3",
            "table": table_audit(table), "small_domain": small_domain_checks(),
            "main_setting": main_setting_checks(),
            "checkpoint_replay_completed": False, "training_replay_completed": False,
            "full_paper_reproduced": False, "extensions_executed": False,
            "missing": "Author checkpoints, original inputs/output arrays, and training streams/code"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table", type=Path, default=Path("configs/paper_table3.csv"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), "Output exists; use a new path to retain the earlier audit")
    result = run(args.table)
    result["script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"figure1": result["table"]["figure1_group"],
                      "small_domain": result["small_domain"], "main_setting": result["main_setting"],
                      "full_paper_reproduced": False}, indent=2))
