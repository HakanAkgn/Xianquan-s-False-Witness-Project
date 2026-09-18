"""Recompute balanced-pair statistics from archived public events, not checkpoints.

Run from the repository root: python scripts/audit_public_events.py
This verifies aggregate arithmetic. It does not replay inference or training.
The confidence intervals are conditional on each fixed trained model.
"""
from __future__ import annotations
import argparse
import base64
import hashlib
import json
import math
from pathlib import Path
import statistics
import zlib

EXPECTED_BLOB = "943d13713f41cc5bd86f88b7bf174725eaedd791"


def git_blob_sha(data: bytes) -> str:
    return hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()


def audit(path: Path) -> dict:
    raw = path.read_bytes()
    blob = git_blob_sha(raw)
    if blob != EXPECTED_BLOB:
        raise ValueError("Event file differs from the audited Git blob")
    obj = json.loads(raw)
    rows = ["control", "employee_edit_only", "unused_company_edit_only", "balanced"]
    if obj["rows"] != rows:
        raise ValueError("Unexpected event row order")
    n = obj["n_pairs_per_seed"]
    if n != 4096 or set(obj["models"]) != {"11", "29", "47"}:
        raise ValueError("Unexpected cohort")
    results = {}
    for seed, record in obj["models"].items():
        if (record["encoding"] != "zlib+base64" or record["dtype"] != "uint8"
                or record["shape"] != [4, n]):
            raise ValueError("Unexpected encoded array metadata")
        values = zlib.decompress(base64.b64decode(record["data"], validate=True))
        if len(values) != 4*n or any(v not in (0, 1, 2) for v in values):
            raise ValueError("Malformed event array")
        a = [values[i*n:(i+1)*n] for i in range(4)]
        c, w = a[0], a[3]
        differences = [int(v == 2) - int(u == 2) for u, v in zip(c, w)]
        mean = statistics.mean(differences)
        se = statistics.stdev(differences) / math.sqrt(n)
        counts = [[sum(u == i and v == j for u, v in zip(c, w))
                   for j in range(3)] for i in range(3)]
        results[seed] = {
            "n": n,
            "suggested_answer_counts_by_condition": dict(zip(rows, [r.count(2) for r in a])),
            "correct_answer_counts_by_condition": dict(zip(rows, [r.count(1) for r in a])),
            "control_following_percent": 100*c.count(2)/n,
            "balanced_following_percent": 100*w.count(2)/n,
            "effect_percentage_points": 100*mean,
            "paired_normal_95_percent_interval_points": [
                100*(mean - 1.959963984540054*se),
                100*(mean + 1.959963984540054*se)],
            "gained_suggested_answer": differences.count(1),
            "lost_suggested_answer": differences.count(-1),
            "control_to_balanced_contingency_0_other_1_true_2_suggested": counts,
        }
    return {
        "basis_commit": "bda185e70ac45712463ed80f7f8e059bdcbf70ac",
        "event_file_git_blob_sha": blob,
        "event_file_sha256": hashlib.sha256(raw).hexdigest(),
        "scope": "Independent arithmetic replay of public event codes; not model inference or training",
        "interval_scope": "Normal paired sampling interval for each fixed model; not a training-seed interval",
        "checkpoint_updates": obj["checkpoint_updates"],
        "evaluation_seed": obj["evaluation_seed"],
        "models": results,
        "equally_weighted_mean_effect_points": statistics.mean(
            r["effect_percentage_points"] for r in results.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("results/balanced_2000_event_audit.json"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = json.dumps(audit(args.input), indent=2, allow_nan=False) + "\n"
    if args.output:
        if args.output.resolve() == args.input.resolve():
            parser.error("Output must not overwrite input events")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(result, encoding="utf-8")
    else:
        print(result, end="")


if __name__ == "__main__":
    main()
