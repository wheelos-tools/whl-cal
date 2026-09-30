"""Numerical equivalence reporting for GRIL frontend traces."""

from __future__ import annotations

import itertools
import sys
from pathlib import Path
from typing import Any


def compare_frontend_traces(
    reference_path: Path,
    candidate_path: Path,
    *,
    epsilon_multiplier: float = 16.0,
) -> dict[str, Any]:
    machine_epsilon = sys.float_info.epsilon
    absolute_tolerance = machine_epsilon
    relative_tolerance = epsilon_multiplier * machine_epsilon
    numeric_values = 0
    inexact_values = 0
    max_absolute_error = 0.0
    max_relative_error = 0.0
    inexact_by_record: dict[str, int] = {}

    with reference_path.open() as reference, candidate_path.open() as candidate:
        for line_number, lines in enumerate(
            itertools.zip_longest(reference, candidate), start=1
        ):
            reference_line, candidate_line = lines
            if reference_line is None or candidate_line is None:
                return {
                    "verdict": "different",
                    "reason": f"trace length differs at line {line_number}",
                }
            reference_tokens = reference_line.split()
            candidate_tokens = candidate_line.split()
            if (
                len(reference_tokens) != len(candidate_tokens)
                or not reference_tokens
                or reference_tokens[0] != candidate_tokens[0]
            ):
                return {
                    "verdict": "different",
                    "reason": f"trace structure differs at line {line_number}",
                }
            record = reference_tokens[0]
            for token_index, (reference_token, candidate_token) in enumerate(
                zip(reference_tokens[1:], candidate_tokens[1:]), start=1
            ):
                try:
                    reference_value = float(reference_token)
                    candidate_value = float(candidate_token)
                except ValueError:
                    if reference_token != candidate_token:
                        return {
                            "verdict": "different",
                            "reason": (
                                "trace token differs at "
                                f"line {line_number}, token {token_index}"
                            ),
                        }
                    continue

                numeric_values += 1
                absolute_error = abs(reference_value - candidate_value)
                scale = max(abs(reference_value), abs(candidate_value))
                relative_error = (
                    absolute_error / scale if scale > 0.0 else absolute_error
                )
                max_absolute_error = max(max_absolute_error, absolute_error)
                max_relative_error = max(max_relative_error, relative_error)
                if absolute_error > 0.0:
                    inexact_values += 1
                    inexact_by_record[record] = inexact_by_record.get(record, 0) + 1
                if absolute_error > (absolute_tolerance + relative_tolerance * scale):
                    return {
                        "verdict": "different",
                        "reason": (
                            "numeric value differs beyond floating-point bound at "
                            f"line {line_number}, token {token_index}"
                        ),
                        "absolute_error": absolute_error,
                        "relative_error": relative_error,
                    }

    return {
        "verdict": "equivalent",
        "comparison": "IEEE-754 double roundoff bound",
        "machine_epsilon": machine_epsilon,
        "epsilon_multiplier": epsilon_multiplier,
        "numeric_values": numeric_values,
        "inexact_values": inexact_values,
        "inexact_by_record": inexact_by_record,
        "max_absolute_error": max_absolute_error,
        "max_relative_error": max_relative_error,
    }
