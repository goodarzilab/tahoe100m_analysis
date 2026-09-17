"""Canonical drug and condition keys, shared by every script that emits one.

Drug names in the Tahoe plate metadata need care when used as join keys:

  * `Erdafitinib ` and `Selinexor ` carry a trailing space in the raw data
  * `Dapagliflozin ((2S)-1,2-propanediol, hydrate)` contains commas, so the
    condition string must be parsed rather than split
  * every step must agree on one key format

Two rules:

  normalize_drug   collapse internal whitespace runs to one space, strip ends.
                   Commas are left alone: `1,2-propanediol` is correct
                   chemical nomenclature.
  drug_dose_key    "<normalised drug>__<dose>", the one key format in use.

`parse_condition` handles the tuple-formatted `drugname_drugconc` strings the
plate files carry, e.g. "[('Drug', 0.05, 'uM')]", using ast.literal_eval rather
than string splitting so embedded commas are safe.
"""

import re
import ast

__all__ = ["normalize_drug", "drug_dose_key", "parse_condition",
           "condition_to_key", "match_key"]


def normalize_drug(name) -> str:
    """Collapse internal whitespace and strip the ends. Nothing else."""
    return re.sub(r"\s+", " ", str(name)).strip()


def drug_dose_key(drug, dose) -> str:
    """The canonical condition key: "<drug>__<dose>"."""
    return f"{normalize_drug(drug)}__{dose}"


def parse_condition(condition: str):
    """Parse "[('Drug', 0.05, 'uM')]" into (drug, dose, unit).

    Uses ast.literal_eval, so drug names containing commas survive intact.
    """
    drug, dose, unit = ast.literal_eval(str(condition))[0]
    return normalize_drug(drug), dose, unit


def condition_to_key(condition: str) -> str:
    """"[('Drug', 0.05, 'uM')]" -> "Drug__0.05"."""
    drug, dose, _unit = parse_condition(condition)
    return drug_dose_key(drug, dose)


def match_key(key) -> str:
    """A comparison-only form for joining against externally-written labels.

    Strips whitespace entirely and folds case, so a key still matches a label
    that differs only by stray spacing. Use this for lookups; drug_dose_key gives the
    canonical spelling for output.
    """
    return re.sub(r"\s+", "", str(key)).lower()
