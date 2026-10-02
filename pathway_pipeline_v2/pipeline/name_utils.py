"""Name normalization for metabolite/feature matching.

Mirrors the canonical normalization used by the outlier-detection pipeline's
endogenous keep-list (``outlier_detection_pipeline/pipeline/data_loader.py``)
so a feature column and an HMDB name/synonym match under the exact same rules:
NFKC folding, Greek symbols AND spelled-out Greek words collapsed to a single
spelled-out English token, lipid-shorthand 'W' between digits folded to OMEGA,
and uppercasing. This keeps feature matching consistent across the two
pipelines (e.g. 'PS(18:2w6/24:1w9)' == 'PS(18:2W6/24:1W9)' == 'PS(18:2omega6)').

Bump ``_NORMALIZATION_VERSION`` whenever the normalized output changes so the
HMDB name index cache is rebuilt instead of reusing a stale index.
"""

import re
import unicodedata


_GREEK_SYMBOL_TO_WORD = {
    "α": "ALPHA", "Α": "ALPHA",
    "β": "BETA", "Β": "BETA",
    "γ": "GAMMA", "Γ": "GAMMA",
    "δ": "DELTA", "Δ": "DELTA",
    "ε": "EPSILON", "Ε": "EPSILON",
    "ζ": "ZETA", "Ζ": "ZETA",
    "η": "ETA", "Η": "ETA",
    "θ": "THETA", "Θ": "THETA", "ϑ": "THETA",
    "ι": "IOTA", "Ι": "IOTA",
    "κ": "KAPPA", "Κ": "KAPPA", "ϰ": "KAPPA",
    "λ": "LAMBDA", "Λ": "LAMBDA",
    "μ": "MU", "Μ": "MU",
    "ν": "NU", "Ν": "NU",
    "ξ": "XI", "Ξ": "XI",
    "ο": "OMICRON", "Ο": "OMICRON",
    "π": "PI", "Π": "PI", "ϖ": "PI",
    "ρ": "RHO", "Ρ": "RHO", "ϱ": "RHO",
    "σ": "SIGMA", "Σ": "SIGMA", "ς": "SIGMA",
    "τ": "TAU", "Τ": "TAU",
    "υ": "UPSILON", "Υ": "UPSILON",
    "φ": "PHI", "Φ": "PHI", "ϕ": "PHI",
    "χ": "CHI", "Χ": "CHI",
    "ψ": "PSI", "Ψ": "PSI",
    "ω": "OMEGA", "Ω": "OMEGA",
}

_GREEK_WORD_PATTERNS = [
    (re.compile(r"(?i)\balpha\b"), "ALPHA"),
    (re.compile(r"(?i)\bbeta\b"), "BETA"),
    (re.compile(r"(?i)\bgamma\b"), "GAMMA"),
    (re.compile(r"(?i)\bdelta\b"), "DELTA"),
    (re.compile(r"(?i)\bepsilon\b"), "EPSILON"),
    (re.compile(r"(?i)\bzeta\b"), "ZETA"),
    (re.compile(r"(?i)\beta\b"), "ETA"),
    (re.compile(r"(?i)\btheta\b"), "THETA"),
    (re.compile(r"(?i)\biota\b"), "IOTA"),
    (re.compile(r"(?i)\bkappa\b"), "KAPPA"),
    (re.compile(r"(?i)\blambda\b"), "LAMBDA"),
    (re.compile(r"(?i)\bmu\b"), "MU"),
    (re.compile(r"(?i)\bnu\b"), "NU"),
    (re.compile(r"(?i)\bxi\b"), "XI"),
    (re.compile(r"(?i)\bomicron\b"), "OMICRON"),
    (re.compile(r"(?i)\bpi\b"), "PI"),
    (re.compile(r"(?i)\brho\b"), "RHO"),
    (re.compile(r"(?i)\bsigma\b"), "SIGMA"),
    (re.compile(r"(?i)\btau\b"), "TAU"),
    (re.compile(r"(?i)\bupsilon\b"), "UPSILON"),
    (re.compile(r"(?i)\bphi\b"), "PHI"),
    (re.compile(r"(?i)\bchi\b"), "CHI"),
    (re.compile(r"(?i)\bpsi\b"), "PSI"),
    (re.compile(r"(?i)\bomega\b"), "OMEGA"),
]

_LIPID_W = re.compile(r"(?<=\d)W(?=\d)", re.IGNORECASE)

_NON_ALNUM = re.compile(r"[^A-Z0-9]+")

_NORMALIZATION_VERSION = 3


def normalize_name(name: str) -> str:
    """
    Normalize a metabolite or feature-column name for exact matching.

    Strips a leading UTF-8 BOM, applies NFKC, folds Greek symbols and
    spelled-out Greek words to a single canonical English token, folds
    lipid-shorthand 'W' between digits to OMEGA, and uppercases.

    Returns the normalized name. Exact (not partial) matching is preserved.
    """
    if name is None:
        return ""
    s = str(name)
    if s.startswith("\ufeff"):
        s = s[1:]
    s = unicodedata.normalize("NFKC", s)
    s = "".join(_GREEK_SYMBOL_TO_WORD.get(ch, ch) for ch in s)
    for pattern, repl in _GREEK_WORD_PATTERNS:
        s = pattern.sub(repl, s)
    s = _LIPID_W.sub("OMEGA", s)
    return s.strip().upper()


def normalize_loose(name: str) -> str:
    """
    Aggressive normalization used as a fallback for endogenous matching.

    Applies :func:`normalize_name` and then removes every non-alphanumeric
    character, collapsing hyphenation/spacing/punctuation differences (e.g.
    'Coproporphyrin III' == 'Coproporphyrin-III' == 'COPROPORPHYRINIII').
    """
    s = normalize_name(name)
    s = _NON_ALNUM.sub("", s)
    return s
