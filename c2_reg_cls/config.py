"""Anatomical labels and non-sensitive algorithm defaults.

Machine-specific paths and atlas identifiers intentionally do not live in this
module.  Reviewer-facing paths are supplied by ``configs/pipeline.example.json``
or explicit CLI arguments.
"""

# ---------------------------------------------------------------------------
# Anatomical labels for the C2 vertebra sub-segmentation
# ---------------------------------------------------------------------------
REGION_LABELS = {
    3: "Dens - Tip",
    4: "Dens - Mid",
    5: "Odontoid Body",
    6: "Lateral Mass - Right",
    7: "Lateral Mass - Left",
    8: "Superior Articular Facet - Right",
    9: "Superior Articular Facet - Left",
    10: "Transverse Process - Right",
    11: "Transverse Process - Left",
    14: "Pars Interarticularis - Right",
    15: "Pars Interarticularis - Left",
    16: "Inferior Articular Process/Facet - Right",
    17: "Inferior Articular Process/Facet - Left",
    20: "Lamina - Right",
    21: "Lamina - Left",
    22: "Spinous Process",
}

ALL_LABELS = sorted(REGION_LABELS)

# Five anatomical groups used for the paper's primary localization analysis.
ANATOMICAL_GROUPS = {
    "Odontoid Process": (3, 4),
    "Subdental Axis Body and Superior Lateral Mass Complex": (5, 6, 7, 8, 9),
    "Transverse Process Complex": (10, 11),
    "Pars-Inferior Facet Complex": (14, 15, 16, 17),
    "Posterior Arch Complex": (20, 21, 22),
}

LABEL_TO_ANATOMICAL_GROUP = {
    label: group
    for group, labels in ANATOMICAL_GROUPS.items()
    for label in labels
}

DEFAULT_OUTPUT_SHAPE = (96, 96, 96)
