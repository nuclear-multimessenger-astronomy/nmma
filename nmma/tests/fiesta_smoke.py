"""Smoke test for the fiesta-surrogates pipeline.

Single sanity check that ``FiestaKilonovaModel`` can instantiate against a
locally-available copy of the
`fiesta-surrogates <https://huggingface.co/nuclear-multimessenger-astronomy/fiesta-surrogates>`_
HuggingFace repo. Replaces the GitLab-fetched SVD-model tests retired in this PR.

Skipped automatically when ``fiesta`` is not installed or the surrogates dir
isn't pointed to by ``$NMMA_FIESTA_SURROGATES`` (CI sets this after the
``huggingface-cli download`` step).
"""

import os
import pytest


pytest.importorskip(
    "fiesta", reason="fiesta not installed; surrogate pipeline untested"
)

FIESTA_SURROGATES = os.environ.get("NMMA_FIESTA_SURROGATES")


@pytest.mark.skipif(
    not FIESTA_SURROGATES or not os.path.isdir(FIESTA_SURROGATES),
    reason="NMMA_FIESTA_SURROGATES not set or not a directory",
)
def test_fiesta_kilonova_loads():
    """Instantiate the canonical kilonova surrogate; verify the model advertises parameters."""
    from nmma.em.model import FiestaKilonovaModel

    # Try the most recent surrogate first; fall back to Bu2025 if the runtime
    # doesn't have Bu2026 yet.
    last_err = None
    for model_name in ("Bu2026_MLP", "Bu2025_MLP"):
        try:
            kn_model = FiestaKilonovaModel(
                model=model_name, surrogate_dir=FIESTA_SURROGATES
            )
            break
        except (OSError, ValueError) as e:
            last_err = e
    else:
        raise last_err

    assert kn_model.model_parameters, "fiesta model did not expose any parameters"
    assert kn_model.filters, "fiesta model did not advertise any filters"
