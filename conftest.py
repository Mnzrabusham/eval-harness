"""Root pytest configuration.

Validation replication count: smoke default is 200 (fast, wide Monte Carlo
tolerance bands); full-scale is 10,000 per docs/statistics-spec.md §11.
Set either

    pytest -m validation --validation-reps=10000
    EVALKIT_VALIDATION_REPS=10000 pytest -m validation

Validation studies are excluded by default (addopts in pyproject.toml).
"""

import os

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--validation-reps",
        action="store",
        type=int,
        default=None,
        help="replications per validation cell (default: EVALKIT_VALIDATION_REPS env var or 200; full scale: 10000)",
    )


@pytest.fixture(scope="session")
def reps(request) -> int:
    opt = request.config.getoption("--validation-reps")
    if opt is not None:
        return int(opt)
    return int(os.environ.get("EVALKIT_VALIDATION_REPS", "200"))
