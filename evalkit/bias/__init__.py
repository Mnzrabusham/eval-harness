"""bias/: judge systematic-error measurement (docs/statistics-spec.md §14).

Three estimands, each a point estimate with a CI from the §2 engine:
position bias (§14.2), the length-preference association and controlled
verbosity bias (§14.3), and self-preference bias (§14.4).

Naming rule (F14): the observational length quantity is an ASSOCIATION and
is named as such; the only public "verbosity_bias" symbol is the
controlled-study estimator.
"""

from ._core import BiasReport, OrderBalancedReduction, build_bias_report, order_balanced_reduce
from .position import position_bias
from .self_preference import self_preference_bias
from .verbosity import (
    VerbosityControlledReport,
    verbosity_association,
    verbosity_bias_controlled,
)

__all__ = [
    "BiasReport",
    "OrderBalancedReduction",
    "VerbosityControlledReport",
    "build_bias_report",
    "order_balanced_reduce",
    "position_bias",
    "self_preference_bias",
    "verbosity_association",
    "verbosity_bias_controlled",
]
