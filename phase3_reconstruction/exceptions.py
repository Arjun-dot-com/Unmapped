"""Typed exceptions for Phase 3.

The CLI (``phase3_reconstruction.run``) catches :class:`Phase3Error` at the top
level and prints a single clean, actionable line instead of a raw traceback.
Anything that is genuinely a bug in our code is allowed to crash loudly.
"""

from __future__ import annotations


class Phase3Error(Exception):
    """Base class for all *expected* Phase 3 failures (bad input, missing dep...)."""


class Phase3InputError(Phase3Error):
    """Upstream data (Phase 1 / Phase 2) is missing, malformed, or inconsistent."""


class Phase3ConfigError(Phase3Error):
    """The YAML/CLI configuration is invalid."""


class Phase3DependencyError(Phase3Error):
    """An optional heavy dependency (torch / gsplat / transformers) is required
    for the requested mode but is not importable."""
