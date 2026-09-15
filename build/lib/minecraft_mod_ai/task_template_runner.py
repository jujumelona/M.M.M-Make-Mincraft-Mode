"""Template-runtime compatibility facade.

Record extraction is host-owned in :mod:`design_record_runtime`; deterministic
artifact rendering is isolated in :mod:`artifact_template_runner`.  No model-owned
record/done/applicability/continuation loop lives in this module.
"""

from .artifact_template_runner import (
    _STANDARD_PORT_DEFINITIONS,
    _bind_job_dependencies,
    _job_value,
    _logical_port,
    execute_artifact_template,
)
from .bounded_record_template import record_batch_response_schema as record_response_schema
from .design_record_runtime import run_record_template
from .fixed_template_generation import generate_fixed_template_value
from .template_errors import TemplateBlocked

__all__ = [
    "TemplateBlocked",
    "_STANDARD_PORT_DEFINITIONS",
    "_bind_job_dependencies",
    "_job_value",
    "_logical_port",
    "execute_artifact_template",
    "generate_fixed_template_value",
    "record_response_schema",
    "run_record_template",
]
