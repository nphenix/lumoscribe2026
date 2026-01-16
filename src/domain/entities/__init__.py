"""领域实体模块。"""

from .source_file import SourceFile
from .template import Template
from .intermediate_artifact import IntermediateArtifact
from .target_file import TargetFile

__all__ = [
    "SourceFile",
    "Template",
    "IntermediateArtifact",
    "TargetFile",
]
