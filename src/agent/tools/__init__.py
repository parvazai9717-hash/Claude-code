"""The tool layer: the agent's hands.

`build_default_tools` is the single place that decides which tools exist in a
run. Nothing registers itself, and the model cannot add to this list.
"""

from __future__ import annotations

from .base import Tool, ToolContext
from .filesystem import ListFilesTool, ReadFileTool, WriteFileTool, initialize_workspace
from .memory_tools import RememberFactTool
from .registry import ToolRegistry, validate_arguments
from .search import SearchFilesTool
from .shell import RunShellTool
from .time_tool import GetCurrentTimeTool
from .verification import VerifyResultTool

__all__ = [
    "GetCurrentTimeTool",
    "ListFilesTool",
    "ReadFileTool",
    "RememberFactTool",
    "RunShellTool",
    "SearchFilesTool",
    "Tool",
    "ToolContext",
    "ToolRegistry",
    "VerifyResultTool",
    "WriteFileTool",
    "build_default_tools",
    "initialize_workspace",
    "validate_arguments",
]


def build_default_tools(*, include_shell: bool = True, include_memory: bool = True) -> list[Tool]:
    """The first-release tool set.

    Args:
        include_shell: Register `run_shell`. Disable to remove the capability
            entirely rather than relying on the allowlist alone.
        include_memory: Register `remember_fact`. Requires a fact store in context.
    """
    tools: list[Tool] = [
        GetCurrentTimeTool(),
        ListFilesTool(),
        ReadFileTool(),
        SearchFilesTool(),
        WriteFileTool(),
        VerifyResultTool(),
    ]
    if include_shell:
        tools.append(RunShellTool())
    if include_memory:
        tools.append(RememberFactTool())
    return tools
