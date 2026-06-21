"""Native tool registry."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class NativeToolResult:
    """Structured result returned by a native tool handler."""

    content: str = ""
    data: Mapping[str, object] = field(default_factory=dict)
    refs: tuple[str, ...] = field(default_factory=tuple)
    attachments: tuple[Mapping[str, object], ...] = field(default_factory=tuple)
    schema_additions: tuple[Mapping[str, object], ...] = field(default_factory=tuple)

    def has_output(self) -> bool:
        """Return whether the tool produced content, data, or references."""
        return bool(self.content.strip() or self.data or self.refs or self.attachments)


NativeToolHandler = Callable[[Mapping[str, object]], NativeToolResult]


@dataclass(frozen=True, slots=True)
class NativeTool:
    """Native tool schema and handler owned by Deepmate."""

    name: str
    description: str
    input_schema: Mapping[str, object]
    handler: NativeToolHandler
    read_only: bool = True
    requires_shell: bool = False
    exposed_by_default: bool = True

    def is_ready(self) -> bool:
        """Return whether the tool has the minimum callable contract."""
        return bool(
            self.name.strip()
            and self.description.strip()
            and callable(self.handler)
        )

    def schema(self) -> Mapping[str, object]:
        """Return a provider-neutral schema reference for future surfaces."""
        return {
            "name": self.name.strip(),
            "description": self.description.strip(),
            "input_schema": self.input_schema,
        }

    def call(self, arguments: Mapping[str, object]) -> NativeToolResult:
        """Run the handler with already-allowed arguments."""
        return self.handler(arguments)


class NativeToolRegistry:
    """Registry for Deepmate-owned native tools."""

    def __init__(self, tools: Iterable[NativeTool] = ()) -> None:
        self._tools: dict[str, NativeTool] = {}
        for tool in tools:
            self.register(tool)

    def register(self, tool: NativeTool) -> None:
        """Register one native tool by normalized name."""
        if not tool.is_ready():
            raise ValueError("NativeTool requires name, description, and handler")
        name = tool.name.strip()
        if name in self._tools:
            raise ValueError(f"Duplicate native tool name: {name}")
        self._tools[name] = tool

    def get(self, name: str) -> NativeTool | None:
        """Return one native tool by exact normalized name."""
        return self._tools.get(name.strip())

    def list_tools(self) -> tuple[NativeTool, ...]:
        """Return registered native tools in registration order."""
        return tuple(self._tools.values())

    def schemas(self, *, include_hidden: bool = False) -> tuple[Mapping[str, object], ...]:
        """Return provider-neutral schema references for registered tools."""
        return tuple(
            tool.schema()
            for tool in self._tools.values()
            if include_hidden or tool.exposed_by_default
        )
