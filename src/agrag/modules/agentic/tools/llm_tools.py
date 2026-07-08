"""LLM-backed tools that reuse the existing GeneratorModule.

Both tools call the single configured generator (per the design's "reuse one
generator" decision). They are gated behind config flags and are off or optional
by default. Neither produces evidence; they transform queries/context.
"""

import logging
from typing import List

from agrag.constants import LOGGER_NAME
from agrag.modules.agentic.tools.base import Tool, ToolResult

logger = logging.getLogger(LOGGER_NAME)

_REWRITE_INSTRUCTION = (
    "Rewrite the following search query to improve document retrieval. "
    "Return only the rewritten query with no preamble.\n\nQuery: "
)

_COMPRESS_INSTRUCTION = (
    "Compress the following context into a concise, self-contained summary that "
    "preserves all facts needed to answer the query. Return only the summary.\n\n"
)


class QueryRewriteTool(Tool):
    """Improve a query after weak retrieval, using the generator."""

    name = "QueryRewriteTool"

    def __init__(self, generator_module):
        self.generator_module = generator_module

    def run(self, query: str, **kwargs) -> ToolResult:
        prompt = f"{_REWRITE_INSTRUCTION}{query}"
        rewritten = self.generator_module.generate_response(prompt).strip()
        logger.debug("%s rewrote %r -> %r", self.name, query, rewritten)
        return self._result(output=rewritten, summary="rewrote query")


class ContextCompressionTool(Tool):
    """Compress many chunks into smaller context using the generator."""

    name = "ContextCompressionTool"

    def __init__(self, generator_module):
        self.generator_module = generator_module

    def run(self, query: str, texts: List[str], **kwargs) -> ToolResult:
        joined = "\n\n".join(texts)
        prompt = f"{_COMPRESS_INSTRUCTION}Query: {query}\n\nContext:\n{joined}"
        compressed = self.generator_module.generate_response(prompt).strip()
        logger.debug("%s compressed %d chunks", self.name, len(texts))
        return self._result(output=compressed, summary=f"compressed {len(texts)} chunks")
