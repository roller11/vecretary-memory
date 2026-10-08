# SPDX-License-Identifier: MPL-2.0
"""Metadata-only formatting for the standalone MCP process's stderr log."""

import logging


class MCPPrivacyFormatter(logging.Formatter):
    def format(self, record):
        message = record.getMessage() if record.name == 'instinct-mcp' else 'Runtime event (details withheld)'
        component = 'instinct-mcp' if record.name == 'instinct-mcp' else 'runtime'
        safe_record = logging.LogRecord(component, record.levelno, '', 0, message, (), None)
        safe_record.created = record.created
        safe_record.msecs = record.msecs
        return super().format(safe_record)
