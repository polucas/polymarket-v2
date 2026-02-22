"""Tests for structlog file output configuration."""
from __future__ import annotations

import logging
import pytest

# Import src.main to trigger module-level structlog and handler configuration.
import src.main  # noqa: F401


class TestStructlogConfig:
    def test_structlog_uses_stdlib_factory(self):
        """Verify structlog is configured with stdlib LoggerFactory, not PrintLoggerFactory."""
        import structlog
        config = structlog.get_config()
        factory = config.get("logger_factory")
        # Should be stdlib LoggerFactory, not PrintLoggerFactory
        assert factory is not None
        assert "stdlib" in type(factory).__module__ or "stdlib" in str(type(factory))

    def test_root_logger_has_file_handler(self):
        """Verify root logger has a FileHandler for bot.log."""
        root = logging.getLogger()
        file_handlers = [h for h in root.handlers if isinstance(h, logging.FileHandler)]
        assert len(file_handlers) >= 1, "No FileHandler found on root logger"
        assert any("bot.log" in str(h.baseFilename) for h in file_handlers)
