"""Tests for src/manage.py CLI tool."""
from __future__ import annotations

import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.manage import main


@pytest.fixture
def mock_deps():
    """Mock _init_deps to return fake db and learning managers."""
    db = AsyncMock()
    cal = MagicMock()
    mt = MagicMock()
    st = MagicMock()
    with patch("src.manage._init_deps", new_callable=AsyncMock, return_value=(db, cal, mt, st)) as init:
        yield init, db, cal, mt, st


class TestMainModelSwap:
    def test_model_swap_parsed_correctly(self, mock_deps, monkeypatch):
        """main() parses model_swap command with --old-model, --new-model, --reason."""
        init, db, cal, mt, st = mock_deps
        handle = AsyncMock()
        monkeypatch.setattr(sys, "argv", [
            "manage", "model_swap",
            "--old-model", "grok-2",
            "--new-model", "grok-3",
            "--reason", "upgrade",
        ])
        with patch("src.manage.handle_model_swap", handle, create=True):
            with patch("src.learning.model_swap.handle_model_swap", handle, create=True):
                main()
        handle.assert_awaited_once_with("grok-2", "grok-3", "upgrade", cal, mt, db)
        db.close.assert_awaited_once()


class TestMainVoidTrade:
    def test_void_trade_parsed_correctly(self, mock_deps, monkeypatch):
        """main() parses void_trade command with --trade-id, --reason."""
        init, db, cal, mt, st = mock_deps
        void_fn = AsyncMock()
        monkeypatch.setattr(sys, "argv", [
            "manage", "void_trade",
            "--trade-id", "trade-xyz",
            "--reason", "bad data",
        ])
        with patch("src.learning.model_swap.void_trade", void_fn, create=True):
            main()
        void_fn.assert_awaited_once_with("trade-xyz", "bad data", db, cal, mt, st)
        db.close.assert_awaited_once()


class TestMainStartExperiment:
    def test_start_experiment_parsed_correctly(self, mock_deps, monkeypatch):
        """main() parses start_experiment command with --description, --model."""
        init, db, cal, mt, st = mock_deps
        start_fn = AsyncMock()
        monkeypatch.setattr(sys, "argv", [
            "manage", "start_experiment",
            "--description", "test new model",
            "--model", "grok-3-fast",
        ])
        with patch("src.learning.experiments.start_experiment", start_fn, create=True):
            main()
        start_fn.assert_awaited_once()
        call_args = start_fn.await_args
        # run_id is generated dynamically: exp_{model}_{timestamp}
        assert call_args[0][0].startswith("exp_grok-3-fast_")
        assert call_args[0][1] == "test new model"
        assert call_args[0][2] == {}
        assert call_args[0][3] == "grok-3-fast"
        assert call_args[0][4] is db
        db.close.assert_awaited_once()


class TestMainEndExperiment:
    def test_end_experiment_parsed_correctly(self, mock_deps, monkeypatch):
        """main() parses end_experiment command with --run-id."""
        init, db, cal, mt, st = mock_deps
        end_fn = AsyncMock()
        monkeypatch.setattr(sys, "argv", [
            "manage", "end_experiment",
            "--run-id", "exp_grok-3-fast_20240101_120000",
        ])
        with patch("src.learning.experiments.end_experiment", end_fn, create=True):
            main()
        end_fn.assert_awaited_once_with("exp_grok-3-fast_20240101_120000", {}, db)
        db.close.assert_awaited_once()


class TestMainRecalculateLearning:
    def test_recalculate_learning_parsed_correctly(self, mock_deps, monkeypatch):
        """main() parses recalculate_learning command (no extra args)."""
        init, db, cal, mt, st = mock_deps
        recalc_fn = AsyncMock()
        monkeypatch.setattr(sys, "argv", [
            "manage", "recalculate_learning",
        ])
        with patch("src.learning.model_swap.recalculate_learning_from_scratch", recalc_fn, create=True):
            main()
        recalc_fn.assert_awaited_once_with(db, cal, mt, st)
        db.close.assert_awaited_once()
