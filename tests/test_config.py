from __future__ import annotations

import pytest

import config as config_module
from config import load_config


@pytest.fixture
def base_env(monkeypatch):
    """Minimal required env for load_config() to succeed, with no .env file
    interference (load_dotenv() has already run at import time; we only
    control os.environ here per-test)."""
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "test-token")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.delenv("MEMORY_LIMIT", raising=False)
    monkeypatch.delenv("ALLOWED_CHAT_IDS", raising=False)
    return monkeypatch


class TestAllowedChatIds:
    def test_valid_allowed_chat_ids_parsed(self, base_env):
        base_env.setenv("ALLOWED_CHAT_IDS", "111, 222,333")
        cfg = load_config()
        assert cfg.allowed_chat_ids == frozenset({111, 222, 333})

    def test_empty_allowed_chat_ids_means_unrestricted(self, base_env):
        base_env.setenv("ALLOWED_CHAT_IDS", "")
        cfg = load_config()
        assert cfg.allowed_chat_ids is None

    def test_absent_allowed_chat_ids_means_unrestricted(self, base_env):
        cfg = load_config()
        assert cfg.allowed_chat_ids is None

    def test_whitespace_only_allowed_chat_ids_means_unrestricted(self, base_env):
        base_env.setenv("ALLOWED_CHAT_IDS", "   ")
        cfg = load_config()
        assert cfg.allowed_chat_ids is None

    def test_invalid_chat_id_fails_at_startup(self, base_env):
        base_env.setenv("ALLOWED_CHAT_IDS", "111, not-a-number, 333")
        with pytest.raises(EnvironmentError):
            load_config()

    def test_trailing_comma_is_tolerated(self, base_env):
        base_env.setenv("ALLOWED_CHAT_IDS", "111,222,")
        cfg = load_config()
        assert cfg.allowed_chat_ids == frozenset({111, 222})


class TestMemoryLimitValidation:
    def test_default_memory_limit_is_20(self, base_env):
        cfg = load_config()
        assert cfg.memory_limit == 20

    def test_valid_even_memory_limit_accepted(self, base_env):
        base_env.setenv("MEMORY_LIMIT", "10")
        cfg = load_config()
        assert cfg.memory_limit == 10

    def test_zero_memory_limit_rejected(self, base_env):
        base_env.setenv("MEMORY_LIMIT", "0")
        with pytest.raises(EnvironmentError):
            load_config()

    def test_negative_memory_limit_rejected(self, base_env):
        base_env.setenv("MEMORY_LIMIT", "-4")
        with pytest.raises(EnvironmentError):
            load_config()

    def test_odd_memory_limit_rejected(self, base_env):
        base_env.setenv("MEMORY_LIMIT", "7")
        with pytest.raises(EnvironmentError):
            load_config()

    def test_memory_limit_of_one_rejected(self, base_env):
        # Odd AND below the minimum of 2 -- must fail either way.
        base_env.setenv("MEMORY_LIMIT", "1")
        with pytest.raises(EnvironmentError):
            load_config()

    def test_memory_limit_default_not_silently_doubled(self):
        # Guards against "fixing" the old misleading .env.example wording
        # by doubling the default -- the effective meaning (max individual
        # messages retained) must stay the same as before this stage.
        assert config_module._validate_memory_limit(20) == 20


class TestRequiredFields:
    def test_missing_telegram_token_raises(self, monkeypatch):
        monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        with pytest.raises(EnvironmentError):
            load_config()
