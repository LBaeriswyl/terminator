"""Shared test fixtures."""

import pytest

from terminator.config import AppConfig, ModelConfig, SafetyConfig, ContextConfig


@pytest.fixture
def default_config():
    return AppConfig()


@pytest.fixture
def model_config():
    return ModelConfig()


@pytest.fixture
def safety_config():
    return SafetyConfig()


@pytest.fixture
def context_config():
    return ContextConfig()
