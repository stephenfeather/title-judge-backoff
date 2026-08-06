"""Guards the checked-in backends.toml itself — a malformed slate should fail
the suite, not a live run.
"""

import re
from pathlib import Path

import pytest

from judge.client import VALID_APIS, VALID_ROLES, load_backends

BACKENDS = Path(__file__).resolve().parent.parent / "backends.toml"


@pytest.fixture(scope="module")
def slate():
    return load_backends(BACKENDS)


def test_slate_loads_and_is_not_empty(slate):
    assert len(slate) >= 1


def test_every_backend_has_valid_api_role_and_key_env(slate):
    for backend in slate:
        assert backend.api in VALID_APIS
        assert backend.role in VALID_ROLES
        assert backend.api_key_env.endswith("_API_KEY")
        assert backend.rpm > 0


def test_backend_names_and_model_ids_are_unique(slate):
    names = [b.name for b in slate]
    assert len(names) == len(set(names))
    model_ids = [b.model_id for b in slate]
    assert len(model_ids) == len(set(model_ids))


def test_every_nvidia_backend_is_marked_eval_only(slate):
    """NVIDIA's free tier is testing-only per ToS — never production."""
    for backend in slate:
        if "integrate.api.nvidia.com" in backend.base_url:
            assert backend.eval_only, f"{backend.name} must be eval_only"


def test_slate_has_exactly_one_floor_and_several_contenders(slate):
    assert sum(b.role == "floor" for b in slate) == 1
    assert sum(b.role == "contender" for b in slate) >= 4


def test_no_api_keys_are_committed_in_the_slate():
    """Keys are env-only. Match key-SHAPED tokens (prefix + a long secret
    body), not bare prefixes — a backend named `xai-grok-4.1-fast` legitimately
    contains a provider prefix.
    """
    text = BACKENDS.read_text()
    key_shaped = re.compile(r"\b(nvapi-|sk-ant-|sk-proj-|sk-|xai-|AIza)[A-Za-z0-9_\-]{20,}")
    match = key_shaped.search(text)
    assert match is None, f"possible key material in backends.toml: {match.group(0)[:12]}..."
