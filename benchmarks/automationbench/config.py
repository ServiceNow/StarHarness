"""Run configuration for the AutomationBench-AA replication.

The agent under test defaults to gpt-5.4 via the OpenAI API (OPENAI_API_KEY in
the repository's `.env`), matching the AutomationBench-AA setup: API toolset, one run per
task, 50-turn cap, no LLM judge.

Override any value via the environment (AGENT_LLM, AGENT_BASE_URL, ...).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _load_dotenv() -> None:
    """Load the repository's .env into os.environ (without overriding existing vars).

    Tiny KEY=VALUE parser so we don't add a python-dotenv dependency.
    """
    # config.py -> automationbench -> benchmarks -> repository root
    env_path = Path(__file__).resolve().parents[2] / ".env"
    if not env_path.is_file():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key, val = key.strip(), val.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = val


_load_dotenv()

DEFAULT_BASE_URL = "https://api.openai.com/v1"
DEFAULT_MODEL = "gpt-5.4"


def _parse_base_urls() -> list[str]:
    """Resolve the agent endpoint(s).

    AGENT_BASE_URLS (comma-separated) load-balances units across several
    OpenAI-compatible replicas; each unit is pinned to one endpoint (see
    RunConfig.endpoint_for) so prefix-cache reuse across its turns is preserved.
    Falls back to the single AGENT_BASE_URL (then DEFAULT_BASE_URL).
    """
    raw = os.environ.get("AGENT_BASE_URLS", "").strip()
    if raw:
        urls = [u.strip() for u in raw.split(",") if u.strip()]
        if urls:
            return urls
    return [os.environ.get("AGENT_BASE_URL", DEFAULT_BASE_URL)]


# AutomationBench-AA methodology constants.
AA_MAX_TURNS = 50  # 50-turn cap per task
AA_REPEATS = 1     # 1 run per task

_GPT5_PREFIXES = ("gpt-5",)


def _is_gpt5(model: str) -> bool:
    return model.split("/", 1)[-1].lower().startswith(_GPT5_PREFIXES)


@dataclass
class RunConfig:
    """All knobs for a run. Defaults reproduce the AutomationBench-AA setup."""

    # --- model under test ---
    model: str = field(default_factory=lambda: os.environ.get("AGENT_LLM", DEFAULT_MODEL))
    base_urls: list[str] = field(default_factory=_parse_base_urls)
    api_key: str = field(default_factory=lambda: os.environ.get("AGENT_API_KEY", "") or os.environ.get("OPENAI_API_KEY", "") or "EMPTY")

    @property
    def base_url(self) -> str:
        """Primary endpoint (first of base_urls) — used for logging + single-endpoint paths."""
        return self.base_urls[0]

    def endpoint_for(self, unit_index: int) -> str:
        """Endpoint for a given unit, round-robin over base_urls (affinity per conversation)."""
        return self.base_urls[unit_index % len(self.base_urls)]

    # --- eval loop ---
    harness: str = field(default_factory=lambda: os.environ.get("AGENT_HARNESS", "stirrup"))
    max_turns: int = field(default_factory=lambda: int(os.environ.get("MAX_TURNS", AA_MAX_TURNS)))
    repeats: int = field(default_factory=lambda: int(os.environ.get("REPEATS", AA_REPEATS)))
    max_concurrency: int = field(default_factory=lambda: int(os.environ.get("MAX_CONCURRENCY", "8")))
    domain: str = field(default_factory=lambda: os.environ.get("AB_DOMAIN", "finance"))
    # Per-unit wall-clock timeout (seconds); 0 = disabled. A unit exceeding this is
    # cancelled and scored 0.0 with a note (AA: infrastructure errors score 0).
    unit_timeout: float = field(default_factory=lambda: float(os.environ.get("UNIT_TIMEOUT", "0")))

    @property
    def litellm_model(self) -> str:
        """LiteLLM identifier (openai/ prefix works for both OpenAI and vLLM endpoints)."""
        m = self.model
        return m if m.startswith(("openai/", "hosted_vllm/")) else f"openai/{m}"

    def register_zero_cost_model(self) -> None:
        """Register a self-hosted model with $0 cost in LiteLLM (silences price-map noise).

        Safe no-op for OpenAI models (already in the price map) or if litellm is absent.
        """
        name = self.model.split("/", 1)[-1] if self.model.startswith(("openai/", "hosted_vllm/")) else self.model
        try:
            import litellm

            litellm.register_model({
                name: {"input_cost_per_token": 0.0, "output_cost_per_token": 0.0,
                       "litellm_provider": "openai", "mode": "chat"}
            })
        except Exception:
            pass

    @property
    def reasoning_effort(self) -> str | None:
        """Reasoning effort for GPT-5 family models (passed to LiteLLMClient, not litellm_kwargs)."""
        if _is_gpt5(self.model):
            return os.environ.get("AGENT_REASONING_EFFORT", "medium") or "medium"
        return None

    def litellm_kwargs(self, base_url: str | None = None) -> dict:
        """Extra acompletion kwargs: api_base (+ temperature for non-reasoning models).

        GPT-5 models do not accept temperature; reasoning_effort is passed separately
        via LiteLLMClient to avoid a duplicate-kwarg error.
        """
        kw: dict = {"api_base": base_url or self.base_url}
        if not _is_gpt5(self.model):
            kw["temperature"] = 0.0
        return kw
