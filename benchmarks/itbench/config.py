"""Run configuration for the ITBench-AA replication."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _load_dotenv() -> None:
    """Load the repository .env without overriding existing variables.

    This small KEY=VALUE parser avoids a python-dotenv dependency.
    """
    # config.py -> itbench -> benchmarks -> repository root
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

    AGENT_BASE_URLS (comma-separated) lets the agent fan out across several
    OpenAI-compatible replicas of the same model for throughput — each unit (a
    whole multi-turn conversation) is pinned to one endpoint (see
    RunConfig.endpoint_for), so vLLM prefix-cache reuse across a unit's turns is
    preserved. Falls back to the single AGENT_BASE_URL (then DEFAULT_BASE_URL),
    so an unset value uses one endpoint.
    """
    raw = os.environ.get("AGENT_BASE_URLS", "").strip()
    if raw:
        urls = [u.strip() for u in raw.split(",") if u.strip()]
        if urls:
            return urls
    return [os.environ.get("AGENT_BASE_URL", DEFAULT_BASE_URL)]

# ITBench-AA methodology constants.
AA_MAX_TURNS = 100  # 100-turn cap per task
AA_REPEATS = 3      # 3 repeats per task; primary score averages them

# Python packages AA's agent is told are available in its sandbox (see prompts.py).
# Baked into the boxlite microVM at box start via `pip install` (the stock image
# disk is sized by `boxlite_disk_gb` so numpy/pandas fit). Pinned exactly to AA.
AA_SANDBOX_PACKAGES = ("drain3==0.9.11", "numpy==2.4.5", "pandas==3.0.3")

_GEMMA_PREFIXES = ("gemma",)
_GPT5_PREFIXES = ("gpt-5",)


def _is_gemma(model: str) -> bool:
    return model.split("/", 1)[-1].lower().startswith(_GEMMA_PREFIXES)


def _is_gpt5(model: str) -> bool:
    return model.split("/", 1)[-1].lower().startswith(_GPT5_PREFIXES)


@dataclass
class RunConfig:
    """Runtime settings for an ITBench SRE evaluation."""

    # --- model under test ---
    model: str = field(default_factory=lambda: os.environ.get("AGENT_LLM", DEFAULT_MODEL))
    # One or more OpenAI-compatible endpoints serving `model`. Multiple = load
    # balanced per unit (AGENT_BASE_URLS). See `_parse_base_urls` / `endpoint_for`.
    base_urls: list[str] = field(default_factory=_parse_base_urls)
    api_key: str = field(default_factory=lambda: os.environ.get("AGENT_API_KEY", "") or os.environ.get("OPENAI_API_KEY", "") or "EMPTY")

    @property
    def base_url(self) -> str:
        """Primary endpoint (first of base_urls) — used for logging + single-endpoint paths."""
        return self.base_urls[0]

    def endpoint_for(self, unit_index: int) -> str:
        """Endpoint for a given unit, round-robin over base_urls (affinity per conversation).

        A whole (task, repeat) unit sticks to one endpoint for all its turns so the
        endpoint's prefix cache is reused as the conversation grows; units are
        spread evenly across endpoints by their global index.
        """
        return self.base_urls[unit_index % len(self.base_urls)]

    # --- eval loop ---
    max_turns: int = field(default_factory=lambda: int(os.environ.get("MAX_TURNS", AA_MAX_TURNS)))
    repeats: int = field(default_factory=lambda: int(os.environ.get("REPEATS", AA_REPEATS)))
    # Max (task, repeat) units run concurrently. Conservative default — all units
    # share one inference endpoint, so this is the main load multiplier.
    max_concurrency: int = field(default_factory=lambda: int(os.environ.get("MAX_CONCURRENCY", "4")))
    shell_timeout: int = field(default_factory=lambda: int(os.environ.get("SHELL_TIMEOUT", "120")))
    split: str = field(default_factory=lambda: os.environ.get("SPLIT", "public"))  # public | private | all
    # This adapter evaluates the vendored Stirrup harness.
    harness: str = field(default_factory=lambda: os.environ.get("AGENT_HARNESS", "stirrup"))

    # --- sandbox backend ---
    # "local": Stirrup's LocalCodeExecToolProvider (a host temp dir; the default).
    # "boxlite": a self-hosted BoxLite microVM per unit (KVM isolation), reproducing
    #            AA's E2B `/home/user` contract. Requires a reachable `boxlite serve`
    #            endpoint (BOXLITE_REST_URL) on a host exposing /dev/kvm.
    backend: str = field(default_factory=lambda: os.environ.get("SANDBOX_BACKEND", "local").lower())
    boxlite_url: str = field(default_factory=lambda: os.environ.get("BOXLITE_REST_URL", ""))
    boxlite_image: str = field(default_factory=lambda: os.environ.get("BOXLITE_IMAGE", "python:3.12-slim"))
    # COW overlay default (~256MB) can't hold numpy+pandas; 8GB leaves room for the
    # staged snapshot (the prompt notes snapshots can be hundreds of MB) + outputs.
    boxlite_disk_gb: int = field(default_factory=lambda: int(os.environ.get("BOXLITE_DISK_GB", "8")))
    boxlite_cpus: int = field(default_factory=lambda: int(os.environ.get("BOXLITE_CPUS", "2")))
    boxlite_memory_mib: int = field(default_factory=lambda: int(os.environ.get("BOXLITE_MEMORY_MIB", "2048")))

    def boxlite_setup_commands(self) -> list[str]:
        """Commands run once at box start to provision AA's sandbox packages.

        Override the whole list via BOXLITE_SETUP (commands separated by `;;`);
        set BOXLITE_SETUP to an empty string to run NO setup (the packages are
        already baked into BOXLITE_IMAGE — see the baked-image optimization);
        if BOXLITE_SETUP is unset, pip-install the AA-pinned packages
        (see AA_SANDBOX_PACKAGES).
        """
        override = os.environ.get("BOXLITE_SETUP")
        if override is not None:  # explicitly set (even to "") — honor it verbatim
            return [c.strip() for c in override.split(";;") if c.strip()]
        return [f"pip install --quiet {' '.join(AA_SANDBOX_PACKAGES)}"]
    # Per-unit wall-clock timeout (seconds); 0 = disabled. A unit exceeding this is
    # cancelled and scored 0.0 with a note — stops one non-converging task (which can
    # run to the 100-turn cap, ~4h) from stretching a dev-subset loop unboundedly.
    unit_timeout: float = field(default_factory=lambda: float(os.environ.get("UNIT_TIMEOUT", "0")))

    # --- judge (LLM entity-normalization, AA-faithful) ---
    # A model judge normalizes predicted entities to ground-truth IDs before scoring.
    # Leave JUDGE_MODEL empty to use the deterministic filter-regex matcher.
    judge_model: str = field(default_factory=lambda: os.environ.get("JUDGE_MODEL", ""))
    judge_base_url: str = field(default_factory=lambda: os.environ.get("JUDGE_BASE_URL", ""))
    judge_api_key: str = field(
        default_factory=lambda: os.environ.get("JUDGE_API_KEY", "") or os.environ.get("OPENAI_API_KEY", "")
    )
    judge_reasoning_effort: str = field(
        default_factory=lambda: os.environ.get("JUDGE_REASONING_EFFORT", "medium")
    )

    def build_judge(self):
        """Construct the LLM judge, or None when no judge model is configured."""
        if not self.judge_model:
            return None
        from .judge import LLMJudge

        return LLMJudge(
            model=self.judge_model,
            api_key=self.judge_api_key,
            base_url=self.judge_base_url,
            reasoning_effort=self.judge_reasoning_effort,
        )

    # --- data ---
    data_root: str = field(default_factory=lambda: os.environ.get("ITBENCH_DATA", ""))

    @property
    def litellm_model(self) -> str:
        """LiteLLM identifier for an OpenAI-compatible (vLLM) endpoint."""
        m = self.model
        return m if m.startswith(("openai/", "hosted_vllm/")) else f"openai/{m}"

    def register_zero_cost_model(self) -> None:
        """Register the self-hosted model with $0 cost in LiteLLM.

        Self-hosted models may not be in LiteLLM's price map. Registering the model
        at zero cost prevents a cost-lookup warning on each request. This is a safe
        no-op if LiteLLM is absent.
        """
        name = self.model.split("/", 1)[-1] if self.model.startswith(("openai/", "hosted_vllm/")) else self.model
        try:
            import litellm

            litellm.register_model({
                name: {"input_cost_per_token": 0.0, "output_cost_per_token": 0.0,
                       "litellm_provider": "openai", "mode": "chat"}
            })
        except Exception:
            pass  # cost lookup already degrades to $0; nothing else depends on this

    @property
    def reasoning_effort(self) -> str | None:
        """Reasoning effort for GPT-5 family models (passed to LiteLLMClient, not litellm_kwargs)."""
        if _is_gpt5(self.model):
            return os.environ.get("AGENT_REASONING_EFFORT", "medium") or "medium"
        return None

    def litellm_kwargs(self, base_url: str | None = None) -> dict:
        """Extra acompletion kwargs: api_base + reasoning temperature/thinking.

        `base_url` overrides the primary endpoint (used to pin a unit to one of
        several load-balanced replicas); defaults to `self.base_url`.
        GPT-5 models do not accept temperature; reasoning_effort is passed separately
        via LiteLLMClient to avoid a duplicate-kwarg error.
        """
        kw: dict = {"api_base": base_url or self.base_url}
        if _is_gemma(self.model):
            kw["temperature"] = 1.0
            kw["extra_body"] = {"chat_template_kwargs": {"enable_thinking": True}}
        elif not _is_gpt5(self.model):
            kw["temperature"] = 0.0
        return kw
