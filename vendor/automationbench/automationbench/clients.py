# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""Custom API clients for the verifiers Client abstraction."""

from __future__ import annotations

import asyncio
import random
import time
from typing import Any, cast

import anthropic
import openai
from openai import AsyncOpenAI

from verifiers.clients.client import Client
from verifiers.clients import AnthropicMessagesClient, OpenAIChatCompletionsClient
from verifiers.errors import OverlongPromptError
from verifiers.types import (
    AssistantMessage,
    ClientConfig,
    FinishReason,
    Messages,
    Response,
    ResponseMessage,
    SamplingArgs,
    SystemMessage,
    TextMessage,
    Tool,
    ToolCall,
    ToolMessage,
    Usage,
    UserMessage,
)
from verifiers.utils.client_utils import setup_openai_client


# A single transient blip (rate limit, overloaded, gateway/server 5xx, dropped
# connection, timeout) must never kill a rollout. When an exception escapes the
# per-turn API call, the agentic loop ends mid-work and the partial transcript
# is recorded with a dangling assistant tool-call tail scored 0 — what we call
# an "abort". Retrying the full transient family with capped exponential backoff
# + jitter is the root-cause fix; only a genuinely non-transient error (e.g. a
# 400 bad request) or exhausting every attempt is allowed to propagate.
_RETRY_MAX_ATTEMPTS = 40

# 4xx client errors that will never succeed on retry — fail fast on these so a
# misconfiguration surfaces immediately instead of burning the full backoff.
# Everything NOT in this set is treated as a transient blip and retried (see
# RetryingOpenAIChatCompletionsClient), which is what makes the proxy/alpha path
# resilient to LiteLLM-wrapped exception types outside the openai.* hierarchy.
_NON_RETRYABLE_CHAT = (
    openai.BadRequestError,
    openai.AuthenticationError,
    openai.PermissionDeniedError,
    openai.NotFoundError,
    openai.UnprocessableEntityError,
    OverlongPromptError,
)


def _retry_delay(attempt: int, retry_after: float | None = None) -> float:
    """Capped exponential backoff with jitter, honoring a server Retry-After."""
    base = retry_after if retry_after is not None else min(60.0, 2.0**attempt)
    return base + random.uniform(0, 1)


def _parse_retry_after(err: Any) -> float | None:
    headers = getattr(getattr(err, "response", None), "headers", None)
    if not headers:
        return None
    ra = headers.get("retry-after")
    if not ra:
        return None
    try:
        return float(ra)
    except ValueError:
        return None


def _perf(state: Any) -> dict | None:
    """Return the per-task performance accumulator on `state`, creating it if the
    task carries a mutable state dict. Returns None when there is no state to record
    into (e.g. ad-hoc calls outside a rollout). Each rollout has its own state, so
    accumulation here is concurrency-safe."""
    if not isinstance(state, dict):
        return None
    p = state.get("_perf")
    if p is None:
        p = {
            "model_time_s": 0.0,
            "model_calls": 0,
            "cached_input_tokens": 0,
            "reasoning_tokens": 0,
            "tool_time_s": 0.0,
            "tool_calls": 0,
        }
        state["_perf"] = p
    return p


def _record_model_call(state: Any, elapsed_s: float, native_response: Any) -> None:
    """Accumulate model-call wall time + cached-input / reasoning token counts from a
    native provider response into state['_perf']. Tolerates missing usage fields and
    different provider shapes (OpenAI Responses, Chat, Anthropic)."""
    p = _perf(state)
    if p is None:
        return
    p["model_time_s"] += elapsed_s
    p["model_calls"] += 1
    u = getattr(native_response, "usage", None)
    if u is None:
        return
    # OpenAI Responses: usage.input_tokens_details.cached_tokens / output_tokens_details.reasoning_tokens
    itd = getattr(u, "input_tokens_details", None)
    if itd is not None and getattr(itd, "cached_tokens", None) is not None:
        p["cached_input_tokens"] += int(itd.cached_tokens)
    otd = getattr(u, "output_tokens_details", None)
    if otd is not None and getattr(otd, "reasoning_tokens", None) is not None:
        p["reasoning_tokens"] += int(otd.reasoning_tokens)
    # OpenAI Chat: usage.prompt_tokens_details.cached_tokens
    ptd = getattr(u, "prompt_tokens_details", None)
    if ptd is not None and getattr(ptd, "cached_tokens", None) is not None:
        p["cached_input_tokens"] += int(ptd.cached_tokens)
    # Anthropic: usage.cache_read_input_tokens. Unlike OpenAI, Anthropic's
    # usage.input_tokens EXCLUDES cache reads/writes, so track those separately
    # for export to add back into the input total ("extra_input_tokens").
    cr = getattr(u, "cache_read_input_tokens", None)
    if cr is not None:
        p["cached_input_tokens"] += int(cr)
        cw = getattr(u, "cache_creation_input_tokens", None) or 0
        p["extra_input_tokens"] = p.get("extra_input_tokens", 0) + int(cr) + int(cw)


class StreamingAnthropicClient(AnthropicMessagesClient):
    """AnthropicMessagesClient that uses streaming to avoid 10-minute timeout."""

    async def get_native_response(self, prompt, model, sampling_args, tools=None, **kwargs):
        from anthropic.types import Message as AnthropicMessage

        state = kwargs.pop("state", None)

        def normalize_sampling_args(sa):
            sa = dict(sa)
            max_tokens = sa.pop("max_tokens", None)
            sa.pop("n", None)
            sa.pop("stop", None)
            reasoning_effort = sa.pop("reasoning_effort", None)
            if max_tokens is None:
                max_tokens = 4096
            if reasoning_effort is not None:
                sa["thinking"] = {"type": "adaptive"}
                sa["output_config"] = {"effort": reasoning_effort}
                sa["temperature"] = 1.0
                sa.pop("top_p", None)
            sa["max_tokens"] = max_tokens
            return {k: v for k, v in sa.items() if v is not None}

        normalized = normalize_sampling_args(sampling_args)
        create_kwargs: dict[str, Any] = {
            "model": model,
            "messages": prompt,
            **normalized,
            **kwargs,
        }
        if tools:
            create_kwargs["tools"] = tools

        # Enable interleaved thinking so the model thinks between tool calls,
        # not just on turn 0 of the agentic loop.
        if "thinking" in normalized:
            existing_headers = create_kwargs.pop("extra_headers", {}) or {}
            create_kwargs["extra_headers"] = {
                **existing_headers,
                "anthropic-beta": "interleaved-thinking-2025-05-14",
            }

        # Retry the full transient-error family (rate limit, overloaded, any 5xx
        # gateway/server error, dropped connection, timeout) with capped
        # exponential backoff + jitter, honoring Retry-After. See _RETRY_MAX_ATTEMPTS.
        for attempt in range(_RETRY_MAX_ATTEMPTS):
            try:
                t0 = time.monotonic()
                async with self.client.messages.stream(**create_kwargs) as stream:
                    response: AnthropicMessage = await stream.get_final_message()
                _record_model_call(state, time.monotonic() - t0, response)
                return response
            except (anthropic.APIConnectionError, anthropic.APITimeoutError):
                if attempt == _RETRY_MAX_ATTEMPTS - 1:
                    raise
                await asyncio.sleep(_retry_delay(attempt))
            except anthropic.APIStatusError as e:
                status = getattr(e, "status_code", None)
                err_str = str(e)
                is_retryable = (
                    (status is not None and (status == 429 or status >= 500))
                    or "overloaded_error" in err_str
                    or "rate_limit_error" in err_str
                )
                if not is_retryable or attempt == _RETRY_MAX_ATTEMPTS - 1:
                    raise
                await asyncio.sleep(_retry_delay(attempt, _parse_retry_after(e)))
        raise RuntimeError("unreachable")


class OpenAIResponsesClient(Client[AsyncOpenAI, list[dict], Any, dict]):
    """Client for the OpenAI Responses API (supports reasoning_effort + tools)."""

    def setup_client(self, config: ClientConfig) -> AsyncOpenAI:
        return setup_openai_client(config)

    async def close(self) -> None:
        await self.client.close()

    async def to_native_tool(self, tool: Tool) -> dict:
        """Convert vf.Tool to Responses API flat tool format."""
        result: dict[str, Any] = {"type": "function", "name": tool.name}
        if tool.description:
            result["description"] = tool.description
        result["parameters"] = tool.parameters
        return result

    async def to_native_prompt(self, messages: Messages) -> tuple[list[dict], dict]:
        """Convert vf.Messages to Responses API input items + instructions kwarg."""
        instructions_parts: list[str] = []
        items: list[dict] = []

        for msg in messages:
            if isinstance(msg, SystemMessage):
                content = msg.content
                if isinstance(content, str):
                    instructions_parts.append(content)
                elif isinstance(content, list):
                    for part in content:
                        if hasattr(part, "text"):
                            instructions_parts.append(cast(str, part.text))
                        elif isinstance(part, dict) and part.get("type") == "text":
                            instructions_parts.append(part.get("text", ""))
            elif isinstance(msg, (UserMessage, TextMessage)):
                items.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AssistantMessage):
                if msg.content:
                    items.append({"role": "assistant", "content": msg.content})
                if msg.tool_calls:
                    for tc in msg.tool_calls:
                        items.append(
                            {
                                "type": "function_call",
                                "call_id": tc.id,
                                "name": tc.name,
                                "arguments": tc.arguments,
                            }
                        )
            elif isinstance(msg, ToolMessage):
                output = msg.content if isinstance(msg.content, str) else str(msg.content)
                items.append(
                    {
                        "type": "function_call_output",
                        "call_id": msg.tool_call_id,
                        "output": output,
                    }
                )

        extra_kwargs: dict[str, Any] = {}
        if instructions_parts:
            extra_kwargs["instructions"] = "\n\n".join(instructions_parts)

        return items, extra_kwargs

    async def get_native_response(
        self,
        prompt: list[dict],
        model: str,
        sampling_args: SamplingArgs,
        tools: list[dict] | None = None,
        **kwargs,
    ) -> Any:
        """Call the Responses API, retrying on rate limits."""
        # Keep state (not an API arg) so we can record per-task perf into it.
        state = kwargs.pop("state", None)

        call_kwargs: dict[str, Any] = {"model": model, "input": prompt}
        if tools:
            call_kwargs["tools"] = tools

        # Pass instructions from to_native_prompt
        if "instructions" in kwargs:
            call_kwargs["instructions"] = kwargs.pop("instructions")

        # Forward sampling args, skipping chat-completions-only keys
        _skip = {"extra_body", "max_completion_tokens", "max_tokens", "n", "stop"}
        for key, val in sampling_args.items():
            if key not in _skip and val is not None:
                call_kwargs[key] = val

        # Map reasoning_effort to Responses API reasoning param
        if sampling_args.get("reasoning_effort"):
            call_kwargs["reasoning"] = {"effort": sampling_args["reasoning_effort"]}
            call_kwargs.pop("reasoning_effort", None)

        for attempt in range(_RETRY_MAX_ATTEMPTS):
            try:
                t0 = time.monotonic()
                resp = await self.client.responses.create(**call_kwargs)
                _record_model_call(state, time.monotonic() - t0, resp)
                return resp
            except openai.BadRequestError as e:
                error_text = getattr(e, "message", str(e)).lower()
                if "context length" in error_text or "too long" in error_text:
                    raise OverlongPromptError from e
                raise
            except (
                openai.RateLimitError,
                openai.APIConnectionError,
                openai.APITimeoutError,
                openai.InternalServerError,
            ) as e:
                if attempt == _RETRY_MAX_ATTEMPTS - 1:
                    raise
                await asyncio.sleep(_retry_delay(attempt, _parse_retry_after(e)))
            except openai.APIStatusError as e:
                # Any other 5xx is transient; 4xx (except handled 400/429) is not.
                status = getattr(e, "status_code", None)
                if status is None or status < 500 or attempt == _RETRY_MAX_ATTEMPTS - 1:
                    raise
                await asyncio.sleep(_retry_delay(attempt, _parse_retry_after(e)))
        raise RuntimeError("unreachable")

    async def raise_from_native_response(self, response: Any) -> None:
        pass

    async def from_native_response(self, response: Any) -> Response:
        """Convert a Responses API response to vf.Response."""
        text_content = ""
        tool_calls: list[ToolCall] = []

        for item in getattr(response, "output", []):
            item_type = getattr(item, "type", None)
            if item_type == "message":
                for part in getattr(item, "content", []):
                    if getattr(part, "type", None) == "output_text":
                        text_content += getattr(part, "text", "")
            elif item_type == "function_call":
                tool_calls.append(
                    ToolCall(
                        id=getattr(item, "call_id", getattr(item, "id", "")),
                        name=getattr(item, "name", ""),
                        arguments=getattr(item, "arguments", ""),
                    )
                )

        finish_reason: FinishReason = "tool_calls" if tool_calls else "stop"

        raw_usage = getattr(response, "usage", None)
        usage: Usage | None = None
        if raw_usage is not None:
            input_toks = getattr(raw_usage, "input_tokens", 0)
            output_toks = getattr(raw_usage, "output_tokens", 0)
            usage = Usage(
                prompt_tokens=input_toks,
                completion_tokens=output_toks,
                reasoning_tokens=0,
                total_tokens=input_toks + output_toks,
            )

        return Response(
            id=getattr(response, "id", ""),
            created=getattr(response, "created_at", 0),
            model=getattr(response, "model", ""),
            usage=usage,
            message=ResponseMessage(
                content=text_content or None,
                tool_calls=tool_calls or None,
                finish_reason=finish_reason,
                is_truncated=False,
            ),
        )


class RetryingOpenAIChatCompletionsClient(OpenAIChatCompletionsClient):
    """Chat Completions client that retries the full transient-error family.

    The stock verifiers client lets connection/timeout/5xx errors propagate,
    which ends the rollout mid-work (a dangling tool-call tail scored 0). This is
    the path used for LiteLLM-proxied and alpha models, where transient proxy 5xx
    and dropped connections are the dominant abort cause — so it needs the same
    backoff guard as the Anthropic and Responses clients.
    """

    async def get_native_response(self, *args, **kwargs):
        state = kwargs.get("state")
        for attempt in range(_RETRY_MAX_ATTEMPTS):
            try:
                t0 = time.monotonic()
                resp = await super().get_native_response(*args, **kwargs)
                _record_model_call(state, time.monotonic() - t0, resp)
                return resp
            except _NON_RETRYABLE_CHAT:
                # Genuine client errors (bad request, auth, not-found, …) won't
                # change on retry — fail fast and loud.
                raise
            except Exception as e:  # noqa: BLE001
                # Everything else is treated as a transient blip and retried.
                # This deliberately broad net catches LiteLLM-/proxy-wrapped
                # exception types that don't subclass the openai.* classes
                # (the dominant abort cause on the proxy/alpha path), as well as
                # connection/timeout/5xx/rate-limit. After the last attempt it
                # propagates, and the --ensure-complete gate re-runs the task.
                if attempt == _RETRY_MAX_ATTEMPTS - 1:
                    raise
                await asyncio.sleep(_retry_delay(attempt, _parse_retry_after(e)))
        raise RuntimeError("unreachable")
