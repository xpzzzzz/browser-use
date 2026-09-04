"""Bounded retry and reliability telemetry for post-processing model calls.

Phase 9B showed that a live run can fail after the browser work is already good: the agent produced an
answer, the evidence was captured, and the analysis still stopped because one model call in claim
extraction, semantic reranking or claim verification hit a transient provider error. Re-running the same
answer against the same evidence often succeeded, which points at the transport rather than the algorithm.

This module addresses that with a bounded number of extra attempts and a deterministic backoff, and it
records what happened as counters. It is deliberately not a fallback. A retry that eventually succeeds
returns the real model's answer; a retry budget that runs out re-raises the original exception so the
calling stage still fails exactly as it did before. Nothing here can turn a failed model call into an
empty claim set, a lexical score, or a default verdict, which is what keeps the strict failure semantics
of Phases 3 to 8 intact.

The wrapper satisfies ``BaseChatModel``, so ``ClaimExtractor``, ``SemanticEvidenceReranker`` and
``ClaimVerifier`` accept it unchanged and stay unaware that retry exists. Retry is a transport concern,
so it lives here rather than being copied into each stage.
"""

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any, TypeVar

from pydantic import BaseModel, ConfigDict, Field, model_validator

from browser_use.llm.base import BaseChatModel
from browser_use.llm.messages import BaseMessage
from browser_use.llm.views import ChatInvokeCompletion

T = TypeVar('T', bound=BaseModel)

# The monotonic counters that stats_delta subtracts. exception_type_counts is handled separately
# because it is a mapping rather than a single number.
_DELTA_FIELDS: tuple[str, ...] = (
	'logical_invocation_count',
	'attempt_count',
	'retry_count',
	'recovered_invocation_count',
	'failed_invocation_count',
)


class LLMRetryPolicy(BaseModel):
	"""How many attempts a post-processing model call gets, and how long to wait between them.

	``max_attempts`` counts total attempts, not extra retries: ``3`` means the initial call plus at most
	two retries. The delays are a plain geometric sequence with no jitter, so a run that waited 1.0s then
	2.0s is reproducible and explainable rather than merely plausible.
	"""

	model_config = ConfigDict(frozen=True)

	max_attempts: int = Field(default=3, ge=1, description='Total attempts per logical call, retries included')
	initial_delay_seconds: float = Field(default=1.0, ge=0.0, description='Delay before the first retry')
	backoff_multiplier: float = Field(default=2.0, ge=1.0, description='Factor applied to the delay per retry')
	max_delay_seconds: float = Field(default=8.0, ge=0.0, description='Ceiling on any single delay')

	@model_validator(mode='after')
	def _check_delay_bounds(self) -> 'LLMRetryPolicy':
		"""A ceiling below the first delay would silently defeat the backoff it claims to bound."""
		if self.max_delay_seconds < self.initial_delay_seconds:
			raise ValueError(
				f'max_delay_seconds ({self.max_delay_seconds}) must be at least initial_delay_seconds '
				f'({self.initial_delay_seconds})'
			)
		return self

	def delay_before_retry(self, retry_number: int) -> float:
		"""Delay in seconds before retry ``retry_number``, where the first retry is 1.

		That is ``initial_delay_seconds * backoff_multiplier ** (retry_number - 1)``, clamped to
		``max_delay_seconds``. Retry numbers below 1 are a caller error.
		"""
		if retry_number < 1:
			raise ValueError(f'retry_number is 1-based, got {retry_number}')
		delay = self.initial_delay_seconds * (self.backoff_multiplier ** (retry_number - 1))
		return min(delay, self.max_delay_seconds)

	def retry_delays(self) -> tuple[float, ...]:
		"""Every delay this policy can produce, in order, across its full retry budget."""
		return tuple(self.delay_before_retry(number) for number in range(1, self.max_attempts))


class LLMRetryStats(BaseModel):
	"""Counters describing how many model calls a wrapper actually made.

	Only counts and exception class names belong here. Messages, prompts, answers, evidence text,
	completions, credentials and provider exception messages must never be recorded: a provider error
	typically echoes the request it was given, and these counters end up in benchmark files that get
	shared.

	A "logical invocation" is one request from a stage; an "attempt" is one call that reached the
	provider. An invocation that failed once and then succeeded counts as logical 1, attempts 2,
	retries 1, recovered 1, failed 0.
	"""

	model_config = ConfigDict(frozen=True)

	logical_invocation_count: int = Field(default=0, ge=0)
	attempt_count: int = Field(default=0, ge=0, description='Calls that actually reached the underlying model')
	retry_count: int = Field(default=0, ge=0, description='Attempts made after the first, across all invocations')
	recovered_invocation_count: int = Field(
		default=0, ge=0, description='Invocations that succeeded only after at least one failed attempt'
	)
	failed_invocation_count: int = Field(default=0, ge=0, description='Invocations that exhausted their attempts')
	exception_type_counts: dict[str, int] = Field(
		default_factory=dict, description='Exception class name to count, never an exception message'
	)


def stats_delta(before: LLMRetryStats, after: LLMRetryStats) -> LLMRetryStats:
	"""Counters for the work done between two snapshots of one wrapper.

	A wrapper is usually shared by every run of a benchmark, so a per-run record has to subtract the
	snapshot taken before the run from the one taken after it. Without that, each run would report the
	cumulative total since the benchmark started.
	"""

	# Every counter only ever increases, so a negative difference means the two snapshots did not come
	# from one wrapper in order. That is a caller bug, and naming the field beats a pydantic error about
	# a negative value reaching the model.
	counters: dict[str, int] = {}
	for field in _DELTA_FIELDS:
		before_value = getattr(before, field)
		after_value = getattr(after, field)
		if after_value < before_value:
			raise ValueError(f'{field} went backwards between snapshots: {before_value} then {after_value}')
		counters[field] = after_value - before_value

	exception_type_counts: dict[str, int] = {}
	for name, count in after.exception_type_counts.items():
		increment = count - before.exception_type_counts.get(name, 0)
		if increment > 0:
			exception_type_counts[name] = increment
		elif increment < 0:
			raise ValueError(f'exception type {name!r} count went backwards between snapshots')

	return LLMRetryStats(**counters, exception_type_counts=exception_type_counts)


class RetryingChatModel(BaseChatModel):
	"""BaseChatModel wrapper that retries a bounded number of times and counts what it did.

	Args:
		llm: the model every attempt goes through, usually the same ``ChatOpenAI`` the agent used.
		policy: attempt budget and backoff. Defaults to ``LLMRetryPolicy()``.
		sleep: injected only so tests can assert the backoff sequence without waiting for it; the
			production default is :func:`asyncio.sleep`.

	Only ``Exception`` is caught, so ``KeyboardInterrupt``, ``SystemExit``, ``GeneratorExit`` and task
	cancellation propagate immediately rather than being retried. When the budget runs out the original
	exception is re-raised untouched: no ``RetryError`` is invented, because claim extraction, reranking
	and verification already raise their own error types and the pipeline already wraps those.

	Attempts are not classified by reading exception text. Provider hierarchies drift, and a substring
	test such as ``'timeout' in str(e)`` both leaks message content into the decision and changes
	meaning when a provider rewords an error, so any ``Exception`` is retried within the small budget.
	"""

	def __init__(
		self,
		llm: BaseChatModel,
		*,
		policy: LLMRetryPolicy | None = None,
		sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
	) -> None:
		"""Args:
		llm: the underlying model that each attempt actually calls.
		policy: retry budget and deterministic backoff; None means the default policy.
		sleep: awaitable delay function, a seam for tests that check delays without spending them.
		"""
		self._llm = llm
		self._policy = policy or LLMRetryPolicy()
		self._sleep = sleep
		self._logical_invocations = 0
		self._attempts = 0
		self._retries = 0
		self._recovered_invocations = 0
		self._failed_invocations = 0
		self._exception_type_counts: dict[str, int] = {}

	@property
	def policy(self) -> LLMRetryPolicy:
		"""The policy this wrapper retries under, which callers may read but not edit."""
		return self._policy

	@property
	def wrapped_model(self) -> BaseChatModel:
		"""The model underneath, exposed for callers that need to reach past the wrapper."""
		return self._llm

	@property
	def model(self) -> str:
		return self._llm.model

	@property
	def provider(self) -> str:
		return self._llm.provider

	@property
	def name(self) -> str:
		return self._llm.name

	@property
	def model_name(self) -> str:
		return self._llm.model_name

	def snapshot_stats(self) -> LLMRetryStats:
		"""A copy of the counters so far. Callers get a snapshot, never the live state."""
		return LLMRetryStats(
			logical_invocation_count=self._logical_invocations,
			attempt_count=self._attempts,
			retry_count=self._retries,
			recovered_invocation_count=self._recovered_invocations,
			failed_invocation_count=self._failed_invocations,
			exception_type_counts=dict(self._exception_type_counts),
		)

	def reset_stats(self) -> None:
		"""Zero the counters for callers that own this wrapper and want a fresh window.

		The benchmark does not use this: it shares one wrapper across runs, so it takes snapshot
		differences with :func:`stats_delta` instead of resetting mid-flight.
		"""
		self._logical_invocations = 0
		self._attempts = 0
		self._retries = 0
		self._recovered_invocations = 0
		self._failed_invocations = 0
		self._exception_type_counts = {}

	async def ainvoke(
		self,
		messages: list[BaseMessage],
		output_format: type[T] | None = None,
		**kwargs: Any,
	) -> ChatInvokeCompletion[T] | ChatInvokeCompletion[str]:
		"""Call the underlying model, retrying failures within the policy.

		Messages, the structured output schema and any extra keyword arguments are forwarded untouched,
		and a successful completion is returned exactly as the underlying model produced it.
		"""
		self._logical_invocations += 1
		attempt = 1

		while True:
			self._attempts += 1
			try:
				completion = await self._llm.ainvoke(messages, output_format=output_format, **kwargs)
			except Exception as e:
				# Only the class name is recorded. str(e) routinely echoes the request, and the
				# request carries the answer under verification and the scraped page text.
				name = type(e).__name__
				self._exception_type_counts[name] = self._exception_type_counts.get(name, 0) + 1
				if attempt >= self._policy.max_attempts:
					self._failed_invocations += 1
					raise
				retry_number = attempt
				self._retries += 1
				attempt += 1
				await self._sleep(self._policy.delay_before_retry(retry_number))
				continue

			if attempt > 1:
				self._recovered_invocations += 1
			return completion


__all__ = ['LLMRetryPolicy', 'LLMRetryStats', 'RetryingChatModel', 'stats_delta']
