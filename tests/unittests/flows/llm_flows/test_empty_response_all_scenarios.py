# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Comprehensive tests for empty model response retry across all scenarios.

The resume message is appended directly to the session (not yielded),
so it reaches the model on retry but never leaks to the UI/SSE stream.
We verify retries via mock_model.response_index and confirm no user
events are yielded.
"""

from google.adk.agents.llm_agent import Agent
from google.adk.events.event import Event
from google.adk.flows.llm_flows.base_llm_flow import _has_meaningful_content
from google.adk.flows.llm_flows.base_llm_flow import _MAX_EMPTY_RESPONSE_RETRIES
from google.adk.flows.llm_flows.base_llm_flow import BaseLlmFlow
from google.adk.models.llm_response import LlmResponse
from google.genai import types
import pytest

from ... import testing_utils


class BaseLlmFlowForTesting(BaseLlmFlow):
  pass


def _collect_resume_leaks(events):
  """Return any resume nudge events that leaked to the output stream."""
  return [
      e
      for e in events
      if e.author == 'user'
      and e.content
      and e.content.parts
      and any(
          'previous response was empty' in (p.text or '')
          for p in e.content.parts
      )
  ]


# ---------------------------------------------------------------------------
# Scenario 1: Non-streaming empty response, then recovery
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_scenario1_non_streaming_empty_then_recovery():
  """Non-streaming: model returns empty parts, retry recovers."""
  empty = LlmResponse(
      content=types.Content(role='model', parts=[]),
      partial=False,
  )
  good = LlmResponse(
      content=types.Content(
          role='model',
          parts=[types.Part.from_text(text='Here is the answer.')],
      ),
      partial=False,
  )
  mock_model = testing_utils.MockModel.create(responses=[empty, good])
  agent = Agent(name='test_agent', model=mock_model)
  ctx = await testing_utils.create_invocation_context(
      agent=agent, user_content='test'
  )

  events = []
  async for event in BaseLlmFlowForTesting().run_async(ctx):
    events.append(event)

  # Model called twice (empty + good)
  assert mock_model.response_index == 1
  # Resume message must NOT leak to UI
  assert len(_collect_resume_leaks(events)) == 0
  # Good response should be in output
  good_texts = [
      p.text
      for e in events
      if e.content and e.content.parts
      for p in e.content.parts
      if p.text
  ]
  assert any('answer' in t for t in good_texts)


# ---------------------------------------------------------------------------
# Scenario 2: Thought-only final response triggers retry
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_scenario2_thought_only_final_response_retried():
  """Thought-only non-partial response triggers retry."""
  thought_only = LlmResponse(
      content=types.Content(
          role='model',
          parts=[types.Part(text='Let me think...', thought=True)],
      ),
      partial=False,
  )
  good = LlmResponse(
      content=types.Content(
          role='model',
          parts=[types.Part.from_text(text='The answer is 42.')],
      ),
      partial=False,
  )
  mock_model = testing_utils.MockModel.create(responses=[thought_only, good])
  agent = Agent(name='test_agent', model=mock_model)
  ctx = await testing_utils.create_invocation_context(
      agent=agent, user_content='test'
  )

  events = []
  async for event in BaseLlmFlowForTesting().run_async(ctx):
    events.append(event)

  assert mock_model.response_index == 1, 'Expected 2 LLM calls (retry)'
  assert len(_collect_resume_leaks(events)) == 0


# ---------------------------------------------------------------------------
# Scenario 3a: No events yielded at all (last_event=None)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_scenario3a_no_events_at_all_retried():
  """When _run_one_step yields nothing, retry fires."""
  empty_responses = [
      LlmResponse(content=None, partial=False)
      for _ in range(_MAX_EMPTY_RESPONSE_RETRIES + 1)
  ]
  mock_model = testing_utils.MockModel.create(responses=empty_responses)
  agent = Agent(name='test_agent', model=mock_model)
  ctx = await testing_utils.create_invocation_context(
      agent=agent, user_content='test'
  )

  events = []
  async for event in BaseLlmFlowForTesting().run_async(ctx):
    events.append(event)

  # Model called initial + retries times
  assert mock_model.response_index == _MAX_EMPTY_RESPONSE_RETRIES
  assert len(_collect_resume_leaks(events)) == 0


# ---------------------------------------------------------------------------
# Scenario 3b: Partial event with no meaningful content
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_scenario3b_partial_empty_content_retried():
  """Partial event with empty parts triggers retry."""
  partial_empty = LlmResponse(
      content=types.Content(role='model', parts=[]),
      partial=True,
  )
  good = LlmResponse(
      content=types.Content(
          role='model',
          parts=[types.Part.from_text(text='Here is the answer.')],
      ),
      partial=False,
  )
  mock_model = testing_utils.MockModel.create(responses=[partial_empty, good])
  agent = Agent(name='test_agent', model=mock_model)
  ctx = await testing_utils.create_invocation_context(
      agent=agent, user_content='test'
  )

  events = []
  async for event in BaseLlmFlowForTesting().run_async(ctx):
    events.append(event)

  assert mock_model.response_index == 1, 'Expected retry on partial empty'
  assert len(_collect_resume_leaks(events)) == 0


@pytest.mark.asyncio
async def test_scenario3b_partial_thought_only_retried():
  """Partial event with thought-only content triggers retry."""
  partial_thought = LlmResponse(
      content=types.Content(
          role='model',
          parts=[types.Part(text='Thinking...', thought=True)],
      ),
      partial=True,
  )
  good = LlmResponse(
      content=types.Content(
          role='model',
          parts=[types.Part.from_text(text='Done thinking, here it is.')],
      ),
      partial=False,
  )
  mock_model = testing_utils.MockModel.create(responses=[partial_thought, good])
  agent = Agent(name='test_agent', model=mock_model)
  ctx = await testing_utils.create_invocation_context(
      agent=agent, user_content='test'
  )

  events = []
  async for event in BaseLlmFlowForTesting().run_async(ctx):
    events.append(event)

  assert mock_model.response_index == 1, 'Expected retry on partial thought'
  assert len(_collect_resume_leaks(events)) == 0


# ---------------------------------------------------------------------------
# Scenario 4: Partial event WITH meaningful content (should NOT retry)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_scenario4_partial_with_meaningful_content_not_retried():
  """Partial event with real text content should NOT trigger retry."""
  partial_with_text = LlmResponse(
      content=types.Content(
          role='model',
          parts=[types.Part.from_text(text='Partial but real content')],
      ),
      partial=True,
  )
  mock_model = testing_utils.MockModel.create(responses=[partial_with_text])
  agent = Agent(name='test_agent', model=mock_model)
  ctx = await testing_utils.create_invocation_context(
      agent=agent, user_content='test'
  )

  events = []
  async for event in BaseLlmFlowForTesting().run_async(ctx):
    events.append(event)

  # Only 1 LLM call — no retry
  assert mock_model.response_index == 0
  partial_events = [e for e in events if e.partial]
  assert len(partial_events) == 1


# ---------------------------------------------------------------------------
# Scenario 5: Empty response exhausts max retries
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_scenario5_empty_exhausts_max_retries():
  """Empty responses stop after max retries."""
  empty_responses = [
      LlmResponse(
          content=types.Content(role='model', parts=[]),
          partial=False,
      )
      for _ in range(_MAX_EMPTY_RESPONSE_RETRIES + 1)
  ]
  mock_model = testing_utils.MockModel.create(responses=empty_responses)
  agent = Agent(name='test_agent', model=mock_model)
  ctx = await testing_utils.create_invocation_context(
      agent=agent, user_content='test'
  )

  events = []
  async for event in BaseLlmFlowForTesting().run_async(ctx):
    events.append(event)

  # Model called initial + retries = MAX_RETRIES + 1
  assert mock_model.response_index == _MAX_EMPTY_RESPONSE_RETRIES
  assert len(_collect_resume_leaks(events)) == 0


# ---------------------------------------------------------------------------
# Scenario 6: Empty -> Empty -> Good (recovery after multiple retries)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_scenario6_multiple_empty_then_recovery():
  """Multiple empty responses followed by good response."""
  responses = [
      LlmResponse(
          content=types.Content(role='model', parts=[]),
          partial=False,
      ),
      LlmResponse(
          content=types.Content(role='model', parts=[]),
          partial=False,
      ),
      LlmResponse(
          content=types.Content(
              role='model',
              parts=[types.Part.from_text(text='Finally recovered!')],
          ),
          partial=False,
      ),
  ]
  mock_model = testing_utils.MockModel.create(responses=responses)
  agent = Agent(name='test_agent', model=mock_model)
  ctx = await testing_utils.create_invocation_context(
      agent=agent, user_content='test'
  )

  events = []
  async for event in BaseLlmFlowForTesting().run_async(ctx):
    events.append(event)

  # All 3 responses consumed
  assert mock_model.response_index == 2
  assert len(_collect_resume_leaks(events)) == 0
  final_texts = [
      p.text
      for e in events
      if e.content and e.content.parts
      for p in e.content.parts
      if p.text and not getattr(p, 'thought', False)
  ]
  assert any('recovered' in t.lower() for t in final_texts)


# ---------------------------------------------------------------------------
# Scenario 7: lite_llm streaming fallback
# ---------------------------------------------------------------------------


def test_scenario7_litellm_fallback_response_is_not_partial():
  """Verify the fallback LlmResponse from lite_llm has partial=False."""
  fallback = LlmResponse(
      content=types.Content(role='model', parts=[]),
      partial=False,
      finish_reason=types.FinishReason.STOP,
      model_version='test-model',
  )
  event = Event(
      invocation_id='test',
      author='test_agent',
      content=fallback.content,
  )
  assert event.is_final_response() is True
  assert _has_meaningful_content(event) is False


# ---------------------------------------------------------------------------
# Scenario 8: Whitespace-only text response
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_scenario8_whitespace_only_response_retried():
  """Response with only whitespace text triggers retry."""
  whitespace = LlmResponse(
      content=types.Content(
          role='model',
          parts=[types.Part.from_text(text='   \n  \t  ')],
      ),
      partial=False,
  )
  good = LlmResponse(
      content=types.Content(
          role='model',
          parts=[types.Part.from_text(text='Actual content.')],
      ),
      partial=False,
  )
  mock_model = testing_utils.MockModel.create(responses=[whitespace, good])
  agent = Agent(name='test_agent', model=mock_model)
  ctx = await testing_utils.create_invocation_context(
      agent=agent, user_content='test'
  )

  events = []
  async for event in BaseLlmFlowForTesting().run_async(ctx):
    events.append(event)

  assert mock_model.response_index == 1, 'Expected retry on whitespace'
  assert len(_collect_resume_leaks(events)) == 0


# ---------------------------------------------------------------------------
# Scenario 9: Function call is meaningful (not retried)
# ---------------------------------------------------------------------------


def test_scenario9_function_call_is_meaningful():
  """A function call event is meaningful and would not trigger retry."""
  event = Event(
      invocation_id='test',
      author='test_agent',
      content=types.Content(
          role='model',
          parts=[
              types.Part(
                  function_call=types.FunctionCall(
                      name='some_tool', args={'key': 'value'}
                  )
              )
          ],
      ),
  )
  assert _has_meaningful_content(event) is True
  assert event.is_final_response() is False


# ---------------------------------------------------------------------------
# Scenario 10: Partial empty then partial with content
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_scenario10_partial_empty_then_partial_with_content():
  """Partial empty retries, then partial with content breaks normally."""
  partial_empty = LlmResponse(
      content=types.Content(role='model', parts=[]),
      partial=True,
  )
  partial_real = LlmResponse(
      content=types.Content(
          role='model',
          parts=[types.Part.from_text(text='Streaming chunk')],
      ),
      partial=True,
  )
  mock_model = testing_utils.MockModel.create(
      responses=[partial_empty, partial_real]
  )
  agent = Agent(name='test_agent', model=mock_model)
  ctx = await testing_utils.create_invocation_context(
      agent=agent, user_content='test'
  )

  events = []
  async for event in BaseLlmFlowForTesting().run_async(ctx):
    events.append(event)

  # Both responses consumed (retry on first, break on second)
  assert mock_model.response_index == 1
  assert len(_collect_resume_leaks(events)) == 0
