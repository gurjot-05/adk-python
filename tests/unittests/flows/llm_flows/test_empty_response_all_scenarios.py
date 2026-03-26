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

Covers:
  Scenario 1: Non-streaming empty response (parts: [], is_final_response=True)
  Scenario 2: Streaming + thinking, thought-only final (is_final_response=True)
  Scenario 3a: No events yielded at all (last_event=None)
  Scenario 3b: Partial event with no meaningful content (last_event.partial=True)
  Scenario 4: Partial event WITH meaningful content (should NOT retry)
  Scenario 5: Empty response after max retries (should stop)
  Scenario 6: Empty then good response (recovery)
  Scenario 7: lite_llm streaming fallback (empty non-partial response yielded)
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


# ---------------------------------------------------------------------------
# Scenario 1: Non-streaming empty response (the original bug from adk_combined.log)
#   Model returns parts: [], partial=False, finish_reason=STOP
#   is_final_response() -> True, _has_meaningful_content() -> False
#   Expected: retry with resume message, then succeed
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

  # Should see: empty model event, resume nudge, good model event
  resume_events = [e for e in events if e.author == 'user']
  model_events = [
      e
      for e in events
      if e.author == 'test_agent' and e.content and e.content.parts
  ]
  assert len(resume_events) == 1, 'Expected exactly 1 resume nudge'
  good_texts = [p.text for e in model_events for p in e.content.parts if p.text]
  assert any(
      'answer' in t for t in good_texts
  ), 'Expected good response after retry'


# ---------------------------------------------------------------------------
# Scenario 2: Streaming + thinking, thought-only final response
#   Model returns thought parts only, partial=False
#   is_final_response() -> True, _has_meaningful_content() -> False (thought-only)
#   Expected: retry with resume message
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

  resume_events = [e for e in events if e.author == 'user']
  assert len(resume_events) == 1, 'Expected retry on thought-only response'


# ---------------------------------------------------------------------------
# Scenario 3a: No events yielded at all (last_event=None)
#   _postprocess_async filters out LlmResponse with content=None
#   Expected: retry with resume message
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_scenario3a_no_events_at_all_retried():
  """When _run_one_step yields nothing, retry fires."""
  # content=None means _postprocess_async returns without yielding
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

  resume_events = [e for e in events if e.author == 'user']
  assert (
      len(resume_events) == _MAX_EMPTY_RESPONSE_RETRIES
  ), f'Expected {_MAX_EMPTY_RESPONSE_RETRIES} resume nudges'


# ---------------------------------------------------------------------------
# Scenario 3b: Partial event with no meaningful content
#   Streaming + thinking: last event is partial with thought-only content
#   Expected: retry with resume message
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

  resume_events = [e for e in events if e.author == 'user']
  assert len(resume_events) == 1, 'Expected retry on partial empty event'


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

  resume_events = [e for e in events if e.author == 'user']
  assert len(resume_events) == 1, 'Expected retry on partial thought-only event'


# ---------------------------------------------------------------------------
# Scenario 4: Partial event WITH meaningful content (should NOT retry)
#   This is a normal streaming state — partial + real text content.
#   Expected: break with warning, no retry
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

  resume_events = [e for e in events if e.author == 'user']
  assert (
      len(resume_events) == 0
  ), 'Partial event with real content should NOT trigger retry'
  # The partial event itself should be yielded
  partial_events = [e for e in events if e.partial]
  assert len(partial_events) == 1


# ---------------------------------------------------------------------------
# Scenario 5: Empty response exhausts max retries
#   Model keeps returning empty — should stop after _MAX_EMPTY_RESPONSE_RETRIES
#   Expected: exactly _MAX_EMPTY_RESPONSE_RETRIES resume nudges, then break
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

  resume_events = [e for e in events if e.author == 'user']
  assert len(resume_events) == _MAX_EMPTY_RESPONSE_RETRIES

  # Model should have been called initial + retries times
  assert (
      mock_model.response_index == _MAX_EMPTY_RESPONSE_RETRIES
  ), f'Expected {_MAX_EMPTY_RESPONSE_RETRIES + 1} LLM calls total'


# ---------------------------------------------------------------------------
# Scenario 6: Empty -> Empty -> Good (recovery after multiple retries)
#   Expected: 2 resume nudges, then good response
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

  resume_events = [e for e in events if e.author == 'user']
  assert len(resume_events) == 2, 'Expected 2 retries before recovery'

  final_texts = [
      p.text
      for e in events
      if e.content and e.content.parts
      for p in e.content.parts
      if p.text and not getattr(p, 'thought', False)
  ]
  assert any('recovered' in t.lower() for t in final_texts)


# ---------------------------------------------------------------------------
# Scenario 7: lite_llm streaming fallback — verify the empty non-partial
#   LlmResponse is what downstream code would see
# ---------------------------------------------------------------------------


def test_scenario7_litellm_fallback_response_is_not_partial():
  """Verify the fallback LlmResponse from lite_llm has partial=False."""
  # Simulates what lite_llm.py now produces when streaming yields nothing
  fallback = LlmResponse(
      content=types.Content(role='model', parts=[]),
      partial=False,
      finish_reason=types.FinishReason.STOP,
      model_version='test-model',
  )
  # This should be treated as a final response
  event = Event(
      invocation_id='test',
      author='test_agent',
      content=fallback.content,
      # partial comes from LlmResponse merge
  )
  assert event.is_final_response() is True
  assert _has_meaningful_content(event) is False


# ---------------------------------------------------------------------------
# Scenario 8: Whitespace-only text response (edge case)
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

  resume_events = [e for e in events if e.author == 'user']
  assert len(resume_events) == 1, 'Whitespace-only should trigger retry'


# ---------------------------------------------------------------------------
# Scenario 9: Function call response is NOT retried (meaningful content)
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
  # is_final_response() would be False (has function calls), so the
  # retry check would never fire for this event anyway.
  assert event.is_final_response() is False


# ---------------------------------------------------------------------------
# Scenario 10: Mixed partial+empty then partial+content (no false positive)
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

  resume_events = [e for e in events if e.author == 'user']
  assert len(resume_events) == 1, (
      'First partial empty should retry, second partial with content should'
      ' break'
  )
