import asyncio
import os
import time
import typing as T

from openai import AsyncOpenAI

from src import logfire
from src.logic import random_string
from src.models import Attempt, Model, ModelUsage


def text_only_messages(messages: list[dict[str, T.Any]]) -> list[dict[str, T.Any]]:
    """Convert messages to text-only format for models that don't support multimodal."""
    new_messages = []
    for message in messages:
        content_strs: list[str] = []
        if isinstance(message["content"], str):
            content_strs.append(message["content"])
        else:
            for content in message["content"]:
                if content["type"] == "text":
                    content_strs.append(content["text"])
        if content_strs:
            new_messages.append(
                {
                    "role": message["role"],
                    "content": "\n".join(content_strs),
                }
            )
    return new_messages


async def get_next_message_openrouter(
    messages: list[dict[str, T.Any]],
    model: Model,
    temperature: float,
    retry_secs: int = 15,
    max_retries: int = 15,
) -> tuple[str, ModelUsage] | None:
    """Get a single response from OpenRouter."""
    openrouter_client = AsyncOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ["OPENROUTER_API_KEY"],
    )

    # Handle special cases for different models
    reasoning_models = [
        Model.openrouter_o1,
        Model.openrouter_o1_mini,
        Model.openrouter_o3,
        Model.openrouter_o4_mini,
    ]

    # Models that might be using preview/text-only versions
    # Note: This may need adjustment based on what OpenRouter actually supports
    text_only_models = [
        Model.openrouter_o1_mini,  # o1-mini was originally text-only
    ]

    # Clean up messages based on model requirements
    processed_messages = messages.copy()

    # Convert system messages to developer role for reasoning models
    if (
        processed_messages
        and processed_messages[0]["role"] == "system"
        and model in reasoning_models
    ):
        processed_messages[0]["role"] = "developer"

    # Use text-only messages for models that don't support multimodal
    if model in text_only_models:
        processed_messages = text_only_messages(processed_messages)

    # Set up parameters
    params = {
        "messages": processed_messages,
        "model": model.value,
        # "max_tokens": 20_000 if model in reasoning_models else 10_000,
        "max_tokens": 40_000,
        "timeout": 120,
    }

    # Only add temperature for non-reasoning models
    # (o1/o3 models typically don't support temperature parameter)
    if model not in reasoning_models:
        params["temperature"] = temperature

    retry_count = 0
    while retry_count < max_retries:
        try:
            request_id = random_string()
            start = time.time()
            logfire.debug(f"[{request_id}] calling openrouter with model {model.value}")

            response = await openrouter_client.chat.completions.create(**params)

            took_ms = (time.time() - start) * 1000

            # Extract usage information
            if response.usage:
                usage = ModelUsage(
                    cache_creation_input_tokens=0,
                    cache_read_input_tokens=0,
                    input_tokens=response.usage.prompt_tokens,
                    output_tokens=response.usage.completion_tokens,
                )
            else:
                # Fallback if no usage info provided
                usage = ModelUsage(
                    cache_creation_input_tokens=0,
                    cache_read_input_tokens=0,
                    input_tokens=0,
                    output_tokens=0,
                )

            logfire.debug(
                f"[{request_id}] got response from openrouter, took {took_ms:.2f}ms, "
                f"usage={usage}, cost_cents={Attempt.cost_cents_from_usage(model=model, usage=usage)}"
            )

            return response.choices[0].message.content, usage

        except Exception as e:
            logfire.debug(
                f"OpenRouter error: {str(e)}, retrying in {retry_secs} seconds "
                f"({retry_count + 1}/{max_retries})..."
            )
            retry_count += 1
            if retry_count < max_retries:
                await asyncio.sleep(retry_secs)
            else:
                return None


async def get_next_messages(
    messages: list[dict[str, T.Any]], model: Model, temperature: float, n_times: int
) -> list[tuple[str, ModelUsage]] | None:
    """Get multiple responses from OpenRouter in parallel."""
    if n_times <= 0:
        return []

    # Run all requests in parallel
    tasks = [
        get_next_message_openrouter(
            messages=messages,
            model=model,
            temperature=temperature,
        )
        for _ in range(n_times)
    ]

    results = await asyncio.gather(*tasks)

    # Filter out None results (failed requests)
    return [result for result in results if result is not None]
