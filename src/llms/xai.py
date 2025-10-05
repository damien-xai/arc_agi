"""
xAI Grok API integration for ARC-AGI benchmarking.
Supports grok-4-fast-reasoning and grok-4-fast-non-reasoning models.
"""
import asyncio
import os
import typing as T

import httpx
from devtools import debug

from src import logfire
from src.models import Model, ModelUsage


async def get_next_messages_xai(
    *,
    messages: list[dict[str, T.Any]],
    model: Model,
    temperature: float,
    n_times: int,
) -> list[tuple[str, ModelUsage]]:
    """
    Get N completions from xAI Grok API.
    
    Args:
        messages: List of message dicts with role and content
        model: Model enum (grok_4_fast_reasoning or grok_4_fast_non_reasoning)
        temperature: Sampling temperature (0-1)
        n_times: Number of completions to generate
        
    Returns:
        List of (response_text, usage) tuples
    """
    api_key = os.environ.get("XAI_API_KEY")
    if not api_key:
        raise ValueError("XAI_API_KEY not found in environment variables")
    
    # xAI API endpoint (OpenAI-compatible)
    base_url = "https://api.x.ai/v1"
    
    # Convert messages to xAI format (OpenAI-compatible)
    # Handle system messages and content formats
    formatted_messages = []
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        
        # Handle different content formats
        if isinstance(content, str):
            formatted_messages.append({"role": role, "content": content})
        elif isinstance(content, list):
            # Extract text from content blocks (ignore images for now)
            text_parts = []
            for block in content:
                if block.get("type") == "text":
                    text_parts.append(block.get("text", ""))
            if text_parts:
                formatted_messages.append({
                    "role": role,
                    "content": "\n\n".join(text_parts)
                })
    
    # Prepare API request
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    
    payload = {
        "model": model.value,
        "messages": formatted_messages,
        "temperature": temperature,
        "stream": False,
    }
    
    # Make N parallel requests with retry logic
    async def make_request() -> tuple[str, ModelUsage] | None:
        retry_count = 0
        max_retries = 50
        
        while retry_count < max_retries:
            try:
                async with httpx.AsyncClient(timeout=300.0) as client:
                    response = await client.post(
                        f"{base_url}/chat/completions",
                        headers=headers,
                        json=payload,
                    )
                    response.raise_for_status()
                    data = response.json()
                    
                    # Extract response text
                    response_text = data["choices"][0]["message"]["content"]
                    
                    # Extract usage information
                    usage_data = data.get("usage", {})
                    usage = ModelUsage(
                        cache_creation_input_tokens=0,  # xAI doesn't report cache separately
                        cache_read_input_tokens=0,
                        input_tokens=usage_data.get("prompt_tokens", 0),
                        output_tokens=usage_data.get("completion_tokens", 0),
                    )
                    
                    logfire.debug(
                        f"xAI {model.value} response",
                        input_tokens=usage.input_tokens,
                        output_tokens=usage.output_tokens,
                        response_length=len(response_text),
                    )
                    
                    return (response_text, usage)
                    
            except httpx.HTTPStatusError as e:
                # Check if it's a rate limit error (429)
                if e.response.status_code == 429:
                    retry_count += 1
                    # Gradual backoff: 1s, 3s, 5s, then exponential (7.5s, 11.25s, ...)
                    if retry_count == 1:
                        wait_time = 1
                    elif retry_count == 2:
                        wait_time = 3
                    elif retry_count == 3:
                        wait_time = 5
                    else:
                        # Exponential backoff from 5s: 7.5, 11.25, 16.875...
                        wait_time = 5 * (1.5 ** (retry_count - 3))
                    
                    logfire.debug(
                        f"xAI rate limit (429), retrying in {wait_time:.1f}s ({retry_count}/{max_retries})"
                    )
                    await asyncio.sleep(wait_time)
                else:
                    # Other HTTP errors, don't retry
                    logfire.debug(f"xAI API error: {e.response.status_code} - {e.response.text}")
                    return None
            except Exception as e:
                logfire.debug(f"xAI request failed: {e}")
                return None
        
        # Max retries exceeded
        logfire.debug(f"xAI request failed after {max_retries} retries")
        return None
    
    # Run N requests in parallel
    results = await asyncio.gather(*[make_request() for _ in range(n_times)])
    
    # Filter out failed requests
    successful_results = [r for r in results if r is not None]
    
    if not successful_results:
        logfire.debug("All xAI requests failed")
        return []
    
    return successful_results

