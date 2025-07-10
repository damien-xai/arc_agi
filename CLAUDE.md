# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is Jeremy Berman's implementation of an Evolutionary Test-time Compute approach for solving ARC-AGI (Abstraction and Reasoning Corpus) puzzles. The project achieved 53.6% accuracy on the public leaderboard by using LLMs to generate and evolve Python transform functions over multiple generations.

## Development Commands

### Setup
```bash
# Install dependencies with Poetry
poetry shell
poetry install

# Create .env file with required API keys (see .example_env)
cp .example_env .env
# Edit .env to add OPENAI_API_KEY and ANTHROPIC_API_KEY at minimum
```

### Running the Project
```bash
# Main execution - solve ARC challenges
python run.py

# Run specific challenges
python run.py --challenge_ids "challenge1,challenge2"

# Run with limits
python run.py --limit 10 --offset 0

# Run FastAPI server
fastapi run src/app.py --port 80 --host 0.0.0.0
```

### Development Tools
```bash
# Linting
ruff check .
ruff format .

# Type checking (MyPy strict mode enabled)
mypy src/
```

## Architecture Overview

### Core Components

1. **Evolutionary Algorithm (`src/logic.py`)**
   - Implements the core evolutionary test-time compute approach
   - `solve_challenge()` - Main entry point for solving individual challenges
   - Manages multiple generations of transform functions
   - Handles both single-parent and pooled multi-parent evolution

2. **Model Integration (`src/models.py`)**
   - Unified interface for multiple LLM providers (OpenAI, Anthropic, Google, DeepSeek, etc.)
   - `get_llm_output()` - Primary function for LLM calls
   - Model pricing and token counting
   - Supports caching via Redis

3. **Tree Configurations (`src/trees/`)**
   - Different experimental setups defined as tree structures
   - Key files: `o3.py`, `deepseek.py`, `prod.py`, `experiments.py`
   - Trees define model choices, generation counts, and evolution parameters

4. **Prompt System (`src/prompts/`)**
   - Templates for generating transform functions
   - Chain-of-thought prompting strategies
   - Multiple grid representations (ASCII, array, image)

5. **Execution Engine (`src/run_python.py`)**
   - Safe Python code execution in isolated environment
   - Transform function validation
   - Grid manipulation utilities

### Key Patterns

- **Async-first design**: Extensive use of asyncio for concurrent operations
- **Tree-based configuration**: Hierarchical experiment configuration system
- **Fitness-based selection**: Solutions compete based on grid transformation accuracy
- **Multi-model support**: Can use different LLMs for diversity in evolution

### Important Files

- `run.py` - Main entry point for solving challenges
- `src/data.py` - Challenge loading and solution validation
- `src/app.py` - FastAPI application (if running as API)
- `src/db.py` - Database operations (PostgreSQL via asyncpg)

## Working with the Codebase

### Adding New Models
To add a new LLM provider, modify `src/models.py`:
1. Add the provider to the model configuration
2. Implement the API call logic in `get_llm_output()`
3. Add pricing information if tracking costs

### Creating New Tree Configurations
Create a new file in `src/trees/` following the pattern in existing files. Trees define:
- Model selections for each generation
- Number of attempts per generation
- Evolution parameters (single-parent vs pooled)

### Modifying Prompts
Edit files in `src/prompts/` to change how the LLM generates transform functions. Key prompts:
- `prompt.py` - Main generation prompt
- `revision_prompt.py` - For evolving existing functions
- `pooling_prompt.py` - For multi-parent evolution

### Environment Variables
Required in `.env`:
- `OPENAI_API_KEY` - For GPT models
- `ANTHROPIC_API_KEY` - For Claude models
- Optional: `GEMINI_API_KEY`, `OPENROUTER_API_KEY`, `AZURE_OPENAI_API_KEY`, etc.
- `LOGFIRE_TOKEN` - For logging (optional but recommended)

## Testing Approach

The project uses integration testing by running the full pipeline on training data. Use `python run.py` with specific challenge IDs to test changes.