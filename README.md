# Local Agent

This project provides a simple, self-contained local agent system supporting multiple Large Language Model (LLM) providers (such as Ollama, OpenAI, Anthropic, Gemini, and llama.cpp). The agent is capable of interacting with your local machine to perform tasks like reading files, listing directories, and launching applications. The entire system is exposed via a REST API built with FastAPI.

## Codebase Architecture

The project is structured into modular components inside `src/`:

- `main.py`: The entry point for running the FastAPI application.
- `src/config.py`: Contains configuration settings (e.g., default LLM model name).
- `src/agent.py`: Contains the core LLM execution loop (`run_agent_loop`).
- `src/providers/`: Modular LLM provider integrations (`ollama`, `openai`, `anthropic`, `gemini`, `llamacpp`).
- `src/tools/`: Tool definitions and implementations (`read_file`, `list_directory`, `launch_app`).

## Features

-   **Multi-Provider LLM Support**: Supports multiple LLM backends including Ollama, OpenAI, Anthropic, Gemini, and llama.cpp.
-   **Flexible Configuration**: Select active providers dynamically using the `LLM_PROVIDER` environment variable.
-   **Tool-Enabled**: The agent can use a predefined set of tools to interact with the operating system.
    -   `read_file`: Reads the content of a specified file.
    -   `list_directory`: Lists the contents of a directory.
    -   `launch_app`: Launches common desktop applications (e.g., Chrome, VSCode, Terminal).
-   **API-Driven**: A FastAPI server exposes the agent's capabilities, allowing for easy integration with other applications.
-   **Simple & Extensible**: The core logic is modular and easy to understand, modify, and extend with new providers and tools.

## Requirements

-   Python 3.12+
-   [uv](https://github.com/astral-sh/uv) (recommended) or `pip`
-   For local Ollama provider: [Ollama](https://ollama.com/) installed and running with a pulled model (e.g., `llama3.2:latest`).
-   API keys for cloud providers if enabled (e.g., `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GEMINI_API_KEY`).

## Setup & Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd local-agent
    ```

2.  **Install Python dependencies:**
    This project uses `uv` for package management.
    ```bash
    uv sync
    ```
    Or with `pip`:
    ```bash
    pip install -e .
    ```

3.  **Setup Provider (e.g., Ollama):**
    If using Ollama, ensure the application is running and pull the model:
    ```bash
    ollama pull llama3.2:latest
    ```

## Running the Application

To start the FastAPI server with a specified provider (default is `dummy`):

```bash
LLM_PROVIDER=ollama uv run uvicorn main:app --host "0.0.0.0" --port 8000
```

Or execute directly with Python:

```bash
LLM_PROVIDER=ollama python main.py
```

Available providers for `LLM_PROVIDER`: `dummy`, `ollama`, `openai`, `anthropic`, `gemini`, `llamacpp` (or `llama.cpp`).

The server will be accessible at `http://localhost:8000`.

## Running Tests

Run unit tests using `uv`:

```bash
uv run pytest
```

## API Endpoints

### Health Check

Check if the active LLM provider and model are online.

-   **Endpoint**: `GET /`
-   **Success Response** (`200 OK`):
    ```json
    {
        "status": "Agent is online",
        "is_online": true,
        "model": "llama3.2:latest",
        "provider": "OllamaProvider"
    }
    ```

### Agent Chat

Interact with the agent by sending a prompt.

-   **Endpoint**: `POST /agent/chat`
-   **Request Body**:
    ```json
    {
        "prompt": "Your instruction for the agent"
    }
    ```
-   **Example using `curl`**:
    ```bash
    curl -X POST "http://localhost:8000/agent/chat" \
         -H "Content-Type: application/json" \
         -d '{"prompt": "Read the first 10 lines of the README.md file."}'
    ```
-   **Response**:
    The response will be a JSON object containing the agent's final answer after potentially using its tools.
    ```json
    {
        "response": "Here are the first 10 lines of the README.md file..."
    }
    ```
