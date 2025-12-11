# Local Agent

This project provides a simple, self-contained local agent system powered by a local Large Language Model (LLM) through [Ollama](https://ollama.com/). The agent is capable of interacting with your local machine to perform tasks like reading files, listing directories, and launching applications. The entire system is exposed via a REST API built with FastAPI.

## Features

-   **Local LLM Interaction**: Leverages Ollama to run an LLM agent entirely on your local machine.
-   **Tool-Enabled**: The agent can use a predefined set of tools to interact with the operating system.
    -   `read_file`: Reads the content of a specified file.
    -   `list_directory`: Lists the contents of a directory.
    -   `launch_app`: Launches common desktop applications (e.g., Chrome, VSCode, Terminal).
-   **API-Driven**: A FastAPI server exposes the agent's capabilities, allowing for easy integration with other applications.
-   **Simple & Extensible**: The core logic is straightforward, making it easy to understand, modify, and extend with new tools.

## Requirements

-   Python 3.12+
-   [Ollama](https://ollama.com/) installed and running.
-   An Ollama model pulled. The default is `llama3.2:latest`.

## Setup & Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd local-agent
    ```

2.  **Install Python dependencies:**
    This project uses `uv` for package management. You can install the dependencies from `pyproject.toml`:
    ```bash
    pip install -e .
    ```

3.  **Setup Ollama:**
    Ensure the Ollama application is running. Then, pull the model specified in `main.py`:
    ```bash
    ollama pull llama3.2:latest
    ```

## Running the Application

To start the FastAPI server, run the following command in the project root:

```bash
uvicorn main:app --host "0.0.0.0" --port 8000
```

The server will be accessible at `http://localhost:8000`.

## API Endpoints

### Health Check

Check if the agent and the required model are running correctly.

-   **Endpoint**: `GET /`
-   **Success Response** (`200 OK`):
    ```json
    {
        "status": "Agent is online",
        "model": "llama3.2:latest"
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