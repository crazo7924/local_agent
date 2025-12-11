import os
import subprocess

import ollama
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Local LLM Agent System")

# --- 1. CONFIGURATION ---
MODEL_NAME = "llama3.2:latest"  # Ensure you ran `ollama pull <model name>`

# --- 2. TOOL DEFINITIONS ---
# These functions are "tools" the LLM can see and decide to call.


def list_directory(directory_path: str) -> str:
    """
    Lists the contents of a directory.
    Args:
        directory_path: The parent directory of which the contents are returned comma separated.
    """
    return ", ".join(os.listdir(directory_path))


def read_file(file_path: str) -> str:
    """
    Reads the content of a file from the local file system.
    Args:
        file_path: The absolute or relative path to the file.
    """
    try:
        if not os.path.exists(file_path):
            return f"Error: File '{file_path}' does not exist."

        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read(2000)  # Limit to 2k chars to prevent context overflow
            if len(content) == 2000:
                content += "\n...[truncated]..."
            return content
    except Exception as e:
        return f"Error reading file: {str(e)}"


def launch_app(application_name: str, args: str) -> str:
    """
    Launches a desktop application in the Linux environment.
    Supported shortcuts: 'chrome', 'vscode', 'terminal'.
    Args:
        application_name: The name of the app to launch.
    """
    # Map common names to actual linux commands
    app_map = {
        "chrome": "google-chrome",
        "google chrome": "google-chrome",
        "vscode": "code",
        "code": "code",
        "terminal": "xdg-terminal-exec",
    }

    command = app_map.get(application_name.lower(), application_name)

    command_with_args = command + " " + args

    try:
        # Popen ensures the server doesn't hang waiting for the app to close
        subprocess.Popen(command_with_args, shell=True, start_new_session=True)
        return f"Successfully launched {application_name} (PID: running in background)."
    except Exception as e:
        return f"Failed to launch {application_name}: {str(e)}"


# Registry of tools to pass to Ollama
available_tools = {
    "read_file": read_file,
    "launch_app": launch_app,
    "list_directory": list_directory,
}

# --- 3. AGENT LOGIC ---


def run_agent_loop(user_prompt: str) -> str:
    """
    The 'Brain' of the agent. It enters a loop:
    1. Send user prompt to LLM.
    2. Check if LLM wants to use a tool.
    3. If yes, execute tool -> feed result back to LLM -> Repeat.
    4. If no, return final answer.
    """
    messages = [{"role": "user", "content": user_prompt}]

    print(f"🤖 Processing: {user_prompt}")

    # Max turns to prevent infinite loops
    for _ in range(5):
        response = ollama.chat(
            model=MODEL_NAME,
            messages=messages,
            tools=[read_file, launch_app, list_directory],  # Pass actual functions
        )

        message = response["message"]
        messages.append(message)  # Add assistant's thought to history

        # CASE A: The Model wants to call tools
        if message.get("tool_calls"):
            for tool in message["tool_calls"]:
                func_name = tool["function"]["name"]
                args = tool["function"]["arguments"]

                print(f"🛠️  Agent calling tool: {func_name} with {args}")

                # Execute the actual Python function
                func_to_call = available_tools.get(func_name, None)
                if func_to_call:
                    tool_output = func_to_call(**args)
                else:
                    tool_output = "Error: Tool not found"

                # Add tool output to chat history so the model knows what happened
                messages.append(
                    {
                        "role": "tool",
                        "content": str(tool_output),
                    }
                )

        # CASE B: The Model simply replied (no tools needed)
        else:
            return message["content"]

    return "Agent stopped: Max iterations reached."


# --- 4. API ENDPOINTS ---


class AgentRequest(BaseModel):
    prompt: str


@app.post("/agent/chat")
async def agent_chat(request: AgentRequest):
    response = run_agent_loop(request.prompt)
    return {"response": response}


@app.get("/")
def health_check():
    ps_response = ollama.ps()
    for model in ps_response.models:
        if model.name == MODEL_NAME:
            return {"status": "Agent is online", "model": MODEL_NAME}

    return {"status": "Agent is NOT running", "model": MODEL_NAME}


if __name__ == "__main__":
    import uvicorn

    # Run on 0.0.0.0 to be accessible, port 8000
    uvicorn.run(app, host="0.0.0.0", port=8000)
