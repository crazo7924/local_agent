"""LLM Agent execution loop."""

import ollama

from src.config import MODEL_NAME
from src.tools import (
    available_tools,
    launch_app,
    list_directory,
    list_path_executables,
    read_file,
)


def run_agent_loop(user_prompt: str) -> str:
    """The 'Brain' of the agent.

    It enters a loop:
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
            tools=[read_file, launch_app, list_directory, list_path_executables],
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
