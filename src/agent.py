"""LLM Agent execution loop."""

from src.providers import DummyProvider, LLMProvider
from src.tools import (
    available_tools,
    launch_app,
    list_directory,
    read_file,
)


def run_agent_loop(
    user_prompt: str,
    provider: LLMProvider = DummyProvider(),
) -> str:
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
        response = provider.chat(
            messages=messages,
            tools=[read_file, launch_app, list_directory],
        )

        content = response.content
        tool_calls = response.tool_calls

        # Append assistant message history
        assistant_msg: dict = {"role": "assistant"}
        if content:
            assistant_msg["content"] = content
        if tool_calls:
            assistant_msg["tool_calls"] = [
                tc.model_dump(exclude_none=True) for tc in tool_calls
            ]
        messages.append(assistant_msg)

        # CASE A: The Model wants to call tools
        if tool_calls:
            for tool in tool_calls:
                func_name = tool.name
                args = tool.arguments

                print(f"🛠️  Agent calling tool: {func_name} with {args}")

                # Execute the actual Python function
                func_to_call = available_tools.get(func_name, None)
                if func_to_call:
                    tool_output = func_to_call(**args)
                else:
                    tool_output = "Error: Tool not found"

                # Add tool output to chat history so the model knows what happened
                tool_msg = {
                    "role": "tool",
                    "content": str(tool_output),
                }
                if tool.id:
                    tool_msg["tool_call_id"] = tool.id
                messages.append(tool_msg)

        # CASE B: The Model simply replied (no tools needed)
        else:
            return content or ""

    return "Agent stopped: Max iterations reached."
