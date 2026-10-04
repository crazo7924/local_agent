"""Tool schema conversion utilities."""

import inspect
from typing import Any, Callable


def callable_to_openai_tool(func: Callable) -> dict[str, Any]:
    """Converts a Python function to an OpenAI/llama.cpp tool definition schema."""
    doc = inspect.getdoc(func) or ""
    sig = inspect.signature(func)

    properties = {}
    required = []

    for param_name, param in sig.parameters.items():
        param_type = "string"
        if param.annotation is int:
            param_type = "integer"
        elif param.annotation is float:
            param_type = "number"
        elif param.annotation is bool:
            param_type = "boolean"
        elif param.annotation in (dict, list):
            param_type = "object"

        properties[param_name] = {
            "type": param_type,
            "description": f"Parameter {param_name}",
        }
        if param.default == inspect.Parameter.empty:
            required.append(param_name)

    return {
        "type": "function",
        "function": {
            "name": func.__name__,
            "description": doc.strip(),
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        },
    }


def callable_to_anthropic_tool(func: Callable) -> dict[str, Any]:
    """Converts a Python function to an Anthropic tool definition schema."""
    doc = inspect.getdoc(func) or ""
    sig = inspect.signature(func)

    properties = {}
    required = []

    for param_name, param in sig.parameters.items():
        param_type = "string"
        if param.annotation is int:
            param_type = "integer"
        elif param.annotation is float:
            param_type = "number"
        elif param.annotation is bool:
            param_type = "boolean"

        properties[param_name] = {
            "type": param_type,
            "description": f"Parameter {param_name}",
        }
        if param.default == inspect.Parameter.empty:
            required.append(param_name)

    return {
        "name": func.__name__,
        "description": doc.strip(),
        "input_schema": {
            "type": "object",
            "properties": properties,
            "required": required,
        },
    }
