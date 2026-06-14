"""FabellaVLLM - LangChain BaseChatModel wrapping vLLM endpoint.

Uses vLLM's native tool-calling pipeline for Gemma 4. The server is started
with ``--enable-auto-tool-choice --tool-call-parser gemma4`` (see
``modal_app.py``), which makes vLLM parse the model's native
``<|tool_call>...<tool_call|>`` markers into OpenAI-spec ``tool_calls`` JSON.
This client passes the tool specs in OpenAI format and reads the parsed
``tool_calls`` straight off the response.
"""

import os
import sys
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field, PrivateAttr
from openai import OpenAI


class FabellaVLLM(BaseChatModel):
    """LangChain chat model backed by vLLM OpenAI-compatible API."""

    base_url: str = Field(default="https://khoitruong071510--fabella-serve-drafter.modal.run")
    model_name: str = "gemma-4"
    temperature: float = 0.9
    top_p: float = 0.95
    max_tokens: int = 4096
    seed: int = 0

    _client: Any = PrivateAttr(default=None)
    _tools: list[dict] = PrivateAttr(default_factory=list)
    _tool_call_id: int = PrivateAttr(default=0)

    @property
    def _llm_type(self) -> str:
        return "fabella-vllm"

    @property
    def _identifying_params(self) -> dict:
        return {
            "base_url": self.base_url,
            "model_name": self.model_name,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
            "seed": self.seed,
        }

    def _get_client(self) -> OpenAI:
        if self._client is None:
            self._client = OpenAI(
                base_url=f"{self.base_url}/v1",
                api_key="EMPTY",
            )
        return self._client

    def bind_tools(self, tools: list, **kwargs):  # type: ignore[override]
        specs = []
        for t in tools:
            specs.append(_to_openai_tool_spec(t))
        object.__setattr__(self, "_tools", specs)
        object.__setattr__(self, "_tool_call_id", 0)
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        client = self._get_client()

        system, non_system = _split_system(messages)
        api_messages = []
        if system:
            api_messages.append({"role": "system", "content": system})
        api_messages.extend(_to_api_messages(non_system))

        request: dict[str, Any] = {
            "model": self.model_name,
            "messages": api_messages,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
        }
        if self.seed:
            request["seed"] = self.seed
        if self._tools:
            request["tools"] = self._tools

        response = client.chat.completions.create(**request)
        message = response.choices[0].message

        ai_message = _parse_response_message(message, state=self)
        return ChatResult(generations=[ChatGeneration(message=ai_message)])


def _split_system(messages) -> tuple[str, list]:
    system_parts: list[str] = []
    rest: list = []
    for m in messages:
        if isinstance(m, SystemMessage):
            content = m.content if isinstance(m.content, str) else str(m.content)
            system_parts.append(content)
        else:
            rest.append(m)
    return "\n".join(system_parts), rest


def _to_api_messages(messages) -> list[dict]:
    """Translate LangChain messages to OpenAI chat-completions format."""
    out: list[dict] = []
    for m in messages:
        if isinstance(m, HumanMessage):
            content = m.content if isinstance(m.content, str) else str(m.content)
            out.append({"role": "user", "content": content})
        elif isinstance(m, AIMessage):
            entry: dict[str, Any] = {"role": "assistant"}
            content = m.content if isinstance(m.content, str) else str(m.content)
            if content:
                entry["content"] = content
            if m.tool_calls:
                entry["tool_calls"] = [
                    {
                        "id": tc.get("id", f"call_{i}"),
                        "type": "function",
                        "function": {
                            "name": tc.get("name", ""),
                            "arguments": _dump_args(tc.get("args", {})),
                        },
                    }
                    for i, tc in enumerate(m.tool_calls)
                ]
            out.append(entry)
        elif isinstance(m, ToolMessage):
            content = m.content if isinstance(m.content, str) else str(m.content)
            entry = {
                "role": "tool",
                "tool_call_id": m.tool_call_id,
                "content": content,
            }
            out.append(entry)
        else:
            content = getattr(m, "content", "")
            content = content if isinstance(content, str) else str(content)
            out.append({"role": "user", "content": content})
    return out


def _parse_response_message(message, *, state: "FabellaVLLM") -> AIMessage:
    content = message.content or ""
    if not message.tool_calls:
        return AIMessage(content=content)

    tool_calls = []
    for tc in message.tool_calls:
        state._tool_call_id += 1
        raw_args = tc.function.arguments
        args = _loads_args(raw_args)
        tool_calls.append(
            {
                "name": tc.function.name,
                "args": args,
                "id": tc.id or f"call_{state._tool_call_id}",
                "type": "tool_call",
            }
        )
    return AIMessage(content=content, tool_calls=tool_calls)


def _to_openai_tool_spec(tool_obj) -> dict:
    """Build an OpenAI-spec tool entry from a LangChain tool."""
    name = getattr(tool_obj, "name", None) or getattr(tool_obj, "__name__", "tool")
    description = (getattr(tool_obj, "description", "") or (tool_obj.__doc__ or "")).strip()
    parameters = _extract_parameters(tool_obj)
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": parameters,
        },
    }


def _extract_parameters(tool_obj) -> dict:
    try:
        from langchain_core.tools import BaseTool

        if isinstance(tool_obj, BaseTool):
            schema = tool_obj.args
            properties = {
                name: _normalize_schema(field)
                for name, field in schema.items()
            }
            required = [
                name
                for name, field in schema.items()
                if field.get("type") != "null" and name not in (schema.get("additionalProperties") or {})
            ]
            return {
                "type": "object",
                "properties": properties,
                "required": list(schema.keys()),
            }
    except Exception:
        pass

    if hasattr(tool_obj, "args_schema") and tool_obj.args_schema is not None:
        try:
            model = tool_obj.args_schema
            from pydantic import BaseModel  # type: ignore

            if isinstance(model, type) and issubclass(model, BaseModel):
                return model.model_json_schema()
            if hasattr(model, "model_json_schema"):
                return model.model_json_schema()
            if hasattr(model, "schema"):
                return model.schema()
        except Exception:
            pass

    return {"type": "object", "properties": {}}


def _normalize_schema(field: dict) -> dict:
    out = {"type": field.get("type", "string")}
    if "description" in field:
        out["description"] = field["description"]
    if "enum" in field:
        out["enum"] = field["enum"]
    return out


def _dump_args(args: Any) -> str:
    import json

    if isinstance(args, str):
        return args
    return json.dumps(args, ensure_ascii=False)


def _loads_args(raw: Any) -> Any:
    import json

    if isinstance(raw, dict):
        return raw
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except (TypeError, ValueError):
        return {"input": raw}
