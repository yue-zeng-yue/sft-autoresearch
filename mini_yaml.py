"""Tiny YAML subset used when PyYAML is unavailable.

It supports the simple config style used by `train_config.yaml`: nested
mappings, block lists, inline empty lists/dicts, strings, booleans and numbers.
For production use, PyYAML is still preferred.
"""

from __future__ import annotations

import json
from typing import Any, TextIO


def _strip_comment(line: str) -> str:
    in_single = False
    in_double = False
    for idx, char in enumerate(line):
        if char == "'" and not in_double:
            in_single = not in_single
        elif char == '"' and not in_single:
            in_double = not in_double
        elif char == "#" and not in_single and not in_double:
            return line[:idx]
    return line


def _parse_scalar(value: str) -> Any:
    value = value.strip()
    if value == "":
        return ""
    if value == "[]":
        return []
    if value == "{}":
        return {}
    if value in {"true", "True"}:
        return True
    if value in {"false", "False"}:
        return False
    if value in {"null", "None", "~"}:
        return None
    if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
        return value[1:-1]
    try:
        if any(ch in value for ch in [".", "e", "E"]):
            return float(value)
        return int(value)
    except ValueError:
        return value


def _preprocess(text: str) -> list[tuple[int, str]]:
    rows: list[tuple[int, str]] = []
    for raw in text.splitlines():
        line = _strip_comment(raw).rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        rows.append((indent, line.strip()))
    return rows


def _parse_block(rows: list[tuple[int, str]], index: int, indent: int) -> tuple[Any, int]:
    if index >= len(rows):
        return {}, index

    is_list = rows[index][0] == indent and rows[index][1].startswith("- ")
    container: Any = [] if is_list else {}

    while index < len(rows):
        row_indent, content = rows[index]
        if row_indent < indent:
            break
        if row_indent > indent:
            raise ValueError(f"unexpected indentation near: {content}")

        if is_list:
            if not content.startswith("- "):
                break
            item = content[2:].strip()
            if item == "":
                value, index = _parse_block(rows, index + 1, indent + 2)
                container.append(value)
                continue
            if ":" in item and not item.startswith(("'", '"')):
                key, raw_value = item.split(":", 1)
                obj: dict[str, Any] = {}
                if raw_value.strip():
                    obj[key.strip()] = _parse_scalar(raw_value)
                    index += 1
                else:
                    value, index = _parse_block(rows, index + 1, indent + 2)
                    obj[key.strip()] = value
                container.append(obj)
                continue
            container.append(_parse_scalar(item))
            index += 1
            continue

        if content.startswith("- "):
            break
        if ":" not in content:
            raise ValueError(f"expected key: value near: {content}")
        key, raw_value = content.split(":", 1)
        key = key.strip()
        raw_value = raw_value.strip()
        if raw_value:
            container[key] = _parse_scalar(raw_value)
            index += 1
        else:
            if index + 1 >= len(rows) or rows[index + 1][0] <= indent:
                container[key] = {}
                index += 1
            else:
                value, index = _parse_block(rows, index + 1, rows[index + 1][0])
                container[key] = value

    return container, index


def safe_load(stream: TextIO | str) -> Any:
    text = stream.read() if hasattr(stream, "read") else str(stream)
    rows = _preprocess(text)
    if not rows:
        return None
    value, index = _parse_block(rows, 0, rows[0][0])
    if index != len(rows):
        raise ValueError("could not parse entire YAML document")
    return value


def safe_dump(data: Any, stream: TextIO | None = None, **_: Any) -> str | None:
    text = json.dumps(data, indent=2, ensure_ascii=False)
    if stream is not None:
        stream.write(text)
        stream.write("\n")
        return None
    return text + "\n"
