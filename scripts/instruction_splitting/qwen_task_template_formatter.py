#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
usage:
给定输入 JSON + 模板 JSON
   export DASHSCOPE_API_KEY=sk-dbbf0cc993e34a85bf57c8fa0a67bbc5 \
   python scripts/qwen_task_template_formatter.py \
     --input /path/to/input.json \
     --template /path/to/template.json \
     --output /path/to/out.json

    python scripts/instruction_splitting/qwen_task_template_formatter.py --input /home/hhl/TravelUAV/TravelUAV/scripts/instruction_splitting/test_input_instruction.json --template /home/hhl/TravelUAV/TravelUAV/scripts/instruction_splitting/instruction_unit_template.json --output /home/hhl/TravelUAV/TravelUAV/scripts/instruction_splitting/test_output.json
    env -u ALL_PROXY -u all_proxy -u HTTP_PROXY -u http_proxy -u HTTPS_PROXY -u https_proxy -u NO_PROXY -u no_proxy python scripts//task_template_formatter/qwen_task_template_formatter.py   --input /home/liz/TravelUAV/scripts/test_raw_prompt.json   --template /home/liz/TravelUAV/scripts/content_unit_template_example.json   --output /home/liz/TravelUAV/scripts/test_output.json

环境变量：
- DASHSCOPE_API_KEY   
- QWEN_BASE_URL       默认 https://dashscope.aliyuncs.com/compatible-mode/v1
- QWEN_MODEL          默认 qwen3-32b
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import httpx
from openai import OpenAI

JsonType = Union[Dict[str, Any], List[Any], str, int, float, bool, None]


DEFAULT_QWEN_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_QWEN_MODEL = "qwen3-32b"


@dataclass
class QwenConfig:
    api_key: str
    base_url: str
    model: str
    timeout_s: float = 120.0
    max_retries: int = 2
    temperature: float = 0.0
    top_p: float = 1.0


def _load_json(path: str) -> JsonType:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _dump_json(obj: JsonType, path: Optional[str]) -> None:
    text = json.dumps(obj, ensure_ascii=False, indent=2)
    if path:
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
            f.write("\n")
    else:
        sys.stdout.write(text)
        sys.stdout.write("\n")


def _get_by_dotted_path(data: JsonType, dotted: str) -> JsonType:
    """Very small helper: dotted path like a.b.c or a.0.b"""
    cur: JsonType = data
    for part in dotted.split("."):
        if isinstance(cur, list):
            try:
                idx = int(part)
            except ValueError as e:
                raise KeyError(f"Expected list index at '{part}' in path '{dotted}'") from e
            if idx < 0 or idx >= len(cur):
                raise KeyError(f"Index {idx} out of range for path '{dotted}'")
            cur = cur[idx]
        elif isinstance(cur, dict):
            if part not in cur:
                raise KeyError(f"Key '{part}' not found for path '{dotted}'")
            cur = cur[part]
        else:
            raise KeyError(f"Cannot traverse into non-container at '{part}' for path '{dotted}'")
    return cur


def _iter_instructions_auto(
    data: JsonType,
    candidate_keys: Tuple[str, ...],
) -> Iterable[str]:
    """Recursively search for strings under candidate keys.

    - dict with key in candidate_keys:
      - if value is str -> yield
      - if value is list -> yield each str inside (and also recurse into dicts)
    - always recurse into dict/list children.
    """
    if isinstance(data, dict):
        for k, v in data.items():
            if k in candidate_keys:
                if isinstance(v, str) and v.strip():
                    yield v
                elif isinstance(v, list):
                    for item in v:
                        if isinstance(item, str) and item.strip():
                            yield item
                        else:
                            yield from _iter_instructions_auto(item, candidate_keys)
                elif isinstance(v, dict):
                    yield from _iter_instructions_auto(v, candidate_keys)
            else:
                yield from _iter_instructions_auto(v, candidate_keys)
    elif isinstance(data, list):
        for item in data:
            yield from _iter_instructions_auto(item, candidate_keys)


def _extract_instructions(
    input_json: JsonType,
    path: Optional[str],
    key: Optional[str],
) -> List[str]:
    """Extract instruction strings from input JSON.

    Priority:
    1) If --path is provided: select that node, then extract strings.
    2) Else if --key is provided: recursive search for that key.
    3) Else: heuristic keys.
    """

    node = input_json
    if path:
        node = _get_by_dotted_path(input_json, path)

    # Special-case: chat-style list like [{"from":"human","value":"..."}, {"from":"gpt","value":""}]
    if isinstance(node, list) and node and all(isinstance(x, dict) for x in node):
        has_from_value = any(("from" in x and "value" in x) for x in node)
        if has_from_value:
            human_values: List[str] = []
            for x in node:
                if x.get("from") == "human":
                    v = x.get("value")
                    if isinstance(v, str) and v.strip():
                        human_values.append(v)
            if human_values:
                node = human_values if len(human_values) > 1 else human_values[0]

    if key:
        keys = (key,)
    else:
        keys = (
            "instruction",
            "instructions",
            "task",
            "tasks",
            "command",
            "commands",
            "query",
            "goal",
            "text",
        )

    instructions = list(_iter_instructions_auto(node, keys))

    # If the selected node itself is a string, treat it as one instruction.
    if not instructions and isinstance(node, str) and node.strip():
        instructions = [node]

    # If the selected node is a list of strings, treat them directly.
    if not instructions and isinstance(node, list) and all(isinstance(x, str) for x in node):
        instructions = [x for x in node if x.strip()]

    # De-duplicate while preserving order
    seen = set()
    uniq: List[str] = []
    for ins in instructions:
        if ins not in seen:
            seen.add(ins)
            uniq.append(ins)
    return uniq


def _replace_placeholders(template: JsonType, mapping: Dict[str, str]) -> JsonType:
    """Deep-replace any string exactly equal to a placeholder token."""
    if isinstance(template, dict):
        return {k: _replace_placeholders(v, mapping) for k, v in template.items()}
    if isinstance(template, list):
        return [_replace_placeholders(v, mapping) for v in template]
    if isinstance(template, str):
        return mapping.get(template, template)
    return template


def _structure_signature(x: JsonType) -> JsonType:
    """Return a signature of structure for strict comparison.

    - dict: keys -> signatures
    - list: list of signatures (preserve length)
    - scalar: use a marker of type (so template placeholders won't require same content)

    NOTE: We validate *structure*, not exact values.
    """
    if isinstance(x, dict):
        return {k: _structure_signature(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_structure_signature(v) for v in x]
    # scalar
    return type(x).__name__


def _validate_same_structure(template: JsonType, candidate: JsonType) -> Tuple[bool, str]:
    """Strict check: candidate must have identical dict keys and list lengths recursively."""

    def _walk(t: JsonType, c: JsonType, path: str) -> Tuple[bool, str]:
        if isinstance(t, dict):
            if not isinstance(c, dict):
                return False, f"Type mismatch at {path}: template=dict, candidate={type(c).__name__}"
            tkeys = set(t.keys())
            ckeys = set(c.keys())
            if tkeys != ckeys:
                missing = sorted(tkeys - ckeys)
                extra = sorted(ckeys - tkeys)
                return False, f"Key mismatch at {path}: missing={missing}, extra={extra}"
            for k in t.keys():
                ok, msg = _walk(t[k], c[k], f"{path}.{k}" if path else k)
                if not ok:
                    return ok, msg
            return True, ""
        if isinstance(t, list):
            if not isinstance(c, list):
                return False, f"Type mismatch at {path}: template=list, candidate={type(c).__name__}"
            if len(t) != len(c):
                return False, f"List length mismatch at {path}: template={len(t)}, candidate={len(c)}"
            for i, (tv, cv) in enumerate(zip(t, c)):
                ok, msg = _walk(tv, cv, f"{path}[{i}]")
                if not ok:
                    return ok, msg
            return True, ""
        # scalar: we don't require same type/value; only require candidate exists.
        return True, ""

    return _walk(template, candidate, "")


_CONTENT_UNIT_KEYS = ("id", "position", "environment", "action")


def _is_content_unit_template(template: JsonType) -> bool:
    if not isinstance(template, dict):
        return False
    if set(template.keys()) != set(_CONTENT_UNIT_KEYS):
        return False
    return True


def _validate_content_unit_list(template_unit: Dict[str, Any], candidate: JsonType) -> Tuple[bool, str]:
    if not isinstance(candidate, list):
        return False, f"Type mismatch at root: expected list, got {type(candidate).__name__}"
    if len(candidate) == 0:
        return False, "List is empty: expected at least 1 content unit"

    for i, item in enumerate(candidate):
        ok, msg = _validate_same_structure(template_unit, item)
        if not ok:
            return False, f"Unit[{i}] invalid: {msg}"
    return True, ""


def _normalize_content_units(units: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    # Ensure id is sequential by the output order.
    for idx, unit in enumerate(units, start=1):
        unit["id"] = idx

        vc = unit.get("visit_constraint")
        if isinstance(vc, dict):
            # Normalize mode casing and k type.
            mode = vc.get("mode")
            if isinstance(mode, str):
                vc["mode"] = mode.strip().upper()

            k = vc.get("k")
            if isinstance(k, str):
                if k.strip().upper() in {"NULL", "NONE"}:
                    vc["k"] = None
                else:
                    try:
                        vc["k"] = int(k)
                    except ValueError:
                        pass
            elif isinstance(k, float) and k.is_integer():
                vc["k"] = int(k)

    return units


def _create_openai_client(cfg: QwenConfig) -> OpenAI:
    """Create OpenAI-compatible client for DashScope/Qwen.

    Some environments set proxy env vars like ALL_PROXY=socks://... which httpx rejects
    unless socks support is installed and a valid scheme (e.g., socks5://) is used.
    We fall back to ignoring env proxies to keep the script usable out-of-the-box.
    """
    try:
        return OpenAI(api_key=cfg.api_key, base_url=cfg.base_url)
    except ValueError as e:
        if "Unknown scheme for proxy URL" not in str(e):
            raise

        print(
            "[WARN] Detected invalid proxy URL from environment (e.g., socks://...). "
            "Falling back to ignore proxy env vars (trust_env=False).",
            file=sys.stderr,
        )
        http_client = httpx.Client(timeout=cfg.timeout_s, trust_env=False)
        return OpenAI(api_key=cfg.api_key, base_url=cfg.base_url, http_client=http_client)


def _qwen_chat_completion(cfg: QwenConfig, messages: List[Dict[str, str]]) -> str:
    client = _create_openai_client(cfg)

    last_err: Optional[str] = None
    for attempt in range(cfg.max_retries):
        try:
            completion = client.chat.completions.create(
                model=cfg.model,
                messages=messages,
                temperature=cfg.temperature,
                top_p=cfg.top_p,
                timeout=cfg.timeout_s,
                extra_body={"enable_thinking": False},
            )
            content = completion.choices[0].message.content
            if not isinstance(content, str) or not content.strip():
                raise ValueError("Unexpected response format: empty/non-string content")
            return content
        except Exception as e:
            last_err = str(e)
            time.sleep(1.0 * (attempt + 1))

    raise RuntimeError(f"Qwen API call failed after retries. Last error: {last_err}")


def _build_messages(template: JsonType, raw_instruction: str, instruction_ph: str, formatted_ph: str) -> List[Dict[str, str]]:
    template_text = json.dumps(template, ensure_ascii=False, indent=2)

    if _is_content_unit_template(template):
        system = (
            "你是一个严格的JSON生成器。"
            "你必须输出合法JSON，且只能输出JSON。"
            "禁止输出解释、Markdown、代码块、注释或任何JSON以外的内容。"
            "你将把一条任务指令拆分成多个instruction unit，并输出为JSON数组。"
            "数组中每个元素必须严格匹配给定instruction unit模板的字段结构。"
            "在生成JSON前，你必须在内部完成路径约束求解、地标编号、共指解析、unit筛选和最终校验，但不得输出这些过程。"
        )

        user = (
            f"instruction unit 模板如下。每个unit必须严格包含模板中的所有字段；若instruction未提供某字段内容，则该字段value使用模板允许的空值：\n"
            f"{template_text}\n\n"

            f"任务：\n"
            f"将原始任务指令拆分为多个instruction unit，并输出JSON数组。\n\n"

            f"核心原则：\n"
            f"- 生成JSON前，必须先在内部建立真实路径序列，不得直接按原文子句顺序输出。\n"
            f"- unit顺序必须按真实路径经过顺序排列。\n"
            f"- 若文本顺序与空间关系冲突，以空间关系、序数关系、路径约束为准。\n"
            f"- 只输出具有明确空间锚点的unit。\n"
            f"- 空间锚点包括：intersection、landmark、目标点、观察对象、明确回指位置。\n"
            f"- 普通行进过程不是空间锚点，不得单独输出unit。\n"
            f"- inferred unit只允许用于补全缺失的关键空间锚点，不得用于表示普通继续前进。\n"
            f"- 同一物理地标在同一次经过中只能输出一个unit；再次经过同一物理地标时，才再次输出一个unit，但environment编号必须复用。\n\n"

            f"路径约束规则：\n"
            f"- 必须同时使用显式先后词、空间介词关系、序数地标关系来确定路径序列。\n"
            f"- 显式先后词包括 then, after that, before that, until, go back, return 等。\n"
            f"- 空间介词关系优先级高于原文出现顺序。\n"
            f"- after A 表示 A 必须早于当前对象。\n"
            f"- before A 表示当前对象必须早于 A。\n"
            f"- between A and B 表示当前对象位于 A 与 B 之间。\n"
            f"- until A 表示持续当前动作直到 A，A 是该动作的目标空间锚点。\n"
            f"- go back to A / return to A 表示再次经过或到达 A，A 必须作为新的unit出现，但environment编号复用此前同一物理位置的编号。\n"
            f"- 若观察对象位于两个已知路径锚点之间，必须将该观察对象unit插入这两个锚点之间。\n\n"

            f"序数与编号规则：\n"
            f"- 地标编号必须按真实路径首次经过顺序分配，而不是按文本出现顺序分配。\n"
            f"- first/second/third 等序数直接约束同类地标编号。\n"
            f"- 若出现第k个同类地标，必须保证第1到第k个同类地标都在路径序列中存在；缺失者用inferred unit补全。\n"
            f"- next 同类地标表示沿当前路径继续前进时遇到的下一个同类地标，编号为最近已确定同类地标编号加1。\n"
            f"- previous、last、same、where 等回指表达必须复用已有编号，不得新建编号。\n"
            f"- last 不表示新的地标；它必须回指到当前路径上下文中最后一个已存在或可推断的同类地标。\n"
            f"- 同一类别的多个物体按真实路径首次出现顺序编号。\n"
            f"- 不同类别分别编号，例如 intersection、a_red_car、a_blue_car、a_man_in_red_clothes 分属不同类别。\n"
            f"- 同一个物体多次出现时必须复用同一编号。\n\n"

            f"观察对象规则：\n"
            f"- see, notice, find, pass by, on your left, on your right 等描述的对象属于观察对象unit。\n"
            f"- 观察对象unit的顺序由其空间关系决定，不由该子句在原文中的位置决定。\n"
            f"- 若观察对象被描述为 after 某个地标，则该地标unit必须排在观察对象unit之前。\n"
            f"- 若观察对象被描述为 before 某个地标，则观察对象unit必须排在该地标unit之前。\n"
            f"- 若观察对象本身不是intersection，则environment只写观察对象本身，不写作为参照物的intersection。\n"
            f"- 若观察对象没有转向动作，action使用 pass。\n\n"

            f"inferred unit规则：\n"
            f"- inferred unit的text必须以'inferred: '开头。\n"
            f"- inferred unit必须表示具体空间锚点，例如缺失的前序intersection、被序数隐含的landmark、被路径约束必然经过的目标点。\n"
            f"- 禁止生成普通动作后的继续前进unit。\n"
            f"- 禁止生成仅表示continue、follow、move forward、after turning left、after turning right的inferred unit。\n"
            f"- 若一个显性unit已经覆盖某个空间锚点，不得再为同一次经过生成重复的inferred unit。\n\n"

            f"字段要求：\n"
            f"- id: int。第一个unit的id必须是0；第n个unit的id必须等于n；不得从1开始，不得跳号。\n"
            f"- position: 按模板要求输出。若instruction未提供坐标，则使用模板允许的空值，通常为空list。\n"
            f"- environment: list[str]。 表示该unit实际所在位置、目标位置或当前位置绑定的关键地标。\n"
            f"  - environment不得包含仅作为时间条件、起点、背景、参照物或已经离开的地点。\n"
            f"  - 多个地标作为list中的多个字符串元素输出，不得用逗号拼成一个字符串。\n"
            f"  - 单个地标命名格式为 <landmark_name>_i，其中i表示该landmark类别内的编号。\n"
            f"  - 回指unit的environment必须继承被回指unit的完整environment。\n"
            f"- action: str。必须且只能从以下集合中选择：\"turn left\", \"turn right\", \"go straight\", \"turn around\", \"pass\", \"reach\"。\n"
            f"  - action 必须表示抵达该 unit 之后要执行的动作，而非表示如何抵达该 unit 的动作。"
            f"  - 左转使用 \"turn left\"。\n"
            f"  - 右转使用 \"turn right\"。\n"
            f"  - 在intersection处继续直行使用 \"go straight\"。\n"
            f"  - 经过无转向动作的landmark或观察对象使用 \"pass\"。\n"
            f"  - 返回已出现位置时，返回动作使用 \"turn around\"。\n"
            f"  - \"reach\" 只能用于整条instruction执行完后的最终终点，也就是最后一个 unit。\n"
            f"  - 若某个地点只是中途到达点，后面仍有go back、return、turn、continue等动作，则该地点不得使用 \"reach\"。\n"
            f"  - 禁止输出 observe、continue、return、go back、follow、stop 等白名单外动作。\n\n"

            f"共指与信息融合规则：\n"
            f"- 必须先解析共指关系，再写environment。\n"
            f"- 若某个unit通过历史动作回指已有位置，例如 where you [action]、the place where you [action]、the previous landmark、the same landmark、the last landmark，必须找到先前对应unit并复用其environment编号。\n"
            f"- 回指unit必须继承被回指unit的完整environment，而不是只写其中一个地标。\n"
            f"- 若多个unit指向同一物理位置，必须共享一致且完整的environment。\n"
            f"- 较早unit中的环境线索要补充到较晚共指unit；较晚unit中的回指信息也要用于校验较早unit的编号和environment。\n\n"

            f"输出前内部校验规则，不得输出校验过程：\n"
            f"- 输出必须是JSON数组，不能有外层对象。\n"
            f"- 每个unit必须严格包含模板中的所有字段。\n"
            f"- 第一个unit的id必须是0；第n个unit的id必须等于n。\n"
            f"- unit顺序必须满足所有after、before、between、then、until、go back、return等路径约束。\n"
            f"- 若出现第k个同类地标，则第1到第k个同类地标必须全部存在。\n"
            f"- first、second、third、next、previous、last、same、where 等表达不得导致编号冲突。\n"
            f"- after 某地标的对象必须排在该地标之后；before 某地标的对象必须排在该地标之前。\n"
            f"- 普通继续前进不得成为inferred unit。\n"
            f"- 同一物理地标单次经过不得重复输出。\n"
            f"- environment不得包含仅作为时间条件、起点、参照物或已离开位置的地点。\n"
            f"- 回指unit必须复用被回指unit的完整environment。\n"
            f"- action必须属于白名单：turn left, turn right, go straight, turn around, pass, reach。\n"
            f"- reach只能用于整条instruction执行完后的最终终点；若后面还有任何动作，不得使用reach。\n\n"

            f"原始任务指令如下。注意：'<image>'只是占位符，可作为上下文，不要当作动作步骤本身：\n"
            f"{raw_instruction}\n\n"

            f"请只输出JSON数组。"
        )
    else:
        raise ValueError("Instruction_unit_template cannot be used")
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def _call_and_parse_json(cfg: QwenConfig, template: JsonType, raw_instruction: str, instruction_ph: str, formatted_ph: str) -> JsonType:
    messages = _build_messages(template, raw_instruction, instruction_ph, formatted_ph)
    content = _qwen_chat_completion(cfg, messages)

    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        # Common failure: model wraps output with ```json ...```
        stripped = content.strip()
        if stripped.startswith("```"):
            stripped = stripped.strip("`")
            # remove a possible leading 'json' token
            stripped = stripped.lstrip().removeprefix("json").lstrip()
        parsed = json.loads(stripped)

    if _is_content_unit_template(template):
        if isinstance(parsed, dict):
            parsed = [parsed]
        ok, msg = _validate_content_unit_list(template, parsed)
        if ok:
            assert isinstance(parsed, list)
            normalized = _normalize_content_units([x for x in parsed if isinstance(x, dict)])
            return normalized
    else:
        ok, msg = _validate_same_structure(template, parsed)
        if ok:
            return parsed

    # Retry once with explicit error feedback
    retry_messages = messages + [
        {
            "role": "user",
            "content": (
                "你的输出没有严格匹配模板结构，错误如下：\n"
                f"{msg}\n\n"
                "请重新输出：必须是合法JSON，且结构（key/层级/数组长度）与模板完全一致。"
            ),
        }
    ]
    content2 = _qwen_chat_completion(cfg, retry_messages)
    parsed2 = json.loads(content2)
    if _is_content_unit_template(template):
        if isinstance(parsed2, dict):
            parsed2 = [parsed2]
        ok2, msg2 = _validate_content_unit_list(template, parsed2)
        if not ok2:
            raise ValueError(f"Model output still does not match content-unit list structure: {msg2}")
        assert isinstance(parsed2, list)
        normalized2 = _normalize_content_units([x for x in parsed2 if isinstance(x, dict)])
        return normalized2

    ok2, msg2 = _validate_same_structure(template, parsed2)
    if not ok2:
        raise ValueError(f"Model output still does not match template structure: {msg2}")
    return parsed2


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract instructions from JSON, call Qwen3 API, and fill a JSON template strictly.")
    parser.add_argument("--input", required=True, help="输入 JSON 文件路径")
    parser.add_argument("--template", required=True, help="模板 JSON 文件路径")
    parser.add_argument("--output", required=True, help="输出 JSON 文件路径")

    parser.add_argument("--path", default=None, help="可选: 从输入JSON中用点路径定位要提取的节点, 例如 tasks.0.instruction")
    parser.add_argument("--key", default=None, help="可选：指定任务指令字段名，例如 instruction")

    parser.add_argument("--instruction-placeholder", default="{{INSTRUCTION}}", help="模板中表示‘原始任务指令’的位置占位符")
    parser.add_argument("--formatted-placeholder", default="{{FORMATTED}}", help="模板中表示‘整理后的任务指令’的位置占位符")

    parser.add_argument("--dry-run", action="store_true", help="不调用API, 只做提取并用占位符替换成原始指令（整理后字段保持占位符）")

    parser.add_argument("--base-url", default=os.getenv("QWEN_BASE_URL", DEFAULT_QWEN_BASE_URL), help="Qwen OpenAI兼容接口 base url")
    parser.add_argument("--model", default=os.getenv("QWEN_MODEL", DEFAULT_QWEN_MODEL), help="模型名")
    parser.add_argument("--timeout", type=float, default=120.0, help="请求超时(秒)")
    parser.add_argument("--retries", type=int, default=2, help="请求重试次数")

    args = parser.parse_args()

    input_json = _load_json(args.input)
    template_json = _load_json(args.template)

    instructions = _extract_instructions(input_json, path=args.path, key=args.key)
    if not instructions:
        print(
            "[ERROR] 未在输入JSON中找到任务指令。你可以：\n"
            "- 用 --key 指定字段名（例如 --key instruction）\n"
            "- 或用 --path 指定更精确的位置（例如 --path tasks）\n",
            file=sys.stderr,
        )
        return 2

    if args.dry_run:
        results: List[JsonType] = []
        for ins in instructions:
            filled = _replace_placeholders(
                template_json,
                {
                    args.instruction_placeholder: ins,
                    # formatted_placeholder 保持不变，方便你后续检查模板位置
                },
            )
            ok, msg = _validate_same_structure(template_json, filled)
            if not ok:
                print(f"[ERROR] dry-run 填充后结构不匹配模板：{msg}", file=sys.stderr)
                return 3
            results.append(filled)
        _dump_json(results if len(results) > 1 else results[0], args.output)
        return 0

    # Align with DashScope/Qwen docs first, but keep backward compatibility.
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        print("[ERROR] 环境变量 DASHSCOPE_API_KEY 未设置", file=sys.stderr)
        return 2

    cfg = QwenConfig(
        api_key=api_key,
        base_url=args.base_url,
        model=args.model,
        timeout_s=args.timeout,
        max_retries=max(1, args.retries),
    )

    outputs: List[JsonType] = []
    for i, ins in enumerate(instructions, start=1):
        print(f"[INFO] Processing instruction {i}/{len(instructions)}", file=sys.stderr)

        # First do a local placeholder fill for raw instruction (so even if model fails, placeholders exist)
        template_for_call = copy.deepcopy(template_json)
        # We do NOT locally replace formatted placeholder; model will fill it.
        template_for_call = _replace_placeholders(
            template_for_call,
            {args.instruction_placeholder: ins},
        )

        # Call model to produce final JSON (structure must equal *original* template)
        # Important: We pass the *original* template, so placeholders are in the prompt
        out = _call_and_parse_json(cfg, template_json, ins, args.instruction_placeholder, args.formatted_placeholder)
        outputs.append(out)

    if _is_content_unit_template(template_json):
        # Flatten across multiple instructions if needed.
        flattened: List[JsonType] = []
        for out in outputs:
            if isinstance(out, list):
                flattened.extend(out)
            else:
                flattened.append(out)
        # Re-assign ids globally by final order.
        normalized = _normalize_content_units([x for x in flattened if isinstance(x, dict)])
        _dump_json(normalized, args.output)
        return 0

    _dump_json(outputs if len(outputs) > 1 else outputs[0], args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
