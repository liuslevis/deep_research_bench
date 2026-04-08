#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import locale
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_QUERY_FILE = REPO_ROOT / "data" / "prompt_data" / "query.jsonl"
DEFAULT_RAW_DATA_DIR = REPO_ROOT / "data" / "test_data" / "raw_data"
DEFAULT_STRUCTURED_DIR = REPO_ROOT / "results" / "tmp"
COMMON_ZH_CHARS = set("的一是在不了有人这中大为上个国我以要他时来们生到作地于出就分对成会可主发年动同工也能下过子说产种面而方后多定行学法所民得经十三之进着等部度家电力里如水化高自二理起小物现实加量都两体制机当使点从业本去把性好应开它合还因由其些然前外天政四日那社义事平形相全表间样与关各重新线内数正心反你明看原又么利比或但质气第向道命此变条只没结解问意建月公无系军很情者最立代想已通并提直题党程展果料象员革位入常文总次品式活设及管特件长求老头基资边流路级少图山统接知较将组见计别她手角期根论运农指几九区强放决西被干做必战先回则任取据处队南给色光门即保治北造百规热领七海口东导器压志世金增争济阶油思术极交受联什认六共权收证改清己美再采转更单风切打白教速花带安场身车例真务具万每目至达走积示议声报斗完类八离华名确才科张信马节话米整空元况今集温传土许步群广石记需段研界拉林律叫且究观越织装影算低持音众书布复容儿须际商非验连断深难近矿千周委素技备半办青省列习响约支般史感劳便团往酸历市克何除消构府称太准精值号率族维划选标写存候毛亲快效斯院查江型眼王按格养易置派层片始却专状育厂京识适属圆包火住调满县局照参红细引听该铁价严龙飞")
ZH_PUNCTUATION = "，。！？；：、（）《》“”‘’"
DEEPFIND_JSON_SNIPPET = (
    "import sys\n"
    "from deepfind.cli import main\n"
    "use_gpu = sys.argv[1] == '1'\n"
    "num_agent = sys.argv[2]\n"
    "max_iter = sys.argv[3]\n"
    "query = sys.stdin.buffer.read().decode('utf-8')\n"
    "argv = [\n"
    "    query,\n"
    "    '--json',\n"
    "    '--long-report-mode',\n"
    "    '--quiet',\n"
    "    '--once',\n"
    "    '--num-agent', num_agent,\n"
    "    '--max-iter-per-agent', max_iter,\n"
    "]\n"
    "if use_gpu:\n"
    "    argv.append('--gpu')\n"
    "raise SystemExit(main(argv))\n"
)


@dataclass(frozen=True)
class DeepFindRun:
    article: str
    payload: dict[str, Any]
    raw_output: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run DeepResearch Bench with ../deepfind-cli on the first N questions."
    )
    parser.add_argument(
        "-n",
        "--num-questions",
        type=int,
        required=True,
        help="Target cumulative question count. Example: -n 10 runs 1-10, then -n 20 resumes and only fills missing items within 1-20.",
    )
    parser.add_argument(
        "--deepfind-dir",
        type=Path,
        default=Path("../deepfind-cli"),
        help="Path to deepfind-cli repo (default: ../deepfind-cli).",
    )
    parser.add_argument(
        "--num-agent",
        type=int,
        default=4,
        help="Number of sub agents for deepfind-cli. Default: 4.",
    )
    parser.add_argument(
        "--max-iter-per-agent",
        type=int,
        default=50,
        help="Max tool/response rounds per deepfind agent.",
    )
    parser.add_argument(
        "--timeout-sec",
        type=int,
        default=900,
        help="Timeout for each deepfind query in seconds.",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=2,
        help="Retries per question if deepfind command fails.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=10,
        help="Worker count used by benchmark evaluation scripts.",
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Run deepfind-cli with --gpu so it uses the local Ollama model instead of the remote Qwen API.",
    )
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="Only generate raw_data, skip RACE and FACT evaluation.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="Output model name. Default: deepfind-cli-json-lr-a{agent}-n{N} or deepfind-cli-json-lr-gpu-a{agent}-n{N}. Reuse the same name to resume in-place.",
    )
    return parser.parse_args()


def load_runtime_env(dotenv_path: Path) -> dict[str, str]:
    env: dict[str, str] = dict(os.environ)
    if not dotenv_path.exists():
        return env

    for raw in dotenv_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            continue
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        env[key] = value
    return env


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f"{path.name}.tmp")
    with temp_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temp_path.replace(path)


def normalize_task_id(value: Any) -> str:
    return str(value)


def index_rows_by_id(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    rows_by_id: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        task_id = row.get("id")
        if task_id is None:
            continue
        rows_by_id[normalize_task_id(task_id)] = row
    return rows_by_id


def load_indexed_jsonl(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    return index_rows_by_id(load_jsonl(path))


def merge_missing_rows(
    destination: dict[str, dict[str, Any]],
    source: dict[str, dict[str, Any]],
) -> int:
    merged = 0
    for task_id, row in source.items():
        if task_id in destination:
            continue
        destination[task_id] = row
        merged += 1
    return merged


def sort_rows_for_output(
    rows_by_id: dict[str, dict[str, Any]],
    query_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    question_order = {
        normalize_task_id(row.get("id")): index
        for index, row in enumerate(query_rows)
        if isinstance(row, dict) and row.get("id") is not None
    }
    default_index = len(question_order)

    def sort_key(row: dict[str, Any]) -> tuple[int, str]:
        task_id = row.get("id")
        normalized = normalize_task_id(task_id) if task_id is not None else ""
        return question_order.get(normalized, default_index), normalized

    return sorted(rows_by_id.values(), key=sort_key)


def completed_selected_ids(
    selected: list[dict[str, Any]],
    raw_rows_by_id: dict[str, dict[str, Any]],
    structured_rows_by_id: dict[str, dict[str, Any]],
) -> set[str]:
    selected_ids = {
        normalize_task_id(task.get("id"))
        for task in selected
        if isinstance(task, dict) and task.get("id") is not None
    }
    return {
        task_id
        for task_id in selected_ids
        if task_id in raw_rows_by_id and task_id in structured_rows_by_id
    }


def discover_prior_checkpoint_names(model_name: str, current_n: int) -> list[str]:
    match = re.fullmatch(r"(.+)-n(\d+)", model_name)
    if match is None:
        return []

    prefix = match.group(1)
    raw_pattern = re.compile(rf"{re.escape(prefix)}-n(\d+)\.jsonl$")
    structured_pattern = re.compile(rf"{re.escape(prefix)}-n(\d+)\.structured\.jsonl$")
    counts: set[int] = set()

    for path in DEFAULT_RAW_DATA_DIR.glob(f"{prefix}-n*.jsonl"):
        matched = raw_pattern.fullmatch(path.name)
        if matched is None:
            continue
        count = int(matched.group(1))
        if 0 < count < current_n:
            counts.add(count)

    for path in DEFAULT_STRUCTURED_DIR.glob(f"{prefix}-n*.structured.jsonl"):
        matched = structured_pattern.fullmatch(path.name)
        if matched is None:
            continue
        count = int(matched.group(1))
        if 0 < count < current_n:
            counts.add(count)

    return [f"{prefix}-n{count}" for count in sorted(counts)]


def persist_outputs(
    *,
    raw_data_path: Path,
    structured_path: Path,
    subset_query_path: Path,
    completed_query_path: Path,
    query_rows: list[dict[str, Any]],
    selected: list[dict[str, Any]],
    raw_rows_by_id: dict[str, dict[str, Any]],
    structured_rows_by_id: dict[str, dict[str, Any]],
) -> None:
    write_jsonl(raw_data_path, sort_rows_for_output(raw_rows_by_id, query_rows))
    write_jsonl(structured_path, sort_rows_for_output(structured_rows_by_id, query_rows))
    write_jsonl(subset_query_path, selected)

    done_ids = completed_selected_ids(selected, raw_rows_by_id, structured_rows_by_id)
    completed_selected = [
        task
        for task in selected
        if isinstance(task, dict)
        and task.get("id") is not None
        and normalize_task_id(task.get("id")) in done_ids
    ]
    write_jsonl(completed_query_path, completed_selected)


def run_command(
    cmd: list[str],
    cwd: Path,
    *,
    timeout_sec: int | None = None,
    env: dict[str, str] | None = None,
    stdin_bytes: bytes | None = None,
) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        cmd,
        cwd=str(cwd),
        capture_output=True,
        text=False,
        input=stdin_bytes,
        timeout=timeout_sec,
        env=env,
    )


def build_deepfind_env(env: dict[str, str] | None) -> dict[str, str]:
    command_env = dict(env or os.environ)
    # Avoid nested `uv run` inheriting the outer temporary environment.
    command_env.pop("VIRTUAL_ENV", None)
    command_env["PYTHONIOENCODING"] = "utf-8"
    command_env["PYTHONUTF8"] = "1"
    return command_env


def count_cjk_chars(text: str) -> int:
    return sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")


def text_quality_score(text: str, *, language: str | None) -> float:
    if not text:
        return -1e9

    replacement_count = text.count("\ufffd")
    suspicious_mojibake = sum(text.count(token) for token in ("Ã", "Â", "�"))

    if language == "zh":
        cjk_count = count_cjk_chars(text)
        common_zh_count = sum(1 for ch in text if ch in COMMON_ZH_CHARS)
        zh_punc_count = sum(1 for ch in text if ch in ZH_PUNCTUATION)
        latin_ext_count = sum(1 for ch in text if 0x0100 <= ord(ch) <= 0x024F)
        return (
            common_zh_count * 3.0
            + zh_punc_count * 2.0
            + cjk_count * 0.1
            - latin_ext_count * 4.0
            - replacement_count * 8.0
            - suspicious_mojibake * 6.0
        )

    ascii_printable_count = sum(1 for ch in text if 32 <= ord(ch) < 127)
    non_ascii_count = len(text) - ascii_printable_count
    return ascii_printable_count - non_ascii_count * 0.2 - replacement_count * 8.0 - suspicious_mojibake * 6.0


def dedupe_encodings(encodings: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for enc in encodings:
        if not enc:
            continue
        normalized = enc.lower()
        if normalized in seen:
            continue
        seen.add(normalized)
        result.append(enc)
    return result


def repair_mojibake_variants(text: str, *, language: str | None) -> str:
    variants: list[str] = [text]
    try:
        variants.append(text.encode("utf-8").decode("gb18030", errors="replace"))
    except UnicodeError:
        pass
    try:
        variants.append(text.encode("gb18030", errors="replace").decode("utf-8", errors="replace"))
    except UnicodeError:
        pass
    return max(variants, key=lambda item: text_quality_score(item, language=language))


def decode_command_output(raw: bytes, *, language: str | None) -> str:
    if not raw:
        return ""
    preferred = locale.getpreferredencoding(False)
    candidates = dedupe_encodings(["utf-8", preferred, "gb18030", "cp936"])
    decoded_texts: list[str] = []

    for encoding in candidates:
        try:
            decoded_texts.append(raw.decode(encoding, errors="strict"))
        except UnicodeDecodeError:
            decoded_texts.append(raw.decode(encoding, errors="replace"))

    repaired = [repair_mojibake_variants(text, language=language) for text in decoded_texts]
    return max(repaired, key=lambda item: text_quality_score(item, language=language))


def _extract_json_object(text: str) -> dict[str, Any]:
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError("deepfind-cli did not return JSON output") from None
        parsed = json.loads(text[start : end + 1])

    if not isinstance(parsed, dict):
        raise ValueError("deepfind-cli JSON output must be an object")
    return parsed


def _reference_label(citation: dict[str, Any], index: int, *, language: str | None) -> str:
    title = str(citation.get("title") or "").strip()
    publisher = str(citation.get("publisher") or "").strip()
    if title:
        return title
    if publisher:
        return publisher
    return (f"来源{index}" if language == "zh" else f"Source {index}")


def _reference_lookup(payload: dict[str, Any]) -> tuple[dict[str, dict[str, Any]], dict[str, int]]:
    citations = payload.get("citations_dedup")
    if not isinstance(citations, list):
        return {}, {}

    citation_by_id: dict[str, dict[str, Any]] = {}
    index_by_id: dict[str, int] = {}
    for index, item in enumerate(citations, start=1):
        if not isinstance(item, dict):
            continue
        citation_id = str(item.get("id") or "").strip()
        url = str(item.get("url") or item.get("canonical_url") or "").strip()
        if not citation_id or not url:
            continue
        citation_by_id[citation_id] = item
        index_by_id[citation_id] = index
    return citation_by_id, index_by_id


def _inline_reference_links(
    citation_ids: Any,
    citation_by_id: dict[str, dict[str, Any]],
    index_by_id: dict[str, int],
    *,
    language: str | None,
) -> str:
    if not isinstance(citation_ids, list):
        return ""

    links: list[str] = []
    seen: set[str] = set()
    for citation_id in citation_ids:
        cid = str(citation_id).strip()
        if not cid or cid in seen:
            continue
        citation = citation_by_id.get(cid)
        if citation is None:
            continue
        seen.add(cid)
        url = str(citation.get("url") or citation.get("canonical_url") or "").strip()
        index = index_by_id.get(cid)
        if not url or index is None:
            continue
        label = _reference_label(citation, index, language=language)
        links.append(f"[{label}]({url})")
    return " ".join(links)


def _looks_like_json_blob(text: str) -> bool:
    stripped = text.strip()
    return (stripped.startswith("{") and stripped.endswith("}")) or (
        stripped.startswith("[") and stripped.endswith("]")
    )


def _render_claim_list(
    claims: Any,
    citation_by_id: dict[str, dict[str, Any]],
    index_by_id: dict[str, int],
    *,
    language: str | None,
) -> list[str]:
    if not isinstance(claims, list):
        return []

    lines: list[str] = []
    for claim in claims:
        if not isinstance(claim, dict):
            continue
        text = str(claim.get("text") or "").strip()
        if not text:
            continue
        inline_links = _inline_reference_links(
            claim.get("citation_ids"),
            citation_by_id,
            index_by_id,
            language=language,
        )
        lines.append(f"- {text}" + (f" {inline_links}" if inline_links else ""))
    return lines


def render_markdown_report(payload: dict[str, Any], *, language: str | None) -> str:
    title = "# 调研报告" if language == "zh" else "# Research Report"
    summary_title = "## 摘要" if language == "zh" else "## Summary"
    key_points_title = "## 关键发现" if language == "zh" else "## Key Findings"
    notes_title = "## 分项研究" if language == "zh" else "## Research Notes"
    references_title = "## 参考资料" if language == "zh" else "## References"

    lead = payload.get("lead") if isinstance(payload.get("lead"), dict) else {}
    overview = str(lead.get("overview_md") or "").strip() if isinstance(lead, dict) else ""
    key_points = lead.get("key_points") if isinstance(lead, dict) else []
    agents = payload.get("agents") if isinstance(payload.get("agents"), list) else []
    citation_by_id, index_by_id = _reference_lookup(payload)

    sections: list[str] = [title]

    if overview:
        sections.append(f"{summary_title}\n{overview}")

    key_point_lines = _render_claim_list(
        key_points,
        citation_by_id,
        index_by_id,
        language=language,
    )
    if key_point_lines:
        sections.append(f"{key_points_title}\n" + "\n".join(key_point_lines))

    agent_sections: list[str] = []
    for agent in agents:
        if not isinstance(agent, dict):
            continue
        task = str(agent.get("task") or "").strip()
        summary = str(agent.get("summary") or "").strip()
        claim_lines = _render_claim_list(
            agent.get("claims"),
            citation_by_id,
            index_by_id,
            language=language,
        )
        body_parts: list[str] = []
        if summary and not _looks_like_json_blob(summary):
            body_parts.append(summary)
        if claim_lines:
            body_parts.append("\n".join(claim_lines))
        if not task or not body_parts:
            continue
        agent_sections.append(f"### {task}\n" + "\n\n".join(body_parts))

    if agent_sections:
        sections.append(f"{notes_title}\n" + "\n\n".join(agent_sections))

    references: list[str] = []
    for citation_id, index in sorted(index_by_id.items(), key=lambda item: item[1]):
        citation = citation_by_id[citation_id]
        url = str(citation.get("url") or citation.get("canonical_url") or "").strip()
        if not url:
            continue
        label = _reference_label(citation, index, language=language)
        publisher = str(citation.get("publisher") or "").strip()
        suffix = f" - {publisher}" if publisher and publisher != label else ""
        references.append(f"{index}. [{label}]({url}){suffix}")
    if references:
        sections.append(f"{references_title}\n" + "\n".join(references))

    article = "\n\n".join(section.strip() for section in sections if section.strip()).strip()
    if not article:
        raise ValueError("deepfind-cli JSON output did not contain any renderable content")
    return article


def ask_deepfind(
    deepfind_dir: Path,
    prompt: str,
    *,
    language: str | None,
    use_gpu: bool,
    num_agent: int,
    max_iter_per_agent: int,
    timeout_sec: int,
    retries: int,
    env: dict[str, str] | None = None,
) -> DeepFindRun:
    cmd = [
        "uv",
        "run",
        "python",
        "-c",
        DEEPFIND_JSON_SNIPPET,
        "1" if use_gpu else "0",
        str(num_agent),
        str(max_iter_per_agent),
    ]
    command_env = build_deepfind_env(env)
    prompt_bytes = prompt.encode("utf-8")

    last_err = ""
    max_attempt = max(1, retries + 1)
    for attempt in range(1, max_attempt + 1):
        try:
            completed = run_command(
                cmd,
                cwd=deepfind_dir,
                timeout_sec=timeout_sec,
                env=command_env,
                stdin_bytes=prompt_bytes,
            )
        except subprocess.TimeoutExpired:
            last_err = f"timeout after {timeout_sec}s"
            print(f"    attempt {attempt}/{max_attempt} failed: {last_err}", file=sys.stderr, flush=True)
            continue

        stdout = decode_command_output(completed.stdout, language=language).strip()
        stderr = decode_command_output(completed.stderr, language=language).strip()
        if completed.returncode == 0 and stdout:
            try:
                payload = _extract_json_object(stdout)
                article = render_markdown_report(payload, language=language)
                return DeepFindRun(article=article, payload=payload, raw_output=stdout)
            except Exception as exc:
                last_err = f"invalid JSON output: {exc}"
                print(
                    f"    attempt {attempt}/{max_attempt} failed: {last_err}",
                    file=sys.stderr,
                    flush=True,
                )
                continue

        last_err = f"exit={completed.returncode}; stderr={stderr[:300]}"
        print(f"    attempt {attempt}/{max_attempt} failed: {last_err}", file=sys.stderr, flush=True)

    raise RuntimeError(f"deepfind query failed after {max_attempt} attempts: {last_err}")


def run_checked(cmd: list[str], cwd: Path, *, env: dict[str, str] | None = None) -> None:
    printable = " ".join(cmd)
    print(f"\n$ {printable}", flush=True)
    result = subprocess.run(cmd, cwd=str(cwd), env=env)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed: {printable}")


def main() -> int:
    args = parse_args()

    if args.num_questions <= 0:
        raise ValueError("-n/--num-questions must be > 0")
    if args.num_agent < 1 or args.num_agent > 4:
        raise ValueError("--num-agent must be between 1 and 4")
    if args.max_iter_per_agent < 1:
        raise ValueError("--max-iter-per-agent must be >= 1")

    deepfind_dir = (REPO_ROOT / args.deepfind_dir).resolve()
    if not deepfind_dir.exists():
        raise FileNotFoundError(f"deepfind dir not found: {deepfind_dir}")

    runtime_env = load_runtime_env(REPO_ROOT / ".env")
    query_rows = load_jsonl(DEFAULT_QUERY_FILE)
    total = len(query_rows)
    if args.num_questions > total:
        raise ValueError(f"-n {args.num_questions} is larger than total questions ({total})")

    selected = query_rows[: args.num_questions]
    model_name = args.model_name or (
        f"deepfind-cli-json-lr-gpu-a{args.num_agent}-n{args.num_questions}"
        if args.gpu
        else f"deepfind-cli-json-lr-a{args.num_agent}-n{args.num_questions}"
    )
    raw_data_path = DEFAULT_RAW_DATA_DIR / f"{model_name}.jsonl"
    structured_path = DEFAULT_STRUCTURED_DIR / f"{model_name}.structured.jsonl"
    subset_query_path = REPO_ROOT / "results" / "tmp" / f"{model_name}.query.jsonl"
    completed_query_path = REPO_ROOT / "results" / "tmp" / f"{model_name}.completed.query.jsonl"
    selected_positions = {
        normalize_task_id(task["id"]): index
        for index, task in enumerate(selected, start=1)
        if "id" in task
    }

    print(f"Selected {len(selected)} / {total} questions", flush=True)
    print(f"Using deepfind-cli at: {deepfind_dir}", flush=True)
    print(
        "Deepfind mode: --json --gpu" if args.gpu else "Deepfind mode: --json",
        flush=True,
    )
    print("Deepfind long report mode: on", flush=True)
    print(f"Deepfind agents: {args.num_agent}", flush=True)
    print(f"Output raw data: {raw_data_path}", flush=True)
    print(f"Output structured data: {structured_path}", flush=True)
    print(f"Subset query file: {subset_query_path}", flush=True)
    print(f"Completed query file: {completed_query_path}", flush=True)

    raw_rows_by_id = load_indexed_jsonl(raw_data_path)
    structured_rows_by_id = load_indexed_jsonl(structured_path)

    imported_checkpoints: list[str] = []
    imported_raw_count = 0
    imported_structured_count = 0
    if args.model_name is None:
        for checkpoint_name in discover_prior_checkpoint_names(model_name, args.num_questions):
            checkpoint_raw_path = DEFAULT_RAW_DATA_DIR / f"{checkpoint_name}.jsonl"
            checkpoint_structured_path = DEFAULT_STRUCTURED_DIR / f"{checkpoint_name}.structured.jsonl"
            added_raw = merge_missing_rows(raw_rows_by_id, load_indexed_jsonl(checkpoint_raw_path))
            added_structured = merge_missing_rows(
                structured_rows_by_id,
                load_indexed_jsonl(checkpoint_structured_path),
            )
            if added_raw or added_structured:
                imported_checkpoints.append(checkpoint_name)
                imported_raw_count += added_raw
                imported_structured_count += added_structured

    if imported_checkpoints:
        print(
            "Imported checkpoint data from: " + ", ".join(imported_checkpoints),
            flush=True,
        )
        print(
            f"Recovered raw rows: {imported_raw_count}; structured rows: {imported_structured_count}",
            flush=True,
        )

    persist_outputs(
        raw_data_path=raw_data_path,
        structured_path=structured_path,
        subset_query_path=subset_query_path,
        completed_query_path=completed_query_path,
        query_rows=query_rows,
        selected=selected,
        raw_rows_by_id=raw_rows_by_id,
        structured_rows_by_id=structured_rows_by_id,
    )

    completed_ids = completed_selected_ids(selected, raw_rows_by_id, structured_rows_by_id)
    pending_tasks = [
        task
        for task in selected
        if normalize_task_id(task["id"]) not in completed_ids
    ]

    print(f"Already completed: {len(completed_ids)} / {len(selected)}", flush=True)
    print(f"Remaining to generate: {len(pending_tasks)}", flush=True)

    for task in pending_tasks:
        task_id = task["id"]
        prompt = task["prompt"]
        language = task.get("language")
        task_pos = selected_positions.get(normalize_task_id(task_id), "?")
        print(f"[{task_pos}/{len(selected)}] id={task_id}", flush=True)

        result = ask_deepfind(
            deepfind_dir=deepfind_dir,
            prompt=prompt,
            language=language,
            use_gpu=args.gpu,
            num_agent=args.num_agent,
            max_iter_per_agent=args.max_iter_per_agent,
            timeout_sec=args.timeout_sec,
            retries=args.retries,
            env=runtime_env,
        )
        task_key = normalize_task_id(task_id)
        raw_rows_by_id[task_key] = {
            "id": task_id,
            "prompt": prompt,
            "article": result.article,
        }
        structured_rows_by_id[task_key] = {
            "id": task_id,
            "prompt": prompt,
            "payload": result.payload,
        }
        persist_outputs(
            raw_data_path=raw_data_path,
            structured_path=structured_path,
            subset_query_path=subset_query_path,
            completed_query_path=completed_query_path,
            query_rows=query_rows,
            selected=selected,
            raw_rows_by_id=raw_rows_by_id,
            structured_rows_by_id=structured_rows_by_id,
        )
        current_done = len(completed_selected_ids(selected, raw_rows_by_id, structured_rows_by_id))
        print(f"    checkpoint saved ({current_done}/{len(selected)})", flush=True)

    print(
        f"\nAvailable answers: {len(completed_selected_ids(selected, raw_rows_by_id, structured_rows_by_id))} / {len(selected)}",
        flush=True,
    )

    if args.skip_eval:
        print("Skip evaluation because --skip-eval is set.", flush=True)
        return 0

    race_output_dir = REPO_ROOT / "results" / "race" / model_name
    fact_output_dir = REPO_ROOT / "results" / "fact" / model_name
    race_output_dir.mkdir(parents=True, exist_ok=True)
    fact_output_dir.mkdir(parents=True, exist_ok=True)

    run_checked(
        [
            sys.executable,
            "-u",
            "deepresearch_bench_race.py",
            model_name,
            "--raw_data_dir",
            str(DEFAULT_RAW_DATA_DIR),
            "--max_workers",
            str(args.max_workers),
            "--query_file",
            str(subset_query_path),
            "--output_dir",
            str(race_output_dir),
            "--force",
        ],
        cwd=REPO_ROOT,
        env=runtime_env,
    )

    run_checked(
        [
            sys.executable,
            "-u",
            "-m",
            "utils.extract",
            "--raw_data_path",
            str(raw_data_path),
            "--output_path",
            str(fact_output_dir / "extracted.jsonl"),
            "--query_data_path",
            str(subset_query_path),
            "--n_total_process",
            str(args.max_workers),
        ],
        cwd=REPO_ROOT,
        env=runtime_env,
    )
    run_checked(
        [
            sys.executable,
            "-u",
            "-m",
            "utils.deduplicate",
            "--raw_data_path",
            str(fact_output_dir / "extracted.jsonl"),
            "--output_path",
            str(fact_output_dir / "deduplicated.jsonl"),
            "--query_data_path",
            str(subset_query_path),
            "--n_total_process",
            str(args.max_workers),
        ],
        cwd=REPO_ROOT,
        env=runtime_env,
    )
    run_checked(
        [
            sys.executable,
            "-u",
            "-m",
            "utils.scrape",
            "--raw_data_path",
            str(fact_output_dir / "deduplicated.jsonl"),
            "--output_path",
            str(fact_output_dir / "scraped.jsonl"),
            "--n_total_process",
            str(args.max_workers),
        ],
        cwd=REPO_ROOT,
        env=runtime_env,
    )
    run_checked(
        [
            sys.executable,
            "-u",
            "-m",
            "utils.validate",
            "--raw_data_path",
            str(fact_output_dir / "scraped.jsonl"),
            "--output_path",
            str(fact_output_dir / "validated.jsonl"),
            "--query_data_path",
            str(subset_query_path),
            "--n_total_process",
            str(args.max_workers),
        ],
        cwd=REPO_ROOT,
        env=runtime_env,
    )
    run_checked(
        [
            sys.executable,
            "-u",
            "-m",
            "utils.stat",
            "--input_path",
            str(fact_output_dir / "validated.jsonl"),
            "--output_path",
            str(fact_output_dir / "fact_result.txt"),
        ],
        cwd=REPO_ROOT,
        env=runtime_env,
    )

    print("\nBenchmark finished.", flush=True)
    print(f"RACE result: {race_output_dir / 'race_result.txt'}", flush=True)
    print(f"FACT result: {fact_output_dir / 'fact_result.txt'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
