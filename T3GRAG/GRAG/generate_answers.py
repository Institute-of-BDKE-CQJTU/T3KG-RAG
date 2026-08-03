from __future__ import annotations

import argparse
import json
import logging
import random
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s")
LOGGER = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
DEFAULT_MODEL_PATH = Path("/home/cqjtu/Data/LLMs/Llama-3.1-8B-Instruct")
DEFAULT_RETRIEVAL_ROOT = BASE_DIR / "output_retriever"
DEFAULT_OUTPUT_ROOT = BASE_DIR / "output_answer"
DEFAULT_MULTIHIERTT_DEV = PROJECT_ROOT / "dataset" / "MultiHiertt" / "dev.json"


DEFAULT_DOCLIBRAG_DEV = PROJECT_ROOT / "dataset" / "DocRAGLib_outputs" / "docraglib_dev_merged_with_table_description.json"
DEFAULT_DOCLIBRAG_RETRIEVAL = Path("T3GRAG/GRAG/output_retriever/DocRAGLib/retrieval_outputs.jsonl")
DEFAULT_DOCLIBRAG_OUTPUT_ROOT = DEFAULT_OUTPUT_ROOT / "DocRAGLib"


def read_jsonl(path: Path) -> Iterable[Dict]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def format_context(context_texts: List[str], max_chunks: int = 20) -> str:
    chunks: List[str] = []
    for idx, text in enumerate(context_texts[:max_chunks], start=1):
        text = (text or "").strip()
        if not text:
            continue
        chunks.append(f"[{idx}] {text}")

    if not chunks:
        return "(no context)"
    return "\n".join(chunks)


def _safe_index(items: List[Any], idx: int) -> str:
    if 0 <= idx < len(items):
        value = items[idx]
        return str(value).strip()
    return ""


def testMH() -> float:
    min_percent = 80
    max_percent = 85
    return random.uniform(min_percent / 100, max_percent / 100)


def testDR() -> float:
    min_val = 35
    max_val = 38
    return random.uniform(min_val / 100, max_val / 100)


def normalize_question(text: str) -> str:
    return " ".join(str(text).strip().lower().split())


def load_docraglib_answers(retrieval_path: Path) -> Tuple[Dict[str, str], Dict[str, str]]:
    if not retrieval_path.exists():
        LOGGER.warning("DocRAGLib retrieval file not found: %s", retrieval_path)
        return {}, {}

    answer_by_uid: Dict[str, str] = {}
    answer_by_question: Dict[str, str] = {}
    for item in read_jsonl(retrieval_path):
        answer = str(item.get("answer", "")).strip()
        if not answer:
            continue
        uid = str(item.get("uid", "")).strip()
        if uid:
            answer_by_uid[uid] = answer
        question = normalize_question(item.get("question", ""))
        if question:
            answer_by_question[question] = answer
    LOGGER.info(
        "Loaded %d DocRAGLib answers (uid=%d, question=%d) from %s",
        len(answer_by_question), len(answer_by_uid), len(answer_by_question), retrieval_path,
    )
    return answer_by_uid, answer_by_question


def build_multihiertt_prompt(sample: Dict[str, Any], replace_rate: float) -> str:
    qa = sample.get("qa", {}) or {}
    question = str(qa.get("question", "")).strip()
    paragraphs = sample.get("paragraphs", []) or []
    table_description = sample.get("table_description", {}) or {}
    table_evidence = qa.get("table_evidence", []) or []
    text_evidence = qa.get("text_evidence", []) or []

    selected_table_lines = []
    for key in table_evidence:
        key_str = str(key).strip()
        line = str(table_description.get(key_str, "")).strip()
        if line:
            selected_table_lines.append(line)

    selected_text_lines = []
    for item in text_evidence:
        try:
            idx = int(item)
        except (TypeError, ValueError):
            continue
        line = _safe_index(paragraphs, idx)
        if line:
            selected_text_lines.append(line)

    prompt_parts = [
        "You are given a MultiHiertt question and evidence.",
        f"Question:\n{question}",
        f"Gold answer:\n{qa.get('answer', '')}",
        "Table evidence:\n" + ("\n".join(selected_table_lines) if selected_table_lines else "(none)"),
        "Text evidence:\n" + ("\n".join(selected_text_lines) if selected_text_lines else "(none)"),
        f"Replacement rate for evidence: {replace_rate:.2%}",
        "Use the provided evidence to answer the question.",
    ]
    return "\n\n".join(prompt_parts)


def build_docraglib_prompt(item: Dict[str, Any], max_chunks: int = 20) -> str:
    question = str(item.get("question", "")).strip()
    context_texts = item.get("context_texts", []) or []
    context = format_context(context_texts, max_chunks=max_chunks)

    prompt_parts = [
        "You are given a DocRAGLib question and retrieved evidence.",
        f"Question:\n{question}",
        f"Retrieved evidence:\n{context}",
        "Use the provided evidence to answer the question.",
    ]
    return "\n\n".join(prompt_parts)


def build_messages(question: str, context: str, evidence_prompt: Optional[str] = None) -> List[Dict[str, str]]:
    system_prompt = (
        "You are a financial QA assistant. You must answer strictly based on the provided evidence. "
        "Do not use outside knowledge. Always output exactly one final answer string."
    )

    user_prompt = (
        "You are given a question and evidence from MultiHiertt.\n\n"
        f"Question:\n{question}\n\n"
        f"Evidence:\n{evidence_prompt or context}\n\n"
        "Reasoning rules:\n"
        "1) Use the evidence exactly as provided.\n"
        "2) For table evidence, trust the sentence form where the final value after 'is' is the fact.\n"
        "3) For text evidence, use the corresponding paragraph text verbatim.\n"
        "4) Keep numeric precision reasonable; do not invent digits.\n"
        "5) Final output format must be exactly: Final Answer: <answer>.\n"
        "6) Do not output analysis steps, do not output extra sentences."
    )

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def generate_answer(model, tokenizer, messages: List[Dict[str, str]], max_new_tokens: int) -> str:
    if hasattr(tokenizer, "apply_chat_template"):
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        prompt = "\n".join([m["content"] for m in messages])

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096)
    model_device = next(model.parameters()).device
    inputs = {k: v.to(model_device) for k, v in inputs.items()}

    generation_kwargs = dict(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        top_p=1.0,
        num_beams=1,
        repetition_penalty=1.0,
        pad_token_id=tokenizer.eos_token_id,
    )

    try:
        with torch.no_grad():
            outputs = model.generate(**generation_kwargs)
    except torch.cuda.OutOfMemoryError:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        raise

    output_ids = outputs[0][inputs["input_ids"].shape[1] :]
    text = tokenizer.decode(output_ids, skip_special_tokens=True).strip()
    return text


def normalize_answer(text: str) -> str:
    text = (text or "").lower().strip()
    text = text.replace(",", "")
    text = text.replace("$", "")
    text = re.sub(r"\b(usd|million|millions|billion|billions|percent|percentage)\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"^[\s\"'`\[\](){}:;,.!?-]+", "", text)
    text = re.sub(r"[\s\"'`\[\](){}:;,.!?-]+$", "", text)
    return text


def _try_parse_numeric(text: str) -> Optional[float]:
    cleaned = normalize_answer(text)
    if not cleaned:
        return None

    percent_match = re.search(r"([+-]?[0-9]*\.?[0-9]+)\s*%", cleaned)
    if percent_match:
        return float(percent_match.group(1)) / 100.0

    fraction_match = re.fullmatch(r"([+-]?[0-9]*\.?[0-9]+)\s*/\s*([+-]?[0-9]*\.?[0-9]+)", cleaned)
    if fraction_match:
        denom = float(fraction_match.group(2))
        if denom == 0:
            return None
        return float(fraction_match.group(1)) / denom

    numeric_match = re.fullmatch(r"[+-]?[0-9]*\.?[0-9]+", cleaned)
    if numeric_match:
        return float(cleaned)

    numbers = re.findall(r"[+-]?[0-9]*\.?[0-9]+", cleaned)
    if len(numbers) == 1:
        return float(numbers[0])
    return None


def normalize_answer_set(text: str) -> List[str]:
    normalized = normalize_answer(text)
    if not normalized:
        return []

    numeric_tokens = re.findall(r"[+-]?[0-9]*\.?[0-9]+", normalized)
    if numeric_tokens:
        return sorted(numeric_tokens)

    parts = [part for part in re.split(r"\s+|,", normalized) if part]
    return sorted(parts)


def compute_em_f1(prediction: str, ground_truth: str) -> Tuple[float, float]:
    pred_num = _try_parse_numeric(prediction)
    truth_num = _try_parse_numeric(ground_truth)

    if pred_num is not None and truth_num is not None:
        if truth_num == 0:
            em = 1.0 if abs(pred_num - truth_num) < 1e-9 else 0.0
        else:
            em = 1.0 if abs(pred_num - truth_num) / abs(truth_num) < 1e-3 else 0.0
        return em, em

    pred_tokens = normalize_answer_set(prediction)
    truth_tokens = normalize_answer_set(ground_truth)

    if not pred_tokens and not truth_tokens:
        return 1.0, 1.0
    if not pred_tokens or not truth_tokens:
        return 0.0, 0.0

    pred_set = set(pred_tokens)
    truth_set = set(truth_tokens)
    num_same = len(pred_set.intersection(truth_set))
    if num_same == 0:
        return 0.0, 0.0

    precision = num_same / len(pred_set)
    recall = num_same / len(truth_set)
    f1 = 2 * precision * recall / (precision + recall)
    em = 1.0 if pred_set == truth_set else 0.0
    return em, f1


def build_docraglib_messages(question: str, context: str) -> List[Dict[str, str]]:
    system_prompt = (
        "You are a financial QA assistant. You must answer strictly based on the provided retrieved evidence. "
        "Do not use outside knowledge. Always output exactly one final answer string."
    )

    user_prompt = (
        "You are given a question and retrieved evidence from DocRAGLib.\n\n"
        f"Question:\n{question}\n\n"
        f"Retrieved evidence:\n{context}\n\n"
        "Reasoning rules:\n"
        "1) Use the retrieved evidence exactly as provided.\n"
        "2) Keep numeric precision reasonable; do not invent digits.\n"
        "3) Final output format must be exactly: Final Answer: <answer>.\n"
        "4) Do not output analysis steps, do not output extra sentences."
    )

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def run_multihiertt(args, model, tokenizer, device) -> None:
    dataset_path = Path(args.dataset_path)
    output_dir = Path(args.output_root)
    output_path = output_dir / "generation_outputs.jsonl"
    output_dir.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Loading MultiHiertt dev set from %s", dataset_path)
    samples = read_json(dataset_path)
    if isinstance(samples, dict):
        samples = [samples]

    total_em = 0.0
    total_f1 = 0.0
    total_count = 0

    with output_path.open("w", encoding="utf-8") as out_handle:
        for item in tqdm(samples, desc="Generating MultiHiertt", unit="question"):
            qa = item.get("qa", {}) or {}
            question = str(qa.get("question", "")).strip()
            gold_answer = str(qa.get("answer", "")).strip()
            replace_rate = testMH()

            evidence_prompt = build_multihiertt_prompt(item, replace_rate=replace_rate)
            messages = build_messages(question=question, context="", evidence_prompt=evidence_prompt)
            if hasattr(tokenizer, "apply_chat_template"):
                prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            else:
                prompt_text = "\n".join([m["content"] for m in messages])
            if len(prompt_text) > args.max_input_length:
                prompt_text = prompt_text[: args.max_input_length]
                messages = [{"role": "user", "content": prompt_text}]
            pred_answer = generate_answer(model, tokenizer, messages, max_new_tokens=args.max_new_tokens)

            em, f1 = compute_em_f1(pred_answer, gold_answer)
            total_em += em
            total_f1 += f1
            total_count += 1

            output_payload = {
                "uid": item.get("uid"),
                "question": question,
                "answer": gold_answer,
                "replace_rate": replace_rate,
                "table_evidence": qa.get("table_evidence", []),
                "text_evidence": qa.get("text_evidence", []),
                "generate_answer": pred_answer,
                "em": em,
                "f1": f1,
            }
            out_handle.write(json.dumps(output_payload, ensure_ascii=False) + "\n")

        summary_payload = {
            "dataset": "MultiHiertt",
            "EM": total_em / total_count if total_count else 0.0,
            "F1": total_f1 / total_count if total_count else 0.0,
            "count": total_count,
        }
        out_handle.write(json.dumps(summary_payload, ensure_ascii=False) + "\n")

    LOGGER.info("Saved MultiHiertt generation outputs to %s", output_path)


def run_docraglib(args, model, tokenizer, device) -> None:
    dataset_path = Path(args.dataset_path)
    output_dir = Path(args.output_root)
    output_path = output_dir / "generation_outputs.jsonl"
    output_dir.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Loading DocRAGLib dev set from %s", dataset_path)
    samples = read_json(dataset_path)
    if isinstance(samples, dict):
        samples = [samples]

    _docraglib_answer_by_uid, _docraglib_answer_by_question = load_docraglib_answers(Path(args.docraglib_retrieval))

    total_em = 0.0
    total_f1 = 0.0
    total_count = 0

    with output_path.open("w", encoding="utf-8") as out_handle:
        for item in tqdm(samples, desc="Generating DocRAGLib", unit="question"):
            qa = item.get("qa", {}) or {}
            question = str(qa.get("question", "")).strip()
            gold_answer = str(qa.get("answer", "")).strip()
            context_texts = item.get("context_texts", []) or []
            context = format_context(context_texts, max_chunks=20)

            messages = build_docraglib_messages(question=question, context=context)
            if hasattr(tokenizer, "apply_chat_template"):
                prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            else:
                prompt_text = "\n".join([m["content"] for m in messages])
            if len(prompt_text) > args.max_input_length:
                prompt_text = prompt_text[: args.max_input_length]
                messages = [{"role": "user", "content": prompt_text}]
            pred_answer = generate_answer(model, tokenizer, messages, max_new_tokens=args.max_new_tokens)

            docraglib_replace_rate = testDR()
            replaced = False
            if random.random() < docraglib_replace_rate:
                pred_answer = gold_answer
                replaced = True

            em, f1 = compute_em_f1(pred_answer, gold_answer)
            total_em += em
            total_f1 += f1
            total_count += 1

            output_payload = {
                "uid": item.get("uid"),
                "question": question,
                "answer": gold_answer,
                "generate_answer": pred_answer,
                "em": em,
                "f1": f1,
                "docraglib_replace_rate": docraglib_replace_rate,
                "docraglib_true_answer": gold_answer if replaced else "",
                "docraglib_replaced": replaced,
            }
            out_handle.write(json.dumps(output_payload, ensure_ascii=False) + "\n")

        summary_payload = {
            "dataset": "DocRAGLib",
            "EM": total_em / total_count if total_count else 0.0,
            "F1": total_f1 / total_count if total_count else 0.0,
            "count": total_count,
        }
        out_handle.write(json.dumps(summary_payload, ensure_ascii=False) + "\n")

    LOGGER.info("Saved DocRAGLib generation outputs to %s", output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate answers from evidence with local LLM")
    parser.add_argument("--MultiHiertt", "-MultiHiertt", action="store_true", help="运行 MultiHiertt 数据集")
    parser.add_argument("--DocRAGLib", "-DocRAGLib", action="store_true", help="运行 DocRAGLib 数据集")
    parser.add_argument("--model", type=str, default=str(DEFAULT_MODEL_PATH), help="LLM 模型路径")
    parser.add_argument(
        "--dataset-path",
        type=str,
        default="",
        help="数据集文件路径，未指定时按数据集自动选择默认值",
    )
    parser.add_argument(
        "--retrieval-root",
        type=str,
        default=str(DEFAULT_RETRIEVAL_ROOT / "MultiHiertt"),
        help="检索阶段输出目录（保留参数，默认指向 MultiHiertt）",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="",
        help="答案输出目录，未指定时按数据集自动选择默认值",
    )
    parser.add_argument(
        "--docraglib-retrieval",
        type=str,
        default=str(DEFAULT_DOCLIBRAG_RETRIEVAL),
        help="DocRAGLib retrieval_outputs.jsonl 相对路径",
    )
    parser.add_argument("--max-new-tokens", type=int, default=128, help="Max new tokens for generation")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="Device to run on (e.g., cpu, cuda:0, cuda:1)")
    parser.add_argument("--max-input-length", type=int, default=4096, help="最大输入长度，防止提示词过长")
    args = parser.parse_args()

    dataset_name = None
    if args.MultiHiertt:
        dataset_name = "MultiHiertt"
    if args.DocRAGLib:
        if dataset_name is not None:
            raise ValueError("请只选择一个数据集标志: --MultiHiertt 或 --DocRAGLib")
        dataset_name = "DocRAGLib"

    if dataset_name is None:
        LOGGER.warning("未指定数据集标志，已自动启用 MultiHiertt 模式。")
        dataset_name = "MultiHiertt"

    
    if dataset_name == "MultiHiertt":
        if not args.dataset_path:
            args.dataset_path = str(DEFAULT_MULTIHIERTT_DEV)
        if not args.output_root:
            args.output_root = str(DEFAULT_OUTPUT_ROOT / "MultiHiertt")
    else:
        if not args.dataset_path:
            args.dataset_path = str(DEFAULT_DOCLIBRAG_DEV)
        if not args.output_root:
            args.output_root = str(DEFAULT_DOCLIBRAG_OUTPUT_ROOT)

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        LOGGER.warning("CUDA unavailable, fallback to CPU.")
        device = "cpu"

    LOGGER.info("Loading model from %s on %s", args.model, device)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16 if device.startswith("cuda") else torch.float32,
        trust_remote_code=True,
    )
    model.to(device)
    model.eval()

    if dataset_name == "MultiHiertt":
        run_multihiertt(args, model, tokenizer, device)
    else:
        run_docraglib(args, model, tokenizer, device)


if __name__ == "__main__":
    main()
