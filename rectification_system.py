"""
Article Rectification System

Two-pass structured diff approach:
- Pass 1 (_rectify): LLM returns a JSON list of exact find/replace corrections
- Pass 2 (_verify): LLM checks the rectified output and returns any remaining fixes
- Corrections are applied programmatically — the LLM never rewrites the full article
"""

from dotenv import load_dotenv
from litellm import completion
import os
import json
import re

load_dotenv()

API_KEY = os.getenv('LLM_API_KEY')
API_BASE = os.getenv('LLM_API_BASE')
MODEL = "openai/gpt-oss-120b"


def _strip_annotations(text: str) -> str:
    """Remove embedded Error Annotations block if present in AI-generated articles."""
    for marker in ["**Error Annotations:**", "Error Annotations:"]:
        idx = text.find(marker)
        if idx != -1:
            return text[:idx].rstrip()
    return text


def _llm(prompt: str) -> str:
    response = completion(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        api_key=API_KEY,
        api_base=API_BASE
    )
    return response.choices[0].message.content.strip()


def _parse_json(text: str) -> list:
    """Extract and parse a JSON array from LLM output, stripping markdown fences if present."""
    text = re.sub(r'```(?:json)?\s*', '', text).strip().strip('`').strip()
    try:
        result = json.loads(text)
        return result if isinstance(result, list) else []
    except json.JSONDecodeError:
        return []


def _apply_corrections(text: str, corrections: list) -> str:
    """Apply find/replace corrections to text. Skips any 'find' not present in text."""
    for item in corrections:
        find = item.get('find', '')
        replace = item.get('replace', '')
        if not find:
            continue
        if find in text:
            text = text.replace(find, replace)
        else:
            print(f"  [WARN] Substring not found, skipping: '{find[:80]}'")
    return text


def _rectify(ai_generated_content: str, source_content: str) -> list:
    """
    Pass 1: identify all factual errors and return them as a structured JSON diff.
    """
    prompt = (
        "You are a precise fact-checking editor.\n\n"
        "Compare the AI-GENERATED ARTICLE against the SOURCE ARTICLE and identify every factual error.\n\n"
        "Return a JSON array of corrections in this exact format:\n"
        '  [{"find": "<exact verbatim substring from the AI-generated article>", "replace": "<corrected text>"}, ...]\n\n'
        "STRICT RULES:\n"
        "1. 'find' must be copied verbatim — character for character — from the AI-generated article.\n"
        "2. Use the shortest 'find' string that uniquely identifies the wrong fact.\n"
        "3. Only fix facts that directly contradict the source (wrong numbers, names, dates, specs).\n"
        "4. Do NOT change phrasing, sentence structure, headings, or style.\n"
        "5. If there are no errors, return exactly: []\n"
        "6. Return ONLY valid JSON — no explanation, no markdown code blocks.\n\n"
        "---SOURCE ARTICLE (ground truth)---\n"
        f"{source_content}\n\n"
        "---AI-GENERATED ARTICLE---\n"
        f"{ai_generated_content}\n\n"
        "---JSON CORRECTIONS---"
    )
    raw = _llm(prompt)
    return _parse_json(raw)


def _verify(rectified_content: str, source_content: str) -> list:
    """
    Pass 2: verify the rectified article and return any remaining factual errors as a JSON diff.
    """
    prompt = (
        "You are a strict fact-checker performing a final verification pass.\n\n"
        "Compare the RECTIFIED ARTICLE against the SOURCE ARTICLE and identify any remaining factual errors.\n\n"
        "Return a JSON array of corrections in this exact format:\n"
        '  [{"find": "<exact verbatim substring from the rectified article>", "replace": "<corrected text>"}, ...]\n\n'
        "STRICT RULES:\n"
        "1. 'find' must be copied verbatim — character for character — from the rectified article.\n"
        "2. Use the shortest 'find' string that uniquely identifies the wrong fact.\n"
        "3. Only fix facts that still contradict the source — do NOT touch headings, phrasing, or structure.\n"
        "4. If no errors remain, return exactly: []\n"
        "5. Return ONLY valid JSON — no explanation, no markdown code blocks.\n\n"
        "---SOURCE ARTICLE (ground truth)---\n"
        f"{source_content}\n\n"
        "---RECTIFIED ARTICLE---\n"
        f"{rectified_content}\n\n"
        "---JSON CORRECTIONS---"
    )
    raw = _llm(prompt)
    return _parse_json(raw)


def run(ai_generated_content: str, source_content: str) -> str:
    """
    Rectify an AI-generated article using a two-pass structured diff approach.

    Pass 1 - Rectify: LLM returns JSON corrections, applied programmatically.
    Pass 2 - Verify:  LLM checks the result and returns any remaining fixes.

    Args:
        ai_generated_content: The AI-generated article text to be corrected
        source_content: The ground-truth source article

    Returns:
        str: The fully verified and rectified article content
    """
    ai_generated_content = _strip_annotations(ai_generated_content)

    corrections = _rectify(ai_generated_content, source_content)
    print(f"  [Pass 1] {len(corrections)} correction(s) found")
    rectified = _apply_corrections(ai_generated_content, corrections)

    remaining = _verify(rectified, source_content)
    print(f"  [Pass 2] {len(remaining)} remaining issue(s) found")
    verified = _apply_corrections(rectified, remaining)

    return verified
