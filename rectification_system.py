"""
Article Rectification System

Two-pass structured diff approach:
- Pass 1 (_rectify): LLM returns a JSON list of exact find/replace corrections
- Pass 2 (_verify): LLM checks the rectified output and returns any remaining fixes
- Corrections are applied programmatically — the LLM never rewrites the full article

Guards:
- Max-length guard: if a 'find' string exceeds MAX_FIND_LENGTH, it is decomposed into
  multiple shorter surgical corrections via _breakdown_long_correction() instead of skipped
- Content-loss guard: if a pass removes more than CONTENT_LOSS_THRESHOLD of the article,
  that pass is discarded and the previous version is kept
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

MAX_FIND_LENGTH = 120          # Max characters for a 'find' string before decomposition
CONTENT_LOSS_THRESHOLD = 0.05  # Max allowed content reduction per pass (5%)


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


def _breakdown_long_correction(find: str, replace: str) -> list:
    """
    Decompose a long find/replace correction into multiple short surgical pairs.
    Called when a correction's 'find' string exceeds MAX_FIND_LENGTH.
    """
    prompt = (
        "A fact-checker found a text segment containing multiple errors.\n\n"
        "WRONG VERSION (from article):\n"
        f"{find}\n\n"
        "CORRECT VERSION (what it should say):\n"
        f"{replace}\n\n"
        "Identify ONLY the specific words or phrases that differ between the two versions "
        "and return them as minimal surgical find/replace pairs.\n\n"
        "Return a JSON array:\n"
        '[{"find": "<exact wrong word or short phrase>", "replace": "<correct word or short phrase>"}, ...]\n\n'
        "RULES:\n"
        "1. Each 'find' must be the shortest unique substring that identifies the wrong fact.\n"
        "2. Do NOT include correct surrounding words in 'find' unless needed for uniqueness.\n"
        "3. Do NOT rewrite whole sentences — only the wrong words.\n"
        "4. Return ONLY valid JSON — no explanation, no markdown.\n"
    )
    raw = _llm(prompt)
    subs = _parse_json(raw)
    if subs:
        print(f"    [BREAK] Decomposed into {len(subs)} sub-correction(s)")
    else:
        print(f"    [BREAK] Could not decompose — skipping")
    return subs


def _apply_corrections(text: str, corrections: list) -> str:
    """
    Apply find/replace corrections to text.
    - If 'find' exceeds MAX_FIND_LENGTH, decomposes it via _breakdown_long_correction()
    - Skips corrections where 'find' is not found in the text
    """
    for item in corrections:
        find = item.get('find', '')
        replace = item.get('replace', '')
        if not find:
            continue

        if len(find) > MAX_FIND_LENGTH:
            print(f"  [LONG]  {len(find)} chars — decomposing: '{find[:70]}...'")
            sub_corrections = _breakdown_long_correction(find, replace)
            for sub in sub_corrections:
                sub_find = sub.get('find', '')
                sub_replace = sub.get('replace', '')
                if not sub_find:
                    continue
                if sub_find in text:
                    text = text.replace(sub_find, sub_replace)
                else:
                    print(f"    [WARN] Sub not found: '{sub_find[:60]}'")
            continue

        if find in text:
            text = text.replace(find, replace)
        else:
            print(f"  [WARN] Substring not found, skipping: '{find[:80]}'")
    return text


def _content_loss_guard(original: str, corrected: str, pass_name: str) -> str:
    """
    Discard a pass if it removed more than CONTENT_LOSS_THRESHOLD of the original content.
    Returns corrected if safe, original if too much content was lost.
    """
    if not original:
        return corrected
    loss = (len(original) - len(corrected)) / len(original)
    if loss > CONTENT_LOSS_THRESHOLD:
        print(f"  [GUARD] {pass_name} removed {loss*100:.1f}% of content — discarding, keeping previous version")
        return original
    return corrected


def _rectify(ai_generated_content: str, source_content: str) -> list:
    """Pass 1: identify all factual errors and return them as a structured JSON diff."""
    prompt = (
        "You are a precise fact-checking editor.\n\n"
        "Compare the AI-GENERATED ARTICLE against the SOURCE ARTICLE and identify every factual error.\n\n"
        "Return a JSON array of corrections in this exact format:\n"
        '  [{"find": "<exact verbatim substring from the AI-generated article>", "replace": "<corrected text>"}, ...]\n\n'
        "STRICT RULES:\n"
        "1. 'find' must be copied verbatim — character for character — from the AI-generated article.\n"
        "2. Use the SHORTEST possible 'find' string that uniquely identifies the wrong fact — ideally just the wrong word(s), not the whole sentence.\n"
        "3. Only fix facts that directly contradict the source (wrong numbers, names, dates, specs).\n"
        "4. Do NOT change phrasing, sentence structure, headings, or style.\n"
        "5. Do NOT remove any sentences or paragraphs — only replace the wrong fact inline.\n"
        "6. If there are no errors, return exactly: []\n"
        "7. Return ONLY valid JSON — no explanation, no markdown code blocks.\n\n"
        "---SOURCE ARTICLE (ground truth)---\n"
        f"{source_content}\n\n"
        "---AI-GENERATED ARTICLE---\n"
        f"{ai_generated_content}\n\n"
        "---JSON CORRECTIONS---"
    )
    raw = _llm(prompt)
    return _parse_json(raw)


def _verify(rectified_content: str, source_content: str) -> list:
    """Pass 2: verify the rectified article and return any remaining factual errors as a JSON diff."""
    prompt = (
        "You are a strict fact-checker performing a final verification pass.\n\n"
        "Compare the RECTIFIED ARTICLE against the SOURCE ARTICLE and identify any remaining factual errors.\n\n"
        "Return a JSON array of corrections in this exact format:\n"
        '  [{"find": "<exact verbatim substring from the rectified article>", "replace": "<corrected text>"}, ...]\n\n'
        "STRICT RULES:\n"
        "1. 'find' must be copied verbatim — character for character — from the rectified article.\n"
        "2. Use the SHORTEST possible 'find' string that uniquely identifies the wrong fact.\n"
        "3. Only fix facts that still contradict the source — do NOT touch headings, phrasing, or structure.\n"
        "4. Do NOT remove any sentences or paragraphs — only replace the wrong fact inline.\n"
        "5. If no errors remain, return exactly: []\n"
        "6. Return ONLY valid JSON — no explanation, no markdown code blocks.\n\n"
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
    Long corrections are decomposed into surgical sub-corrections instead of skipped.
    Both passes are protected by content-loss guard.

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
    rectified = _content_loss_guard(ai_generated_content, rectified, "Pass 1")

    remaining = _verify(rectified, source_content)
    print(f"  [Pass 2] {len(remaining)} remaining issue(s) found")
    verified = _apply_corrections(rectified, remaining)
    verified = _content_loss_guard(rectified, verified, "Pass 2")

    return verified
