"""
Article Rectification System

This is where you implement your article rectification logic.
The run() function receives AI-generated content and should return the corrected version.

Feel free to:
- Add additional modules, classes, or helper functions
- Load and compare with source articles
- Implement multi-step validation and correction strategies
- Use multiple LLM calls or different models
- Add confidence scoring and logging
"""

from dotenv import load_dotenv
from litellm import completion
import os

load_dotenv()

API_KEY = os.getenv('LLM_API_KEY')
API_BASE = os.getenv('LLM_API_BASE')
MODEL = "openai/gpt-oss-120b"


def _llm(prompt: str) -> str:
    response = completion(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        api_key=API_KEY,
        api_base=API_BASE
    )
    return response.choices[0].message.content.strip()


def _rectify(ai_generated_content: str, source_content: str) -> str:
    prompt = (
        "You are a precise fact-checking editor. Your job is to correct factual errors in an AI-generated article by cross-referencing it with the provided source material.\n\n"
        "STRICT RULES:\n"
        "1. Fix ONLY facts that directly contradict the source (wrong numbers, names, dates, specifications, etc.).\n"
        "2. Do NOT rewrite, rephrase, or restructure any sentence. Change only the minimum words needed.\n"
        "3. Do NOT add new information not present in the AI-generated article.\n"
        "4. Do NOT remove any content from the AI-generated article.\n"
        "5. Preserve all formatting, headings, and structure exactly as-is.\n"
        "6. Return ONLY the corrected article text — no explanations, no markdown code blocks.\n\n"
        "---SOURCE ARTICLE (ground truth)---\n"
        f"{source_content}\n\n"
        "---AI-GENERATED ARTICLE (to be corrected)---\n"
        f"{ai_generated_content}\n\n"
        "---CORRECTED ARTICLE---"
    )
    return _llm(prompt)


def _verify(rectified_content: str, source_content: str) -> str:
    """
    Second pass: compare the rectified article against the source.
    If errors remain, fix them and return the corrected version.
    If everything is accurate, return the article unchanged.
    """
    prompt = (
        "You are a strict fact-checker performing a final verification pass.\n\n"
        "Compare the RECTIFIED ARTICLE against the SOURCE ARTICLE fact by fact.\n\n"
        "STRICT RULES:\n"
        "1. If you find any fact in the rectified article that still contradicts the source, fix ONLY that fact with the minimum word change possible.\n"
        "2. Do NOT rewrite, rephrase, or restructure any sentence.\n"
        "3. Do NOT add new information not present in the rectified article.\n"
        "4. Do NOT remove any content from the rectified article.\n"
        "5. Preserve all formatting, headings, and structure exactly as-is.\n"
        "6. Return ONLY the final article text — no explanations, no markdown code blocks.\n\n"
        "---SOURCE ARTICLE (ground truth)---\n"
        f"{source_content}\n\n"
        "---RECTIFIED ARTICLE (to be verified)---\n"
        f"{rectified_content}\n\n"
        "---FINAL VERIFIED ARTICLE---"
    )
    return _llm(prompt)


def run(ai_generated_content: str, source_content: str) -> str:
    """
    Rectify an AI-generated article using a two-pass approach:
    Pass 1 - Rectify: fix factual errors against the source.
    Pass 2 - Verify: catch any remaining errors missed in pass 1.

    Args:
        ai_generated_content: The AI-generated article text to be corrected
        source_content: The ground-truth source article

    Returns:
        str: The fully verified and rectified article content
    """
    rectified = _rectify(ai_generated_content, source_content)
    verified = _verify(rectified, source_content)
    return verified
