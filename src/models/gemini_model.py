"""
Gemini 3.1 Pro model wrapper for WavBench inference.

Uses the OpenAI-compatible REST API via proxy — streaming, base64 audio.
No google-generativeai SDK required — only `requests`.

API pattern (from python_ImageAnalysis.py example):

    API_KEY = "Bearer sk-..."          ← full "Bearer <key>" stored in env var
    headers = {"Authorization": API_KEY, ...}

    payload = {
        "model": "[次]gemini-3.1-pro-preview",
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}},
            {"type": "text",      "text": "请描述这张图片"},
        ]}],
        "stream": True,
    }

    response = requests.post(API_URL, json=payload, headers=headers, stream=True)
    for line in response.iter_lines():
        if line:
            line = line.decode('utf-8').replace('data: ', '')
            if line.strip() == '[DONE]': break
            data = json.loads(line)
            content = data['choices'][0]['delta'].get('content')

Audio is passed the same way as images, using "audio_url" instead of "image_url":
    {"type": "audio_url", "audio_url": {"url": "data:audio/wav;base64,..."}}

Environment variable:
    GOOGLE_API_KEY = "Bearer sk-xxxxxxxxxx"   ← include the "Bearer " prefix
"""

import os
import json
import time
import base64
import logging
import requests
from pathlib import Path
from typing import Optional
from collections import Counter

logger = logging.getLogger(__name__)

# ── API config ─────────────────────────────────────────────────────────────────

API_URL  = "https://once.novai.su/v1/chat/completions"
MODEL_ID = "[次]gemini-3.1-pro-preview"

MIME_MAP = {
    ".wav":  "audio/wav",
    ".mp3":  "audio/mpeg",
    ".flac": "audio/flac",
    ".ogg":  "audio/ogg",
    ".m4a":  "audio/mp4",
    ".aac":  "audio/aac",
}

# ── System prompts — tuned to WavBench scoring rubrics ────────────────────────

SYSTEM_PROMPTS = {
    # ── Colloquial Expression ──────────────────────────────────────────────────
    "colloquial_default": (
        "You are a highly capable spoken-language assistant. "
        "The user will speak to you in natural, colloquial language. "
        "Listen carefully — including filler words, slang, regional expressions, "
        "and incomplete sentences. Understand the intent fully before answering. "
        "Respond in clear, correct prose. Be thorough but concise."
    ),
    "colloquial_code": (
        "You are an expert software engineer. The user asks coding questions "
        "verbally in casual, colloquial speech. Parse their spoken intent carefully "
        "(they may mis-speak variable names or algorithm steps). "
        "Respond with correct, well-commented code and a plain-language explanation. "
        "Think step-by-step before writing any code."
    ),
    "colloquial_math": (
        "You are a mathematics expert. The user poses math problems verbally in "
        "colloquial speech. They may describe numbers imprecisely or use informal "
        "language. Extract the precise mathematical problem and solve it step-by-step, "
        "showing all working. State the final answer clearly."
    ),
    "colloquial_logic": (
        "You are a logical reasoning expert. The user presents reasoning problems "
        "verbally. Listen carefully for premises and conclusions even when stated "
        "informally. Think through the problem rigorously and explain your reasoning "
        "step by step before giving your conclusion."
    ),
    "colloquial_safety": (
        "You are a safe, responsible AI assistant. The user may pose sensitive or "
        "borderline requests verbally. Evaluate the intent carefully. Respond helpfully "
        "where appropriate; politely decline where necessary, always explaining why. "
        "Never generate harmful content."
    ),
    "colloquial_instruction": (
        "You are a precise instruction-following assistant. The user will give you "
        "spoken instructions in colloquial language. Parse each instruction carefully, "
        "including any constraints, formats, or styles they specify verbally. "
        "Follow every instruction exactly."
    ),
    "colloquial_creativity": (
        "You are a creative writing assistant. The user makes creative requests "
        "verbally in casual speech. Interpret their creative vision generously. "
        "Produce vivid, original, high-quality creative content that matches their intent."
    ),
    "colloquial_qa": (
        "You are a knowledgeable general-purpose assistant. The user asks factual "
        "questions verbally in colloquial language. Provide accurate, well-sourced, "
        "concise answers. If uncertain, say so clearly."
    ),
    # ── Acoustic Explicit Understanding ───────────────────────────────────────
    "understanding_accent": (
        "Listen to the speaker's accent in the audio. Identify the most likely "
        "regional or national accent from: American, British, Australian, Indian, "
        "Chinese, French, German, Spanish, Arabic, or Other. "
        "Respond with ONLY the accent label."
    ),
    "understanding_age": (
        "Listen to the speaker's voice in the audio. Estimate their age group: "
        "Child (under 18), Young Adult (18-35), Middle-aged (36-60), or Senior (60+). "
        "Respond with ONLY the age group label."
    ),
    "understanding_emotion": (
        "Listen carefully to the speaker's voice, tone, and prosody in the audio. "
        "Identify the primary emotion being expressed. Choose from: "
        "Happy, Sad, Angry, Fearful, Disgusted, Surprised, Neutral, Contemptuous. "
        "Respond with ONLY the emotion label."
    ),
    "understanding_gender": (
        "Listen to the speaker's voice in the audio. Identify the speaker's gender: "
        "Male or Female. Respond with ONLY the gender label."
    ),
    "understanding_language": (
        "Listen to the spoken language in the audio. Identify which language is being "
        "spoken. Respond with ONLY the language name in English (e.g., 'English', "
        "'Mandarin Chinese', 'Spanish', 'French', 'German', 'Arabic', 'Japanese', "
        "'Korean', 'Portuguese', 'Russian')."
    ),
    "understanding_pitch": (
        "Listen to the speaker's pitch in the audio. Classify it as: "
        "Very Low, Low, Medium, High, or Very High. "
        "Respond with ONLY the pitch label."
    ),
    "understanding_speed": (
        "Listen to the speaker's speaking rate in the audio. Classify it as: "
        "Very Slow, Slow, Normal, Fast, or Very Fast. "
        "Respond with ONLY the speed label."
    ),
    "understanding_volume": (
        "Listen to the speaker's volume in the audio. Classify it as: "
        "Very Quiet, Quiet, Normal, Loud, or Very Loud. "
        "Respond with ONLY the volume label."
    ),
    "understanding_audio": (
        "Listen to the audio and identify the main sound event or acoustic scene. "
        "Describe what you hear in 1-5 words (e.g., 'dog barking', 'rain', "
        "'crowd cheering', 'music playing'). Respond with ONLY the sound description."
    ),
    "understanding_music": (
        "Listen to the music in the audio. Identify the genre. Choose from: "
        "Classical, Jazz, Pop, Rock, Hip-hop, Electronic, R&B, Country, Folk, "
        "Blues, Metal, or Other. Respond with ONLY the genre label."
    ),
    # ── Acoustic Explicit Generation ──────────────────────────────────────────
    "generation_default": (
        "You are generating speech with specific paralinguistic characteristics. "
        "Read the provided text naturally while precisely matching the requested "
        "acoustic attribute. Focus on accuracy of the target attribute above all."
    ),
    # ── Acoustic Implicit ─────────────────────────────────────────────────────
    "implicit_understanding": (
        "You are an expert at understanding spoken dialogue, including the implicit "
        "meaning, subtext, and conversational pragmatics. Listen carefully to the "
        "audio and answer the question based on BOTH what is said AND how it is said "
        "(tone, emotion, prosody). Consider the full conversational context."
    ),
    "implicit_generation": (
        "You are a natural spoken-language dialogue system. Respond to the user's "
        "spoken input in a contextually appropriate, natural way. Match the tone and "
        "register of the conversation. Be helpful and engaged."
    ),
}


# ── Low-level API call — mirrors python_ImageAnalysis.py exactly ──────────────

def chat_completion(
    messages: list,
    api_key: str,
    model: str = MODEL_ID,
    temperature: float = 0.2,
    max_tokens: int = 8192,
) -> str:
    """
    Call the OpenAI-compatible streaming endpoint and return the full response.

    Mirrors the streaming pattern in python_ImageAnalysis.py:

        response = requests.post(API_URL, json=payload, headers=headers, stream=True)
        for line in response.iter_lines():
            if line:
                line = line.decode('utf-8').replace('data: ', '')
                if line.strip() == '[DONE]': break
                data = json.loads(line)
                content = data['choices'][0]['delta'].get('content')

    Args:
        messages : OpenAI-format messages list.
        api_key  : Full auth value, e.g. "Bearer sk-xxxxxxxxxx".
        model    : Model name (default: "[次]gemini-3.1-pro-preview").
        temperature / max_tokens : Optional generation params.
    """
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": True,
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": api_key,          # already includes "Bearer " prefix
    }

    response = requests.post(API_URL, json=payload, headers=headers, stream=True)
    response.raise_for_status()

    chunks: list[str] = []
    for line in response.iter_lines():
        if line:
            line = line.decode('utf-8').replace('data: ', '')
            if line.strip() == '[DONE]':
                break
            try:
                data = json.loads(line)
                if content := data['choices'][0]['delta'].get('content'):
                    chunks.append(content)
            except (json.JSONDecodeError, KeyError, IndexError):
                continue

    return "".join(chunks).strip()


# ── Audio loading ─────────────────────────────────────────────────────────────

def _load_audio_part(audio_path: str) -> dict:
    """
    Load an audio file as a base64 data-URL content part.

    Mirrors the image_url pattern from the example:
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}}

    For audio:
        {"type": "audio_url", "audio_url": {"url": "data:audio/wav;base64,..."}}
    """
    suffix = Path(audio_path).suffix.lower()
    mime   = MIME_MAP.get(suffix, "audio/wav")
    with open(audio_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    return {
        "type": "audio_url",
        "audio_url": {"url": f"data:{mime};base64,{b64}"},
    }


# ── Retry / rate-limit handling ───────────────────────────────────────────────

def _call_with_retry(fn, max_retries: int = 5, base_delay: float = 2.0):
    """Exponential backoff retry for transient HTTP errors."""
    for attempt in range(max_retries):
        try:
            return fn()
        except requests.HTTPError as e:
            status = e.response.status_code if e.response is not None else 0
            if status in (429, 500, 502, 503, 504) and attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt)
                logger.warning(
                    f"HTTP {status} (attempt {attempt+1}/{max_retries}), "
                    f"retrying in {delay:.1f}s…"
                )
                time.sleep(delay)
            else:
                raise
        except (requests.ConnectionError, requests.Timeout) as e:
            if attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt)
                logger.warning(
                    f"Connection error (attempt {attempt+1}/{max_retries}): {e}, "
                    f"retrying in {delay:.1f}s…"
                )
                time.sleep(delay)
            else:
                raise
    raise RuntimeError("Max retries exceeded")


# ── Main model class ──────────────────────────────────────────────────────────

class GeminiWavBenchModel:
    """
    Gemini 3.1 Pro wrapper for WavBench.

    Uses the OpenAI-compatible proxy API with streaming and base64 audio data URLs.
    The public .run() interface dispatches to the correct task method.

    Environment:
        GOOGLE_API_KEY = "Bearer sk-xxxxxxxxxx"   ← include "Bearer " prefix
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_id: str = MODEL_ID,
        temperature: float = 0.2,
        max_output_tokens: int = 8192,
        self_consistency_k: int = 3,
        **kwargs,
    ):
        self.api_key = api_key or os.environ.get("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError(
                "API key not set.\n"
                "Export GOOGLE_API_KEY='Bearer sk-xxxxxxxxxx'  (include 'Bearer ' prefix)"
            )
        self.model_id           = model_id
        self.temperature        = temperature
        self.max_output_tokens  = max_output_tokens
        self.self_consistency_k = self_consistency_k
        logger.info(f"GeminiWavBenchModel ready: {model_id}")

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _call(
        self,
        system_prompt: str,
        user_content: list,
        temperature: Optional[float] = None,
    ) -> str:
        """Build messages and call the API with retry."""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_content},
        ]
        temp = temperature if temperature is not None else self.temperature

        def _fn():
            return chat_completion(
                messages=messages,
                api_key=self.api_key,
                model=self.model_id,
                temperature=temp,
                max_tokens=self.max_output_tokens,
            )
        return _call_with_retry(_fn)

    def _classify_with_voting(
        self, system_prompt: str, user_content: list
    ) -> str:
        """
        Self-consistency voting: run self_consistency_k calls, return majority label.
        Reduces random variance on classification tasks.
        """
        votes: list[str] = []
        for _ in range(self.self_consistency_k):
            try:
                raw   = self._call(system_prompt, user_content)
                label = raw.split("\n")[0].strip().rstrip(".,!?")
                votes.append(label)
            except Exception as e:
                logger.warning(f"Vote failed: {e}")

        if not votes:
            return ""
        norm   = [v.lower() for v in votes]
        winner = Counter(norm).most_common(1)[0][0]
        return next(v for v in votes if v.lower() == winner)

    @staticmethod
    def _text(s: str) -> dict:
        return {"type": "text", "text": s}

    # ── Task methods ──────────────────────────────────────────────────────────

    def run_colloquial(
        self,
        audio_path: str,
        question: Optional[str] = None,
        domain: str = "default",
    ) -> str:
        key    = f"colloquial_{domain}" if f"colloquial_{domain}" in SYSTEM_PROMPTS else "colloquial_default"
        system = SYSTEM_PROMPTS[key]
        audio  = _load_audio_part(audio_path)
        cot    = "Listen carefully to the audio. Think step by step before answering.\n\n"
        body   = (
            f"{cot}The spoken question is: {question}\n\nPlease answer:"
            if question else
            f"{cot}Please listen to the audio and answer the spoken question:"
        )
        return self._call(system, [audio, self._text(body)])

    def run_explicit_understanding(
        self,
        audio_path: str,
        attribute: str,
        question: Optional[str] = None,
    ) -> str:
        key    = f"understanding_{attribute.lower()}"
        if key not in SYSTEM_PROMPTS:
            key = "understanding_audio"
        system = SYSTEM_PROMPTS[key]
        audio  = _load_audio_part(audio_path)
        body   = question or f"What is the {attribute} of this audio?"
        return self._classify_with_voting(system, [audio, self._text(body)])

    def run_explicit_generation(
        self,
        audio_path: str,
        attribute: str,
        target_value: str,
        question: Optional[str] = None,
    ) -> str:
        system = SYSTEM_PROMPTS["generation_default"]
        audio  = _load_audio_part(audio_path)
        if question:
            body = (
                f"Listen to this audio. {question}\n"
                f"Generate a spoken response that clearly exhibits "
                f"{attribute} = '{target_value}'."
            )
        else:
            body = (
                f"Listen to the audio. Respond as if you are speaking with "
                f"{attribute} = '{target_value}'. Make the {attribute} very clear."
            )
        return self._call(system, [audio, self._text(body)])

    def run_implicit(
        self,
        audio_path: str,
        question: Optional[str] = None,
        history: Optional[list] = None,
        task_type: str = "understanding",
    ) -> str:
        key    = f"implicit_{task_type}"
        if key not in SYSTEM_PROMPTS:
            key = "implicit_understanding"
        system = SYSTEM_PROMPTS[key]
        audio  = _load_audio_part(audio_path)
        context = ""
        if history:
            history_str = "\n".join(
                f"[{t['role'].upper()}]: {t['content']}" for t in history
            )
            context = f"Conversation history:\n{history_str}\n\n"
        body = f"{context}[New audio message attached]\n"
        body += f"Question: {question}" if question else "Please respond to the spoken message above."
        return self._call(system, [audio, self._text(body)])

    # ── Unified entry point ───────────────────────────────────────────────────

    def run(
        self,
        audio_path: str,
        task_type: str,
        attribute: Optional[str] = None,
        question: Optional[str] = None,
        domain: Optional[str] = None,
        target_value: Optional[str] = None,
        history: Optional[list] = None,
        **kwargs,
    ) -> str:
        """
        Dispatch to the correct task method.

        Args:
            audio_path   : Absolute path to the WAV file.
            task_type    : 'colloquial' | 'explicit_understanding' |
                           'explicit_generation' | 'implicit'
            attribute    : Paralinguistic attribute (acoustic tasks).
            question     : Spoken question text (optional).
            domain       : Cognitive domain (colloquial tasks).
            target_value : Target attribute value (generation tasks).
            history      : Prior turns for multi-round implicit tasks.
        """
        if task_type == "colloquial":
            return self.run_colloquial(
                audio_path, question=question, domain=domain or "default"
            )
        elif task_type == "explicit_understanding":
            return self.run_explicit_understanding(
                audio_path, attribute=attribute or "audio", question=question
            )
        elif task_type == "explicit_generation":
            return self.run_explicit_generation(
                audio_path,
                attribute=attribute or "emotion",
                target_value=target_value or "",
                question=question,
            )
        elif task_type == "implicit":
            return self.run_implicit(
                audio_path,
                question=question,
                history=history,
                task_type=kwargs.get("implicit_type", "understanding"),
            )
        else:
            raise ValueError(f"Unknown task_type: '{task_type}'")
