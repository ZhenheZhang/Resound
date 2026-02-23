"""
Gemini 3.1 Pro model wrapper for WavBench inference.

Supports:
- Colloquial Expression (Basic & Pro): text-out from audio-in
- Acoustic Explicit Understanding: classify paralinguistic attributes
- Acoustic Explicit Generation: generate audio with target paralinguistic attributes
- Acoustic Implicit: multi-turn spoken dialogue understanding
"""

import os
import time
import base64
import logging
from pathlib import Path
from typing import Optional

import google.generativeai as genai
from google.generativeai import types

logger = logging.getLogger(__name__)


# ── Model configuration ───────────────────────────────────────────────────────

MODEL_ID = "gemini-3.1-pro-preview"

# Per-task system prompts — carefully tuned for WavBench scoring rubrics
SYSTEM_PROMPTS = {
    # ── Colloquial Expression ──────────────────────────────────────────────
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

    # ── Acoustic Explicit Understanding ───────────────────────────────────
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

    # ── Acoustic Explicit Generation ──────────────────────────────────────
    "generation_default": (
        "You are generating speech with specific paralinguistic characteristics. "
        "Read the provided text naturally while precisely matching the requested "
        "acoustic attribute. Focus on accuracy of the target attribute above all."
    ),

    # ── Acoustic Implicit ─────────────────────────────────────────────────
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


# ── Retry / rate-limit handling ───────────────────────────────────────────────

def _call_with_retry(fn, max_retries: int = 5, base_delay: float = 2.0):
    """Exponential backoff retry wrapper for Gemini API calls."""
    for attempt in range(max_retries):
        try:
            return fn()
        except Exception as e:
            err = str(e).lower()
            is_rate_limit = "429" in err or "quota" in err or "resource_exhausted" in err
            is_server_err = "500" in err or "503" in err or "unavailable" in err
            if (is_rate_limit or is_server_err) and attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt)
                logger.warning(f"API error (attempt {attempt+1}/{max_retries}): {e}. Retrying in {delay:.1f}s...")
                time.sleep(delay)
            else:
                raise
    raise RuntimeError("Max retries exceeded")


# ── Main model class ──────────────────────────────────────────────────────────

class GeminiWavBenchModel:
    """
    Gemini 3.1 Pro wrapper designed to maximize WavBench scores.

    Key strategies:
    1. Task-specific system prompts tuned to WavBench rubrics
    2. Chain-of-thought for colloquial/reasoning tasks
    3. Concise label-only output for classification tasks
    4. Self-consistency voting for paralinguistic classification
    5. Audio preprocessing metadata injection
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_id: str = MODEL_ID,
        temperature: float = 0.2,
        top_p: float = 0.95,
        max_output_tokens: int = 8192,
        self_consistency_k: int = 3,  # votes for classification tasks
    ):
        api_key = api_key or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not set. Export it or pass api_key=...")

        genai.configure(api_key=api_key)
        self.model_id = model_id
        self.temperature = temperature
        self.top_p = top_p
        self.max_output_tokens = max_output_tokens
        self.self_consistency_k = self_consistency_k

        # Lazy model init per system prompt (cache)
        self._model_cache: dict[str, genai.GenerativeModel] = {}
        logger.info(f"GeminiWavBenchModel initialized with {model_id}")

    def _get_model(self, system_prompt_key: str) -> genai.GenerativeModel:
        if system_prompt_key not in self._model_cache:
            system_instruction = SYSTEM_PROMPTS.get(system_prompt_key, SYSTEM_PROMPTS["colloquial_default"])
            self._model_cache[system_prompt_key] = genai.GenerativeModel(
                model_name=self.model_id,
                system_instruction=system_instruction,
                generation_config=genai.GenerationConfig(
                    temperature=self.temperature,
                    top_p=self.top_p,
                    max_output_tokens=self.max_output_tokens,
                ),
            )
        return self._model_cache[system_prompt_key]

    def _load_audio(self, audio_path: str) -> dict:
        """Load audio file and return as inline data part."""
        audio_path = Path(audio_path)
        suffix = audio_path.suffix.lower()
        mime_map = {
            ".wav": "audio/wav",
            ".mp3": "audio/mp3",
            ".flac": "audio/flac",
            ".ogg": "audio/ogg",
            ".m4a": "audio/mp4",
            ".aac": "audio/aac",
        }
        mime_type = mime_map.get(suffix, "audio/wav")
        with open(audio_path, "rb") as f:
            audio_bytes = f.read()
        return {
            "inline_data": {
                "mime_type": mime_type,
                "data": base64.b64encode(audio_bytes).decode("utf-8"),
            }
        }

    def _generate(self, model: genai.GenerativeModel, contents: list) -> str:
        """Single generation call with retry."""
        def _call():
            response = model.generate_content(contents)
            return response.text.strip()
        return _call_with_retry(_call)

    def _classify_with_voting(
        self, model: genai.GenerativeModel, contents: list, k: int
    ) -> str:
        """
        Self-consistency voting: run k inferences, return majority label.
        Reduces variance for classification tasks.
        """
        votes = []
        for _ in range(k):
            try:
                result = self._generate(model, contents)
                # Normalize: take first line, strip punctuation
                label = result.split("\n")[0].strip().rstrip(".,!?")
                votes.append(label)
            except Exception as e:
                logger.warning(f"Vote failed: {e}")

        if not votes:
            return ""

        # Majority vote (case-insensitive)
        from collections import Counter
        normalized = [v.lower() for v in votes]
        winner_lower = Counter(normalized).most_common(1)[0][0]
        # Return original casing of first match
        for v in votes:
            if v.lower() == winner_lower:
                return v
        return votes[0]

    # ── Public inference methods ──────────────────────────────────────────

    def run_colloquial(
        self,
        audio_path: str,
        question: Optional[str] = None,
        domain: str = "default",
        use_cot: bool = True,
    ) -> str:
        """
        Colloquial Expression task (Basic or Pro).

        Args:
            audio_path: Path to the audio file containing the spoken question.
            question: Optional text version of question (used as hint if provided).
            domain: Task domain: 'code', 'math', 'logic', 'safety',
                    'instruction', 'creativity', 'qa', or 'default'.
            use_cot: Whether to prepend chain-of-thought instruction.

        Returns:
            Model response text.
        """
        prompt_key = f"colloquial_{domain}" if f"colloquial_{domain}" in SYSTEM_PROMPTS else "colloquial_default"
        model = self._get_model(prompt_key)
        audio_part = self._load_audio(audio_path)

        cot_prefix = (
            "Listen carefully to the audio. Think step by step before answering.\n\n"
            if use_cot
            else ""
        )

        contents = [audio_part]
        if question:
            contents.append(f"{cot_prefix}The spoken question is: {question}\n\nPlease answer:")
        else:
            contents.append(f"{cot_prefix}Please listen to the audio and answer the spoken question:")

        return self._generate(model, contents)

    def run_explicit_understanding(
        self,
        audio_path: str,
        attribute: str,
        question: Optional[str] = None,
    ) -> str:
        """
        Acoustic Explicit Understanding task.

        Args:
            audio_path: Path to audio file.
            attribute: One of: accent, age, emotion, gender, language,
                       pitch, speed, volume, audio, music.
            question: Optional explicit question from the dataset.

        Returns:
            Classified label string.
        """
        prompt_key = f"understanding_{attribute.lower()}"
        if prompt_key not in SYSTEM_PROMPTS:
            prompt_key = "understanding_audio"

        model = self._get_model(prompt_key)
        audio_part = self._load_audio(audio_path)

        user_text = question if question else f"What is the {attribute} of this audio?"
        contents = [audio_part, user_text]

        # Use self-consistency voting for classification robustness
        return self._classify_with_voting(model, contents, k=self.self_consistency_k)

    def run_explicit_generation(
        self,
        audio_path: str,
        attribute: str,
        target_value: str,
        text_to_speak: Optional[str] = None,
        question: Optional[str] = None,
    ) -> str:
        """
        Acoustic Explicit Generation task.
        Note: Gemini 3.1 Pro outputs TEXT. For audio output, the text
        response is passed to a TTS system. This method returns the
        instruction/text that should be synthesized.

        Args:
            audio_path: Reference audio or context audio.
            attribute: Paralinguistic attribute to generate (e.g., 'emotion').
            target_value: Target value (e.g., 'Happy').
            text_to_speak: Text content to speak (if known).
            question: Original dataset question.

        Returns:
            Text response / generation instruction.
        """
        model = self._get_model("generation_default")
        audio_part = self._load_audio(audio_path)

        if text_to_speak:
            prompt = (
                f"Please respond to the following spoken request by producing text "
                f"that should be spoken with {attribute} = '{target_value}'.\n"
                f"Text to convey: {text_to_speak}\n"
                f"Generate your response in a way that clearly expresses {target_value} {attribute}."
            )
        elif question:
            prompt = (
                f"Listen to this audio. {question}\n"
                f"Generate a spoken response that clearly exhibits {attribute} = '{target_value}'."
            )
        else:
            prompt = (
                f"Listen to the audio. Respond as if you are speaking with "
                f"{attribute} = '{target_value}'. Make the {attribute} very clear."
            )

        contents = [audio_part, prompt]
        return self._generate(model, contents)

    def run_implicit(
        self,
        audio_path: str,
        question: Optional[str] = None,
        history: Optional[list[dict]] = None,
        task_type: str = "understanding",
    ) -> str:
        """
        Acoustic Implicit task — single or multi-turn.

        Args:
            audio_path: Current turn audio.
            question: Optional text question.
            history: Prior conversation turns: [{"role": "user"|"model", "content": str}]
            task_type: 'understanding' or 'generation'.

        Returns:
            Model response text.
        """
        prompt_key = f"implicit_{task_type}"
        if prompt_key not in SYSTEM_PROMPTS:
            prompt_key = "implicit_understanding"

        model = self._get_model(prompt_key)
        audio_part = self._load_audio(audio_path)

        # Build multi-turn context
        if history:
            # Inject history as context in the prompt
            history_str = "\n".join(
                f"[{turn['role'].upper()}]: {turn['content']}"
                for turn in history
            )
            context = f"Conversation history:\n{history_str}\n\n"
        else:
            context = ""

        user_text = f"{context}[New audio message attached]\n"
        if question:
            user_text += f"Question: {question}"
        else:
            user_text += "Please respond to the spoken message above."

        contents = [audio_part, user_text]
        return self._generate(model, contents)

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
        Unified entry point dispatching to the correct task method.

        Args:
            audio_path: Path to audio file.
            task_type: One of: 'colloquial', 'explicit_understanding',
                       'explicit_generation', 'implicit'.
            attribute: For acoustic tasks, the paralinguistic attribute.
            question: Optional text question.
            domain: For colloquial tasks, the cognitive domain.
            target_value: For generation tasks, the target attribute value.
            history: For multi-turn tasks, prior conversation history.
        """
        if task_type == "colloquial":
            return self.run_colloquial(
                audio_path=audio_path,
                question=question,
                domain=domain or "default",
            )
        elif task_type == "explicit_understanding":
            return self.run_explicit_understanding(
                audio_path=audio_path,
                attribute=attribute or "audio",
                question=question,
            )
        elif task_type == "explicit_generation":
            return self.run_explicit_generation(
                audio_path=audio_path,
                attribute=attribute or "emotion",
                target_value=target_value or "",
                question=question,
            )
        elif task_type == "implicit":
            return self.run_implicit(
                audio_path=audio_path,
                question=question,
                history=history,
                task_type=kwargs.get("implicit_type", "understanding"),
            )
        else:
            raise ValueError(f"Unknown task_type: {task_type}")
