"""VLM annotation via the google-genai SDK with constrained structured output."""

import logging
import os
import random
import time

from google import genai
from google.genai import types
from PIL import Image

from gcaption.prompt import PROMPT
from gcaption.schema import SequenceCaption

logger = logging.getLogger(__name__)

_MEDIA = {
    "default": None,
    "low": types.MediaResolution.MEDIA_RESOLUTION_LOW,
    "medium": types.MediaResolution.MEDIA_RESOLUTION_MEDIUM,
    "high": types.MediaResolution.MEDIA_RESOLUTION_HIGH,
}

# benign gait footage of walking people -> relax false-positive content blocks
_SAFETY = [
    types.SafetySetting(category=c, threshold=types.HarmBlockThreshold.BLOCK_ONLY_HIGH)
    for c in (
        types.HarmCategory.HARM_CATEGORY_HARASSMENT,
        types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
        types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
        types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
    )
]


def _reason(rsp) -> str:
    """Why a response produced no parse (finish_reason / prompt block)."""
    try:
        pf = getattr(rsp, "prompt_feedback", None)
        if pf is not None and getattr(pf, "block_reason", None):
            return f"prompt_block={pf.block_reason}"
        cands = getattr(rsp, "candidates", None) or []
        if cands:
            return f"finish={getattr(cands[0], 'finish_reason', None)}"
        return "no_candidates"
    except Exception:
        return "unknown"


class Annotator:
    """Wraps a google-genai client; one call annotates all frames of a sequence."""

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        max_retries: int = 5,
        media_res: str = "default",
    ):
        key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not key:
            raise RuntimeError("set GEMINI_API_KEY (or GOOGLE_API_KEY) for the genai client")
        self.client = genai.Client(api_key=key)
        self.model = model
        self.max_retries = max_retries
        self._cfg = types.GenerateContentConfig(
            system_instruction=PROMPT,
            response_mime_type="application/json",
            response_schema=SequenceCaption,
            media_resolution=_MEDIA[media_res],
            safety_settings=_SAFETY,
        )

    def annotate(self, frames: list[Image.Image]) -> SequenceCaption:
        """Return per-frame captions. Retries with exponential backoff + jitter."""
        contents = list(frames)
        last: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                rsp = self.client.models.generate_content(
                    model=self.model, contents=contents, config=self._cfg
                )
                parsed: SequenceCaption = rsp.parsed
                if parsed is None or not parsed.frames:
                    raise ValueError(f"empty parse [{_reason(rsp)}]")
                if len(parsed.frames) != len(frames):
                    logger.warning(
                        "frame count mismatch: got %d for %d frames",
                        len(parsed.frames),
                        len(frames),
                    )
                return parsed
            except Exception as e:  # transient API / parse / quota
                last = e
                sleep = min(2**attempt, 30) + random.uniform(0, 1)
                logger.warning(
                    "annotate attempt %d/%d failed: %s; retry in %.1fs",
                    attempt + 1,
                    self.max_retries,
                    e,
                    sleep,
                )
                time.sleep(sleep)
        raise RuntimeError(f"annotate failed after {self.max_retries} retries: {last}")


class OpenAIAnnotator:
    """Same .annotate() interface via OpenAI vision + structured output.

    Fallback for sequences Gemini hard-blocks (PROHIBITED_CONTENT).
    Note: gpt-4o-mini inflates image tokens ~33x, so it is NOT cheaper than gpt-4o here.
    """

    def __init__(self, model: str, api_key: str | None = None, max_retries: int = 5):
        from openai import OpenAI

        key = api_key or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("set OPENAI_API_KEY for the OpenAI client")
        self.client = OpenAI(api_key=key)
        self.model = model
        self.max_retries = max_retries

    @staticmethod
    def _b64(img: Image.Image) -> str:
        import base64
        import io

        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        return base64.b64encode(buf.getvalue()).decode()

    def annotate(self, frames: list[Image.Image]) -> SequenceCaption:
        content = [{"type": "input_text", "text": PROMPT}]
        content += [
            {"type": "input_image", "image_url": f"data:image/jpeg;base64,{self._b64(f)}"}
            for f in frames
        ]
        last: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                rsp = self.client.responses.parse(
                    model=self.model,
                    input=[{"role": "user", "content": content}],
                    text_format=SequenceCaption,
                )
                parsed: SequenceCaption = rsp.output_parsed
                if parsed is None or not parsed.frames:
                    raise ValueError("empty parse [openai]")
                if len(parsed.frames) != len(frames):
                    logger.warning(
                        "frame count mismatch: got %d for %d frames",
                        len(parsed.frames),
                        len(frames),
                    )
                return parsed
            except Exception as e:
                last = e
                sleep = min(2**attempt, 30) + random.uniform(0, 1)
                logger.warning(
                    "annotate attempt %d/%d failed: %s; retry in %.1fs",
                    attempt + 1,
                    self.max_retries,
                    e,
                    sleep,
                )
                time.sleep(sleep)
        raise RuntimeError(f"annotate failed after {self.max_retries} retries: {last}")
