"""Generation step of the RAG: make a retrieved prompt identity-neutral for a selfie service.

Default behaviour (ImagineMe): the person's appearance comes only from the reference photos,
so every description of who the person is gets removed, while the look of the style stays.
Appearance details that ARE the style (pink hair, character make-up) are kept and reported,
so the user can decide. `keep` and `wish` let the user override and add edits.
"""

from __future__ import annotations

import json
import os
from typing import Protocol

from pydantic import BaseModel, Field

from .dataset import fill_placeholders

# A pinned stable model: "-latest" aliases are hot-swapped by Google at each release.
DEFAULT_LLM_MODEL = os.environ.get("PROMPT_SCOUT_LLM_MODEL", "gemini-3.8-flash")

ADAPT_SYSTEM = """You prepare image-generation prompts for a selfie-based photo service.
At generation time the model receives one or more reference photos of the SAME person.
Everything about who the person is must come from those photos, so the prompt must work
for anyone: any gender, age, ethnicity, body, hair.

Do this:
1. REMOVE every description of the person's natural appearance: gender words (woman, girl, man, guy,
   she/he), age, ethnicity, skin tone, face shape and features, eye color, body type, height, facial hair,
   and natural hair (color, length, texture, how it falls). Refer to "the person from the reference photos"
   and use gender-neutral wording ("the person", "they", "their"). Hair: "their own hair as in the
   reference photos" unless rule 2 applies.
2. KEEP appearance details that are part of the STYLE rather than the person: unnatural or dyed hair
   colors (pink, silver, neon), a styled hairdo the concept depends on (1960s beehive, braided crown in a
   fantasy look), character or editorial make-up, face paint, glitter, wigs, prosthetics, body paint.
   List each one in kept_as_style. If unsure whether something is style or person, REMOVE it and list it
   in removed with "(possibly style)".
3. Everything in the user's KEEP list stays as written, even if rule 1 would remove it.
4. Keep the look intact: scene, wardrobe, props, pose, framing, lighting, palette, camera, lens, film or
   device simulation, post-processing, text overlays, negative prompts. Keep the prompt's format
   (JSON stays JSON, parameter lists stay lists). Do not add new creative ideas.
5. Wardrobe that clearly assumes a gender (dress, bikini, suit and tie) stays as written; add a warning.
6. Remove names of real people (celebrities, influencers); describe only the styling. Note it in changes.
7. Apply the user's WISH (e.g. "horizontal 16:9", "green instead of yellow") by editing the relevant parts so
   the result is one coherent prompt, never by appending a contradicting sentence. Report it in changes.
8. If the source needs extra reference images (an outfit, a product, a second person), keep them numbered
   after the person's photos and list them in extra_references.
9. Write adapted_prompt in English. Write removed, kept_as_style, changes and warnings in Russian."""


class Adaptation(BaseModel):
    adapted_prompt: str
    removed: list[str] = Field(default_factory=list, description="appearance details removed")
    kept_as_style: list[str] = Field(default_factory=list, description="appearance details kept as style")
    changes: list[str] = Field(default_factory=list, description="other edits: wish, names, format")
    extra_references: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class LLM(Protocol):
    def adapt(self, prompt: str, wish: str = "", keep: str = "", needs_reference: bool = False) -> dict: ...


def adapt_user_message(prompt: str, wish: str = "", keep: str = "", needs_reference: bool = False) -> str:
    return (
        f"SOURCE PROMPT (originally built around extra reference images: {needs_reference}):\n"
        f"{fill_placeholders(prompt)}\n\n"
        f"KEEP (user override, keep as written): {keep.strip() or '(nothing)'}\n"
        f"WISH (extra edits): {wish.strip() or '(none)'}"
    )


class LLMError(RuntimeError):
    pass


class GeminiLLM:
    def __init__(self, model: str = DEFAULT_LLM_MODEL, client=None):
        if client is None:
            from google import genai
            client = genai.Client()
        self.client = client
        self.model = model

    def adapt(self, prompt: str, wish: str = "", keep: str = "", needs_reference: bool = False) -> dict:
        from google.genai import types

        try:
            resp = self.client.models.generate_content(
                model=self.model,
                contents=adapt_user_message(prompt, wish, keep, needs_reference),
                config=types.GenerateContentConfig(
                    system_instruction=ADAPT_SYSTEM,
                    response_mime_type="application/json",
                    response_schema=Adaptation,
                    temperature=0.3,
                ),
            )
        except Exception as exc:  # API errors: wrong model name, quota, key
            raise LLMError(f"Gemini ({self.model}) request failed: {type(exc).__name__}: {exc}") from exc

        if not resp.text:
            reason = ""
            try:
                reason = f" finish_reason={resp.candidates[0].finish_reason}"
            except Exception:
                pass
            feedback = getattr(resp, "prompt_feedback", None)
            raise LLMError(f"Gemini returned no text.{reason} {feedback or ''}".strip())
        try:
            return Adaptation.model_validate(json.loads(resp.text)).model_dump()
        except Exception as exc:
            raise LLMError(f"Gemini returned malformed JSON: {resp.text[:300]}") from exc
