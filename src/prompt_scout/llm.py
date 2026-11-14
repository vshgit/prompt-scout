"""Generation step of the RAG: adapt a retrieved prompt for selfie-based generation.

Uses JSON output so the MCP tool returns predictable fields.
"""

from __future__ import annotations

import json
import os
from typing import Protocol

from .dataset import Record, fill_placeholders

DEFAULT_LLM_MODEL = os.environ.get("PROMPT_SCOUT_LLM_MODEL", "gemini-flash-latest")

ADAPT_SYSTEM = """You adapt a community image-generation prompt for a selfie-based photo service.
The user's own photo(s) are attached to the generation request as reference image 1 (and more if given).
Rules:
1. Identity comes ONLY from the reference photo. Remove from the prompt every description of the subject's face,
   facial features, ethnicity, age, skin tone, body type and, unless the style depends on it, hair.
   Add one clear instruction to keep the person from reference image 1 exactly recognizable.
2. Keep what makes the look: scene, wardrobe, props, pose, lighting, palette, camera, lens, composition, medium,
   post-processing, negative prompts.
3. Make the prompt work for any gender unless the look truly depends on one; then say so in warnings.
4. If the user gave a wish, merge it into ONE coherent prompt (change the relevant parts, do not append a
   contradicting sentence). If the wish conflicts with the look, follow the wish and note it in warnings.
5. If the source prompt relied on extra reference images (outfit, product, second person), keep them as
   numbered references and list them in extra_references.
6. If the source names a real person (celebrity, influencer), remove the name and describe only the look
   (styling, lighting, setting); mention the removal in changes.
7. Write the adapted prompt in English. Write `changes` and `warnings` in the language of the user's wish
   (Russian if the wish is in Russian; English if there is no wish).
Return JSON: {"adapted_prompt": str, "changes": [str], "extra_references": [str], "warnings": [str]}"""


class LLM(Protocol):
    def adapt(self, record: Record, wish: str) -> dict: ...


def adapt_user_message(record: Record, wish: str) -> str:
    return (
        f"SOURCE PROMPT (title: {record.title}; needs extra reference images: {record.needs_reference}):\n"
        f"{fill_placeholders(record.prompt)}\n\n"
        f"USER WISH: {wish.strip() or '(none)'}"
    )


class GeminiLLM:
    def __init__(self, model: str = DEFAULT_LLM_MODEL, client=None):
        if client is None:
            from google import genai
            client = genai.Client()
        self.client = client
        self.model = model

    def adapt(self, record: Record, wish: str) -> dict:
        from google.genai import types

        resp = self.client.models.generate_content(
            model=self.model,
            contents=adapt_user_message(record, wish),
            config=types.GenerateContentConfig(
                system_instruction=ADAPT_SYSTEM,
                response_mime_type="application/json",
                temperature=0.4,
            ),
        )
        return json.loads(resp.text)
