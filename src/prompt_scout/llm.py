"""LLM steps: describe a reference photo's style, and adapt a found prompt.

Both use structured JSON output, so the MCP tools return predictable fields.
"""

from __future__ import annotations

import json
import mimetypes
import os
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from .dataset import Record, fill_placeholders

DEFAULT_LLM_MODEL = os.environ.get("PROMPT_SCOUT_LLM_MODEL", "gemini-flash-latest")

DESCRIBE_SYSTEM = """You analyse a reference image so we can find image-generation prompts that reproduce its LOOK.
Describe the visual style, not the identity of any person in it.
Cover: genre/concept, setting, wardrobe and props, lighting, colour palette, camera and lens feel,
composition and framing, post-processing or medium (film, painting, 3D, etc.), mood.
Never describe a person's face, ethnicity, age or body. Never name real people.
Return JSON: {"style_description": "<4-6 sentences>", "search_query": "<one dense English line of the key visual terms>"}"""

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
    def describe_image(self, image: bytes, mime_type: str) -> dict: ...

    def adapt(self, record: Record, wish: str) -> dict: ...


@dataclass
class ImageInput:
    data: bytes
    mime_type: str


def read_image(path_or_url: str, max_bytes: int = 15 * 1024 * 1024) -> ImageInput:
    if path_or_url.startswith(("http://", "https://")):
        req = urllib.request.Request(path_or_url, headers={"User-Agent": "prompt-scout"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = resp.read(max_bytes + 1)
            mime = resp.headers.get_content_type()
    else:
        p = Path(path_or_url).expanduser()
        data = p.read_bytes()
        mime = mimetypes.guess_type(p.name)[0] or "image/jpeg"
    if len(data) > max_bytes:
        raise ValueError("image is larger than 15 MB")
    if not mime.startswith("image/"):
        raise ValueError(f"not an image: {mime}")
    return ImageInput(data, mime)


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

    def _json(self, system: str, contents) -> dict:
        from google.genai import types

        resp = self.client.models.generate_content(
            model=self.model,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=system,
                response_mime_type="application/json",
                temperature=0.4,
            ),
        )
        return json.loads(resp.text)

    def describe_image(self, image: bytes, mime_type: str) -> dict:
        from google.genai import types

        part = types.Part.from_bytes(data=image, mime_type=mime_type)
        return self._json(DESCRIBE_SYSTEM, [part, "Describe the style of this image."])

    def adapt(self, record: Record, wish: str) -> dict:
        return self._json(ADAPT_SYSTEM, adapt_user_message(record, wish))
