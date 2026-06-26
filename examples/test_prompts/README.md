Last updated: 2026-06-26

# Test prompts

Reverse-caption prompts used to probe Krea 2 (image → dense prompt → regenerate). Each is a dense caption
produced by a vision LLM from a reference image, in Krea's caption style (see the reverse-caption method
and `docs/findings.md`).

- `mushroom_gemini_caption.txt` — Gemini 3.5 Flash caption of a 3D mushroom-figure image.
- `mushroom_qwen_caption.txt` — Qwen3.7-Plus caption of the same image.
- `geisha.txt` — caption of a cyberpunk-anime android-geisha image (text-heavy; useful for the L14/text probe).
