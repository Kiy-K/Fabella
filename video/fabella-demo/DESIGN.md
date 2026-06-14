# DESIGN.md — Fabella Demo Video

## 1. Identity
- **Product:** Fabella — small words for big questions.
- **Track I · Backyard AI** Hackathon demo.
- **One-line vibe:** Warm, parent-friendly, gentle, book-like. Like a kind notebook on a kitchen table, not a SaaS dashboard.

## 2. Color (use these exact hex values)
- Background warm cream: `#F7F4EC`
- Surface (cards): `#FFFFFF`
- Line/border: `#E2DCCD`
- Soft surface / muted bg: `#F3EFE6`
- Primary text: `#1F2330`
- Soft text: `#3A3F4A`
- Muted text: `#6B6F78`
- Brand green (primary accent): `#4F7A4A`
- Brand green strong (CTA, hover): `#2E5A36`
- Brand green soft (highlight wash): `#DCEADC`
- Warm yellow accent (read-aloud, "new"): `#E7B73A` (bookmark yellow, used very sparingly)
- Danger (kept off-screen for this video): `#B04A3A`

Dark mode preview: deep slate `#1A1F25` for the populated-state scene to mirror the actual dark-mode rendering of the live site.

## 3. Typography
- Display / body sans: **Outfit** 400/500/600/700.
- Mono / micro labels: **JetBrains Mono** 400/500.
- Type scale (px on a 1920×1080 canvas):
  - Hero title: 120
  - Section heading: 64
  - Body: 32
  - Caption: 26
  - Mono label: 18 with 0.14em tracking
- Never use display serifs. Never use Fraunces. Never use a tight condensed.

## 4. Layout rules
- 1920×1080 canvas.
- 96 px safe margin on all sides.
- Content blocks are rounded 18 px with 1 px `--line` border.
- Generous vertical rhythm — at least 48 px between blocks.
- All on-screen text is left-aligned within its container. Center only for hero titles.
- No drop shadows beyond `--shadow`: a single soft 30 px–22 px blur. No neon.

## 5. Motion
- Default easing: `cubic-bezier(0.2, 0.7, 0.2, 1)` (matches the live app).
- Fade-up on entry: 24 px, 600 ms.
- Cross-fade transitions between beats: 350 ms.
- No bouncing, no rotation, no glow. Soft and steady.
- Captions: bottom-third, white text on a 50% black card, fade in 200 ms.

## 6. Voice
- Calm adult woman, soft and warm.
- Pace: ~150 wpm. Short breaths.
- Reasonable warmth; never theatrical.
- Generate via the Hyperframes media skill (Kokoro TTS).
