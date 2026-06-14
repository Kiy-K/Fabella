# STORYBOARD — Fabella Demo (90s)

Total duration: 90.0 s. 1920×1080. 30 fps. 2700 frames.
One parent `index.html`. Sub-compositions mounted via `data-composition-src` so each beat is independently editable.
Track layout: T0 background, T1 main content, T2 captions, T3 narration audio, T4 ambient TTS waveform decoration.

| Beat | Time | Visuals | Captions | Real recording clip | SFX |
|------|------|---------|----------|--------------------|-----|
| 1. Title | 0.0–4.0s | Cream wash. Centered "Fabella" 156 px Outfit 700 in `#2E5A36`. Below: "small words for big questions" 34 px. Forest-green dot pulses. | (none) | none | Page-turn whoosh |
| 2. Problem | 4.0–13.0s | Left 40%: left-line "When something *hard* happens, the right words are hard to find." Right 60%: the **real welcome-state recording** (`recording-clips/welcome.mp4`) shown with gentle Ken Burns. | "When something hard happens, a hospital, a move, a loss, parents need the right words. And they need them right now." | welcome.mp4 | Soft heartbeat |
| 3. Product flow | 13.0–22.0s | The **real flow recording** (`recording-clips/flow.mp4`) shows the actual typing → Draft → reply flow. A thin caption bar at the bottom labels the action: "Type → Draft → Reply". | "Type a sentence about the situation. Pick the age. Pick the tone. Fabella drafts a short, kind, age-appropriate explanation." | flow.mp4 | Soft UI clicks |
| 4. Output shape | 22.0–31.0s | The **real populated recording** (`recording-clips/populated.mp4`) on the right. Left 40%: four conversation-style heading cards slide in one by one — Where to begin, The explanation, How to land it, If they ask another question. | "It comes back in four parts. Where to begin. The explanation. How to land it. And what to say if your child asks another question." | populated.mp4 | Calm chime on each card |
| 5. Read aloud | 31.0–40.0s | The **real read-aloud recording** (`recording-clips/readaloud.mp4`) shows the "Read aloud" button click and the "Warming VoxCPM2 and preparing narration..." status. VoxCPM2 dot pulses green. | "Tap Read aloud, and a friendly voice reads it for you. VoxCPM2 runs on demand. Nothing leaves the bucket that shouldn't." | readaloud.mp4 | TTS audio swell |
| 6. AI pipeline | 40.0–52.0s | Three badges appear left-to-right with GSAP: Gemma 4 E4B drafter → Nemotron 3 Nano judge → VoxCPM2 read aloud. Connecting green gradient lines. | "Behind the scenes, three small models work together. Gemma 4 E4B writes. Nemotron 3 Nano checks. VoxCPM2 reads it aloud." | none | Soft whoosh per badge |
| 7. Memory & follow-up | 52.0–64.0s | Two columns. Left: a stylized memory card showing durable facts ("Grandma had surgery in March 2026") and a rolling summary. Right: a follow-up turn — parent asks "What if she asks if grandma will die?" and Fabella answers consistently. | "Fabella remembers this family. Durable facts and a rolling summary travel with the parent. So when a follow-up question lands, the answer stays consistent with the first explanation." | none | Calm chime |
| 8. Privacy & storage | 64.0–76.0s | Cream wash. Center: stylized JSON block rendering `user-<owner_key>.memory.json` with the keys visible. Tiny lock icon. A small "HF Bucket" pill at the top. | "Nothing is sent to a third-party database. History and memory live as one JSON file per parent in the HF Bucket. No external cloud DB. No tracking. Just the family, and the words." | none | Soft whoosh |
| 9. Closing | 76.0–90.0s | Cream wash. Centered 60 px Outfit 600: "A second pair of eyes, when the right words are hard to find." Below in JetBrains Mono 22: "Fabella · Track I · Backyard AI". Slow fade to brand green wash. | (none — title card style) | none | Brand-tone pad swells then fades |

## Caption rules
- Bottom-third band, 1080 px wide centered, 12 px radius, 50% black.
- White Outfit 500, 26 px, 1.4 line height, max 80 chars per line.
- 200 ms fade in, hold for the duration of the caption, 200 ms fade out.
- Captions are time-locked to narration. Beats 1 and 9 have no captions.

## Asset map
- `rendering-clips/welcome.mp4` (4s) — real recording, 0–4s
- `rendering-clips/flow.mp4` (6s) — real recording, 8–14s
- `rendering-clips/populated.mp4` (4s) — real recording, 15–19s
- `rendering-clips/readaloud.mp4` (6s) — real recording, 60–66s
- All UI mockups (form, badges, output cards, memory, JSON) are drawn in HTML/CSS, no extra image assets needed.

## Animation rules
- Every clip carries `data-start`, `data-duration`, `data-track-index`, `class="clip"`.
- GSAP timeline paused and registered on `window.__timelines["main"]`.
- No `Date.now()`, no `Math.random()`.
- Per-beat sub-compositions are independent so future edits stay scoped.
