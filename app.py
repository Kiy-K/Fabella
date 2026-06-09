"""Fabella — personalized storytelling companion. Gradio Blocks UI."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import gradio as gr

from generator import USE_MOCK, generate_story
from schema import StoryRequest
from safety import (
    has_profanity,
    sanitize_moral,
    sanitize_name,
    sanitize_themes,
)

THEME_CHOICES = [
    "dinosaurs",
    "robots",
    "space",
    "fantasy",
    "adventure",
    "animals",
    "ocean",
    "friends",
]

MORAL_PRESETS = [
    "being kind to others",
    "being brave when things feel hard",
    "telling the truth",
    "sharing what you have",
    "trying again after a mistake",
    "listening before you speak",
]


def _mode_badge() -> str:
    return "Mock mode" if USE_MOCK else "Real model"


def _generate(
    name: str,
    age: int,
    themes: list[str] | None,
    moral_choice: str,
    moral_text: str,
    length: str,
    seed: int,
) -> tuple[str, str, str]:
    clean_name = sanitize_name(name) or "Friend"
    clean_themes = sanitize_themes(themes or []) or ["friends"]
    clean_moral = sanitize_moral(moral_text or moral_choice or "")

    if has_profanity(clean_name) or has_profanity(clean_moral):
        return (
            "Oops",
            "Some of the words used aren't allowed. Please try different words.",
            _mode_badge(),
        )

    req = StoryRequest(
        name=clean_name,
        age=int(age),
        themes=clean_themes,
        moral=clean_moral,
        length=length or "medium",
        seed=int(seed or 0),
    )
    title, body, mode = generate_story(req)
    return title, body, mode


def _regenerate(
    name: str,
    age: int,
    themes: list[str] | None,
    moral_choice: str,
    moral_text: str,
    length: str,
    current_seed: int,
) -> tuple[str, str, str]:
    return _generate(name, age, themes, moral_choice, moral_text, length, (current_seed or 0) + 1)


def build_ui() -> gr.Blocks:
    badge = _mode_badge()
    with gr.Blocks(title="Fabella") as demo:
        gr.Markdown(
            "# Fabella\n"
            "_A personalized storytelling companion for children._\n\n"
            f"**Current mode:** `{badge}` — set the `USE_MOCK` env var to change."
        )

        with gr.Row():
            with gr.Column(scale=1):
                name = gr.Textbox(label="Child's name", placeholder="e.g. Mia", max_length=30)
                age = gr.Slider(6, 10, step=1, value=7, label="Age")
                themes = gr.CheckboxGroup(
                    choices=THEME_CHOICES,
                    value=["adventure"],
                    label="Favorite themes (up to 3)",
                )
                moral_choice = gr.Dropdown(
                    choices=MORAL_PRESETS,
                    value=MORAL_PRESETS[0],
                    label="Moral lesson (preset)",
                )
                moral_text = gr.Textbox(
                    label="Or write your own moral (optional)",
                    placeholder="e.g. it's okay to ask for help",
                    max_length=120,
                )
                length = gr.Radio(
                    choices=["short", "medium", "long"],
                    value="medium",
                    label="Story length",
                )
                seed = gr.State(value=0)
                with gr.Row():
                    submit = gr.Button("Tell me a story", variant="primary")
                    regen = gr.Button("New story")

            with gr.Column(scale=2):
                out_title = gr.Markdown("### Your story will appear here")
                out_body = gr.Markdown("")
                out_mode = gr.Markdown(f"_{badge}_")

        submit.click(
            fn=_generate,
            inputs=[name, age, themes, moral_choice, moral_text, length, seed],
            outputs=[out_title, out_body, out_mode],
        )
        regen.click(
            fn=_regenerate,
            inputs=[name, age, themes, moral_choice, moral_text, length, seed],
            outputs=[out_title, out_body, out_mode],
        )

    return demo


demo = build_ui()


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)), theme=gr.themes.Soft())
