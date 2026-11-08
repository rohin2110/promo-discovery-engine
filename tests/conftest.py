"""Shared fixtures: synthetic screenplays with generic placeholder characters."""
from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pytest

from config.settings import AppConfig, LLMSettings, ScoringConfig


def ts(sec: float) -> str:
    m, s = divmod(sec, 60)
    return f"{int(m):02d}:{s:05.2f}"


def block(label: str, start: float, end: float, text: str) -> str:
    return f"**{label}:**\nStart: ({ts(start)}) | End: ({ts(end)})\n{text}\n"


def make_screenplay(ep: int, *, names=("MORGAN", "RILEY"), dramatic: bool = True, n_lines: int = 14,
                    step: float = 3.5, metadata: bool = True, scene_heading: str = "INT. HALL – NIGHT",
                    extra_scenes: int = 1) -> str:
    """Build a screenplay in the bold-label / Start-End format."""
    a, b = names
    hot = [
        (a, "Furious", '"You lied to me about everything!"'),
        (b, "Cold", '"I did what I had to do."'),
        (a, "Shocked", '"You stole the money? Why?"'),
        (b, "Menacing", '"Because the truth would destroy you."'),
    ]
    calm = [(a, "Calm", '"The tea is ready."'), (b, "Polite", '"Thank you, it smells nice."')]
    lines = hot if dramatic else calm
    out = []
    if metadata:
        out += [f"EPISODE NUMBER: Episode {ep}",
                f"EPISODE LOGLINE: {a.title()} confronts {b.title()} in episode {ep}.",
                f"CONTINUITY NOTES: Follows directly from Episode {ep - 1}." if ep > 1 else "CONTINUITY NOTES: First episode.",
                ""]
    out.append("--- SCRIPT START ---\n")
    t = 0.0
    for sc in range(1, extra_scenes + 1):
        start_scene = t
        body = []
        body.append(block("CAMERA", t, t + 2.0, "Close-up on a door as it slams open."))
        body.append(block("ACTION", t + 0.5, t + 2.0, f"{a.title()} storms into the room and grabs {b.title()}'s arm."))
        t += 2.0
        for i in range(n_lines):
            who, par, line = lines[i % len(lines)]
            body.append(block("CAMERA", t, t + step, f"Medium shot on {who.title()}."))
            body.append(block(f"{who} ({par})", t + 0.2, t + step - 0.3, line))
            t += step
        if dramatic:
            body.append(block(f"{b} (Whispering)", t, t + 2.5, '"Then who really killed her...?"'))
            body.append(block("SOUND FX", t + 1.0, t + 3.0, "Ominous drone swells to a cliffhanger sting."))
            t += 3.0
        out.append(f"SCENE {sc:02d}\nScene heading: {scene_heading}\nScene start: {ts(start_scene)}\nScene end: {ts(t)}\n\nActions and dialogue:\n")
        out += body
    out.append("--- SCRIPT END ---")
    return "\n".join(out)


def zip_bytes(files: dict[str, str | bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, data in files.items():
            zf.writestr(name, data)
    return buf.getvalue()


def make_dataset(n: int, **kw) -> list[tuple[str, bytes]]:
    files = {f"show/episode-{i:03d}.txt": make_screenplay(i, **kw) for i in range(1, n + 1)}
    return [("dataset.zip", zip_bytes(files))]


@pytest.fixture
def app_cfg(tmp_path: Path) -> AppConfig:
    from config.settings import load_app_config
    base = load_app_config()
    return base.model_copy(update={
        "output_dir": str(tmp_path / "outputs"), "raw_output_dir": str(tmp_path / "raw_outputs"),
        "feedback_path": str(tmp_path / "feedback.jsonl"),
    })


@pytest.fixture
def scoring() -> ScoringConfig:
    return ScoringConfig()


@pytest.fixture
def heuristic_llm() -> LLMSettings:
    return LLMSettings(provider="heuristic", max_retries=1)


@pytest.fixture
def mock_llm() -> LLMSettings:
    return LLMSettings(provider="mock", model="mock-model", max_retries=1)
