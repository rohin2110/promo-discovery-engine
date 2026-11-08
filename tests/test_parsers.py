import io
import json
import zipfile

import pytest

from core.normalizer import load_dataset
from models.schemas import EventType
from tests.conftest import make_screenplay, zip_bytes


def _load(app_cfg, files):
    return load_dataset([(n, d.encode() if isinstance(d, str) else d) for n, d in files], app_cfg)


def test_scene_dialogue_action_camera_parsing(app_cfg):
    eps, rep, _ = _load(app_cfg, [("episode-007.txt", make_screenplay(7, extra_scenes=2))])
    assert rep.episodes_detected == 1 and not rep.errors
    ep = eps[0]
    assert ep.episode_id == "EP007" and ep.episode_number == 7
    assert ep.logline.startswith("Morgan confronts Riley")
    assert ep.continuity_notes.startswith("Follows directly from Episode 6")
    assert len(ep.scenes) == 2
    assert ep.scenes[0].heading == "INT. HALL – NIGHT" and ep.scenes[0].bounds_source == "source"
    types = {e.event_type for e in ep.timeline_events}
    assert {EventType.camera, EventType.action, EventType.dialogue, EventType.other} <= types
    d = next(e for e in ep.timeline_events if e.event_type == EventType.dialogue)
    assert d.speaker == "MORGAN" and d.parenthetical == "Furious"
    assert d.text == '"You lied to me about everything!"'  # original text preserved
    assert d.start_seconds == pytest.approx(2.2) and d.start_timestamp == "00:02.20"
    a = next(e for e in ep.timeline_events if e.event_type == EventType.action)
    assert "storms into the room" in a.text
    sfx = next(e for e in ep.timeline_events if e.event_type == EventType.other)
    assert sfx.label == "SOUND FX"


def test_offscreen_parentheticals_and_multiword_speakers(app_cfg):
    text = ("SCENE 1\nScene heading: EXT. ROAD – DAY\nScene start: 00:00.00\nScene end: 00:10.00\n\n"
            "**THE OLD SERGEANT (O.S.) (Firm):**\nStart: (00:01.00) | End: (00:03.00)\n\"Halt.\"\n")
    eps, _, _ = _load(app_cfg, [("ep-2.txt", text)])
    ev = eps[0].timeline_events[0]
    assert ev.speaker == "THE OLD SERGEANT" and ev.parenthetical == "O.S.; Firm"


def test_screenplay_without_episode_metadata_uses_filename(app_cfg):
    eps, rep, _ = _load(app_cfg, [("my-show_ep12_final.txt", make_screenplay(12, metadata=False))])
    assert eps[0].episode_id == "EP012"
    assert eps[0].logline == ""
    eps2, rep2, _ = _load(app_cfg, [("pilot.txt", make_screenplay(1, metadata=False))])
    assert eps2[0].episode_id.startswith("EP-") and eps2[0].episode_number is None
    assert any(i.code == "episode_id" for i in rep2.issues)


def test_missing_timestamps_are_flagged_and_not_invented(app_cfg):
    text = ("EPISODE NUMBER: Episode 3\n\nSCENE 01\nScene heading: INT. ROOM – DAY\n\n"
            "**ALEX (Angry):**\n\"Where were you?\"\n\n**ACTION:**\nAlex slams the table.\n")
    eps, rep, _ = _load(app_cfg, [("ep3.txt", text)])
    ep = eps[0]
    assert len(ep.timeline_events) == 2
    assert all(e.timestamp_status == "missing" and e.start_seconds is None for e in ep.timeline_events)
    assert not ep.usable_for_promos
    assert any(i.code == "no_timestamps" for i in rep.issues)


def test_identical_scene_names_get_unique_ids(app_cfg):
    eps, _, _ = _load(app_cfg, [("e1.txt", make_screenplay(1)), ("e2.txt", make_screenplay(2))])
    s1, s2 = eps[0].scenes[0], eps[1].scenes[0]
    assert s1.heading == s2.heading and s1.label == s2.label == "SCENE 01"
    assert s1.scene_id != s2.scene_id
    ids = [e.event_id for ep in eps for e in ep.timeline_events]
    assert len(ids) == len(set(ids))


def test_overlapping_camera_and_dialogue_not_double_counted(app_cfg):
    eps, _, _ = _load(app_cfg, [("e1.txt", make_screenplay(1))])
    ep = eps[0]
    summed = sum(e.duration for e in ep.timed_events)
    assert summed > ep.content_duration_seconds
    assert ep.content_duration_seconds == pytest.approx(ep.timeline_end - ep.timeline_start, abs=0.01)


def test_invalid_and_inconsistent_timestamps_flagged(app_cfg):
    text = ("EPISODE NUMBER: Episode 4\nSCENE 01\nScene heading: INT. A – DAY\nScene start: 00:00.00\nScene end: 00:20.00\n\n"
            "**ALEX:**\nStart: (00:75.00) | End: (00:76.00)\n\"Bad minute.\"\n\n"
            "**SAM:**\nStart: (00:09.00) | End: (00:05.00)\n\"Backwards.\"\n\n"
            "**ALEX:**\nStart: (00:10.00) | End: (00:30.00)\n\"Past the scene end.\"\n\n"
            "**SAM:**\nStart: (00:11.00) | End: (00:12.00)\n\"Fine.\"\n\n**SAM:**\nStart: (00:12.00) | End: (00:13.00)\n\"Fine again.\"\n")
    eps, rep, _ = _load(app_cfg, [("ep4.txt", text)])
    evs = eps[0].timeline_events
    assert evs[0].timestamp_status == "invalid" and evs[0].start_seconds is None
    assert evs[1].timestamp_status == "inconsistent" and "end_before_start" in evs[1].flags
    assert "outside_scene_bounds" in evs[2].flags
    codes = {i.code for i in rep.issues}
    assert {"event_timestamp_invalid", "event_timestamp_inconsistent", "event_outside_scene"} <= codes


def test_unsupported_format_reported(app_cfg):
    eps, rep, ing = _load(app_cfg, [("notes.xyz", "hello"), ("e1.txt", make_screenplay(1))])
    assert len(eps) == 1
    rec = next(f for f in rep.files if f.display_name == "notes.xyz")
    assert rec.status == "unsupported"
    assert any(i.code == "unsupported_format" for i in rep.errors)


def test_empty_and_malformed_files_reported(app_cfg):
    eps, rep, _ = _load(app_cfg, [("empty.txt", ""), ("bad.json", "{not json"), ("prose.txt", "Just a story with no structure.\nAnother line."),
                                  ("e1.txt", make_screenplay(1))])
    assert len(eps) == 1
    status = {f.display_name: f.status for f in rep.files}
    assert status["empty.txt"] == "failed" and status["bad.json"] == "failed" and status["prose.txt"] == "failed"
    codes = {i.code for i in rep.errors}
    assert {"empty_file", "malformed_json", "unrecognised_structure"} <= codes


def test_empty_dataset(app_cfg):
    eps, rep, _ = load_dataset([], app_cfg)
    assert eps == [] and any(i.code == "empty_upload" for i in rep.errors)


def test_markdown_inline_format(app_cfg):
    md = ("# Episode 5: The Letter\n\n**Logline:** A letter arrives.\n\n## Scene 1 - INT. STUDY – NIGHT\n\n"
          "[00:00.0 - 00:03.5] **CAMERA:** Slow push in on the desk.\n"
          "[00:01.0 - 00:03.0] JORDAN (Shaken): \"Who sent this?\"\n"
          "[00:03.5 - 00:06.0] The candle gutters out.\n")
    eps, rep, _ = _load(app_cfg, [("letter.md", md)])
    ep = eps[0]
    assert ep.episode_id == "EP005" and ep.episode_title == "The Letter" and ep.logline == "A letter arrives."
    kinds = [(e.event_type, e.speaker) for e in ep.timeline_events]
    assert kinds == [(EventType.camera, ""), (EventType.dialogue, "JORDAN"), (EventType.description, "")]
    assert ep.timeline_events[1].start_seconds == pytest.approx(1.0)


def test_srt_parsing(app_cfg):
    srt = "1\n00:00:01,000 --> 00:00:03,500\nALEX: Open the door!\n\n2\n00:00:04,000 --> 00:00:06,000\n[thunder rumbles]\n"
    eps, rep, _ = _load(app_cfg, [("episode_9.srt", srt)])
    ep = eps[0]
    assert ep.episode_id == "EP009" and len(ep.timeline_events) == 2
    assert ep.timeline_events[0].speaker == "ALEX" and ep.timeline_events[0].end_seconds == pytest.approx(3.5)
    assert ep.timeline_events[1].event_type == EventType.other


def test_json_normalised_roundtrip(app_cfg):
    eps, _, _ = _load(app_cfg, [("episode-002.txt", make_screenplay(2))])
    doc = json.dumps([eps[0].model_dump(mode="json")])
    eps2, rep2, _ = _load(app_cfg, [("export.json", doc)])
    assert not rep2.errors
    a, b = eps[0].timeline_events, eps2[0].timeline_events
    assert [(e.text, e.start_seconds, e.end_seconds, e.event_type) for e in a] == \
           [(e.text, e.start_seconds, e.end_seconds, e.event_type) for e in b]


def test_csv_with_field_mapping_and_missing_columns(app_cfg):
    csv_ok = ("Episode,Scene,Character,Line,In,Out\n1,1,ALEX,\"You lied!\",00:01.0,00:02.5\n"
              "1,1,,The door slams.,00:02.5,00:03.0\n2,1,SAM,\"Run.\",00:00.5,00:01.5\n")
    eps, rep, _ = _load(app_cfg, [("lines.csv", csv_ok)])
    assert [e.episode_id for e in eps] == ["EP001", "EP002"]
    assert eps[0].timeline_events[0].event_type == EventType.dialogue
    assert eps[0].timeline_events[1].event_type == EventType.description
    eps2, rep2, _ = _load(app_cfg, [("weird.csv", "foo,bar\n1,2\n")])
    assert eps2 == [] and any(i.code == "field_mapping_required" for i in rep2.errors)


def test_docx_parsing(app_cfg):
    docx = pytest.importorskip("docx")
    d = docx.Document()
    for line in make_screenplay(6).replace("**", "").split("\n"):
        d.add_paragraph(line)
    buf = io.BytesIO()
    d.save(buf)
    eps, rep, _ = _load(app_cfg, [("episode-6.docx", buf.getvalue())])
    assert not rep.errors and eps[0].episode_id == "EP006"
    assert any(e.speaker == "MORGAN" for e in eps[0].timeline_events)


def _minimal_pdf(lines: list[str]) -> bytes:
    esc = [l.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)") for l in lines]
    stream = "BT /F1 10 Tf 40 800 Td 12 TL " + " ".join(f"({l}) Tj T*" for l in esc) + " ET"
    objs = [
        "<< /Type /Catalog /Pages 2 0 R >>",
        "<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 842] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>",
        f"<< /Length {len(stream)} >>\nstream\n{stream}\nendstream",
        "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out, offsets = "%PDF-1.4\n", []
    for i, o in enumerate(objs, start=1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n{o}\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objs) + 1}\n0000000000 65535 f \n" + "".join(f"{o:010d} 00000 n \n" for o in offsets)
    out += f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF"
    return out.encode("latin-1")


def test_pdf_parsing_and_bad_pdf(app_cfg):
    pytest.importorskip("pypdf")
    lines = ["EPISODE NUMBER: Episode 8", "SCENE 01", "Scene heading: INT. VAULT - NIGHT", "Scene start: 00:00.00",
             "Scene end: 00:10.00", "ALEX (Afraid):", "Start: (00:01.00) | End: (00:03.00)", "Somebody is inside the vault.",
             "ACTION:", "Start: (00:03.00) | End: (00:05.00)", "The lights go out."]
    eps, rep, _ = _load(app_cfg, [("ep8.pdf", _minimal_pdf(lines))])
    assert not rep.errors, rep.errors
    assert eps[0].episode_id == "EP008"
    assert [e.event_type for e in eps[0].timeline_events] == [EventType.dialogue, EventType.action]
    _, rep2, _ = _load(app_cfg, [("broken.pdf", b"%PDF-1.4 garbage")])
    assert any(i.code in ("bad_pdf", "pdf_no_text") for i in rep2.errors)


# ---------------------------------------------------------------------------
# ZIP safety
# ---------------------------------------------------------------------------
def test_nested_zip_duplicates_and_system_files(app_cfg):
    inner = zip_bytes({"deep/episode-003.txt": make_screenplay(3)})
    outer = zip_bytes({
        "a/episode-001.txt": make_screenplay(1),
        "b/episode-001.txt": make_screenplay(1),  # identical content -> duplicate
        "c/episode-002.txt": make_screenplay(2),
        "d/episode-002.txt": make_screenplay(2, names=("CASEY", "DREW")),  # same basename, different content
        "__MACOSX/a/._episode-001.txt": "junk",
        "nested.zip": inner,
    })
    eps, rep, ing = load_dataset([("data.zip", outer)], app_cfg)
    statuses = {f.display_name: f.status for f in rep.files}
    assert statuses["data.zip/b/episode-001.txt"] == "duplicate"
    assert statuses["data.zip/__MACOSX/a/._episode-001.txt"] == "skipped"
    assert "data.zip/nested.zip/deep/episode-003.txt" in statuses
    ids = sorted(e.episode_id for e in eps)
    assert ids == ["EP001", "EP002", "EP002-B", "EP003"]
    assert any(i.code == "duplicate_episode_id" for i in rep.issues)


def test_unsafe_archive_paths_rejected(app_cfg):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("../../etc/evil.txt", make_screenplay(1))
        zf.writestr("/abs/evil2.txt", make_screenplay(2))
        zf.writestr("ok/episode-003.txt", make_screenplay(3))
    eps, rep, _ = load_dataset([("x.zip", buf.getvalue())], app_cfg)
    assert [e.episode_id for e in eps] == ["EP003"]
    assert sum(1 for i in rep.errors if i.code == "unsafe_archive_path") == 2


def test_size_limits(app_cfg):
    small = app_cfg.model_copy(update={"max_upload_mb": 0.001})
    eps, rep, _ = load_dataset([("e1.txt", make_screenplay(1).encode())], small)
    assert eps == [] and any(i.code == "upload_too_large" for i in rep.errors)
    tight = app_cfg.model_copy(update={"max_file_mb": 0.002})
    eps, rep, _ = load_dataset([("d.zip", zip_bytes({"e1.txt": make_screenplay(1)}))], tight)
    assert eps == [] and any(i.code == "file_too_large" for i in rep.errors)
    shallow = app_cfg.model_copy(update={"max_zip_depth": 1})
    nested = zip_bytes({"inner.zip": zip_bytes({"e1.txt": make_screenplay(1)})})
    eps, rep, _ = load_dataset([("d.zip", nested)], shallow)
    assert eps == [] and any(i.code == "zip_depth" for i in rep.errors)


def test_bad_zip_reported(app_cfg):
    eps, rep, _ = load_dataset([("broken.zip", b"PK\x03\x04 not really a zip")], app_cfg)
    assert eps == [] and any(i.code == "bad_zip" for i in rep.errors)
