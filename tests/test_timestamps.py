import pytest

from core.timestamp_utils import (
    extract_timestamps, format_timestamp, merge_intervals, parse_time_range, parse_timestamp,
    segments_total_duration, union_duration,
)


@pytest.mark.parametrize("raw,expected", [
    ("00:11.30", 11.3), ("01:23.70", 83.7), ("1:05", 65.0), ("(02:18.00)", 138.0),
    ("00:01:02.500", 62.5), ("00:00:01,250", 1.25), ("1:02:03", 3723.0), ("01:00:00.5", 3600.5),
    ("12.5", 12.5), ("12.5s", 12.5), ("12 sec", 12.0), ("1m 05.5s", 65.5), ("1h2m3s", 3723.0), ("90 seconds", 90.0),
])
def test_valid_formats_convert_to_seconds(raw, expected):
    p = parse_timestamp(raw)
    assert p.status == "valid", p.message
    assert p.seconds == pytest.approx(expected)


@pytest.mark.parametrize("raw", ["00:75.00", "00:61:00", "1:60:00", "12:3x", "1:2:3:4:5", "99:99:99"])
def test_invalid_timestamps_are_flagged_not_guessed(raw):
    p = parse_timestamp(raw)
    assert p.status == "invalid"
    assert p.seconds is None


@pytest.mark.parametrize("raw", [None, "", "  ", "N/A", "none", "--", "TBD"])
def test_missing_timestamps_never_invented(raw):
    p = parse_timestamp(raw)
    assert p.status == "missing"
    assert p.seconds is None


def test_frame_timecode_is_ambiguous_without_fps():
    assert parse_timestamp("00:00:01:12").status == "ambiguous"
    p = parse_timestamp("00:00:01:12", fps=24)
    assert p.status == "valid" and p.seconds == pytest.approx(1.5)
    assert parse_timestamp("00:00:01:30", fps=24).status == "invalid"


def test_extract_timestamps_from_text():
    assert extract_timestamps("Start: (00:01.00) | End: (00:04.80)") == ["00:01.00", "00:04.80"]
    assert extract_timestamps("00:00:01,000 --> 00:00:04,800") == ["00:00:01,000", "00:00:04,800"]
    assert extract_timestamps("no times here") == []


@pytest.mark.parametrize("line,start,end,rest", [
    ("Start: (00:01.00) | End: (00:04.80)", "00:01.00", "00:04.80", ""),
    ("start: 00:01.00, end: 00:04.80", "00:01.00", "00:04.80", ""),
    ("00:00:01,000 --> 00:00:04,800", "00:00:01,000", "00:00:04,800", ""),
    ("[00:12.5 - 00:14.0] ALEX: Hello", "00:12.5", "00:14.0", "ALEX: Hello"),
    ("Start: (00:75.00) | End: (00:80.00)", "00:75.00", "00:80.00", ""),
])
def test_parse_time_range(line, start, end, rest):
    assert parse_time_range(line) == (start, end, rest)


def test_parse_time_range_rejects_plain_text():
    assert parse_time_range("She walks to the window.") is None
    assert parse_time_range('"Meet me at 5 tonight."') is None


def test_format_timestamp():
    assert format_timestamp(83.7) == "01:23.70"
    assert format_timestamp(0) == "00:00.00"
    assert format_timestamp(3723.5) == "1:02:03.50"
    assert format_timestamp(None) == ""


def test_union_duration_does_not_double_count_overlaps():
    # camera 0-4.8, action 1-4.8 (overlaps), camera 4.8-6.8 (adjacent)
    ivs = [(0.0, 4.8), (1.0, 4.8), (4.8, 6.8)]
    assert union_duration(ivs) == pytest.approx(6.8)
    assert sum(b - a for a, b in ivs) > 6.8
    assert merge_intervals([(5, 6), (0, 1), (0.5, 2)]) == [(0, 2), (5, 6)]


def test_multi_segment_duration_is_sum_not_span():
    segs = [(10.0, 20.0), (40.0, 55.0)]
    assert segments_total_duration(segs) == pytest.approx(25.0)
    assert segments_total_duration(segs) != pytest.approx(55.0 - 10.0)
