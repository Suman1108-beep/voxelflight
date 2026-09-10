"""Normalize unseen SIH video and telemetry into one canonical scene layout."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np


VIDEO_EXTENSIONS = {".mp4", ".mov", ".m4v", ".avi", ".mkv"}
TELEMETRY_EXTENSIONS = {".csv", ".srt", ".json", ".txt"}

FIELD_ALIASES = {
    "latitude": {"latitude", "lat", "gpslatitude", "gpslat"},
    "longitude": {"longitude", "lon", "lng", "long", "gpslongitude", "gpslon"},
    "altitude_m": {
        "altitude",
        "alt",
        "altitudem",
        "gpsaltitude",
        "gpsalt",
        "relativealtitude",
        "height",
    },
    "timestamp": {
        "timestamp",
        "timestamps",
        "time",
        "times",
        "timems",
        "timeus",
        "timestampus",
        "timestampms",
        "timpstemp",
        "datetime",
        "elapsedtime",
    },
    "source_frame": {"sourceframe", "frame", "frameid", "framenumber", "frameindex", "imgid", "imageid"},
    "gps_accuracy_m": {"gpsaccuracy", "gpsaccuracym", "eph", "ephm", "hdop"},
    "gps_fix_type": {"fix", "fixtype", "gpsfix", "gpsfixtype"},
    "satellites": {"satellites", "numsat", "satellitecount", "gpssatellites"},
    "yaw_deg": {"yaw", "yawdeg", "heading", "headingdeg"},
    "pitch_deg": {"pitch", "pitchdeg", "gimbalpitch", "gimbalpitchdeg"},
    "roll_deg": {"roll", "rolldeg", "gimbalroll", "gimbalrolldeg"},
}


def _normalise_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.strip().lower())


def _float(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip().replace("\u2212", "-")
    if not text:
        return None
    match = re.search(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", text)
    if match is None:
        return None
    number = float(match.group(0))
    return number if math.isfinite(number) else None


def _int(value: Any) -> int | None:
    number = _float(value)
    return None if number is None else int(round(number))


def _first(mapping: dict[str, Any], names: Iterable[str]) -> Any:
    for name in names:
        if name in mapping and str(mapping[name]).strip():
            return mapping[name]
    return None


def _canonical_columns(fieldnames: Iterable[str]) -> dict[str, str]:
    normalised = {_normalise_name(name): name for name in fieldnames if name}
    result: dict[str, str] = {}
    for canonical, aliases in FIELD_ALIASES.items():
        # Prefer explicit canonical columns, then deterministic legacy aliases.
        preferred=['timestamps'] if canonical=='timestamp' else [_normalise_name(canonical)]
        for alias in [*preferred,*sorted(aliases)]:
            if alias in normalised:
                result[canonical] = normalised[alias]
                break
    return result


@dataclass(frozen=True)
class VideoInfo:
    path: Path
    frame_count: int
    fps: float
    width: int
    height: int
    duration_s: float


def probe_video(path: Path) -> VideoInfo:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not decode video: {path}")
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    capture.release()
    if frame_count <= 0 or fps <= 0 or width <= 0 or height <= 0:
        raise RuntimeError(f"Invalid video metadata: {path}")
    return VideoInfo(path, frame_count, fps, width, height, frame_count / fps)


def discover_one(root: Path, extensions: set[str], kind: str) -> Path | None:
    candidates = sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in extensions
    )
    if not candidates:
        return None
    if kind == "video":
        candidates.sort(key=lambda path: path.stat().st_size, reverse=True)
        if len(candidates) > 1 and candidates[0].stat().st_size == candidates[1].stat().st_size:
            raise RuntimeError(
                "Multiple equally likely videos found; pass --video explicitly: "
                + ", ".join(str(path) for path in candidates[:5])
            )
    elif kind == "telemetry":
        def telemetry_rank(path: Path) -> tuple[int, int, int, int]:
            name = _normalise_name(path.stem)
            positive = any(token in name for token in ("gps", "telemetry", "flight", "location"))
            negative = any(token in name for token in ("camera", "calibration", "manifest", "config"))
            canonical = name in {"frametelemetry", "telemetry", "gps"}
            depth = len(path.relative_to(root).parts)
            return (int(canonical), int(positive) - int(negative), -depth, path.stat().st_size)

        candidates.sort(key=telemetry_rank, reverse=True)
    return candidates[0]


def _normalise_timestamps(values: np.ndarray, video_duration_s: float) -> tuple[np.ndarray, float]:
    values = values.astype(np.float64)
    values -= np.nanmin(values)
    span = float(np.nanmax(values))
    if span <= 0:
        return values, 1.0
    scales = np.asarray([1.0, 1e-3, 1e-6, 1e-9])
    if video_duration_s > 0:
        score = np.abs(np.log(np.maximum(span * scales, 1e-9) / video_duration_s))
        scale = float(scales[int(np.argmin(score))])
    else:
        scale = 1.0
    return values * scale, scale


def _read_csv_records(path: Path) -> list[dict[str, Any]]:
    with path.open(errors="replace") as handle:
        sample = handle.read(16384)
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
    except csv.Error:
        dialect = csv.excel
    with path.open(newline="", errors="replace") as handle:
        return list(csv.DictReader(handle, dialect=dialect))


def _read_json_records(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(errors="replace"))
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    if isinstance(payload, dict):
        for key in ("records", "telemetry", "samples", "data", "frames"):
            rows = payload.get(key)
            if isinstance(rows, list):
                return [row for row in rows if isinstance(row, dict)]
    raise ValueError(f"No telemetry record list found in {path}")


def _srt_timestamp_seconds(value: str) -> float:
    hours, minutes, remainder = value.replace(",", ".").split(":")
    return int(hours) * 3600 + int(minutes) * 60 + float(remainder)


def _read_srt_records(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(errors="replace")
    blocks = re.split(r"\n\s*\n", text.strip())
    records: list[dict[str, Any]] = []
    for block in blocks:
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        time_line = next((line for line in lines if "-->" in line), None)
        if time_line is None:
            continue
        timestamp = _srt_timestamp_seconds(time_line.split("-->", 1)[0].strip())
        body = " ".join(line for line in lines if line != time_line)
        patterns = {
            "latitude": r"(?:latitude|lat)\s*[:=]\s*([-+]?\d+(?:\.\d+)?)",
            "longitude": r"(?:longitude|lon|lng)\s*[:=]\s*([-+]?\d+(?:\.\d+)?)",
            "altitude_m": r"(?:relative_altitude|altitude|alt)\s*[:=]\s*([-+]?\d+(?:\.\d+)?)",
            "yaw_deg": r"(?:yaw|heading)\s*[:=]\s*([-+]?\d+(?:\.\d+)?)",
            "pitch_deg": r"pitch\s*[:=]\s*([-+]?\d+(?:\.\d+)?)",
            "roll_deg": r"roll\s*[:=]\s*([-+]?\d+(?:\.\d+)?)",
        }
        record: dict[str, Any] = {"timestamp": timestamp}
        for name, pattern in patterns.items():
            match = re.search(pattern, body, flags=re.IGNORECASE)
            if match:
                record[name] = match.group(1)
        if "latitude" in record and "longitude" in record:
            records.append(record)
    return records


def read_telemetry(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".csv" or suffix == ".txt":
        return _read_csv_records(path)
    if suffix == ".json":
        return _read_json_records(path)
    if suffix == ".srt":
        return _read_srt_records(path)
    raise ValueError(f"Unsupported telemetry format: {path.suffix}")


def canonicalise_telemetry(
    records: list[dict[str, Any]], video: VideoInfo
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not records:
        raise ValueError("Telemetry contains no records")
    columns = _canonical_columns(records[0].keys())
    missing = {"latitude", "longitude"} - columns.keys()
    if missing:
        raise ValueError(f"Telemetry is missing required fields: {sorted(missing)}")

    frame_offset = 0
    original_record_count = len(records)
    if "source_frame" in columns:
        canonical_frame_column=_normalise_name(columns['source_frame'])=='sourceframe'
        frame_values = np.asarray(
            [_float(row.get(columns["source_frame"])) for row in records],
            dtype=np.float64,
        )
        segment_match = re.search(
            r"(?:segment|frames?)[_-]?(\d+)[_-](\d+)",
            video.path.stem,
            flags=re.IGNORECASE,
        )
        if segment_match and not canonical_frame_column:
            segment_start, segment_end = map(int, segment_match.groups())
            selected = [
                row
                for row, frame in zip(records, frame_values)
                if np.isfinite(frame) and segment_start <= frame <= segment_end
            ]
            if len(selected) >= 2:
                records = selected
                frame_offset = segment_start
        elif not canonical_frame_column and np.isfinite(frame_values).sum() >= 2:
            finite_frames = frame_values[np.isfinite(frame_values)]
            minimum, maximum = int(finite_frames.min()), int(finite_frames.max())
            span = maximum - minimum + 1
            if minimum != 0 and 0.8 * video.frame_count <= span <= 1.25 * video.frame_count:
                frame_offset = minimum

    raw_timestamps = np.asarray(
        [_float(row.get(columns.get("timestamp", ""))) for row in records],
        dtype=np.float64,
    )
    has_time = "timestamp" in columns and np.isfinite(raw_timestamps).sum() >= 2
    if has_time:
        finite = np.isfinite(raw_timestamps)
        fill = np.interp(
            np.arange(len(records)), np.flatnonzero(finite), raw_timestamps[finite]
        )
        if _normalise_name(columns['timestamp'])=='timestamps':
            # Canonical timestamps are already relative to the video. A log
            # beginning at t=2s must not silently be shifted back to zero.
            timestamp_s,time_scale=fill,1.0
        else:
            timestamp_s, time_scale = _normalise_timestamps(fill, video.duration_s)
    else:
        timestamp_s = np.linspace(0.0, video.duration_s, len(records))
        time_scale = None

    result: list[dict[str, Any]] = []
    for index, (row, timestamp) in enumerate(zip(records, timestamp_s)):
        lat = _float(row.get(columns["latitude"]))
        lon = _float(row.get(columns["longitude"]))
        if lat is None or lon is None or not (-90 <= lat <= 90 and -180 <= lon <= 180):
            continue
        explicit_frame = _int(row.get(columns.get("source_frame", "")))
        if explicit_frame is not None:
            explicit_frame -= frame_offset
        source_frame = explicit_frame if explicit_frame is not None else round(timestamp * video.fps)
        canonical = {
            "sample_index": index,
            "timestamp_s": round(float(timestamp), 6),
            "source_frame": min(max(int(source_frame), 0), video.frame_count - 1),
            "latitude": lat,
            "longitude": lon,
            "altitude_m": _float(row.get(columns.get("altitude_m", ""))),
            "gps_accuracy_m": _float(row.get(columns.get("gps_accuracy_m", ""))),
            "gps_fix_type": _int(row.get(columns.get("gps_fix_type", ""))),
            "satellites": _int(row.get(columns.get("satellites", ""))),
            "yaw_deg": _float(row.get(columns.get("yaw_deg", ""))),
            "pitch_deg": _float(row.get(columns.get("pitch_deg", ""))),
            "roll_deg": _float(row.get(columns.get("roll_deg", ""))),
        }
        result.append(canonical)
    if len(result) < 2:
        raise ValueError("Fewer than two valid GPS telemetry samples remain")
    metadata = {
        "source_columns": columns,
        "source_records": original_record_count,
        "selected_records": len(records),
        "valid_records": len(result),
        "source_frame_offset": frame_offset,
        "timestamp_scale_to_seconds": time_scale,
        "timestamp_inferred": not has_time,
    }
    return result, metadata


def _sha256_prefix(path: Path, limit: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        digest.update(handle.read(limit))
    return digest.hexdigest()


def prepare_scene(
    input_root: Path,
    output_root: Path,
    *,
    video_path: Path | None = None,
    telemetry_path: Path | None = None,
    calibration_path: Path | None = None,
) -> dict[str, Any]:
    input_root = input_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    video_path = (video_path or discover_one(input_root, VIDEO_EXTENSIONS, "video"))
    telemetry_path = (
        telemetry_path or discover_one(input_root, TELEMETRY_EXTENSIONS, "telemetry")
    )
    if video_path is None:
        raise FileNotFoundError(f"No drone video found below {input_root}")
    if telemetry_path is None:
        raise FileNotFoundError(f"No GPS telemetry found below {input_root}")
    video_path, telemetry_path = video_path.resolve(), telemetry_path.resolve()
    video = probe_video(video_path)
    canonical_rows, telemetry_metadata = canonicalise_telemetry(
        read_telemetry(telemetry_path), video
    )

    link = output_root / f"flight{video_path.suffix.lower()}"
    if link.exists() or link.is_symlink():
        if link.resolve() != video_path:
            raise FileExistsError(f"Canonical video already points elsewhere: {link}")
    else:
        link.symlink_to(video_path)

    fieldnames = list(canonical_rows[0])
    with (output_root / "frame_telemetry.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(canonical_rows)

    calibration_destination = None
    if calibration_path is not None:
        calibration_path = calibration_path.resolve()
        calibration_destination = output_root / f"camera{calibration_path.suffix.lower()}"
        if not calibration_destination.exists():
            calibration_destination.symlink_to(calibration_path)

    manifest = {
        "contract_version": 1,
        "input_root": str(input_root),
        "video": {
            "path": str(link.resolve()),
            "canonical_link": str(link.absolute()),
            "bytes": video_path.stat().st_size,
            "sha256_first_8mib": _sha256_prefix(video_path),
            "frame_count": video.frame_count,
            "fps": video.fps,
            "width": video.width,
            "height": video.height,
            "duration_s": video.duration_s,
        },
        "telemetry": {
            "source": str(telemetry_path),
            "canonical": str((output_root / "frame_telemetry.csv").resolve()),
            **telemetry_metadata,
        },
        "calibration": str(calibration_destination.resolve()) if calibration_destination else None,
        "validation": {
            "has_mandatory_video": True,
            "has_mandatory_gps": True,
            "has_flight_metadata": True,
            "resolution_is_1080p_or_better": video.width >= 1920 and video.height >= 1080,
            "duration_within_official_reference_s": video.duration_s <= 600.0,
        },
    }
    (output_root / "scene_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest
