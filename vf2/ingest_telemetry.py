"""Real drone telemetry -> VoxelFlight per-frame telemetry CSV
(source_frame,timestamp_s,latitude,longitude,altitude_m[,gps_eph_m][,baro_alt_m]), one row per video frame.

  python ingest_telemetry.py --video flight.mp4 --telemetry flight.SRT|.csv|.gpx|.json --output telemetry_v2.csv

Inputs (auto-detected by extension, then content):
  DJI .SRT   every common firmware style: [latitude: ..] [longitude: ..] [rel_alt: .. abs_alt: ..] / [altitude: ..];
             GPS(lon,lat,sats) BAROMETER:/Hb:; Mini/P4 'GPS (lon, lat, sats), D 12.3m, H 30.0m'; M300 GPS(lat,lon,0.0M).
             SRT times are video times. abs_alt/[altitude] -> altitude_m; rel_alt/H/BAROMETER -> baro_alt_m.
  GPX        trkpt lat/lon/ele/time; video start = --video-start or the first trkpt.
  CSV/JSON   lat/lon/alt + time columns (case-insensitive, units in (feet)/[ft]/_ms honoured; AirData and DJI
             flight-record names; AirData isVideo marks the recording start). Pipeline-schema CSVs are passed through.
GPS is interpolated linearly onto frame i at time i / fps (so VFR video uses its mean rate)."""
import argparse, csv, json, math, os, re, shutil, subprocess, sys
from datetime import datetime, timezone
import numpy as np

PIPELINE_COLS = ("source_frame", "timestamp_s", "latitude", "longitude", "altitude_m")
FEET = {"ft", "feet", "foot"}
TIME_UNITS = {"s": 1.0, "sec": 1.0, "second": 1.0, "seconds": 1.0, "ms": 1e-3, "millisecond": 1e-3, "milliseconds": 1e-3,
              "msec": 1e-3, "us": 1e-6, "microsecond": 1e-6, "microseconds": 1e-6}
ALIASES = {  # normalised column name, best first
    "lat": ["latitude", "lat", "gpslatitude", "gpslat"],
    "lon": ["longitude", "lon", "lng", "long", "longtitude", "gpslongitude", "gpslon"],
    "alt": ["altitudeabovesealevel", "altitudemsl", "absalt", "altitude", "alt", "gpsaltitude", "gpsalt", "elevation", "ele"],
    "rel": ["baroalt", "heightabovetakeoff", "relalt", "relativealtitude", "height", "barometer", "baro", "h"],
    "eph": ["gpseph", "eph", "hacc", "horizontalaccuracy", "accuracy"],
    "time": ["timestamp", "time", "t", "flytime", "elapsedtime", "elapsed", "datetime", "updatetime", "gpstime", "date"],
    "frame": ["sourceframe", "frame", "frameindex", "framenumber"],
    "video": ["isvideo"],
}


DURATION = re.compile(r"(?:(\d+)h\s*)?(?:(\d+)m\s*)?(\d+(?:\.\d+)?)s")


class TelemetryError(ValueError):
    pass


def parse_datetime(s):
    """Seconds since the epoch for '2022-08-07 13:40:40,774', '2017.8.5 14:11:51', ISO 8601 with Z/+hh:mm; naive = UTC."""
    m = re.search(r"(\d{4})[-./](\d{1,2})[-./](\d{1,2})[ T](\d{1,2}):(\d{2}):(\d{2})(?:[.,](\d+))?(?:\s*(Z|[+-]\d{2}:?\d{2})\b)?", str(s))
    if not m:
        return None
    y, mo, d, H, M, S, frac, tz = m.groups()
    t = datetime(int(y), int(mo), int(d), int(H), int(M), int(S), tzinfo=timezone.utc).timestamp() + (float("0." + frac) if frac else 0.0)
    if tz and tz != "Z":
        t -= (1 if tz[0] == "+" else -1) * (int(tz[1:3]) * 3600 + int(tz[-2:]) * 60)
    return t


def _num(v):
    s = "" if v is None else str(v).strip()
    s = s.replace(",", ".") if re.fullmatch(r"-?\d+,\d+", s) else s  # decimal comma
    m = re.search(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", s)
    return float(m.group(0)) if m else np.nan


def _colkey(name):
    """'OSD.height [ft]' -> ('height', 'ft'); 'altitude_above_seaLevel(feet)' -> ('altitudeabovesealevel', 'feet'); 'time_ms' -> ('time', 'ms')."""
    low = str(name).strip().lower()
    m = re.search(r"[\[(]\s*([^\])]*?)\s*[\])]", low)
    unit = m.group(1) if m else ""
    base = re.sub(r"[\[(].*?[\])]", "", low).strip().split(".")[-1]
    if not unit:
        m = re.search(r"_(ms|us|s|sec|m|ft|feet|meters|metres)$", base)
        if m:
            unit, base = m.group(1), base[:m.start()]
    return re.sub(r"[^a-z0-9]", "", base), unit


# ---------------------------------------------------------------- readers -> raw samples dict(t, absolute, lat, lon, alt, rel, eph)

def _gps_tuple(body):
    """DJI 'GPS(a, b, c)': (lon, lat, satellites|alt'm') on most firmware, (lat, lon, 'n.nM') on M300; Autel uses W:/N: prefixes."""
    parts = [p.strip() for p in body.split(",")]
    hemi = [re.match(r"([NSEW])\s*:\s*([\d.]+)", p) for p in parts[:2]]
    third = parts[2] if len(parts) > 2 else ""
    if len(parts) >= 2 and all(hemi):
        v = {h[1]: float(h[2]) for h in hemi}
        return v["N"] if "N" in v else -v["S"], v["E"] if "E" in v else -v["W"], third
    a, b = float(parts[0]), float(parts[1])
    lat, lon = (a, b) if re.fullmatch(r"[-+]?\d+(?:\.\d+)?M", third) else (b, a)
    if abs(lat) > 90 >= abs(lon):
        lat, lon = lon, lat
    return lat, lon, third


def read_srt(text):
    recs, sats_like = [], True
    for block in re.split(r"\n\s*\n", text.replace("\ufeff", "").replace("\r", "").strip()):
        b = re.sub(r"<[^>]*>", " ", block)
        kv = {k.lower(): float(v) for k, v in re.findall(r"([A-Za-z_][A-Za-z_.]*)\s*:\s*([-+]?\d+(?:\.\d+)?)", b)}
        lat, lon = kv.get("latitude", kv.get("lat")), kv.get("longitude", kv.get("longtitude", kv.get("lon")))
        g3 = np.nan; alt = kv.get("abs_alt", kv.get("altitude", np.nan))
        g = re.search(r"\b(?:GPS|RTK)\s*\(([^)]*)\)", b)
        if lat is None and g:
            try:
                lat, lon, third = _gps_tuple(g.group(1))
            except (ValueError, IndexError, KeyError):
                continue
            if re.fullmatch(r"[-+]?\d+(?:\.\d+)?\s*m", third):
                alt = float(third[:-1]) if np.isnan(alt) else alt
            elif re.fullmatch(r"[-+]?\d+(?:\.\d+)?", third):
                g3 = float(third); sats_like &= g3 == int(g3) and 0 <= g3 <= 50
        if lat is None or lon is None:
            continue
        h = re.search(r"(?<![\w.])H\s+([-+]?\d+(?:\.\d+)?)\s*m\b", b)
        rel = next((kv[k] for k in ("rel_alt", "barometer", "hb") if k in kv), float(h.group(1)) if h else np.nan)
        m = re.search(r"(\d+):(\d{2}):(\d{2})[,.](\d{1,3})\s*-->", b)
        t = int(m[1]) * 3600 + int(m[2]) * 60 + int(m[3]) + int(m[4].ljust(3, "0")) / 1000 if m else np.nan
        recs.append((t, parse_datetime(b) or np.nan, lat, lon, alt, rel, g3))
    if not recs:
        raise TelemetryError("no valid GPS rows in SRT (no latitude/longitude or GPS(...) fields found)")
    t, clock, lat, lon, alt, rel, g3 = map(np.array, zip(*recs))
    if np.isnan(alt).all() and np.isfinite(g3).any() and not sats_like:
        alt = g3  # unitless 3rd GPS value that cannot be a satellite count -> altitude
    absolute = np.isfinite(t).sum() < 0.5 * len(t)  # no SRT clock ('-->') -> fall back to the per-block date
    tt = clock if absolute else t
    keep = np.isfinite(tt)
    return {"t": tt[keep], "absolute": absolute, "lat": lat[keep], "lon": lon[keep], "alt": alt[keep], "rel": rel[keep], "eph": None}


def read_gpx(path):
    import xml.etree.ElementTree as ET
    rows = []
    root = ET.parse(path).getroot()
    pts = [e for e in root.iter() if e.tag.rsplit("}", 1)[-1] == "trkpt"] or [e for e in root.iter() if e.tag.rsplit("}", 1)[-1] in ("rtept", "wpt")]
    for p in pts:
        child = {c.tag.rsplit("}", 1)[-1]: (c.text or "").strip() for c in p}
        rows.append({"lat": p.get("lat"), "lon": p.get("lon"), "ele": child.get("ele", ""), "time": child.get("time", "")})
    if not rows:
        raise TelemetryError(f"no <trkpt> points in {path}")
    return rows


def read_csv_rows(path):
    with open(path, newline="", encoding="utf-8-sig", errors="replace") as f:
        lines = f.read().splitlines()
    if lines and lines[0].lower().startswith("sep="):  # Excel hint line in some flight-log exports
        lines = lines[1:]
    return list(csv.DictReader(lines, delimiter=max(",;\t|", key=lines[0].count) if lines else ","))


def read_json_rows(path):
    payload = json.load(open(path, encoding="utf-8-sig"))
    if isinstance(payload, dict):
        payload = next((payload[k] for k in ("records", "telemetry", "samples", "data", "frames", "points") if isinstance(payload.get(k), list)), None)
    if not isinstance(payload, list):
        raise TelemetryError(f"{path}: expected a JSON list of {{t, lat, lon, alt}} objects")
    return [r for r in payload if isinstance(r, dict)]


def read_table(rows, fps=None):
    """Rows of dicts (CSV / JSON / GPX) -> raw samples."""
    if not rows:
        raise TelemetryError("no valid GPS rows (empty table)")
    cols = {}
    for name in dict.fromkeys(k for r in rows for k in r if k):
        cols.setdefault(_colkey(name)[0], (name, _colkey(name)[1]))
    pick = lambda f: next((cols[k] for k in ALIASES[f] if k in cols), (None, ""))
    col = lambda name: np.array([_num(r.get(name)) if r.get(name) not in (None, "") else np.nan for r in rows]) if name else None
    (latc, _), (lonc, _) = pick("lat"), pick("lon")
    if latc is None or lonc is None:
        raise TelemetryError(f"no latitude/longitude columns among {list(cols)}")
    def metres(f):
        c, u = pick(f)
        return None if c is None else col(c) * (0.3048 if u in FEET else 1.0)
    out = {"lat": col(latc), "lon": col(lonc), "alt": metres("alt"), "rel": metres("rel"), "eph": metres("eph")}
    (tc, tu), (fc, _) = pick("time"), pick("frame")
    if tc is None and fc is not None and fps:
        out["t"], out["absolute"] = col(fc) / fps, False
    elif tc is None:
        raise TelemetryError(f"no time column among {list(cols)} (expected time/timestamp/time_s/datetime/t)")
    else:
        raw = [str(r.get(tc, "")).strip() for r in rows]
        num = []
        for v in raw:
            try:
                num.append(float(v))
            except ValueError:
                num.append(np.nan)
        num = np.array(num)
        if np.isfinite(num).sum() >= 0.5 * len(num):
            scale = TIME_UNITS.get(tu) or (1e-3 if _colkey(tc)[0].endswith("ms") else None)
            mx = np.nanmax(np.abs(num))
            if scale is None:
                scale = 1e-6 if mx > 1e14 else 1e-3 if mx > 1e11 else 1.0
            out["t"] = num * scale
            out["absolute"] = np.nanmax(np.abs(out["t"])) > 1e8  # epoch seconds
        elif sum(bool(DURATION.fullmatch(v)) for v in raw) >= 0.5 * len(raw):  # DJI flight records: OSD.flyTime '2m 13.4s'
            dur = [DURATION.fullmatch(v) for v in raw]
            out["t"] = np.array([float(m[1] or 0) * 3600 + float(m[2] or 0) * 60 + float(m[3]) if m else np.nan for m in dur])
            out["absolute"] = False
        else:
            out["t"], out["absolute"] = np.array([parse_datetime(v) or np.nan for v in raw]), True
            if np.isnan(out["t"]).all():
                raise TelemetryError(f"cannot parse time column '{tc}' (e.g. '{raw[0]}')")
    vc = pick("video")[0]
    on = lambda v: str(v).strip().lower() in ("true", "yes") or _num(v) > 0     # 1/0 or TRUE/FALSE
    rec = np.array([on(r.get(vc, "")) for r in rows]) & np.isfinite(out["t"]) if vc is not None else np.zeros(0, bool)
    if rec.any():  # AirData / flight-record logs: the video starts where recording starts
        out["video_start"] = float(out["t"][rec][0])
        segments = int(rec[0]) + int((np.diff(rec.astype(int)) == 1).sum())
        if segments > 1:
            print(f"note: log has {segments} recording segments; aligned to the first (use --time-offset for another)", file=sys.stderr)
    return out


# ---------------------------------------------------------------- core

def detect_format(path):
    ext = os.path.splitext(path)[1].lower()
    if ext in (".srt", ".gpx", ".json", ".csv"):
        return ext[1:]
    head = open(path, encoding="utf-8-sig", errors="replace").read(4096)
    if "<gpx" in head or "<trkpt" in head:
        return "gpx"
    if head.lstrip()[:1] in "[{" and head.strip():
        return "json"
    if "-->" in head or re.search(r"\bGPS\s*\(|\[\s*latitude", head):
        return "srt"
    return "csv"


def is_pipeline_csv(path):
    if not str(path).lower().endswith(".csv"):
        return False
    with open(path, newline="") as f:
        header = next(csv.reader(f), [])
    return set(PIPELINE_COLS) <= {h.strip() for h in header}


def probe_video(path):
    """(frame_count, fps) via OpenCV (same count the pipeline uses), falling back to ffprobe."""
    try:
        import cv2
        cap = cv2.VideoCapture(path); n, fps = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), float(cap.get(cv2.CAP_PROP_FPS)); cap.release()
        if n > 0 and fps > 0:
            return n, fps
    except ImportError:
        pass
    try:
        s = json.loads(subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets", "-show_entries",
                                       "stream=nb_read_packets,avg_frame_rate", "-of", "json", path], capture_output=True, text=True).stdout)["streams"][0]
        num, den = s["avg_frame_rate"].split("/")
        n, fps = int(s["nb_read_packets"]), float(num) / float(den)
        if n > 0 and fps > 0:
            return n, fps
    except (OSError, ValueError, KeyError, IndexError, ZeroDivisionError):
        pass
    raise TelemetryError(f"cannot read frame count / fps of {path}; pass --frames and --fps")


def load_samples(path, fps=None):
    fmt = detect_format(path)
    if fmt == "srt":
        return fmt, read_srt(open(path, encoding="utf-8-sig", errors="replace").read())
    rows = read_gpx(path) if fmt == "gpx" else read_json_rows(path) if fmt == "json" else read_csv_rows(path)
    return fmt, read_table(rows, fps)


def to_frames(s, n, fps, video_start=None, time_offset=0.0, alt_offset=0.0, swap_latlon=False):
    """Raw samples -> per-frame dict (frame, t, lat, lon, alt, baro|None, eph|None)."""
    lat, lon = (s["lon"], s["lat"]) if swap_latlon else (s["lat"], s["lon"])
    t = np.asarray(s["t"], float)
    ok = np.isfinite(t) & np.isfinite(lat) & np.isfinite(lon) & (np.abs(lat) <= 90) & (np.abs(lon) <= 180) & ~((lat == 0) & (lon == 0))
    if ok.sum() < 2:
        raise TelemetryError(f"no valid GPS rows ({int(ok.sum())} usable of {len(t)}; 0,0 / out-of-range / missing-time rows are dropped)")
    sel = lambda a: None if a is None else np.asarray(a, float)[ok]
    t, lat, lon, alt, rel, eph = sel(t), sel(lat), sel(lon), sel(s["alt"]), sel(s["rel"]), sel(s["eph"])
    bad = np.flatnonzero(np.diff(t) < 0)
    if len(bad):
        raise TelemetryError(f"timestamps not increasing: sample {bad[0] + 1} at t={t[bad[0] + 1]:.3f} s follows t={t[bad[0]]:.3f} s")
    if s["absolute"]:
        t = t - (video_start if video_start is not None else s.get("video_start") if s.get("video_start") is not None else t[0])
    elif s.get("video_start") is not None:
        t = t - s["video_start"]
    t = t + time_offset
    u = np.concatenate([[True], np.diff(t) > 0])  # duplicate timestamps: keep the first
    t, lat, lon = t[u], lat[u], lon[u]
    alt, rel, eph = [None if a is None else a[u] for a in (alt, rel, eph)]
    dur = n / fps
    cover = min(t[-1], dur) - max(t[0], 0.0)
    if cover < 0.5 * dur:
        raise TelemetryError(f"telemetry covers {max(cover, 0):.1f} s of the {dur:.1f} s video (telemetry spans {t[0]:.1f}..{t[-1]:.1f} s in video time); "
                             f"need at least half. Check --time-offset / --video-start")
    tf = np.arange(n) / fps
    interp = lambda a: None if a is None or np.isfinite(a).sum() < 1 else np.interp(tf, t[np.isfinite(a)], a[np.isfinite(a)])
    baro = interp(rel)
    A = interp(alt)
    if A is None:
        if baro is not None:
            print("note: no absolute altitude in telemetry; altitude_m = relative/barometric height + --alt-offset", file=sys.stderr)
            A = baro + alt_offset
        else:
            print("WARNING: telemetry has no altitude at all; altitude_m = --alt-offset for every frame", file=sys.stderr)
            A = np.full(n, float(alt_offset))
    elif alt_offset:
        A = A + alt_offset
    return {"frame": np.arange(n), "t": tf, "lat": np.interp(tf, t, lat), "lon": np.interp(tf, t, lon), "alt": A,
            "baro": baro, "eph": interp(eph), "span": (float(t[0]), float(t[-1])), "rows": len(t)}


def write_csv(F, out):
    cols = list(PIPELINE_COLS) + (["gps_eph_m"] if F["eph"] is not None else []) + (["baro_alt_m"] if F["baro"] is not None else [])
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.writer(f); w.writerow(cols)
        for i in range(len(F["frame"])):
            row = [int(F["frame"][i]), f"{F['t'][i]:.6f}", f"{F['lat'][i]:.8f}", f"{F['lon'][i]:.8f}", f"{F['alt'][i]:.3f}"]
            row += [f"{F['eph'][i]:.3f}"] if F["eph"] is not None else []
            row += [f"{F['baro'][i]:.3f}"] if F["baro"] is not None else []
            w.writerow(row)


def path_length_m(lat, lon):
    p, l = np.radians(lat), np.radians(lon)
    a = np.sin(np.diff(p) / 2) ** 2 + np.cos(p[:-1]) * np.cos(p[1:]) * np.sin(np.diff(l) / 2) ** 2
    return float((2 * 6371008.8 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))).sum())


def convert(telemetry, output, video=None, frames=None, fps=None, video_start=None, time_offset=0.0, alt_offset=0.0, swap_latlon=False):
    """Convert any supported telemetry file to the per-frame pipeline CSV. Returns the per-frame dict (None on pass-through)."""
    if not os.path.isfile(telemetry):
        raise TelemetryError(f"telemetry file not found: {telemetry}")
    if is_pipeline_csv(telemetry) and not swap_latlon:
        if os.path.abspath(telemetry) != os.path.abspath(output):
            shutil.copyfile(telemetry, output)
        print(f"telemetry: {telemetry} already in pipeline schema -> {output}")
        return None
    if not (frames and fps):
        if not video:
            raise TelemetryError("need --video (or --frames and --fps) to know the frame timestamps")
        n, r = probe_video(video)
        frames, fps = frames or n, fps or r
    vs = None
    if video_start:
        vs = parse_datetime(video_start)
        if vs is None:
            raise TelemetryError(f"cannot parse --video-start '{video_start}' (use ISO 8601, e.g. 2026-10-04T10:15:30Z)")
    fmt, s = load_samples(telemetry, fps)
    F = to_frames(s, int(frames), float(fps), vs, time_offset, alt_offset, swap_latlon)
    write_csv(F, output)
    print(f"telemetry: {fmt} {F['rows']} rows over {F['span'][0]:.1f}..{F['span'][1]:.1f} s -> {frames} frames @ {fps:.3f} fps "
          f"({frames / fps:.1f} s); flown {path_length_m(F['lat'], F['lon']):.1f} m; altitude {F['alt'].min():.1f}..{F['alt'].max():.1f} m"
          f"{'; baro yes' if F['baro'] is not None else ''} -> {output}")
    return F


def ensure_pipeline_telemetry(telemetry, video, output_dir):
    """Pipeline hook: the per-frame CSV to use (input itself if already in the schema, else <output_dir>/telemetry_v2.csv)."""
    out = os.path.join(output_dir, "telemetry_v2.csv")
    try:
        if is_pipeline_csv(telemetry):
            return telemetry
        convert(telemetry, out, video=video)
    except (ValueError, OSError, SyntaxError) as e:  # TelemetryError, bad JSON, bad XML (ParseError is a SyntaxError)
        raise SystemExit(f"telemetry error ({telemetry}): {e}")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", help="video the telemetry belongs to (frame count + fps)")
    ap.add_argument("--telemetry", required=True); ap.add_argument("--output", required=True)
    ap.add_argument("--frames", type=int, help="override video frame count"); ap.add_argument("--fps", type=float, help="override video fps")
    ap.add_argument("--video-start", help="ISO time of the first video frame (GPX / absolute-time logs); default: first sample")
    ap.add_argument("--time-offset", type=float, default=0.0, help="seconds added to telemetry times after alignment (+ = telemetry later)")
    ap.add_argument("--alt-offset", type=float, default=0.0, help="metres added to altitude_m (e.g. take-off elevation when only relative height exists)")
    ap.add_argument("--swap-latlon", action="store_true", help="swap latitude/longitude (for an unusual GPS(...) order)")
    a = ap.parse_args(argv)
    try:
        convert(a.telemetry, a.output, a.video, a.frames, a.fps, a.video_start, a.time_offset, a.alt_offset, a.swap_latlon)
    except (ValueError, OSError, SyntaxError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
