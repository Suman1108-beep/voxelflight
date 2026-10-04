"""Tests for vf2/ingest_telemetry.py.  python3 -m pytest vf2/tests -q   (or: python3 vf2/tests/test_ingest_telemetry.py)"""
import csv, json, os, subprocess, sys, tempfile
HERE = os.path.dirname(os.path.abspath(__file__)); VF2 = os.path.dirname(HERE); DATA = os.path.join(HERE, "data")
sys.path.insert(0, VF2)
import numpy as np
import ingest_telemetry as it

N, FPS = 60, 30.0                                      # synthetic video: 60 frames, 2.0 s
LAT0, LON0, ALT0, REL0 = 47.37, 8.54, 450.0, 30.0
lat_at = lambda t: LAT0 + 1e-4 * t                     # ground truth track used by every synthetic log
lon_at = lambda t: LON0 + 2e-4 * t
alt_at = lambda t: ALT0 + 5.0 * t
rel_at = lambda t: REL0 + 5.0 * t

# One block per DJI firmware style. {k} block no., {t0}/{t1} SRT clock, {ms} millis, values from the track above.
SRT_STYLES = {
    "mavic3": ('<font size="28">FrameCnt: {k}, DiffTime: 100ms\n2024-05-01 10:00:00.{ms:03d}\n[iso: 100] [shutter: 1/1000.0] [fnum: 2.8] '
               '[ev: 0] [ct: 5500] [color_md : default] [focal_len: 24.00] [latitude: {lat:.7f}] [longitude: {lon:.7f}] '
               '[rel_alt: {rel:.3f} abs_alt: {alt:.3f}] </font>'),
    "mavic2_spaced": ('<font size="36">FrameCnt : {k}, DiffTime : 100ms\n2020-04-02 15:19:58,{ms:03d},764\n[iso : 120] [shutter : 1/2500.0] '
                      '[fnum : 280] [ev : 0] [latitude : {lat:.7f}] [longtitude : {lon:.7f}] [altitude: {alt:.6f}] </font>'),
    "phantom_baro": "HOME(8.5400,47.3700) 2017.08.05 14:11:51\nGPS({lon:.7f},{lat:.7f},16) BAROMETER:{rel:.2f}\nISO:100 Shutter:60 EV: Fnum:2.2",
    "gps_alt": "GPS({lon:.7f},{lat:.7f},{alt:.2f}) BAROMETER:{rel:.2f}",
    "mini": "F/2.8, SS 1000, ISO 100, EV 0, GPS ({lon:.7f}, {lat:.7f}, 18), D 12.30m, H {rel:.2f}m, H.S 2.30m/s, V.S 0.00m/s",
    "m300": "2022.06.21 16:06:17\nGPS({lat:.7f},{lon:.7f},0.0M) BAROMETER:{rel:.2f}M",
}
# what each style yields: (altitude_m source, baro_alt_m source)
EXPECT = {"mavic3": ("alt", "rel"), "mavic2_spaced": ("alt", None), "phantom_baro": ("rel", "rel"), "gps_alt": ("alt", "rel"),
          "mini": ("rel", "rel"), "m300": ("rel", "rel")}


def clock(t):
    ms = int(round(t * 1000))
    return f"{ms // 3600000:02d}:{ms // 60000 % 60:02d}:{ms // 1000 % 60:02d},{ms % 1000:03d}"


def make_srt(style, times, override=None):
    out = []
    for k, t in enumerate(times):
        v = dict(k=k + 1, ms=int(round(t * 1000)) % 1000, lat=lat_at(t), lon=lon_at(t), alt=alt_at(t), rel=rel_at(t))
        v.update((override or {}).get(k, {}))
        out.append(f"{k + 1}\n{clock(t)} --> {clock(t + 0.1)}\n" + SRT_STYLES[style].format(**v))
    return "\n\n".join(out) + "\n"


_video_cache = {}
def video(d):
    """Tiny real video file (60 frames @ 30 fps, 64x48) so frame count / fps come from OpenCV."""
    if "path" not in _video_cache:
        import cv2
        path = os.path.join(tempfile.mkdtemp(), "synthetic.avi")
        w = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), FPS, (64, 48))
        for i in range(N):
            img = np.zeros((48, 64, 3), np.uint8); img[:, i % 64] = 255; w.write(img)
        w.release()
        _video_cache["path"] = path
    return _video_cache["path"]


def write(d, name, text):
    p = os.path.join(d, name); open(p, "w").write(text); return p


def read(path):
    rows = list(csv.DictReader(open(path)))
    return rows, {k: np.array([float(r[k]) for r in rows]) for k in rows[0]}


def raises(fn, *a, **k):
    try:
        fn(*a, **k)
    except it.TelemetryError as e:
        return str(e)
    raise AssertionError("expected TelemetryError")


# ---------------------------------------------------------------- tests

def test_probe_synthetic_video(tmp_path):
    assert it.probe_video(video(tmp_path)) == (N, FPS)


def test_dji_srt_all_styles_interpolate_onto_frames(tmp_path):
    times = np.arange(21) * 0.1                                         # 10 Hz SRT covering 0..2.0 s
    t = np.arange(N) / FPS
    for style, (alt_src, baro_src) in EXPECT.items():
        out = os.path.join(tmp_path, style + ".csv")
        it.convert(write(tmp_path, style + ".SRT", make_srt(style, times)), out, video=video(tmp_path))
        rows, c = read(out)
        assert len(rows) == N and list(rows[0])[:5] == list(it.PIPELINE_COLS), style
        assert np.array_equal(c["source_frame"], np.arange(N)) and np.allclose(c["timestamp_s"], t, atol=1e-6), style
        assert np.allclose(c["latitude"], lat_at(t), atol=2e-7), style           # linear interpolation between 0.1 s samples
        assert np.allclose(c["longitude"], lon_at(t), atol=2e-7), style          # (lon, lat) order inside GPS(...) resolved
        assert np.allclose(c["altitude_m"], (alt_at if alt_src == "alt" else rel_at)(t), atol=0.01), style
        if baro_src:
            assert np.allclose(c["baro_alt_m"], rel_at(t), atol=0.01), style
        else:
            assert "baro_alt_m" not in c, style


def test_srt_zero_coordinates_ignored_and_ambiguous_order(tmp_path):
    times = np.arange(21) * 0.1
    srt = make_srt("mini", times, {5: dict(lat=0.0, lon=0.0)})                # a no-fix block mid-flight
    out = os.path.join(tmp_path, "o.csv")
    it.convert(write(tmp_path, "z.srt", srt), out, video=video(tmp_path))
    _, c = read(out)
    assert np.allclose(c["latitude"], lat_at(np.arange(N) / FPS), atol=2e-7)  # bridged by interpolation, no jump to 0,0
    # southern / western hemisphere in GPS(lon, lat, sats): |lon| > 90 so the order is unambiguous either way
    srt = "1\n00:00:00,000 --> 00:00:01,000\nGPS(-121.7458,48.0771,17) BAROMETER:3.0\n\n2\n00:00:01,000 --> 00:00:02,000\nGPS(-121.7459,48.0772,17) BAROMETER:4.0\n"
    it.convert(write(tmp_path, "w.srt", srt), out, video=video(tmp_path))
    _, c = read(out)
    assert abs(c["latitude"][0] - 48.0771) < 1e-9 and abs(c["longitude"][0] + 121.7458) < 1e-9


def test_srt_without_arrow_times_uses_block_clock(tmp_path):
    # early Mavic 2 style: no '-->' line, the date line carries the time
    blocks = [f"{k + 1}\n2017.08.05 14:11:{51 + k:02d},393,525\n[latitude : {lat_at(k):.6f}] [longtitude : {lon_at(k):.6f}] [iso : 100]"
              for k in range(3)]
    out = os.path.join(tmp_path, "o.csv")
    it.convert(write(tmp_path, "m2.SRT", "\n\n".join(blocks)), out, video=video(tmp_path))
    _, c = read(out)
    assert np.allclose(c["latitude"], lat_at(np.arange(N) / FPS), atol=1e-6)
    assert np.allclose(c["altitude_m"], 0.0)                                   # no altitude in this firmware -> --alt-offset


def test_gpx_alignment_video_start_and_offset(tmp_path):
    pts = "".join(f'<trkpt lat="{lat_at(s):.7f}" lon="{lon_at(s):.7f}"><ele>{alt_at(s):.2f}</ele><time>2026-10-04T10:00:0{s}Z</time></trkpt>'
                  for s in range(4))
    gpx = write(tmp_path, "track.gpx", '<?xml version="1.0" encoding="UTF-8"?>\n<gpx version="1.1" creator="t" '
                f'xmlns="http://www.topografix.com/GPX/1/1"><trk><trkseg>{pts}</trkseg></trk></gpx>')
    out = os.path.join(tmp_path, "o.csv"); t = np.arange(N) / FPS
    it.convert(gpx, out, video=video(tmp_path))                                # video starts at the first trkpt
    _, c = read(out)
    assert np.allclose(c["latitude"], lat_at(t), atol=1e-7) and np.allclose(c["altitude_m"], alt_at(t), atol=1e-3)
    it.convert(gpx, out, video=video(tmp_path), video_start="2026-10-04T12:00:01+02:00")   # = 10:00:01Z
    _, c = read(out)
    assert np.allclose(c["latitude"], lat_at(t + 1), atol=1e-7)
    assert it.main(["--video", video(tmp_path), "--telemetry", gpx, "--output", out, "--video-start", "2026-10-04T10:00:01Z", "--time-offset", "0.5"]) == 0
    _, c = read(out)
    assert np.allclose(c["latitude"], lat_at(t + 0.5), atol=1e-7)


def test_generic_csv_case_insensitive_and_iso_semicolon(tmp_path):
    out = os.path.join(tmp_path, "o.csv"); t = np.arange(N) / FPS
    src = write(tmp_path, "log.csv", "Time_s,Lat,Lng,Height\n" + "".join(f"{s},{lat_at(s):.7f},{lon_at(s):.7f},{rel_at(s):.2f}\n" for s in (0, 1, 2)))
    it.convert(src, out, video=video(tmp_path))
    _, c = read(out)
    assert np.allclose(c["latitude"], lat_at(t), atol=1e-7) and np.allclose(c["baro_alt_m"], rel_at(t), atol=1e-3)
    src = write(tmp_path, "log2.csv", "timestamp;LATITUDE;LONGITUDE;Altitude\n" +
                "".join(f"2026-10-04T10:00:0{s}.000Z;{lat_at(s):.7f};{lon_at(s):.7f};{str(round(alt_at(s), 2)).replace('.', ',')}\n" for s in (0, 1, 2)))
    it.convert(src, out, video=video(tmp_path))
    _, c = read(out)
    assert np.allclose(c["longitude"], lon_at(t), atol=1e-7) and np.allclose(c["altitude_m"], alt_at(t), atol=1e-3)


def test_airdata_csv_feet_and_isvideo(tmp_path):
    ft = 0.3048; rows = []
    for k in range(9):                                       # 0.5 s log; recording (isVideo) starts at 1.0 s
        s = k * 0.5; tv = s - 1.0
        rows.append(f"{int(s * 1000)},2026-10-04 10:00:0{int(s)},{lat_at(tv):.7f},{lon_at(tv):.7f},{rel_at(tv) / ft:.3f},{alt_at(tv) / ft:.3f},{int(s >= 1.0)}")
    src = write(tmp_path, "airdata.csv", "time(millisecond),datetime(utc),latitude,longitude,height_above_takeoff(feet),"
                "altitude_above_seaLevel(feet),isVideo\n" + "\n".join(rows) + "\n")
    out = os.path.join(tmp_path, "o.csv"); t = np.arange(N) / FPS
    it.convert(src, out, video=video(tmp_path))
    _, c = read(out)
    assert np.allclose(c["latitude"], lat_at(t), atol=1e-7)
    assert np.allclose(c["altitude_m"], alt_at(t), atol=1e-3) and np.allclose(c["baro_alt_m"], rel_at(t), atol=1e-3)


def test_dji_flight_record_flytime_text_and_camera_isvideo(tmp_path):
    ft = 0.3048; rows = []
    for k in range(9):                                       # 'OSD.flyTime' is text ('0m 1.5s'); a whole-second twin column exists
        s = k * 0.5; tv = s - 1.0
        rows.append(f"{1722648654 + int(s)},0m {s:.1f}s,{int(s)},{lat_at(tv):.7f},{lon_at(tv):.7f},{rel_at(tv) / ft:.3f},{alt_at(tv) / ft:.3f},"
                    f"{'TRUE' if s >= 1.0 else 'FALSE'}")
    src = write(tmp_path, "FlightRecord.csv", "timestamps,OSD.flyTime,OSD.flyTime [s],OSD.latitude,OSD.longitude,OSD.height [ft],"
                "OSD.altitude [ft],CAMERA.isVideo\n" + "\n".join(rows) + "\n")
    out = os.path.join(tmp_path, "o.csv"); t = np.arange(N) / FPS
    it.convert(src, out, video=video(tmp_path))
    _, c = read(out)
    assert np.allclose(c["latitude"], lat_at(t), atol=1e-7) and np.allclose(c["altitude_m"], alt_at(t), atol=1e-3)


def test_json_list(tmp_path):
    out = os.path.join(tmp_path, "o.csv"); t = np.arange(N) / FPS
    for key in ("t", "time"):
        src = write(tmp_path, f"{key}.json", json.dumps([{key: s, "lat": lat_at(s), "lon": lon_at(s), "alt": alt_at(s)} for s in (0, 0.5, 1, 1.5, 2)]))
        it.convert(src, out, video=video(tmp_path))
        _, c = read(out)
        assert np.allclose(c["latitude"], lat_at(t), atol=1e-7) and np.allclose(c["altitude_m"], alt_at(t), atol=1e-3)


def test_pipeline_schema_passthrough_and_hook(tmp_path):
    src = write(tmp_path, "telemetry.csv", "source_frame,timestamp_s,latitude,longitude,altitude_m,gps_eph_m\n0,0.0,47.37,8.54,450,0.5\n1,0.033,47.37,8.54,450,0.5\n")
    assert it.is_pipeline_csv(src) and it.ensure_pipeline_telemetry(src, video(tmp_path), str(tmp_path)) == src
    out = os.path.join(tmp_path, "copy.csv")
    it.convert(src, out)
    assert open(out).read() == open(src).read()
    srt = write(tmp_path, "f.SRT", make_srt("mavic3", np.arange(21) * 0.1))
    run = os.path.join(tmp_path, "run"); os.makedirs(run)
    conv = it.ensure_pipeline_telemetry(srt, video(tmp_path), run)
    assert conv == os.path.join(run, "telemetry_v2.csv") and it.is_pipeline_csv(conv)
    try:                                                       # the pipeline's own loaders accept the result
        from run_pipeline import load_telemetry
        from telemetry import load_video_telemetry
    except ImportError:                                        # pyproj missing locally: check the columns they read
        rows, c = read(conv)
        assert {"source_frame", "timestamp_s", "latitude", "longitude", "altitude_m", "baro_alt_m"} <= set(rows[0])
        return
    tel = load_telemetry(conv)
    assert np.array_equal(tel["frame"], np.arange(N)) and np.isfinite(tel["baro"]).all() and np.allclose(tel["eph"], 5.0)
    frame, ts, lat, lon, alt, eph = load_video_telemetry(conv)
    assert len(frame) == N and np.allclose(lat, lat_at(ts), atol=1e-7)


def test_errors(tmp_path):
    v = video(tmp_path); out = os.path.join(tmp_path, "o.csv")
    zeros = make_srt("mini", np.arange(21) * 0.1, {k: dict(lat=0.0, lon=0.0) for k in range(21)})
    assert "no valid GPS rows" in raises(it.convert, write(tmp_path, "zero.srt", zeros), out, video=v)
    assert "covers" in raises(it.convert, write(tmp_path, "short.srt", make_srt("mavic3", np.arange(6) * 0.1)), out, video=v)
    back = make_srt("mavic3", [0.0, 1.0, 0.5, 2.0])
    assert "not increasing" in raises(it.convert, write(tmp_path, "back.srt", back), out, video=v)
    assert "no valid GPS rows" in raises(it.convert, write(tmp_path, "nogps.srt", "1\n00:00:00,000 --> 00:00:00,033\n[iso : 100] [shutter : 1/240.0]\n"), out, video=v)
    # CLI: non-zero exit + message
    p = subprocess.run([sys.executable, os.path.join(VF2, "ingest_telemetry.py"), "--video", v, "--telemetry", os.path.join(tmp_path, "short.srt"),
                        "--output", out], capture_output=True, text=True)
    assert p.returncode != 0 and "error:" in p.stderr and "covers" in p.stderr


def test_cli_summary_line(tmp_path):
    src = write(tmp_path, "f.SRT", make_srt("mini", np.arange(21) * 0.1))
    out = os.path.join(tmp_path, "telemetry_v2.csv")
    p = subprocess.run([sys.executable, os.path.join(VF2, "ingest_telemetry.py"), "--video", video(tmp_path), "--telemetry", src, "--output", out],
                       capture_output=True, text=True)
    assert p.returncode == 0, p.stderr
    assert "21 rows" in p.stdout and f"{N} frames" in p.stdout and "flown" in p.stdout and "altitude" in p.stdout
    assert len(read(out)[0]) == N


def test_real_dji_srt_samples():
    # (file, lat, lon, alt_m first block, baro first block or None, fps of the clip)
    cases = [("dji_air2s.srt", 41.424724, 2.234156, 117.0, None, 30), ("dji_mavic3_head60.srt", 3.41531, -3.37440, 0.401, -2.4, 50),
             ("dji_p4_rtk.SRT", -34.237922, -58.851745, 85.8, 85.8, 1), ("dji_matrice_300.srt", 36.6146, -6.1120, 0.3, 0.3, 1),
             ("dji_mavic_pro.SRT", -20.2533, 149.0251, 1.9, 1.9, 1)]
    for name, lat, lon, alt, baro, fps in cases:
        fmt, s = it.load_samples(os.path.join(DATA, name))
        assert fmt == "srt" and len(s["t"]) > 10, name
        n = int((s["t"][-1] - min(s["t"][0], 0)) * fps) + 1
        F = it.to_frames(s, n, float(fps))
        assert abs(F["lat"][0] - lat) < 1e-6 and abs(F["lon"][0] - lon) < 1e-6, (name, F["lat"][0], F["lon"][0])
        assert abs(F["alt"][0] - alt) < 1e-3, (name, F["alt"][0])
        assert (F["baro"] is None) if baro is None else abs(F["baro"][0] - baro) < 1e-3, name
        assert np.all(np.abs(np.diff(F["lat"])) < 1e-3), name                 # no jumps / swapped rows


if __name__ == "__main__":  # plain runner when pytest is unavailable
    failed = 0
    for name, fn in sorted((k, v) for k, v in globals().items() if k.startswith("test_")):
        try:
            fn(tempfile.mkdtemp()) if fn.__code__.co_argcount else fn(); print("ok  ", name)
        except Exception as e:
            failed += 1; print("FAIL", name, repr(e))
    sys.exit(1 if failed else 0)
