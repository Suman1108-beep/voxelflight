from __future__ import annotations

from pathlib import Path

from sih3d.ingest import VideoInfo, canonicalise_telemetry


def test_canonical_round_trip_preserves_video_timestamp_and_source_frame():
    video=VideoInfo(path=Path('flight.mp4'),frame_count=1800,fps=30.,width=1920,height=1080,duration_s=60.)
    records=[dict(timestamp_s=t,source_frame=f,latitude=47.3,longitude=8.5,altitude_m=460)
             for t,f in [(2.,60),(4.,120),(59.966667,1799)]]
    rows,meta=canonicalise_telemetry(records,video)
    assert [r['timestamp_s'] for r in rows]==[2.,4.,59.966667]
    assert [r['source_frame'] for r in rows]==[60,120,1799]
    assert meta['timestamp_inferred'] is False
    assert meta['timestamp_scale_to_seconds']==1.


def test_segmented_flight_telemetry_is_filtered_and_rebased() -> None:
    video = VideoInfo(
        path=Path("segment_100_102.mp4"),
        frame_count=3,
        fps=1.0,
        width=1920,
        height=1080,
        duration_s=3.0,
    )
    records = [
        {
            " imgid": str(frame),
            " Timpstemp": str((frame - 99) * 1_000_000),
            " lat": "28.6000",
            " lon": "77.2000",
            " alt": "120.0",
        }
        for frame in range(99, 104)
    ]

    canonical, metadata = canonicalise_telemetry(records, video)

    assert [row["source_frame"] for row in canonical] == [0, 1, 2]
    assert [row["timestamp_s"] for row in canonical] == [0.0, 1.0, 2.0]
    assert metadata["source_records"] == 5
    assert metadata["selected_records"] == 3
    assert metadata["source_frame_offset"] == 100
    assert metadata["timestamp_scale_to_seconds"] == 1e-6


def test_common_dji_style_aliases_are_accepted() -> None:
    video = VideoInfo(
        path=Path("flight.mov"),
        frame_count=91,
        fps=30.0,
        width=3840,
        height=2160,
        duration_s=3.033333,
    )
    records = [
        {
            "GPS Latitude": "12.5",
            "GPS Longitude": "77.5",
            "Relative Altitude": "44.2 m",
            "Time (ms)": str(index * 1000),
            "Heading": "91.0",
        }
        for index in range(4)
    ]

    canonical, metadata = canonicalise_telemetry(records, video)

    assert len(canonical) == 4
    assert canonical[-1]["timestamp_s"] == 3.0
    assert canonical[-1]["source_frame"] == 90
    assert canonical[0]["altitude_m"] == 44.2
    assert canonical[0]["yaw_deg"] == 91.0
    assert metadata["timestamp_inferred"] is False
