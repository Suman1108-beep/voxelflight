# Real DJI SRT samples

All files come from the `samples/` folder of **JuanIrache/DJI_SRT_Parser** (MIT License), commit on `master`
fetched 2026-10-04. Renamed with a `dji_` prefix; contents unchanged except where noted.

| File here | Source URL | Firmware style |
|---|---|---|
| `dji_air2s.srt` | https://raw.githubusercontent.com/JuanIrache/DJI_SRT_Parser/master/samples/air2s.srt | Air 2S: `[latitude: ..] [longitude: ..] [altitude: ..]`, 30 fps blocks |
| `dji_mavic3_head60.srt` | https://raw.githubusercontent.com/JuanIrache/DJI_SRT_Parser/master/samples/MAVIC3.srt | Mavic 3: `[rel_alt: .. abs_alt: ..]`, 50 fps blocks. **Truncated to the first 60 blocks** (original is 3.6 MB) |
| `dji_p4_rtk.SRT` | https://raw.githubusercontent.com/JuanIrache/DJI_SRT_Parser/master/samples/p4_rtk.SRT | Phantom 4 RTK / Mini style: `GPS (lon, lat, sats), D ..m, H ..m`, 1 Hz |
| `dji_matrice_300.srt` | https://raw.githubusercontent.com/JuanIrache/DJI_SRT_Parser/master/samples/matrice_300.srt | M300: `GPS(lat,lon,0.0M) BAROMETER:..M` (latitude first), 1 Hz |
| `dji_mavic_pro.SRT` | https://raw.githubusercontent.com/JuanIrache/DJI_SRT_Parser/master/samples/mavic_pro.SRT | Mavic Pro: `HOME(..) GPS(lon,lat,sats) BAROMETER:..`, 1 Hz, first block at 1 s |

Some coordinates in the upstream samples appear to be anonymised (e.g. the Mavic 3 file sits in the Gulf of Guinea);
the tests only check that each file parses into the location the file itself states.

## License (applies to the files above)

```
MIT License

Copyright (c) 2018 JuanIrache

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
