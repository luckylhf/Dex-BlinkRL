# Third-party notices

`runtime/blinkmimic/` is derived from [BlinkMimic](https://github.com/fuyichen0725/blinkmimic) (Fuyi Chen, MIT),
which in turn contains portions of [supermimic_vibed](https://github.com/zita-ch/supermimic_vibed) (Chong Zhang,
ETH Zurich) and [InstantTrack](https://github.com/Huy-Quang-Dao/instantTrack) (Huy-Quang-Dao). Portions of `tools/`
come from the same two projects. All are used under their MIT Licenses, reproduced below.

## InstantTrack

```
MIT License

Copyright (c) 2026 Huy-Quang-Dao

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

## supermimic_vibed

```
MIT License

Copyright (c) 2026 ETH Zurich, Chong Zhang

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

## Method references

[SuperMimic](https://github.com/HybridRobotics/supermimic) and
[XSoma](https://github.com/nico-bohlinger/xsoma/tree/835410a2646c224addd536b3fb857ca84260ff7a) are cited for method
and interface ideas only. No source code or data from either project is copied into this package.

## Dependencies and data

The Python dependencies (mjlab, MuJoCo, MuJoCo Warp, NVIDIA Warp, PyTorch, matplotlib, Pillow) are installed from
their own distributions under their own licenses (Apache-2.0, BSD-3-Clause and others) and are not redistributed here.

This package ships no motion-capture, retargeted or reference-trajectory data. The recipe in
`configs/success_recipe.json` trains from a randomly initialized actor and world model, without expert motion data
and without pretrained weights. The motion loader in `runtime/blinkmimic/` exists because the runtime is shared with
other tasks; this recipe never reaches it and no `.npz` motion files are included.
