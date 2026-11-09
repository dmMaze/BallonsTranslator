# Experimental screentone preservation

`lama_mpe` and `lama_large_512px` expose **Preserve Screentones
(Experimental)** in their existing module settings. It is off by default;
enabling it does not change the model, download assets, or use a CLI/service.

LaMa runs once with its normal image and mask. After inference and resizing,
`modules/inpaint/screentone.py` analyzes original unmasked pixels at native
resolution. It estimates a repeating lattice from bounded reference crops and
interpolates nearby source pixels separately for each lattice phase. Local
means describe brightness; deviations describe the dot texture. Each phase
needs donor support, and local prediction error reduces confidence where the
pattern does not match.

The local source donors supply both brightness and dot contrast, so generated
brightness blobs are not carried into the restored screen. LaMa supplies the
structural-edge reference. Strong structural edges reduce replacement, retaining
the normal model result around linework. Confidence varies locally; insufficient
support does not change the model input or affect supported areas elsewhere.
Only masked pixels are composited. Unmasked source pixels remain identical.
The debug log reports masked coverage with strong support.

This is a local monochrome-texture method, not semantic manga segmentation.
Missing linework and unsupported areas still depend on LaMa. Nearby source
context is needed for each phase; large holes and different or warped screens may retain
some normal-model artifacts. Colored screens and crops without a reliable
lattice keep the normal result. The integer lattice handles pixel-aligned
screens and small local appearance variations, not arbitrary geometric
warping. Turning the option off restores the existing workflow.

The implementation uses the existing NumPy/OpenCV dependencies. It does not
require the optional native PatchMatch binary. Regression tests use synthetic
masked screens, including dense glyph masks, to check phase, shading, dark
linework, donor exclusion, local fallbacks, lazy settings metadata and the
unchanged LaMa inference path:

```sh
PYTHONPATH=tests python -m unittest test_screentone_inpaint test_lama_padding
```

For quality evaluation, hide a known text-free screen region and compare the
restoration against those held-out pixels. Do not judge quality only by input
and output dimensions or assume generated missing linework is ground truth.
