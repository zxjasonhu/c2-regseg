# De-identified embedded C2 atlases

Each `atlas_XX_embedded.npz` contains one numeric array named `mask` with shape
`(2, 96, 96, 96)`. Channel 0 is a binary C2 mask; channel 1 is the
expert-annotated anatomical subregion mask. The files contain no source image
intensities, DICOM headers, filenames, source UIDs, free text, or other
case-level metadata.

SHA-256 checksums:

```text
9558dbffa7330e3b531ea5c9654f7196bdebb1c9be7db67d898135eeca3eb4a6  atlas_01_embedded.npz
d4f18b19c7ad5fbdc533078c880dfba77fd3dfe17b913dd6b8e11fec6f6bf9a5  atlas_02_embedded.npz
39dfe25ed3feec18cf7e806bf5539284e3492785867eb0153c984bf96af81fcf  atlas_03_embedded.npz
```
