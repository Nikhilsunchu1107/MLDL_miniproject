# Dataset Audit Report

- Scan root: `datasets`
- Dataset sources: 9 immediate subdirectories under `datasets`, each scanned recursively
- Duplicate estimate: exact-match groups on normalized 16x16 grayscale thumbnail hashes within each source
- Brightness flags: grayscale thumbnails flagged as near-black or near-white; segmentation masks and binary annotations can trigger these flags legitimately
- Gender breakdown: inferred only from explicit Male/Female tokens in paths or documented metadata; unresolved numeric gender codes are reported separately

## Consolidated Summary

| Source | Images | Male | Female | Unknown | Unresolved gender codes | Min res | Max res | Top format(s) | Approx dupes | Corrupted | Near-black | Near-white |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| Dataset | 700 | 364 | 336 | 0 | - | 2600x2048 | 3900x3072 | JPG:700 | 0 | 0 | 0 | 0 |
| drive-retinal-vessel-segmentation-pixelwise | 0 | 0 | 0 | 0 | - | - | - | - | 0 | 0 | 0 | 0 |
| fundus-image-registration | 270 | 0 | 0 | 270 | - | 2912x2912 | 2912x2912 | JPG:268, PNG:2 | 144 | 0 | 0 | 0 |
| papila-retinal-fundus-images | 976 | 0 | 0 | 0 | 0:372, 1:604 | 2576x1934 | 2576x1934 | JPG:976 | 0 | 0 | 0 | 0 |
| retina-blood-vessel | 200 | 0 | 0 | 200 | - | 512x512 | 512x512 | PNG:200 | 0 | 0 | 0 | 0 |
| retinal-colorized-oct-images-003 | 52000 | 0 | 0 | 52000 | - | 512x496 | 1536x496 | JPG:52000 | 12126 | 0 | 0 | 0 |
| retinal-disease-detection-002 | 2254 | 0 | 0 | 2254 | - | 1440x960 | 2000x1333 | JPG:1197, PNG:1057 | 4 | 0 | 0 | 0 |
| retinal-vessel-segmentation | 716 | 0 | 0 | 716 | - | 565x584 | 3504x2336 | GIF:60, JPG:73, PNG:56 | 0 | 0 | 0 | 0 |
| UK_Biobank_Dataset | 40 | 0 | 0 | 40 | - | 2048x1536 | 2048x1536 | PNG:40 | 0 | 0 | 28 | 0 |

## Dataset

- Path: `datasets/Dataset`
- Total images: 700
- Per-class breakdown: Male 364, Female 336, Unknown/Unlabeled 0
- Label note: Sex labels are explicit in `Demographics of the participants.xlsx`, and diagnosis labels are balanced at 100 images each for DR, AMD, RVO, PM, Uveitis, RD, and Healthy.
- Resolutions: min 2600x2048, max 3900x3072, most common 3900x3072 (691), 2600x2048 (9)
- Formats present: JPG (700 filenames)
- Encoding note: 500 files decode as JPEG and 200 files decode as TIFF despite the `.jpg` extension; all AMD and DR images fall into this TIFF-in-JPG-name subset.
- Approximate duplicates: 0 duplicate files across 0 hash group(s)
- Corrupted images: 0
- Near-black images: 0
- Near-white images: 0
- Metadata note: `Ground Truth.xlsx` contains aggregate quality labels for all 700 images, and `Individual Quality Assessment.xlsx` contains three annotator-specific quality sheets for the same image IDs.

## drive-retinal-vessel-segmentation-pixelwise

- Path: `datasets/drive-retinal-vessel-segmentation-pixelwise`
- Total images: 0
- Per-class breakdown: Male 0, Female 0, Unknown/Unlabeled 0
- Resolutions: min n/a, max n/a, most common n/a
- Formats present: n/a
- Approximate duplicates: 0 duplicate files across 0 hash group(s)
- Corrupted images: 0
- Near-black images: 0
- Near-white images: 0

## fundus-image-registration

- Path: `datasets/fundus-image-registration`
- Total images: 270
- Per-class breakdown: Male 0, Female 0, Unknown/Unlabeled 270
- Resolutions: min 2912x2912, max 2912x2912, most common 2912x2912 (270)
- Formats present: JPG (268), PNG (2)
- Approximate duplicates: 144 duplicate files across 67 hash group(s)
- Duplicate example 1: `datasets/fundus-image-registration/FIRE/Images/S07_1.jpg`; `datasets/fundus-image-registration/FIRE/Images/S08_1.jpg`; `datasets/fundus-image-registration/FIRE/Images/S09_1.jpg`; `datasets/fundus-image-registration/FIRE/Images/S10_1.jpg`; `datasets/fundus-image-registration/FIRE/Images/S11_1.jpg`
- Duplicate example 2: `datasets/fundus-image-registration/FIRE/Images/S07_2.jpg`; `datasets/fundus-image-registration/FIRE/Images/S15_1.jpg`; `datasets/fundus-image-registration/FIRE/Images/S16_1.jpg`; `datasets/fundus-image-registration/FIRE/Images/S17_1.jpg`; `datasets/fundus-image-registration/FIRE/Images/S18_1.jpg`
- Duplicate example 3: `datasets/fundus-image-registration/FIRE/Images/S08_2.jpg`; `datasets/fundus-image-registration/FIRE/Images/S15_2.jpg`; `datasets/fundus-image-registration/FIRE/Images/S22_1.jpg`; `datasets/fundus-image-registration/FIRE/Images/S23_1.jpg`; `datasets/fundus-image-registration/FIRE/Images/S24_1.jpg`
- Corrupted images: 0
- Near-black images: 0
- Near-white images: 0

## papila-retinal-fundus-images

- Path: `datasets/papila-retinal-fundus-images`
- Total images: 976
- Per-class breakdown: Male 0, Female 0, Unknown/Unlabeled 0
- Unresolved binary gender codes: code 0: 372 image(s), code 1: 604 image(s)
- Label note: Binary gender codes found in PAPILA clinical spreadsheets (`0`/`1`), but the local files do not document which code means Male or Female.
- Resolutions: min 2576x1934, max 2576x1934, most common 2576x1934 (976)
- Formats present: JPG (976)
- Approximate duplicates: 0 duplicate files across 0 hash group(s)
- Corrupted images: 0
- Near-black images: 0
- Near-white images: 0

## retina-blood-vessel

- Path: `datasets/retina-blood-vessel`
- Total images: 200
- Per-class breakdown: Male 0, Female 0, Unknown/Unlabeled 200
- Resolutions: min 512x512, max 512x512, most common 512x512 (200)
- Formats present: PNG (200)
- Approximate duplicates: 0 duplicate files across 0 hash group(s)
- Corrupted images: 0
- Near-black images: 0
- Near-white images: 0

## retinal-colorized-oct-images-003

- Path: `datasets/retinal-colorized-oct-images-003`
- Total images: 52000
- Per-class breakdown: Male 0, Female 0, Unknown/Unlabeled 52000
- Resolutions: min 512x496, max 1536x496, most common 512x496 (22204), 512x512 (13091), 768x496 (11713)
- Formats present: JPG (52000)
- Approximate duplicates: 12126 duplicate files across 4098 hash group(s)
- Duplicate example 1: `datasets/retinal-colorized-oct-images-003/eye/CNV/3D_Rendering/CNV-6652117-518_10.jpg`; `datasets/retinal-colorized-oct-images-003/eye/CNV/3D_Rendering/CNV-6652117-519_10.jpg`; `datasets/retinal-colorized-oct-images-003/eye/CNV/3D_Volume_Rendering/CNV-6652117-518_12.jpg`; `datasets/retinal-colorized-oct-images-003/eye/CNV/3D_Volume_Rendering/CNV-6652117-519_12.jpg`; `datasets/retinal-colorized-oct-images-003/eye/CNV/Basic_Color_Map/CNV-6652117-518_1.jpg`
- Duplicate example 2: `datasets/retinal-colorized-oct-images-003/eye/DME/3D_Rendering/DME-3157783-8_10.jpg`; `datasets/retinal-colorized-oct-images-003/eye/DME/3D_Rendering/DME-3157783-9_10.jpg`; `datasets/retinal-colorized-oct-images-003/eye/DME/3D_Volume_Rendering/DME-3157783-8_12.jpg`; `datasets/retinal-colorized-oct-images-003/eye/DME/3D_Volume_Rendering/DME-3157783-9_12.jpg`; `datasets/retinal-colorized-oct-images-003/eye/DME/Basic_Color_Map/DME-3157783-8_1.jpg`
- Duplicate example 3: `datasets/retinal-colorized-oct-images-003/eye/DME/3D_Rendering/DME-3712405-30_10.jpg`; `datasets/retinal-colorized-oct-images-003/eye/DME/3D_Rendering/DME-3712405-31_10.jpg`; `datasets/retinal-colorized-oct-images-003/eye/DME/3D_Volume_Rendering/DME-3712405-30_12.jpg`; `datasets/retinal-colorized-oct-images-003/eye/DME/3D_Volume_Rendering/DME-3712405-31_12.jpg`; `datasets/retinal-colorized-oct-images-003/eye/DME/Basic_Color_Map/DME-3712405-30_1.jpg`
- Corrupted images: 0
- Near-black images: 0
- Near-white images: 0

## retinal-disease-detection-002

- Path: `datasets/retinal-disease-detection-002`
- Total images: 2254
- Per-class breakdown: Male 0, Female 0, Unknown/Unlabeled 2254
- Resolutions: min 1440x960, max 2000x1333, most common 2000x1329 (614), 2000x1333 (603), 1440x960 (527)
- Formats present: JPG (1197), PNG (1057)
- Approximate duplicates: 4 duplicate files across 4 hash group(s)
- Duplicate example 1: `datasets/retinal-disease-detection-002/Diabetic Retinopathy/test/images/IMAGE_00811.png`; `datasets/retinal-disease-detection-002/Diabetic Retinopathy/train/images/IMAGE_00807.png`
- Duplicate example 2: `datasets/retinal-disease-detection-002/Diabetic Retinopathy/test/images/IMAGE_00812.png`; `datasets/retinal-disease-detection-002/Diabetic Retinopathy/train/images/IMAGE_00808.png`
- Duplicate example 3: `datasets/retinal-disease-detection-002/Diabetic Retinopathy/train/images/IMAGE_00805.png`; `datasets/retinal-disease-detection-002/Diabetic Retinopathy/train/images/IMAGE_00810.png`
- Corrupted images: 0
- Near-black images: 0
- Near-white images: 0

## retinal-vessel-segmentation

- Path: `datasets/retinal-vessel-segmentation`
- Total images: 716
- Per-class breakdown: Male 0, Female 0, Unknown/Unlabeled 716
- Resolutions: min 565x584, max 3504x2336, most common 700x605 (397), 3504x2336 (135), 565x584 (100)
- Formats present: GIF (60), JPG (73), PNG (56), PPM (397), TIF (130)
- Approximate duplicates: 0 duplicate files across 0 hash group(s)
- Corrupted images: 0
- Near-black images: 0
- Near-white images: 0

## UK_Biobank_Dataset

- Path: `datasets/UK_Biobank_Dataset`
- Total images: 40
- Per-class breakdown: Male 0, Female 0, Unknown/Unlabeled 40
- Resolutions: min 2048x1536, max 2048x1536, most common 2048x1536 (40)
- Formats present: PNG (40)
- Approximate duplicates: 0 duplicate files across 0 hash group(s)
- Corrupted images: 0
- Near-black images: 28
- Near-black examples: `datasets/UK_Biobank_Dataset/1033721_21015_0_0_2ndHO.png`; `datasets/UK_Biobank_Dataset/1095850_21015_0_0_2ndHO.png`; `datasets/UK_Biobank_Dataset/1645965_21016_0_0_1stHO.png`; `datasets/UK_Biobank_Dataset/1645965_21016_0_0_2ndHO.png`; `datasets/UK_Biobank_Dataset/1795682_21015_0_0_2ndHO.png`
- Near-white images: 0
