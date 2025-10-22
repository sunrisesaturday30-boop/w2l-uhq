# Wav2Lip UHQ - Mouth-Only Detection Mode

This document describes the mouth-only detection mode for Wav2Lip UHQ, which provides more efficient processing by focusing directly on mouth regions instead of full face detection.

## Overview

The mouth-only mode (`run_local_mouth.py`) skips the traditional face detection pipeline and directly extracts mouth regions using facial landmarks. This approach offers several advantages:

- **Faster processing** - No need for full face detection
- **Lower memory usage** - Process only mouth crops instead of full faces
- **Better focus** - Direct attention to the lip-sync area
- **More robust** - Less affected by face orientation or partial occlusion

## Key Differences from Standard Mode

### Standard Mode (`run_local.py`)
```
Input Video → Face Detection → Wav2Lip → Enhancement → Output
```

### Mouth-Only Mode (`run_local_mouth.py`)
```
Input Video → Mouth Detection → Mouth Crops → Wav2Lip → Enhancement → Reconstruction → Output
```

## Workflow

### 1. Mouth Region Extraction
- **Facial landmark detection** - Use dlib's 68-point model to find facial landmarks
- **Mouth point extraction** - Extract points 48-67 (20 points around the mouth)
- **Bounding box creation** - Create tight bounding box around mouth with padding
- **Crop extraction** - Extract mouth region from each frame

### 2. Mouth-Only Video Creation
- **Standardize dimensions** - Resize all mouth crops to consistent dimensions
- **Create video** - Generate a video containing only mouth regions
- **Handle missing frames** - Use black frames for frames without detectable mouths

### 3. Wav2Lip Processing
- **Process mouth crops** - Run Wav2Lip on mouth-only video (no resize factor needed)
- **Generate lip-synced mouths** - Create synchronized mouth movements
- **Maintain quality** - Preserve mouth region quality during processing

### 4. Enhancement
- **Face restoration** - Apply CodeFormer/GFPGAN to enhance mouth quality
- **Quality improvement** - Enhance details and reduce artifacts
- **Consistent processing** - Apply enhancement to all mouth crops

### 5. Video Reconstruction
- **Frame reconstruction** - Paste enhanced mouth regions back to original frames
- **Coordinate mapping** - Use stored bounding box coordinates for accurate placement
- **Audio integration** - Add original audio to final video

## Usage

### Basic Usage
```bash
python run_local_mouth.py --video input_video.mp4 --audio input_audio.wav
```

### Advanced Usage
```bash
python run_local_mouth.py \
    --video input_video.mp4 \
    --audio input_audio.wav \
    --face-restore codeformer \
    --mask-dilate 5 \
    --mask-blur 5 \
    --low-vram \
    --debug \
    --output result.mp4
```

## Parameters

### Input Files
- `--video` - Path to input video file (required)
- `--audio` - Path to input audio file (required)

### Model Management
- `--download-models` - Download all required models
- `--check-models` - Check if all models are available

### Processing Options
- `--face-restore` - Face restoration method (`codeformer`, `gfpgan`, `none`)
- `--only-mouth` - Process only mouth region (always enabled for mouth-only mode)

### Masking and Padding
- `--mask-dilate` - Mask dilation radius (default: 5)
- `--mask-blur` - Mask blur radius (default: 5)
- `--pad-top` - Top padding (default: 0)
- `--pad-bottom` - Bottom padding (default: 10)
- `--pad-left` - Left padding (default: 0)
- `--pad-right` - Right padding (default: 0)

### Performance Options
- `--low-vram` - Enable low VRAM mode for smaller GPUs
- `--debug` - Enable debug logging

### Output Options
- `--output` - Output video path (optional)

## Key Features

### No Resize Factor
Unlike the standard mode, mouth-only mode doesn't use `--resize-factor` because:
- Mouth crops are already optimized for processing
- No need for additional resizing
- Maintains original mouth proportions

### Automatic Mouth Detection
- **Landmark-based detection** - Uses dlib's 68-point facial landmark model
- **Robust extraction** - Handles various face orientations and expressions
- **Padding included** - Adds context around mouth for better processing

### Efficient Processing
- **Smaller regions** - Process only mouth areas instead of full faces
- **Faster inference** - Reduced computational load
- **Better quality** - More focus on the actual lip-sync area

## Requirements

### Models
- `weights/predicator/shape_predictor_68_face_landmarks.dat` - dlib predictor
- `weights/wav2lip/wav2lip_gan.pth` - Wav2Lip model
- `weights/s3fd/s3fd.pth` - Face detection (for initial face detection)
- `weights/codeformer/` or `weights/gfpgan/` - Face restoration models

### Dependencies
- `dlib` - For facial landmark detection
- `opencv-python` - For video processing
- `torch` - For neural network inference
- `ffmpeg` - For video/audio processing

## Performance Comparison

| Mode | Face Detection | Mouth Detection | Processing Speed | Memory Usage |
|------|---------------|----------------|------------------|--------------|
| Standard | Full face | Full face | Baseline | Baseline |
| Mouth-Only | Mouth landmarks | Mouth crops | ~2x faster | ~50% less |

## Troubleshooting

### Common Issues

1. **No mouth detected**
   - Ensure face is clearly visible in video
   - Check if dlib predictor is properly loaded
   - Verify facial landmarks are detected correctly

2. **Poor quality results**
   - Try different face restoration models
   - Adjust mask dilation and blur parameters
   - Enable debug mode to see processing steps

3. **Memory issues**
   - Use `--low-vram` flag
   - Reduce batch sizes in processor settings
   - Close other applications

### Debug Mode
Enable debug logging to see detailed processing steps:
```bash
python run_local_mouth.py --video input.mp4 --audio input.wav --debug
```

## Examples

### Basic Lip-Sync
```bash
python run_local_mouth.py --video person.mp4 --audio speech.wav
```

### High Quality with CodeFormer
```bash
python run_local_mouth.py \
    --video person.mp4 \
    --audio speech.wav \
    --face-restore codeformer \
    --mask-dilate 8 \
    --mask-blur 3
```

### Low VRAM Mode
```bash
python run_local_mouth.py \
    --video person.mp4 \
    --audio speech.wav \
    --low-vram \
    --debug
```

## Technical Details

### Mouth Region Extraction
- Uses dlib's 68-point facial landmark model
- Extracts points 48-67 for mouth region
- Adds 20-pixel padding around mouth
- Handles missing landmarks gracefully

### Video Processing
- Creates standardized mouth-only video
- Processes with Wav2Lip neural network
- Enhances quality with face restoration
- Reconstructs final video with original audio

### Memory Management
- Processes mouth crops in batches
- Cleans up intermediate files
- Optimizes GPU memory usage
- Handles large videos efficiently

This mouth-only mode provides a more efficient and focused approach to lip-sync processing, making it ideal for applications where speed and resource efficiency are important.
