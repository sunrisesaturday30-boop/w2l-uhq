# Wav2Lip UHQ Local Usage Guide

This guide explains how to use the Wav2Lip UHQ processor locally for lip-sync video generation.

## Prerequisites

- Python 3.11+
- CUDA-compatible GPU (recommended)
- At least 8GB VRAM (or use low VRAM mode)

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Download required models:
```bash
python download_models.py --download
```

3. Verify models are present:
```bash
python download_models.py --check
```

## Usage

### Basic Usage

```bash
python run_local.py --video input_video.mp4 --audio input_audio.wav
```

### Advanced Usage

```bash
python run_local.py \
    --video input_video.mp4 \
    --audio input_audio.wav \
    --output output_video.mp4 \
    --resize-factor 2 \
    --only-mouth \
    --face-restore-model CodeFormer \
    --code-former-weight 0.7 \
    --debug
```

### Persistent Mode

For multiple predictions without reloading models:

```bash
python run_local_persistent.py --low-vram
```

This will start an interactive session where you can process multiple videos.

## Parameters

### Input Parameters
- `--video`: Path to input video file
- `--audio`: Path to input audio file
- `--output`: Path to output video file (optional)

### Wav2Lip Parameters
- `--checkpoint`: Wav2Lip checkpoint (default: wav2lip_gan)
- `--nosmooth`: Disable face smoothing
- `--resize-factor`: Resize factor for video (1-4, default: 1)
- `--pad-top`, `--pad-bottom`, `--pad-left`, `--pad-right`: Padding values

### Enhancement Parameters
- `--face-restore-model`: Face restoration model (GFPGAN/CodeFormer)
- `--mouth-mask-dilatation`: Mouth mask dilation (0-20)
- `--erode-face-mask`: Erode face mask (0-20)
- `--mask-blur`: Mask blur (0-20)
- `--only-mouth`: Only process mouth region
- `--code-former-weight`: CodeFormer weight (0.0-1.0)

### System Parameters
- `--debug`: Enable debug mode
- `--low-vram`: Enable low VRAM mode
- `--device`: Device to use (cuda/cpu)

## Examples

### Example 1: Basic Processing
```bash
python run_local.py --video person.mp4 --audio speech.wav
```

### Example 2: High Quality with CodeFormer
```bash
python run_local.py \
    --video person.mp4 \
    --audio speech.wav \
    --face-restore-model CodeFormer \
    --code-former-weight 0.8 \
    --resize-factor 1
```

### Example 3: Low VRAM Mode
```bash
python run_local.py \
    --video person.mp4 \
    --audio speech.wav \
    --low-vram \
    --resize-factor 2
```

### Example 4: Only Mouth Processing
```bash
python run_local.py \
    --video person.mp4 \
    --audio speech.wav \
    --only-mouth \
    --mouth-mask-dilatation 5
```

## Troubleshooting

### Common Issues

1. **Out of Memory Error**
   - Use `--low-vram` flag
   - Increase `--resize-factor` to 2 or 4
   - Close other GPU applications

2. **Model Not Found Error**
   - Run `python download_models.py --download`
   - Check if models exist in `weights/` directory

3. **Face Not Detected Error**
   - Ensure video contains clear face
   - Try different `--resize-factor` values
   - Check video quality and lighting

4. **Audio Processing Error**
   - Ensure audio file is in supported format (WAV, MP3, etc.)
   - Check audio file is not corrupted

### Performance Tips

1. **For faster processing:**
   - Use `--resize-factor 2` or higher
   - Enable `--low-vram` mode
   - Use `--only-mouth` for mouth-only processing

2. **For better quality:**
   - Use `--resize-factor 1`
   - Use CodeFormer with higher weight
   - Disable `--low-vram` mode

3. **For batch processing:**
   - Use `run_local_persistent.py` to avoid model reloading
   - Process videos with similar characteristics together

## File Structure

```
wav2lip-uhq/
├── core/
│   ├── face_restoration.py
│   └── processor.py
├── wav2lip/
│   ├── w2l.py
│   ├── wav2lip_uhq.py
│   ├── audio.py
│   ├── hparams.py
│   ├── models/
│   └── face_detection/
├── weights/
│   ├── wav2lip/
│   ├── s3fd/
│   ├── predicator/
│   ├── codeformer/
│   └── gfpgan/
├── run_local.py
├── run_local_persistent.py
├── download_models.py
├── example_usage.py
└── requirements.txt
```

## Support

For issues and questions:
1. Check the troubleshooting section above
2. Enable debug mode with `--debug` flag
3. Check the logs for detailed error messages
