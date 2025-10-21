# Wav2Lip UHQ - Local Usage

This directory contains a standalone Wav2Lip UHQ implementation that can be run locally without Cog.

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Download Models (Optional)

```bash
python run_local.py --download-models
```

### 3. Run Basic Example

```bash
python run_local.py input_video.mp4 input_audio.wav output_video.mp4
```

## Usage

### Command Line Interface

```bash
python run_local.py [OPTIONS] VIDEO AUDIO OUTPUT
```

**Required Arguments:**
- `VIDEO`: Path to input video file (mp4, avi, mov)
- `AUDIO`: Path to input audio file (wav, mp3)
- `OUTPUT`: Path to output video file

**Optional Arguments:**
- `--checkpoint {wav2lip,wav2lip_gan}`: Wav2Lip model (default: wav2lip_gan)
- `--face-restore-model {CodeFormer,GFPGAN}`: Face restoration model (default: GFPGAN)
- `--resize-factor {1,2,3,4}`: Video downscaling factor (default: 1)
- `--low-vram`: Enable low VRAM mode for systems with <16GB VRAM
- `--debug`: Enable debug logging
- `--download-models`: Download models if missing

**Advanced Parameters:**
- `--no-smooth`: Disable face detection smoothing
- `--only-mouth`: Track only mouth area (faster)
- `--pad-top N`: Padding above lips (default: 0)
- `--pad-bottom N`: Padding below lips (default: 0)
- `--pad-left N`: Padding left of lips (default: 0)
- `--pad-right N`: Padding right of lips (default: 0)
- `--mouth-mask-dilate N`: Mouth mask dilation (default: 15)
- `--face-mask-erode N`: Face mask erosion (default: 15)
- `--mask-blur N`: Mask blur kernel size (default: 15)
- `--code-former-fidelity FLOAT`: CodeFormer fidelity 0.0-1.0 (default: 0.75)

### Examples

**Basic Usage:**
```bash
python run_local.py video.mp4 audio.wav result.mp4
```

**High Quality (slower):**
```bash
python run_local.py video.mp4 audio.wav result.mp4 \
  --checkpoint wav2lip_gan \
  --face-restore-model CodeFormer \
  --code-former-fidelity 0.5
```

**Fast Processing:**
```bash
python run_local.py video.mp4 audio.wav result.mp4 \
  --resize-factor 2 \
  --only-mouth \
  --low-vram
```

**Debug Mode:**
```bash
python run_local.py video.mp4 audio.wav result.mp4 --debug
```

### Programmatic Usage

```python
from core.processor import Wav2LipProcessor

# Initialize processor
processor = Wav2LipProcessor(
    weights_dir="weights",
    device="cuda",
    low_vram=False,
    debug=False
)

# Process video
result_path = processor.process_video(
    video_path="input_video.mp4",
    audio_path="input_audio.wav",
    checkpoint="wav2lip_gan",
    face_restore_model="GFPGAN",
    resize_factor=1
)

# Cleanup
processor.cleanup()
```

## Performance Tips

### Maximum Speed (Default)
- Uses larger batch sizes
- Keeps models in memory
- Minimal CUDA cache clearing
- Best for high-end GPUs (A100, H100)

### Low VRAM Mode
- Reduces batch sizes
- Unloads models between phases
- Aggressive memory cleanup
- Best for T4, RTX 3080, RTX 4080

### Speed vs Quality Trade-offs
- `resize_factor=2`: 4x faster, slightly lower quality
- `only_mouth=True`: Faster processing, less facial motion
- `low_vram=True`: Slower but uses less memory

## File Structure

```
wav2lip-uhq/
├── run_local.py          # Command line interface
├── example_usage.py      # Programmatic examples
├── predict.py            # Cog predictor (for deployment)
├── core/
│   ├── processor.py      # Core processing logic
│   └── face_restoration.py # Face restoration
├── wav2lip/              # Wav2Lip implementation
└── weights/              # Model weights (auto-downloaded)
    ├── wav2lip/
    ├── s3fd/
    └── predicator/
```

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Use `--low-vram` flag
   - Increase `--resize-factor` to 2 or 3
   - Close other GPU applications

2. **Face Not Detected**
   - Ensure face is visible in all frames
   - Try different `--resize-factor`
   - Check video quality

3. **Poor Lip Sync**
   - Use `wav2lip_gan` checkpoint
   - Adjust `--mouth-mask-dilate`
   - Try `--no-smooth` flag

4. **Slow Processing**
   - Use `--only-mouth` for faster processing
   - Increase `--resize-factor`
   - Ensure GPU is being used

### Debug Mode

Enable debug logging to see detailed processing information:

```bash
python run_local.py video.mp4 audio.wav result.mp4 --debug
```

This will show:
- Model loading progress
- Frame processing status
- Memory usage
- Error details
