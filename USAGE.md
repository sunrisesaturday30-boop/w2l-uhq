# Wav2Lip UHQ Usage Guide

This guide explains how to use the separated Wav2Lip UHQ components.

## 📁 File Structure

```
wav2lip-uhq/
├── predict.py              # Cog predictor (for Replicate deployment)
├── run_local.py            # Local test script
├── download_models.py      # Model download utility
├── example_usage.py        # Programmatic examples
├── test_imports.py         # Import test script
├── core/                   # Core processing modules
├── wav2lip/               # Wav2Lip implementation
└── weights/               # Model weights (auto-downloaded)
```

## 🚀 Quick Start

### 1. Download Models

```bash
# Download all required models
python download_models.py

# Check if models exist without downloading
python download_models.py --check-only

# Download to custom directory
python download_models.py --weights-dir custom_weights
```

### 2. Test Imports

```bash
# Verify all imports work correctly
python test_imports.py
```

### 3. Run Local Test

```bash
# Basic usage
python run_local.py input_video.mp4 input_audio.wav output_video.mp4

# With model download
python run_local.py input_video.mp4 input_audio.wav output_video.mp4 --download-models

# Check models only
python run_local.py --check-models
```

## 📦 Model Management

### `download_models.py` - Standalone Model Downloader

**Purpose**: Download required models for Wav2Lip UHQ processing

**Usage**:
```bash
python download_models.py [OPTIONS]
```

**Options**:
- `--weights-dir DIR`: Directory to store models (default: "weights")
- `--check-only`: Only check if models exist, don't download
- `--verbose`: Enable verbose logging

**Examples**:
```bash
# Download all models
python download_models.py

# Check models without downloading
python download_models.py --check-only

# Download to custom directory
python download_models.py --weights-dir my_models --verbose
```

**Downloaded Models**:
- `weights/wav2lip/wav2lip_gan.pth` - Wav2Lip GAN model
- `weights/wav2lip/wav2lip.pth` - Standard Wav2Lip model
- `weights/s3fd/s3fd-619a316812.pth` - S3FD face detection
- `weights/predicator/shape_predictor_68_face_landmarks.dat` - Dlib landmarks

## 🎬 Video Processing

### `run_local.py` - Local Video Processor

**Purpose**: Process videos locally with full parameter control

**Usage**:
```bash
python run_local.py [OPTIONS] VIDEO AUDIO OUTPUT
```

**Required Arguments**:
- `VIDEO`: Input video file (mp4, avi, mov)
- `AUDIO`: Input audio file (wav, mp3)
- `OUTPUT`: Output video file

**Model Management Options**:
- `--download-models`: Download models if missing
- `--check-models`: Check if models exist (exit if missing)

**Processing Options**:
- `--checkpoint {wav2lip,wav2lip_gan}`: Wav2Lip model
- `--face-restore-model {CodeFormer,GFPGAN}`: Face restoration
- `--resize-factor {1,2,3,4}`: Video downscaling
- `--low-vram`: Enable low VRAM mode
- `--debug`: Enable debug logging

**Advanced Parameters**:
- `--no-smooth`: Disable face detection smoothing
- `--only-mouth`: Track only mouth area
- `--pad-top N`: Padding above lips
- `--pad-bottom N`: Padding below lips
- `--pad-left N`: Padding left of lips
- `--pad-right N`: Padding right of lips
- `--mouth-mask-dilate N`: Mouth mask dilation
- `--face-mask-erode N`: Face mask erosion
- `--mask-blur N`: Mask blur kernel size
- `--code-former-fidelity FLOAT`: CodeFormer fidelity (0.0-1.0)

**Examples**:
```bash
# Basic processing
python run_local.py video.mp4 audio.wav result.mp4

# High quality processing
python run_local.py video.mp4 audio.wav result.mp4 \
  --checkpoint wav2lip_gan \
  --face-restore-model CodeFormer \
  --code-former-fidelity 0.5

# Fast processing with model download
python run_local.py video.mp4 audio.wav result.mp4 \
  --resize-factor 2 \
  --only-mouth \
  --low-vram \
  --download-models

# Debug mode
python run_local.py video.mp4 audio.wav result.mp4 --debug
```

## 🔧 Programmatic Usage

### `example_usage.py` - Usage Examples

**Purpose**: Show how to use the processor programmatically

**Usage**:
```bash
python example_usage.py
```

**Interactive Menu**:
1. Basic usage
2. Advanced usage  
3. Low VRAM usage
4. Batch processing

### Direct Import Usage

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
    face_restore_model="GFPGAN"
)

# Cleanup
processor.cleanup()
```

## 🚀 Deployment

### `predict.py` - Cog Predictor

**Purpose**: Deploy to Replicate using Cog

**Features**:
- Automatic model downloading in setup()
- All parameters exposed as inputs
- Optimized for maximum speed
- Comprehensive error handling

**Deployment**:
```bash
# Build and push to Replicate
cog build
cog push r8.im/your-username/wav2lip-uhq
```

## 🧪 Testing

### `test_imports.py` - Import Verification

**Purpose**: Verify all imports work correctly

**Usage**:
```bash
python test_imports.py
```

**Tests**:
- wav2lip.audio import
- wav2lip.w2l import
- wav2lip.models import
- wav2lip.face_detection import
- core.face_restoration import

## 📊 Performance Tips

### Maximum Speed (Default)
- Uses larger batch sizes (32/256)
- Keeps models in memory
- Minimal CUDA cache clearing
- Best for high-end GPUs (A100, H100)

### Low VRAM Mode
- Reduces batch sizes (8/64)
- Unloads models between phases
- Aggressive memory cleanup
- Best for T4, RTX 3080, RTX 4080

### Speed vs Quality Trade-offs
- `resize_factor=2`: 4x faster, slightly lower quality
- `only_mouth=True`: Faster processing, less facial motion
- `low_vram=True`: Slower but uses less memory

## 🐛 Troubleshooting

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
