# Wav2Lip UHQ Usage Guide

This guide covers the general usage of the Wav2Lip UHQ processor for both local and Cog deployment.

## Overview

Wav2Lip UHQ is a high-quality lip-sync processor that combines Wav2Lip with face restoration and enhancement techniques. It can be used locally or deployed via Cog for cloud processing.

## Features

- **High-Quality Lip-Sync**: Uses Wav2Lip for accurate lip synchronization
- **Face Restoration**: Supports GFPGAN and CodeFormer for face enhancement
- **Advanced Masking**: Sophisticated mouth and face masking options
- **GPU Optimization**: Efficient GPU memory management with low VRAM support
- **Batch Processing**: Support for processing multiple videos
- **Debug Mode**: Detailed logging for troubleshooting

## Model Requirements

The processor requires several pre-trained models:

### Core Models
- **Wav2Lip**: `weights/wav2lip/wav2lip_gan.pth`
- **S3FD Face Detector**: `weights/s3fd/s3fd.pth`
- **Dlib Predictor**: `weights/predicator/shape_predictor_68_face_landmarks.dat`

### Enhancement Models
- **CodeFormer**: `weights/codeformer/codeformer.pth`
- **GFPGAN**: `weights/gfpgan/GFPGANv1.4.pth`

## Model Download

### Automatic Download
```bash
python download_models.py --download
```

### Manual Download
You can manually download models to the `weights/` directory:

```
weights/
├── wav2lip/
│   └── wav2lip_gan.pth
├── s3fd/
│   └── s3fd.pth
├── predicator/
│   └── shape_predictor_68_face_landmarks.dat
├── codeformer/
│   └── codeformer.pth
└── gfpgan/
    └── GFPGANv1.4.pth
```

### Check Models
```bash
python download_models.py --check
```

## Local Usage

### Basic Command
```bash
python run_local.py --video input.mp4 --audio input.wav
```

### Advanced Command
```bash
python run_local.py \
    --video input.mp4 \
    --audio input.wav \
    --output output.mp4 \
    --resize-factor 2 \
    --face-restore-model CodeFormer \
    --code-former-weight 0.7 \
    --only-mouth \
    --debug
```

### Persistent Mode
For multiple predictions without reloading models:
```bash
python run_local_persistent.py
```

## Cog Deployment

### Building the Model
```bash
cog build
```

### Running Predictions
```bash
cog predict -i video=@input.mp4 -i audio=@input.wav
```

### Advanced Cog Usage
```bash
cog predict \
    -i video=@input.mp4 \
    -i audio=@input.wav \
    -i resize_factor=2 \
    -i face_restore_model=CodeFormer \
    -i code_former_weight=0.7 \
    -i only_mouth=true \
    -i debug=true
```

## Parameters

### Input Parameters
- `video`: Input video file
- `audio`: Input audio file
- `output`: Output video file (local only)

### Wav2Lip Parameters
- `checkpoint`: Wav2Lip checkpoint (default: wav2lip_gan)
- `nosmooth`: Disable face smoothing (default: false)
- `resize_factor`: Video resize factor (1-4, default: 1)
- `pad_top`, `pad_bottom`, `pad_left`, `pad_right`: Padding values

### Enhancement Parameters
- `face_restore_model`: Face restoration model (GFPGAN/CodeFormer)
- `mouth_mask_dilatation`: Mouth mask dilation (0-20, default: 0)
- `erode_face_mask`: Erode face mask (0-20, default: 0)
- `mask_blur`: Mask blur (0-20, default: 0)
- `only_mouth`: Only process mouth region (default: false)
- `code_former_weight`: CodeFormer weight (0.0-1.0, default: 0.5)

### System Parameters
- `debug`: Enable debug mode (default: false)
- `low_vram`: Enable low VRAM mode (default: false)
- `device`: Device to use (cuda/cpu, default: cuda)

## Workflow

### Basic Workflow
1. **Input Validation**: Check video and audio files
2. **Face Detection**: Detect faces in video frames
3. **Wav2Lip Processing**: Generate lip-sync frames
4. **Face Restoration**: Enhance face quality
5. **Masking**: Apply sophisticated masking
6. **Output**: Generate final video

### Advanced Workflow
1. **Preprocessing**: Resize and prepare inputs
2. **Face Detection**: Multi-scale face detection
3. **Wav2Lip Processing**: Batch processing with optimization
4. **Face Restoration**: CodeFormer/GFPGAN enhancement
5. **Advanced Masking**: Mouth and face region masking
6. **Postprocessing**: Final video composition

## Performance Optimization

### For Speed
- Use `resize_factor=2` or higher
- Enable `low_vram=true`
- Use `only_mouth=true`
- Process videos with similar characteristics

### For Quality
- Use `resize_factor=1`
- Use CodeFormer with higher weight
- Disable `low_vram` mode
- Use appropriate masking parameters

### For Memory
- Enable `low_vram=true`
- Use smaller batch sizes
- Process shorter videos
- Use CPU mode if needed

## Troubleshooting

### Common Issues

1. **Out of Memory**
   - Enable `low_vram=true`
   - Increase `resize_factor`
   - Close other GPU applications

2. **Face Not Detected**
   - Check video quality
   - Ensure clear face visibility
   - Try different resize factors

3. **Poor Quality**
   - Use `resize_factor=1`
   - Enable face restoration
   - Adjust masking parameters

4. **Slow Processing**
   - Use `resize_factor=2`
   - Enable `low_vram=true`
   - Use `only_mouth=true`

### Debug Mode

Enable debug mode for detailed logging:
```bash
python run_local.py --video input.mp4 --audio input.wav --debug
```

## Examples

### Example 1: Basic Processing
```bash
python run_local.py --video person.mp4 --audio speech.wav
```

### Example 2: High Quality
```bash
python run_local.py \
    --video person.mp4 \
    --audio speech.wav \
    --face-restore-model CodeFormer \
    --code-former-weight 0.8
```

### Example 3: Low VRAM
```bash
python run_local.py \
    --video person.mp4 \
    --audio speech.wav \
    --low-vram \
    --resize-factor 2
```

### Example 4: Mouth Only
```bash
python run_local.py \
    --video person.mp4 \
    --audio speech.wav \
    --only-mouth \
    --mouth-mask-dilatation 5
```

## API Usage

### Programmatic Usage
```python
from core.processor import Wav2LipProcessor

# Initialize processor
processor = Wav2LipProcessor(device='cuda', low_vram=False)

# Process video
result_path = processor.process_video(
    video_path="input.mp4",
    audio_path="input.wav",
    face_restore_model="CodeFormer",
    code_former_weight=0.7
)
```

### Batch Processing
```python
# Process multiple videos
videos = ["video1.mp4", "video2.mp4", "video3.mp4"]
audios = ["audio1.wav", "audio2.wav", "audio3.wav"]

for video, audio in zip(videos, audios):
    result = processor.process_video(video, audio)
    print(f"Processed: {result}")
```

## Support

For issues and questions:
1. Check the troubleshooting section
2. Enable debug mode
3. Check the logs for detailed error messages
4. Ensure all models are downloaded correctly
