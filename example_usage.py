#!/usr/bin/env python3
"""
Example usage of Wav2Lip UHQ processor
"""

import os
import sys
import logging

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.processor import Wav2LipProcessor
from download_models import download_models, check_models


def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def example_basic_usage():
    """Basic usage example"""
    print("=== Basic Usage Example ===")
    
    # Check if models exist
    if not check_models():
        print("Models not found. Downloading...")
        download_models()
    
    # Initialize processor
    processor = Wav2LipProcessor(device='cuda', low_vram=False)
    
    # Process video with default settings
    try:
        result_path = processor.process_video(
            video_path="input_video.mp4",
            audio_path="input_audio.wav"
        )
        print(f"Processing completed! Output: {result_path}")
    except Exception as e:
        print(f"Processing failed: {str(e)}")


def example_advanced_usage():
    """Advanced usage example with custom parameters"""
    print("=== Advanced Usage Example ===")
    
    # Initialize processor with low VRAM mode
    processor = Wav2LipProcessor(device='cuda', low_vram=True)
    
    # Process video with custom settings
    try:
        result_path = processor.process_video(
            video_path="input_video.mp4",
            audio_path="input_audio.wav",
            checkpoint="wav2lip_gan",
            nosmooth=False,
            resize_factor=2,
            pad_top=0,
            pad_bottom=10,
            pad_left=0,
            pad_right=0,
            face_restore_model="CodeFormer",
            mouth_mask_dilatation=5,
            erode_face_mask=2,
            mask_blur=3,
            only_mouth=True,
            code_former_weight=0.7,
            debug=True
        )
        print(f"Processing completed! Output: {result_path}")
    except Exception as e:
        print(f"Processing failed: {str(e)}")


def example_batch_processing():
    """Batch processing example"""
    print("=== Batch Processing Example ===")
    
    # List of video-audio pairs
    video_audio_pairs = [
        ("video1.mp4", "audio1.wav"),
        ("video2.mp4", "audio2.wav"),
        ("video3.mp4", "audio3.wav"),
    ]
    
    # Initialize processor once
    processor = Wav2LipProcessor(device='cuda', low_vram=False)
    
    # Process each pair
    for i, (video_path, audio_path) in enumerate(video_audio_pairs):
        print(f"Processing pair {i+1}/{len(video_audio_pairs)}: {video_path} + {audio_path}")
        
        try:
            result_path = processor.process_video(
                video_path=video_path,
                audio_path=audio_path
            )
            print(f"  Completed: {result_path}")
        except Exception as e:
            print(f"  Failed: {str(e)}")


def example_error_handling():
    """Error handling example"""
    print("=== Error Handling Example ===")
    
    processor = Wav2LipProcessor(device='cuda', low_vram=False)
    
    # Try to process non-existent files
    try:
        result_path = processor.process_video(
            video_path="nonexistent_video.mp4",
            audio_path="nonexistent_audio.wav"
        )
    except FileNotFoundError as e:
        print(f"File not found error handled: {str(e)}")
    except Exception as e:
        print(f"Other error handled: {str(e)}")


if __name__ == "__main__":
    setup_logging()
    
    print("Wav2Lip UHQ Example Usage")
    print("=" * 50)
    
    # Run examples
    example_basic_usage()
    print()
    
    example_advanced_usage()
    print()
    
    example_batch_processing()
    print()
    
    example_error_handling()
    print()
    
    print("Examples completed!")
