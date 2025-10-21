#!/usr/bin/env python3
"""
Example usage script for Wav2Lip UHQ
Shows how to use the processor programmatically
"""

import os
import sys
import logging
from pathlib import Path

# Add core module to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'core'))

from core.processor import Wav2LipProcessor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def example_basic_usage():
    """Basic usage example with minimal parameters"""
    logger.info("=== Basic Usage Example ===")
    
    # Initialize processor
    processor = Wav2LipProcessor(
        weights_dir="weights",
        device="cuda",  # or "cpu"
        low_vram=False,
        debug=False
    )
    
    try:
        # Process video with default settings
        result_path = processor.process_video(
            video_path="input_video.mp4",
            audio_path="input_audio.wav",
            checkpoint="wav2lip_gan",
            face_restore_model="GFPGAN"
        )
        
        logger.info(f"Result saved to: {result_path}")
        
    except Exception as e:
        logger.error(f"Processing failed: {e}")
    finally:
        processor.cleanup()

def example_advanced_usage():
    """Advanced usage example with custom parameters"""
    logger.info("=== Advanced Usage Example ===")
    
    # Initialize processor with custom settings
    processor = Wav2LipProcessor(
        weights_dir="weights",
        device="cuda",
        low_vram=False,  # Set to True for low VRAM systems
        debug=True
    )
    
    try:
        # Process video with custom parameters
        result_path = processor.process_video(
            video_path="input_video.mp4",
            audio_path="input_audio.wav",
            checkpoint="wav2lip_gan",
            face_restore_model="CodeFormer",
            no_smooth=False,
            only_mouth=False,
            resize_factor=2,  # Downscale to half resolution for speed
            pad_top=5,
            pad_bottom=5,
            pad_left=5,
            pad_right=5,
            mouth_mask_dilate=20,
            face_mask_erode=10,
            mask_blur=10,
            code_former_fidelity=0.8
        )
        
        logger.info(f"Result saved to: {result_path}")
        
    except Exception as e:
        logger.error(f"Processing failed: {e}")
    finally:
        processor.cleanup()

def example_low_vram_usage():
    """Low VRAM usage example for systems with limited GPU memory"""
    logger.info("=== Low VRAM Usage Example ===")
    
    # Initialize processor for low VRAM
    processor = Wav2LipProcessor(
        weights_dir="weights",
        device="cuda",
        low_vram=True,  # Enable low VRAM mode
        debug=False
    )
    
    try:
        # Process video with low VRAM settings
        result_path = processor.process_video(
            video_path="input_video.mp4",
            audio_path="input_audio.wav",
            checkpoint="wav2lip_gan",
            face_restore_model="GFPGAN",
            resize_factor=2,  # Downscale for memory efficiency
            only_mouth=True,  # Track only mouth for speed
            mouth_mask_dilate=15,
            face_mask_erode=15,
            mask_blur=15
        )
        
        logger.info(f"Result saved to: {result_path}")
        
    except Exception as e:
        logger.error(f"Processing failed: {e}")
    finally:
        processor.cleanup()

def example_batch_processing():
    """Batch processing example for multiple videos"""
    logger.info("=== Batch Processing Example ===")
    
    # List of video/audio pairs to process
    video_audio_pairs = [
        ("video1.mp4", "audio1.wav", "output1.mp4"),
        ("video2.mp4", "audio2.wav", "output2.mp4"),
        ("video3.mp4", "audio3.wav", "output3.mp4"),
    ]
    
    # Initialize processor once
    processor = Wav2LipProcessor(
        weights_dir="weights",
        device="cuda",
        low_vram=False,
        debug=False
    )
    
    try:
        for video_path, audio_path, output_path in video_audio_pairs:
            if not os.path.exists(video_path) or not os.path.exists(audio_path):
                logger.warning(f"Skipping {video_path} - files not found")
                continue
            
            logger.info(f"Processing {video_path} with {audio_path}")
            
            result_path = processor.process_video(
                video_path=video_path,
                audio_path=audio_path,
                checkpoint="wav2lip_gan",
                face_restore_model="GFPGAN",
                resize_factor=1
            )
            
            # Copy to final output location
            import shutil
            shutil.copy2(result_path, output_path)
            logger.info(f"Saved to {output_path}")
            
    except Exception as e:
        logger.error(f"Batch processing failed: {e}")
    finally:
        processor.cleanup()

if __name__ == "__main__":
    print("Wav2Lip UHQ Example Usage")
    print("=" * 50)
    print("Choose an example to run:")
    print("1. Basic usage")
    print("2. Advanced usage")
    print("3. Low VRAM usage")
    print("4. Batch processing")
    
    choice = input("Enter choice (1-4): ").strip()
    
    if choice == "1":
        example_basic_usage()
    elif choice == "2":
        example_advanced_usage()
    elif choice == "3":
        example_low_vram_usage()
    elif choice == "4":
        example_batch_processing()
    else:
        print("Invalid choice. Running basic example...")
        example_basic_usage()
