"""
Core Wav2Lip UHQ processor for maximum speed optimization
Handles the complete lip-sync workflow with performance tuning
"""

import os
import sys
import torch
import cv2
import numpy as np
import logging
import tempfile
import subprocess
from pathlib import Path
from typing import Optional, Tuple, List
import gc

# Add wav2lip to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'wav2lip'))

from wav2lip.w2l import W2l
from wav2lip.wav2lip_uhq import Wav2LipUHQ
from core.face_restoration import FaceRestoration

# Configure logging
logger = logging.getLogger(__name__)

class Wav2LipProcessor:
    """
    High-performance Wav2Lip UHQ processor optimized for speed
    """
    
    def __init__(self, weights_dir: str = "weights", device: str = "cuda", 
                 low_vram: bool = False, debug: bool = False):
        self.weights_dir = weights_dir
        self.device = device
        self.low_vram = low_vram
        self.debug = debug
        
        # Performance optimization settings
        if low_vram:
            self.face_det_batch_size = 8
            self.wav2lip_batch_size = 64
            self.enhancement_batch_size = 1
        else:
            # Maximum speed settings
            self.face_det_batch_size = 32  # 2x original
            self.wav2lip_batch_size = 256   # 2x original  
            self.enhancement_batch_size = 4
        
        # Enable optimizations for speed
        if not low_vram and torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
        
        self.temp_dir = None
        self.face_restorer = None
        
    def _setup_temp_dir(self):
        """Create temporary directory for processing"""
        self.temp_dir = tempfile.mkdtemp(prefix="wav2lip_")
        logger.info(f"Created temp directory: {self.temp_dir}")
        
    def _cleanup_temp_dir(self):
        """Clean up temporary directory"""
        if self.temp_dir and os.path.exists(self.temp_dir):
            import shutil
            shutil.rmtree(self.temp_dir, ignore_errors=True)
            logger.info(f"Cleaned up temp directory: {self.temp_dir}")
    
    def _validate_inputs(self, video_path: str, audio_path: str) -> Tuple[bool, str]:
        """Validate input video and audio files"""
        try:
            # Validate video
            if not os.path.exists(video_path):
                return False, f"Video file not found: {video_path}"
            
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return False, f"Cannot open video file: {video_path}"
            
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            cap.release()
            
            if frame_count <= 0:
                return False, f"Video has no frames: {video_path}"
            
            if fps <= 0:
                return False, f"Invalid video FPS: {fps}"
            
            # Validate audio
            if not os.path.exists(audio_path):
                return False, f"Audio file not found: {audio_path}"
            
            # Check audio can be loaded
            import librosa
            try:
                audio, sr = librosa.load(audio_path, sr=16000)
                if len(audio) == 0:
                    return False, f"Audio file is empty: {audio_path}"
            except Exception as e:
                return False, f"Cannot load audio file: {e}"
            
            return True, "Inputs validated successfully"
            
        except Exception as e:
            return False, f"Input validation failed: {e}"
    
    def _setup_face_restoration(self, model_name: str, code_former_weight: float):
        """Setup face restoration model"""
        try:
            self.face_restorer = FaceRestoration(
                model_name=model_name,
                code_former_weight=code_former_weight,
                device=self.device,
                low_vram=self.low_vram
            )
            logger.info(f"Face restoration model loaded: {model_name}")
        except Exception as e:
            logger.error(f"Failed to load face restoration model: {e}")
            raise
    
    def _run_wav2lip(self, video_path: str, audio_path: str, checkpoint: str,
                    no_smooth: bool, resize_factor: int, pad_values: List[int]) -> str:
        """Run Wav2Lip processing with optimized settings"""
        try:
            logger.info("Starting Wav2Lip processing...")
            
            # Create W2l instance with optimized settings
            w2l = W2l(
                face=video_path,
                audio=audio_path,
                checkpoint=checkpoint,
                nosmooth=no_smooth,
                resize_factor=resize_factor,
                pad_top=pad_values[0],
                pad_bottom=pad_values[1], 
                pad_left=pad_values[2],
                pad_right=pad_values[3],
                face_swap_img=None
            )
            
            # Override batch sizes for performance
            w2l.face_det_batch_size = self.face_det_batch_size
            w2l.wav2lip_batch_size = self.wav2lip_batch_size
            
            # Set output path
            w2l.outfile = os.path.join(self.temp_dir, "wav2lip_result.mp4")
            
            # Execute Wav2Lip
            w2l.execute()
            
            if not os.path.exists(w2l.outfile):
                raise RuntimeError("Wav2Lip processing failed - no output file generated")
            
            logger.info(f"Wav2Lip processing completed: {w2l.outfile}")
            return w2l.outfile
            
        except Exception as e:
            logger.error(f"Wav2Lip processing failed: {e}")
            raise
    
    def _run_enhancement(self, original_video: str, wav2lip_video: str,
                        face_restore_model: str, code_former_weight: float,
                        mouth_mask_dilate: int, face_mask_erode: int, 
                        mask_blur: int, only_mouth: bool, resize_factor: int) -> str:
        """Run quality enhancement with optimized settings"""
        try:
            logger.info("Starting quality enhancement...")
            
            # Setup face restoration if not already done
            if self.face_restorer is None:
                self._setup_face_restoration(face_restore_model, code_former_weight)
            
            # Create Wav2LipUHQ instance
            w2luhq = Wav2LipUHQ(
                face=original_video,
                face_restore_model=face_restore_model,
                mouth_mask_dilatation=mouth_mask_dilate,
                erode_face_mask=face_mask_erode,
                mask_blur=mask_blur,
                only_mouth=only_mouth,
                face_swap_img=None,
                resize_factor=resize_factor,
                code_former_weight=code_former_weight,
                debug=self.debug
            )
            
            # Set paths
            w2luhq.w2l_video = wav2lip_video
            w2luhq.wav2lip_folder = self.temp_dir
            
            # Create output directories
            os.makedirs(os.path.join(self.temp_dir, "output", "debug"), exist_ok=True)
            os.makedirs(os.path.join(self.temp_dir, "output", "face_enhanced"), exist_ok=True)
            os.makedirs(os.path.join(self.temp_dir, "output", "final"), exist_ok=True)
            
            # Execute enhancement
            result_paths = w2luhq.execute()
            
            if result_paths is None or len(result_paths) < 4:
                raise RuntimeError("Quality enhancement failed - no output generated")
            
            final_video = result_paths[3]  # Final generated video
            if not os.path.exists(final_video):
                raise RuntimeError("Final video not generated")
            
            logger.info(f"Quality enhancement completed: {final_video}")
            return final_video
            
        except Exception as e:
            logger.error(f"Quality enhancement failed: {e}")
            raise
    
    def process_video(self, video_path: str, audio_path: str,
                     checkpoint: str = "wav2lip_gan",
                     face_restore_model: str = "GFPGAN", 
                     no_smooth: bool = False,
                     only_mouth: bool = False,
                     resize_factor: int = 1,
                     pad_top: int = 0,
                     pad_bottom: int = 0, 
                     pad_left: int = 0,
                     pad_right: int = 0,
                     mouth_mask_dilate: int = 15,
                     face_mask_erode: int = 15,
                     mask_blur: int = 15,
                     code_former_fidelity: float = 0.75) -> str:
        """
        Process video with lip-sync and quality enhancement
        
        Args:
            video_path: Path to input video
            audio_path: Path to input audio
            checkpoint: Wav2Lip model ("wav2lip" or "wav2lip_gan")
            face_restore_model: Face restoration model ("CodeFormer" or "GFPGAN")
            no_smooth: Disable face detection smoothing
            only_mouth: Track only mouth area
            resize_factor: Video downscaling factor (1-4)
            pad_top/bottom/left/right: Padding around mouth
            mouth_mask_dilate: Mouth mask dilation
            face_mask_erode: Face mask erosion  
            mask_blur: Mask blur kernel size
            code_former_fidelity: CodeFormer fidelity (0.0-1.0)
            
        Returns:
            Path to final processed video
        """
        try:
            # Setup
            self._setup_temp_dir()
            
            # Validate inputs
            is_valid, message = self._validate_inputs(video_path, audio_path)
            if not is_valid:
                raise ValueError(f"Input validation failed: {message}")
            
            logger.info("Input validation passed")
            
            # Setup face restoration
            self._setup_face_restoration(face_restore_model, code_former_fidelity)
            
            # Run Wav2Lip processing
            pad_values = [pad_top, pad_bottom, pad_left, pad_right]
            wav2lip_video = self._run_wav2lip(
                video_path, audio_path, checkpoint, no_smooth, 
                resize_factor, pad_values
            )
            
            # Run quality enhancement
            final_video = self._run_enhancement(
                video_path, wav2lip_video, face_restore_model, 
                code_former_fidelity, mouth_mask_dilate, face_mask_erode,
                mask_blur, only_mouth, resize_factor
            )
            
            logger.info("Video processing completed successfully")
            return final_video
            
        except Exception as e:
            logger.error(f"Video processing failed: {e}")
            raise
        finally:
            # Cleanup
            if self.face_restorer:
                self.face_restorer.cleanup()
            self._cleanup_temp_dir()
    
    def cleanup(self):
        """Clean up resources"""
        if self.face_restorer:
            self.face_restorer.cleanup()
        self._cleanup_temp_dir()
        
        # Clear GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
