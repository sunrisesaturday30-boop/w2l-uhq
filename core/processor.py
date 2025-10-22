import os
import tempfile
import shutil
import logging
import torch
import cv2
import numpy as np

from .face_restoration import FaceRestoration

# Import wav2lip modules
try:
    from wav2lip.w2l import W2l
    from wav2lip.wav2lip_uhq import Wav2LipUHQ
except ImportError:
    # Fallback for different import contexts
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from wav2lip.w2l import W2l
    from wav2lip.wav2lip_uhq import Wav2LipUHQ

logger = logging.getLogger(__name__)


class Wav2LipProcessor:
    def __init__(self, device='cuda', low_vram=False):
        self.device = device
        self.low_vram = low_vram
        self.temp_dir = None
        self.face_restoration = None
        
        # Dynamic batch sizes based on low_vram setting
        if low_vram:
            self.face_det_batch_size = 4
            self.wav2lip_batch_size = 32
        else:
            self.face_det_batch_size = 16
            self.wav2lip_batch_size = 128

    def _setup_temp_dir(self):
        """Create a temporary directory for processing"""
        self.temp_dir = tempfile.mkdtemp(prefix='wav2lip_')
        logger.info(f"Created temp directory: {self.temp_dir}")
        return self.temp_dir

    def _cleanup_temp_dir(self):
        """Clean up temporary directory"""
        if self.temp_dir and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
            logger.info(f"Cleaned up temp directory: {self.temp_dir}")
            self.temp_dir = None

    def _validate_inputs(self, video_path, audio_path):
        """Validate input files"""
        if not os.path.isfile(video_path):
            raise ValueError(f"Video file not found: {video_path}")
        
        if not os.path.isfile(audio_path):
            raise ValueError(f"Audio file not found: {audio_path}")
        
        # Check video format
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {video_path}")
        cap.release()
        
        # Check audio format
        try:
            import librosa
            librosa.load(audio_path, sr=16000)
        except Exception as e:
            raise ValueError(f"Cannot load audio file: {audio_path}, error: {str(e)}")

    def _setup_face_restoration(self, face_restore_model, code_former_weight):
        """Setup face restoration model"""
        if not self.face_restoration:
            self.face_restoration = FaceRestoration(device=self.device, low_vram=self.low_vram)
        
        # Pre-load the model
        self.face_restoration._load_model(face_restore_model)
        return self.face_restoration

    def _run_wav2lip(self, video_path, audio_path, checkpoint, nosmooth, resize_factor, 
                     pad_top, pad_bottom, pad_left, pad_right, face_swap_img):
        """Run the Wav2Lip processing"""
        logger.info("Starting Wav2Lip processing...")
        
        w2l = W2l(
            face=video_path,
            audio=audio_path,
            checkpoint=checkpoint,
            nosmooth=nosmooth,
            resize_factor=resize_factor,
            pad_top=pad_top,
            pad_bottom=pad_bottom,
            pad_left=pad_left,
            pad_right=pad_right,
            face_swap_img=face_swap_img,
            temp_dir=self.temp_dir
        )
        
        # Update batch sizes
        w2l.face_det_batch_size = self.face_det_batch_size
        w2l.wav2lip_batch_size = self.wav2lip_batch_size
        
        try:
            w2l.execute()
            result_path = w2l.outfile
            logger.info(f"Wav2Lip processing completed: {result_path}")
            return result_path
        except Exception as e:
            logger.error(f"Wav2Lip processing failed: {str(e)}")
            raise

    def _run_enhancement(self, original_video, wav2lip_video, face_restore_model, 
                        mouth_mask_dilatation, erode_face_mask, mask_blur, only_mouth,
                        face_swap_img, resize_factor, code_former_weight, debug):
        """Run the quality enhancement"""
        logger.info("Starting quality enhancement...")
        
        w2luhq = Wav2LipUHQ(
            face=original_video,
            face_restore_model=face_restore_model,
            mouth_mask_dilatation=mouth_mask_dilatation,
            erode_face_mask=erode_face_mask,
            mask_blur=mask_blur,
            only_mouth=only_mouth,
            face_swap_img=face_swap_img,
            resize_factor=resize_factor,
            code_former_weight=code_former_weight,
            debug=debug,
            temp_dir=self.temp_dir
        )
        
        try:
            result_paths = w2luhq.execute()
            if result_paths and len(result_paths) > 3:
                final_path = result_paths[3]  # output_video.mp4
                logger.info(f"Quality enhancement completed: {final_path}")
                return final_path
            else:
                raise ValueError("Quality enhancement did not produce expected output")
        except Exception as e:
            logger.error(f"Quality enhancement failed: {str(e)}")
            raise

    def process_video(self, video_path, audio_path, checkpoint='wav2lip_gan', 
                     nosmooth=False, resize_factor=1, pad_top=0, pad_bottom=10, 
                     pad_left=0, pad_right=0, face_swap_img=None, 
                     face_restore_model='GFPGAN', mouth_mask_dilatation=0, 
                     erode_face_mask=0, mask_blur=0, only_mouth=False, 
                     code_former_weight=0.5, debug=False):
        """Main processing function"""
        
        try:
            # Setup
            self._setup_temp_dir()
            self._validate_inputs(video_path, audio_path)
            self._setup_face_restoration(face_restore_model, code_former_weight)
            
            # Run Wav2Lip
            wav2lip_result = self._run_wav2lip(
                video_path, audio_path, checkpoint, nosmooth, resize_factor,
                pad_top, pad_bottom, pad_left, pad_right, face_swap_img
            )
            
            # Run enhancement
            final_video = self._run_enhancement(
                video_path, wav2lip_result, face_restore_model,
                mouth_mask_dilatation, erode_face_mask, mask_blur, only_mouth,
                face_swap_img, resize_factor, code_former_weight, debug
            )
            
            return final_video
            
        except Exception as e:
            logger.error(f"Video processing failed: {str(e)}")
            raise
        finally:
            # Cleanup
            if self.face_restoration:
                self.face_restoration.model = None
            torch.cuda.empty_cache()
            self._cleanup_temp_dir()
