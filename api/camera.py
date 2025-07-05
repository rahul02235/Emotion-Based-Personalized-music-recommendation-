import cv2
import numpy as np
from django.http import StreamingHttpResponse, JsonResponse
import threading
import time
import json
import logging
import traceback
import sys
import os
from .ml_models_new import load_models, emotion_labels, face_cascade

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class CameraManager:
    def __init__(self):
        self.active_camera = None
        self.is_running = False
        self.lock = threading.Lock()
        self.frame = None
        self.camera_thread = None
        self.last_frame_time = time.time()
        self.current_emotion = None
        self.face_emotion_model = None
        logger.info("CameraManager initialized")

    def get_camera_list(self):
        """Get list of available cameras"""
        camera_list = []
        logger.info("Scanning for available cameras...")
        
        try:
            # Try default backend first
            for i in range(2):  # Try first 2 indices
                try:
                    cap = cv2.VideoCapture(i)
                    if cap.isOpened():
                        ret, frame = cap.read()
                        if ret and frame is not None:
                            camera_list.append(i)
                            logger.info(f"Found camera {i} using default backend")
                    cap.release()
                except Exception as e:
                    logger.debug(f"Default backend failed for camera {i}: {e}")
            
            # If no cameras found, try DirectShow
            if not camera_list:
                for i in range(2):
                    try:
                        cap = cv2.VideoCapture(i + cv2.CAP_DSHOW)
                        if cap.isOpened():
                            ret, frame = cap.read()
                            if ret and frame is not None:
                                camera_list.append(i)
                                logger.info(f"Found camera {i} using DirectShow")
                        cap.release()
                    except Exception as e:
                        logger.debug(f"DirectShow failed for camera {i}: {e}")
        
        except Exception as e:
            logger.error(f"Error scanning cameras: {e}")
            logger.debug(traceback.format_exc())
        
        logger.info(f"Found {len(camera_list)} cameras: {camera_list}")
        return camera_list

    def start_camera(self, camera_index):
        """Start the camera with specified index"""
        with self.lock:
            if self.is_running:
                logger.warning("Camera is already running, stopping first")
                self.stop_camera()
                time.sleep(2)
            
            try:
                # Load emotion detection model if not loaded
                if self.face_emotion_model is None:
                    self.face_emotion_model = load_models()
                
                # Try default backend first
                logger.info(f"Attempting to open camera {camera_index} with default backend")
                cap = cv2.VideoCapture(camera_index)
                
                if not cap.isOpened():
                    logger.info("Default backend failed, trying DirectShow")
                    cap.release()
                    cap = cv2.VideoCapture(camera_index + cv2.CAP_DSHOW)
                
                if cap.isOpened():
                    # Set basic camera properties
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                    cap.set(cv2.CAP_PROP_FPS, 30)
                    
                    # Test frame capture
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        self.active_camera = cap
                        self.is_running = True
                        
                        # Start capture thread
                        if self.camera_thread is None or not self.camera_thread.is_alive():
                            self.camera_thread = threading.Thread(target=self._camera_thread)
                            self.camera_thread.daemon = True
                            self.camera_thread.start()
                        
                        logger.info(f"Camera started successfully with frame size {frame.shape}")
                        return True
                    else:
                        logger.error("Failed to capture test frame")
                        cap.release()
                else:
                    logger.error(f"Failed to open camera {camera_index}")
                    if 'cap' in locals():
                        cap.release()
                
                return False
                
            except Exception as e:
                logger.error(f"Error starting camera: {e}")
                logger.debug(traceback.format_exc())
                if 'cap' in locals():
                    cap.release()
                return False

    def _detect_emotion(self, face_img):
        """Detect emotion from face image"""
        try:
            # Preprocess face image
            face_img = cv2.resize(face_img, (48, 48))
            face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
            face_img = np.expand_dims(face_img, axis=0)
            face_img = np.expand_dims(face_img, axis=-1)
            face_img = face_img / 255.0

            # Predict emotion
            predictions = self.face_emotion_model.predict(face_img)
            emotion_idx = np.argmax(predictions[0])
            emotion = emotion_labels[emotion_idx]
            confidence = predictions[0][emotion_idx]

            return emotion, confidence
        except Exception as e:
            logger.error(f"Error in emotion detection: {e}")
            return None, 0

    def _camera_thread(self):
        """Camera capture thread with emotion detection"""
        logger.info("Camera capture thread started")
        error_count = 0
        MAX_ERRORS = 3
        frame_timeout = 5  # 5 seconds timeout for frame capture
        
        while self.is_running:
            try:
                if self.active_camera:
                    # Check for camera timeout
                    if time.time() - self.last_frame_time > frame_timeout:
                        logger.error("Camera frame timeout")
                        self.stop_camera()
                        break
                    
                    ret, frame = self.active_camera.read()
                    if ret and frame is not None:
                        try:
                            # Convert to grayscale for face detection
                            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                            
                            # Detect faces
                            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
                            
                            # Process each face
                            for (x, y, w, h) in faces:
                                # Draw rectangle around face
                                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                                
                                # Extract face ROI
                                face_roi = frame[y:y+h, x:x+w]
                                
                                # Detect emotion
                                emotion, confidence = self._detect_emotion(face_roi)
                                
                                if emotion and confidence > 0.5:
                                    self.current_emotion = emotion
                                    # Add emotion text
                                    cv2.putText(frame, f"{emotion}: {confidence:.2f}", 
                                              (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 
                                              0.9, (0, 255, 0), 2)
                        except Exception as e:
                            logger.error(f"Error processing frame: {e}")
                            continue
                        
                        # Encode frame
                        _, buffer = cv2.imencode('.jpg', frame)
                        self.frame = buffer.tobytes()
                        self.last_frame_time = time.time()
                        error_count = 0
                    else:
                        error_count += 1
                        logger.warning(f"Failed to capture frame (attempt {error_count}/{MAX_ERRORS})")
                        if error_count >= MAX_ERRORS:
                            logger.error("Too many consecutive errors, stopping camera")
                            self.stop_camera()
                            break
                        time.sleep(0.5)  # Wait before retry
                
                time.sleep(1/15)  # Limit to 15 FPS for better stability
                
            except Exception as e:
                error_count += 1
                logger.error(f"Error in camera thread: {e}")
                if error_count >= MAX_ERRORS:
                    logger.error("Too many errors, stopping camera thread")
                    self.stop_camera()
                    break
                time.sleep(0.5)
        
        logger.info("Camera capture thread stopped")

    def stop_camera(self):
        """Stop the active camera"""
        with self.lock:
            logger.info("Stopping camera")
            self.is_running = False
            
            if self.camera_thread and self.camera_thread.is_alive():
                try:
                    self.camera_thread.join(timeout=2.0)
                except Exception as e:
                    logger.warning(f"Error joining camera thread: {e}")
            
            if self.active_camera:
                try:
                    self.active_camera.release()
                except Exception as e:
                    logger.warning(f"Error releasing camera: {e}")
                self.active_camera = None
            
            self.frame = None
            self.camera_thread = None
            self.current_emotion = None
            logger.info("Camera stopped")

    def get_frame(self):
        """Get the latest frame"""
        return self.frame if self.frame else None

    def get_current_emotion(self):
        """Get the current detected emotion"""
        return self.current_emotion

# Global camera manager instance
camera_manager = CameraManager()

def camera_feed(request):
    """Stream the camera feed"""
    logger.info("Starting camera feed stream")
    def generate():
        while camera_manager.is_running:
            frame = camera_manager.get_frame()
            if frame:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            time.sleep(1/30)
        logger.info("Camera feed stream ended")

    return StreamingHttpResponse(generate(),
                               content_type='multipart/x-mixed-replace; boundary=frame')

def start_camera(request):
    """Start the camera with specified index"""
    try:
        data = json.loads(request.body)
        camera_index = data.get('camera_index', 0)  # Default to laptop camera (index 0)
        logger.info(f"Request to start camera {camera_index}")
        
        # First, ensure any existing camera is properly stopped
        camera_manager.stop_camera()
        time.sleep(2)  # Give time for cleanup
        
        # Try to start the camera
        success = camera_manager.start_camera(camera_index)
        
        if not success:
            # If failed, try alternative camera index
            alt_index = 1 if camera_index == 0 else 0
            logger.info(f"Trying alternative camera index {alt_index}")
            success = camera_manager.start_camera(alt_index)
        
        return JsonResponse({
            'success': success,
            'message': 'Camera started successfully' if success else 'Failed to start camera'
        })
            
    except Exception as e:
        logger.error(f"Error in start_camera: {e}")
        logger.debug(traceback.format_exc())
        return JsonResponse({
            'success': False,
            'message': str(e)
        })

def stop_camera(request):
    """Stop the active camera"""
    logger.info("Request to stop camera")
    camera_manager.stop_camera()
    return JsonResponse({
        'success': True,
        'message': 'Camera stopped'
    })

def get_cameras(request):
    """Get list of available cameras"""
    logger.info("Request to list cameras")
    cameras = camera_manager.get_camera_list()
    response = {
        'success': True,
        'cameras': cameras
    }
    logger.info(f"Found cameras: {cameras}")
    return JsonResponse(response)

def get_current_emotion(request):
    """Get the current detected emotion"""
    emotion = camera_manager.get_current_emotion()
    return JsonResponse({
        'success': True,
        'emotion': emotion
    }) 