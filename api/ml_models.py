import os
import tensorflow as tf
import pandas as pd
from django.conf import settings
import logging
from PIL import Image
import numpy as np
import warnings
import random
import cv2
from django.core.cache import cache
from functools import lru_cache
import time

logger = logging.getLogger(__name__)

# Configure TensorFlow to be more efficient
tf.config.set_soft_device_placement(True)
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

# Suppress numpy warnings that might occur during model loading
warnings.filterwarnings('ignore', category=UserWarning, module='numpy')

# Emotion mapping
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# Load face detection cascade classifier
try:
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    if not os.path.exists(cascade_path):
        raise FileNotFoundError(f"Cascade classifier file not found at {cascade_path}")
        
    face_cascade = cv2.CascadeClassifier(cascade_path)
    if face_cascade.empty():
        raise ValueError("Failed to load cascade classifier")
    logger.info("Successfully loaded face detection cascade classifier")
except Exception as e:
    logger.error(f"Error loading face cascade classifier: {str(e)}")
    raise RuntimeError(f"Failed to initialize face detection: {str(e)}")

# Mood to emotion mapping
mood_to_emotion = {
    'energetic': ['happy', 'surprise'],
    'Chill': ['neutral'],
    'romantic': ['happy'],
    'cheerful': ['happy'],
}

# Lazy loading of models and data
_face_emotion_model = None
_music_df = None
_played_songs = {}  # Dictionary to store played songs for each emotion

@lru_cache(maxsize=100)
def get_cached_recommendations(emotion, timestamp):
    """Cache recommendations for each emotion with a timestamp to expire cache"""
    recommendations = get_music_recommendations(emotion)
    return recommendations

def load_models():
    global _face_emotion_model, _music_df
    
    try:
        # Check if model is cached
        if _face_emotion_model is None:
            cached_model = cache.get('face_emotion_model')
            if cached_model is not None:
                _face_emotion_model = cached_model
                logger.info("Loaded face emotion model from cache")
            else:
                model_path = os.path.join(settings.BASE_DIR, 'resource', 'face_emotion.h5')
                if not os.path.exists(model_path):
                    raise FileNotFoundError(f"Model file not found at {model_path}")
                    
                logger.info(f"Loading face emotion model from {model_path}")
                
                try:
                    # Configure GPU memory growth
                    gpus = tf.config.list_physical_devices('GPU')
                    if gpus:
                        for gpu in gpus:
                            tf.config.experimental.set_memory_growth(gpu, True)
                except Exception as e:
                    logger.warning(f"Could not configure GPU: {str(e)}")
                
                try:
                    # Load model with optimized settings and error handling
                    _face_emotion_model = tf.keras.models.load_model(model_path, compile=False)
                    _face_emotion_model.compile(
                        optimizer='adam',
                        loss='sparse_categorical_crossentropy',
                        metrics=['accuracy']
                    )
                except Exception as e:
                    raise RuntimeError(f"Failed to load model: {str(e)}")
                
                # Cache the model
                try:
                    cache.set('face_emotion_model', _face_emotion_model, timeout=3600)  # Cache for 1 hour
                except Exception as e:
                    logger.warning(f"Could not cache model: {str(e)}")
                    
    except Exception as e:
        logger.error(f"Error loading face emotion model: {str(e)}")
        raise RuntimeError(f"Failed to initialize emotion detection model: {str(e)}")
    
    try:
        # Check if music data is cached
        if _music_df is None:
            cached_df = cache.get('music_df')
            if cached_df is not None:
                _music_df = cached_df
                logger.info("Loaded music data from cache")
            else:
                csv_path = os.path.join(settings.BASE_DIR, 'resource', 'ClassifiedMusic.csv')
                if not os.path.exists(csv_path):
                    raise FileNotFoundError(f"Music data file not found at {csv_path}")
                    
                logger.info(f"Loading music data from {csv_path}")
                _music_df = pd.read_csv(csv_path)
                
                # Ensure required columns exist
                required_columns = ['name', 'artist', 'id', 'mood']
                missing_columns = [col for col in required_columns if col not in _music_df.columns]
                if missing_columns:
                    raise ValueError(f"Missing required columns in CSV: {missing_columns}")
                
                # Convert mood names to lowercase for consistent matching
                _music_df['mood'] = _music_df['mood'].str.lower()
                
                # Cache the dataframe
                try:
                    cache.set('music_df', _music_df, timeout=3600)  # Cache for 1 hour
                except Exception as e:
                    logger.warning(f"Could not cache music data: {str(e)}")
                    
    except Exception as e:
        logger.error(f"Error loading music data: {str(e)}")
        raise RuntimeError(f"Failed to load music recommendations data: {str(e)}")

def detect_and_crop_face(image_path):
    """Detect and crop face from image using OpenCV with improved detection parameters"""
    try:
        # Validate image path
        if not os.path.exists(image_path):
            logger.error(f"Image file not found at path: {image_path}")
            raise FileNotFoundError(f"Image file not found: {image_path}")
            
        # Read image and log its properties
        img = cv2.imread(image_path)
        if img is None:
            logger.error(f"Failed to read image at {image_path}")
            raise ValueError("Failed to read image file. The file may be corrupted or in an unsupported format.")
        
        # Log image properties
        logger.info(f"Image loaded successfully. Shape: {img.shape}, Type: {img.dtype}")
        
        # Basic image validation
        if len(img.shape) != 3 or img.shape[2] != 3:
            logger.error("Invalid image format: Must be a color image")
            raise ValueError("Invalid image format: Must be a color image")
        
        if img.shape[0] < 48 or img.shape[1] < 48:
            logger.error(f"Image too small: {img.shape[0]}x{img.shape[1]} (minimum 48x48)")
            raise ValueError("Image is too small. Minimum size is 48x48 pixels.")
            
        # Convert to grayscale with error checking
        try:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            logger.info("Successfully converted image to grayscale")
        except Exception as e:
            logger.error(f"Grayscale conversion failed: {str(e)}")
            raise ValueError(f"Failed to convert image to grayscale: {str(e)}")
        
        # Save original image for debugging
        debug_dir = os.path.join(os.path.dirname(image_path), 'debug')
        os.makedirs(debug_dir, exist_ok=True)
        cv2.imwrite(os.path.join(debug_dir, 'original.jpg'), img)
        
        # Enhance image for better face detection
        gray_eq = cv2.equalizeHist(gray)
        
        # Try multiple face detection parameters
        detection_params = [
            # More lenient parameters first
            {'scaleFactor': 1.1, 'minNeighbors': 3, 'minSize': (30, 30)},
            {'scaleFactor': 1.2, 'minNeighbors': 4, 'minSize': (30, 30)},
            {'scaleFactor': 1.3, 'minNeighbors': 3, 'minSize': (40, 40)},
            # Original strict parameters as fallback
            {'scaleFactor': 1.1, 'minNeighbors': 5, 'minSize': (int(img.shape[0] * 0.2), int(img.shape[1] * 0.2))}
        ]

        faces = None
        for params in detection_params:
            logger.info(f"Trying face detection with parameters: {params}")
            current_faces = face_cascade.detectMultiScale(
                gray_eq,
                scaleFactor=params['scaleFactor'],
                minNeighbors=params['minNeighbors'],
                minSize=params['minSize']
            )
            
            if len(current_faces) > 0:
                faces = current_faces
                logger.info(f"Found {len(faces)} faces with parameters: {params}")
                break
            
        if faces is None or len(faces) == 0:
            logger.error("No faces detected in the image after trying multiple parameters")
            raise ValueError("No faces detected. Please ensure your image contains a clear, well-lit face looking towards the camera.")
            
        if len(faces) > 1:
            logger.warning(f"Multiple faces detected ({len(faces)}). Using the largest face.")
            
        # Get the largest face
        largest_face = max(faces, key=lambda rect: rect[2] * rect[3])
        x, y, w, h = largest_face
        
        # Add padding around face (20% of face dimensions)
        padding_w = int(0.2 * w)
        padding_h = int(0.2 * h)
        x = max(0, x - padding_w)
        y = max(0, y - padding_h)
        w = min(img.shape[1] - x, w + 2*padding_w)
        h = min(img.shape[0] - y, h + 2*padding_h)
        
        # Crop and validate face
        face = img[y:y+h, x:x+w]
        
        # Save detected face for debugging
        cv2.imwrite(os.path.join(debug_dir, 'detected_face.jpg'), face)
        
        # Draw face rectangle on original image for debugging
        debug_img = img.copy()
        cv2.rectangle(debug_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.imwrite(os.path.join(debug_dir, 'face_detection.jpg'), debug_img)
        
        logger.info(f"Successfully detected and cropped face. Size: {w}x{h}")
        return face
        
    except Exception as e:
        logger.error(f"Error in face detection: {str(e)}")
        raise ValueError(str(e))

def preprocess_image(image_path):
    """Preprocess image for emotion detection using OpenCV and TensorFlow"""
    logger.info(f"Processing image: {image_path}")
    
    try:
        # Detect and crop face
        face = detect_and_crop_face(image_path)
        if face is None:
            raise ValueError("Failed to detect face in image")
        
        # Save original face crop for debugging
        debug_dir = os.path.join(os.path.dirname(image_path), 'debug')
        os.makedirs(debug_dir, exist_ok=True)
        cv2.imwrite(os.path.join(debug_dir, 'original_crop.jpg'), face)
        
        # Convert to grayscale
        if len(face.shape) > 2:
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            cv2.imwrite(os.path.join(debug_dir, 'grayscale.jpg'), face)
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to grayscale image
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        face = clahe.apply(face)
        cv2.imwrite(os.path.join(debug_dir, 'clahe.jpg'), face)
        
        # Resize to model input size (48x48)
        face = cv2.resize(face, (48, 48), interpolation=cv2.INTER_AREA)
        cv2.imwrite(os.path.join(debug_dir, 'resized.jpg'), face)
        
        # Convert to float32 and normalize
        face = face.astype('float32')
        
        # Apply standard normalization (zero mean, unit variance)
        mean = np.mean(face)
        std = np.std(face)
        face = (face - mean) / (std + 1e-7)
        
        # Scale to [0,1] range after normalization
        face = (face - face.min()) / (face.max() - face.min() + 1e-7)
        
        # Save normalized image for debugging
        debug_normalized = (face * 255).astype('uint8')
        cv2.imwrite(os.path.join(debug_dir, 'normalized.jpg'), debug_normalized)
        
        # Add channel dimension first (required for model)
        face = np.expand_dims(face, -1)
        
        # Add batch dimension
        face = np.expand_dims(face, 0)
        
        # Log preprocessing details
        logger.info(f"Preprocessed image shape: {face.shape}")
        logger.info(f"Pixel value range: [{face.min():.3f}, {face.max():.3f}]")
        logger.info(f"Mean: {face.mean():.3f}, Std: {face.std():.3f}")
        
        return face
        
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        raise ValueError(f"Failed to process image: {str(e)}")

def process_image_and_get_recommendations(image_path):
    """Process image and return music recommendations"""
    try:
        # Only load models when needed
        if _face_emotion_model is None or _music_df is None:
            load_models()
        
        # Preprocess image
        processed_image = preprocess_image(image_path)
        logger.info("Image preprocessing completed")
        
        # Get emotion prediction
        emotion_pred = _face_emotion_model.predict(processed_image, verbose=0)
        emotion_probs = emotion_pred[0]
        
        # Log raw predictions for debugging
        logger.info("Raw emotion predictions:")
        for emotion, prob in zip(emotion_labels, emotion_probs):
            logger.info(f"{emotion:10s}: {prob:.4f}")
        
        # Get top 2 predictions
        top_2_idx = np.argsort(emotion_probs)[-2:][::-1]
        primary_emotion = emotion_labels[top_2_idx[0]]
        secondary_emotion = emotion_labels[top_2_idx[1]]
        primary_conf = emotion_probs[top_2_idx[0]]
        secondary_conf = emotion_probs[top_2_idx[1]]
        
        logger.info(f"Top prediction: {primary_emotion} ({primary_conf:.4f})")
        logger.info(f"Second prediction: {secondary_emotion} ({secondary_conf:.4f})")
        
        # Decision logic for emotion
        if primary_conf < 0.4:  # If confidence is very low
            logger.warning(f"Very low confidence ({primary_conf:.4f}), defaulting to neutral")
            emotion = 'neutral'
        elif primary_conf - secondary_conf < 0.15:  # If top emotions are too close
            logger.warning(f"Ambiguous prediction (gap: {primary_conf - secondary_conf:.4f})")
            # Prefer neutral or happy if they're in top 2 and close
            if 'neutral' in [primary_emotion, secondary_emotion]:
                emotion = 'neutral'
            elif 'happy' in [primary_emotion, secondary_emotion]:
                emotion = 'happy'
            else:
                emotion = primary_emotion
        else:
            emotion = primary_emotion
        
        logger.info(f"Final emotion decision: {emotion}")
        
        # Get cached recommendations or generate new ones
        timestamp = int(time.time() / 3600)  # Change cache every hour
        recommendations = get_cached_recommendations(emotion, timestamp)
        logger.info(f"Generated {len(recommendations)} recommendations")
        
        return recommendations, emotion
        
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        raise

def get_music_recommendations(emotion):
    """Get music recommendations based on emotion"""
    global _played_songs
    
    # Ensure models and data are loaded
    load_models()
    
    if _music_df is None or len(_music_df) == 0:
        raise ValueError("No music data available")
    
    try:
        # Map emotion to mood
        emotion = emotion.lower()
        if emotion in ['happy', 'surprise']:
            moods = ['cheerful', 'energetic', 'happy']
        elif emotion in ['sad']:
            moods = ['sad', 'melancholic', 'slow', 'emotional']  # More specific sad moods
        elif emotion in ['fear']:
            moods = ['calm', 'ambient', 'peaceful']  # Calming music for fear
        elif emotion in ['neutral']:
            moods = ['moderate', 'balanced', 'chill']
        elif emotion in ['angry', 'disgust']:
            moods = ['intense', 'powerful', 'energetic']
        else:
            moods = ['moderate']  # Default mood
        
        logger.info(f"Finding songs for emotion: {emotion}, moods: {moods}")
        
        # Filter songs by matching moods (case insensitive)
        emotion_songs = _music_df[_music_df['mood'].str.lower().isin([m.lower() for m in moods])]
        
        if len(emotion_songs) == 0:
            logger.warning(f"No songs found for emotion: {emotion}, moods: {moods}. Falling back to all songs.")
            emotion_songs = _music_df
        else:
            logger.info(f"Found {len(emotion_songs)} songs matching the mood")
        
        # Initialize played songs for this emotion if not exists
        if emotion not in _played_songs:
            _played_songs[emotion] = set()
        
        # Get unplayed songs
        unplayed_songs = emotion_songs[~emotion_songs.index.isin(_played_songs[emotion])]
        
        # If all songs have been played, reset the played songs for this emotion
        if len(unplayed_songs) < 5:
            logger.info(f"Resetting played songs for emotion: {emotion}")
            _played_songs[emotion] = set()
            unplayed_songs = emotion_songs
        
        # Select up to 5 random songs from unplayed songs
        num_recommendations = min(5, len(unplayed_songs))
        selected_songs = unplayed_songs.sample(n=num_recommendations)
        
        # Add selected songs to played songs
        _played_songs[emotion].update(selected_songs.index)
        
        # Prepare the recommendations
        result = []
        for _, song in selected_songs.iterrows():
            song_info = {
                'song_title': song['name'],
                'artist': song['artist'],
                'spotify_link': f"https://open.spotify.com/track/{song['id']}",
                'spotify_embed': f"https://open.spotify.com/embed/track/{song['id']}",
                'emotion': emotion,
                'mood': song['mood']
            }
            result.append(song_info)
            logger.info(f"Recommending song: {song['name']} by {song['artist']} (Mood: {song['mood']})")
        
        return result
    except Exception as e:
        logger.error(f"Error selecting songs: {str(e)}")
        raise ValueError(f"Failed to generate recommendations: {str(e)}") 