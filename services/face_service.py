import cv2
import numpy as np
import base64
import logging
from io import BytesIO
from PIL import Image
from deepface import DeepFace

logger = logging.getLogger(__name__)


class FaceRecognitionService:
    
    def __init__(self, model_name='Facenet512', detector_backend='mtcnn'):
        self.model_name = model_name
        self.detector_backend = detector_backend
        self.embedding_size = 512 if model_name == 'Facenet512' else 128
        logger.info(f"FaceRecognitionService initialized with {model_name}")
    
    def extract_embedding(self, image_data):
      
        try:
            # Convert base64 to image if needed
            if isinstance(image_data, str):
                img = self._base64_to_image(image_data)
            else:
                img = image_data
            
            # Extract embedding using DeepFace
            embedding_objs = DeepFace.represent(
                img_path=img,
                model_name=self.model_name,
                detector_backend=self.detector_backend,
                enforce_detection=True
            )
            
            if not embedding_objs:
                raise ValueError("No face detected in image")
            
            # Get first face embedding
            embedding = np.array(embedding_objs[0]['embedding'])
            
            # Normalize embedding
            embedding = embedding / np.linalg.norm(embedding)
            
            return embedding.astype(np.float32)
            
        except Exception as e:
            logger.error(f"Error extracting embedding: {e}")
            raise ValueError(f"Failed to extract face embedding: {str(e)}")
    
    def extract_multiple_embeddings(self, images_data):
        
        embeddings = []
        failed_count = 0
        
        for img_data in images_data:
            try:
                embedding = self.extract_embedding(img_data)
                embeddings.append(embedding)
            except Exception as e:
                logger.warning(f"Failed to process image: {e}")
                failed_count += 1
                continue
        
        if not embeddings:
            raise ValueError("Failed to extract embeddings from all images")
        
        if failed_count > 0:
            logger.warning(f"Failed to process {failed_count}/{len(images_data)} images")
        
        # Average all embeddings
        avg_embedding = np.mean(embeddings, axis=0)
        
        # Normalize the averaged embedding
        avg_embedding = avg_embedding / np.linalg.norm(avg_embedding)
        
        return avg_embedding.astype(np.float32)
    
    def verify_face(self, img1_data, img2_data, threshold=0.4):
       
        try:
            embedding1 = self.extract_embedding(img1_data)
            embedding2 = self.extract_embedding(img2_data)
            
            # Calculate cosine distance
            distance = self._cosine_distance(embedding1, embedding2)
            
            is_same = distance < threshold
            confidence = 1 - distance
            
            return {
                'verified': is_same,
                'distance': float(distance),
                'confidence': float(confidence),
                'threshold': threshold
            }
            
        except Exception as e:
            logger.error(f"Error verifying faces: {e}")
            raise ValueError(f"Failed to verify faces: {str(e)}")
    
    def detect_faces(self, image_data):
      
        try:
            if isinstance(image_data, str):
                img = self._base64_to_image(image_data)
            else:
                img = image_data
            
            faces = DeepFace.extract_faces(
                img_path=img,
                detector_backend=self.detector_backend,
                enforce_detection=False
            )
            
            return len(faces)
            
        except Exception as e:
            logger.error(f"Error detecting faces: {e}")
            return 0
    
    def _base64_to_image(self, base64_string):
        try:
            # Remove data URL prefix if present
            if ',' in base64_string:
                base64_string = base64_string.split(',')[1]
            
            # Decode base64
            img_data = base64.b64decode(base64_string)
            
            # Convert to PIL Image
            img = Image.open(BytesIO(img_data))
            
            # Convert to RGB if needed
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Convert to numpy array
            img_array = np.array(img)
            
            return img_array
            
        except Exception as e:
            logger.error(f"Error converting base64 to image: {e}")
            raise ValueError(f"Invalid image data: {str(e)}")
    
    def _cosine_distance(self, embedding1, embedding2):
        """Calculate cosine distance between two embeddings"""
        return 1 - np.dot(embedding1, embedding2)


# Global instance
face_service = FaceRecognitionService()
