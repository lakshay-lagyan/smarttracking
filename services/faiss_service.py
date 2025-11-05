import faiss
import numpy as np
import logging
from threading import Lock

logger = logging.getLogger(__name__)


class FAISSService:
    
    def __init__(self, embedding_dim=512):
        self.embedding_dim = embedding_dim
        self.index = None
        self.person_ids = []
        self.person_names = {}
        self.lock = Lock()
        self._initialize_index()
    
    def _initialize_index(self):
        """Initialize FAISS index"""
        try:
            # Use L2 distance (Euclidean)
            self.index = faiss.IndexFlatL2(self.embedding_dim)
            logger.info(f"FAISS index initialized with dimension {self.embedding_dim}")
        except Exception as e:
            logger.error(f"Failed to initialize FAISS index: {e}")
            raise
    
    def add_person(self, person_id, name, embedding):
       
        with self.lock:
            try:
                # Ensure embedding is correct shape and type
                embedding = np.array(embedding, dtype=np.float32)
                if embedding.ndim == 1:
                    embedding = embedding.reshape(1, -1)
                
                if embedding.shape[1] != self.embedding_dim:
                    raise ValueError(f"Embedding dimension mismatch: expected {self.embedding_dim}, got {embedding.shape[1]}")
                
                # Add to index
                self.index.add(embedding)
                self.person_ids.append(person_id)
                self.person_names[person_id] = name
                
                logger.info(f"Added person {name} (ID: {person_id}) to index")
                
            except Exception as e:
                logger.error(f"Failed to add person to index: {e}")
                raise
    
    def remove_person(self, person_id):
        
        with self.lock:
            try:
                if person_id not in self.person_ids:
                    logger.warning(f"Person ID {person_id} not in index")
                    return
                
                # Remove from tracking lists
                idx = self.person_ids.index(person_id)
                self.person_ids.pop(idx)
                if person_id in self.person_names:
                    del self.person_names[person_id]
                
                logger.info(f"Removed person ID {person_id} from index")
                
                # Note: FAISS doesn't support removal, so we need to rebuild
                # For now, just mark it as removed from our tracking
                
            except Exception as e:
                logger.error(f"Failed to remove person from index: {e}")
                raise
    
    def search(self, embedding, k=1, threshold=0.6):
       
        with self.lock:
            try:
                if self.index.ntotal == 0:
                    return []
                
                # Ensure embedding is correct shape
                embedding = np.array(embedding, dtype=np.float32)
                if embedding.ndim == 1:
                    embedding = embedding.reshape(1, -1)
                
                # Search index
                distances, indices = self.index.search(embedding, min(k, self.index.ntotal))
                
                # Process results
                results = []
                for dist, idx in zip(distances[0], indices[0]):
                    if idx >= 0 and idx < len(self.person_ids):
                        # Convert L2 distance to similarity score
                        # Lower distance = higher similarity
                        # Scale distance to 0-1 range
                        confidence = max(0, 1 - (dist / 2.0))
                        
                        if confidence >= (1 - threshold):
                            person_id = self.person_ids[idx]
                            name = self.person_names.get(person_id, 'Unknown')
                            
                            results.append({
                                'person_id': person_id,
                                'name': name,
                                'distance': float(dist),
                                'confidence': float(confidence)
                            })
                
                return results
                
            except Exception as e:
                logger.error(f"Search failed: {e}")
                raise
    
    def rebuild_from_database(self, persons):
        
        with self.lock:
            try:
                # Reinitialize index
                self._initialize_index()
                self.person_ids = []
                self.person_names = {}
                
                if not persons:
                    logger.info("No persons to add to index")
                    return
                
                # Add all persons
                for person_id, name, embedding in persons:
                    try:
                        # Convert bytes to numpy array if needed
                        if isinstance(embedding, bytes):
                            embedding = np.frombuffer(embedding, dtype=np.float32)
                        
                        embedding = np.array(embedding, dtype=np.float32)
                        if embedding.ndim == 1:
                            embedding = embedding.reshape(1, -1)
                        
                        self.index.add(embedding)
                        self.person_ids.append(person_id)
                        self.person_names[person_id] = name
                        
                    except Exception as e:
                        logger.error(f"Failed to add person {person_id} during rebuild: {e}")
                        continue
                
                logger.info(f"Index rebuilt with {self.index.ntotal} persons")
                
            except Exception as e:
                logger.error(f"Failed to rebuild index: {e}")
                raise
    
    def get_total_persons(self):
        """Get total number of persons in index"""
        return self.index.ntotal if self.index else 0
    
    def clear(self):
        """Clear the index"""
        with self.lock:
            self._initialize_index()
            self.person_ids = []
            self.person_names = {}
            logger.info("Index cleared")


# Global instance
faiss_service = FAISSService()
