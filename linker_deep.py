import pandas as pd
import numpy as np
import faiss
import pickle
import time
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, CrossEncoder

# Import Config and KPICalculator
from config import Config
from kpi_calculator import KPICalculator

class DeepEntityLinker:
    """Two-stage entity linker: Bi-Encoder + Cross-Encoder with batching, caching, and NIL handling"""
    
    def __init__(self, confidence_threshold=None):
        print("\n" + "="*80)
        print(" " * 22 + "INITIALIZING DEEP LINKER")
        print("="*80)
        
        self.confidence_threshold = confidence_threshold if confidence_threshold is not None else Config.NIL_CONFIDENCE_THRESHOLD
        self.cache = {} # Implements 'Add caching mechanisms'
        
        # Load KB
        print(f"\n📚 Loading knowledge base...")
        self.kb_df = pd.read_csv(Config.KB_PATH)
        self.canonical_names = self.kb_df['Canonical_Name'].tolist()
        print(f"✅ Loaded {len(self.canonical_names)} entities")
        
        # Load bi-encoder
        print(f"\n🤖 Loading Bi-Encoder...")
        self.bi_encoder = SentenceTransformer(Config.BI_ENCODER_MODEL)
        print("✅ Bi-Encoder ready")

        # Load cross-encoder
        print(f"\n🤖 Loading Cross-Encoder...")
        self.cross_encoder = CrossEncoder(Config.CROSS_ENCODER_PATH)
        print("✅ Cross-Encoder ready")
        
        self.K_CANDIDATES = Config.K_CANDIDATES
        
        # Load FAISS index
        print(f"\n🔍 Loading FAISS index...")
        self.index = faiss.read_index(Config.FAISS_INDEX_PATH)
        # Ensure nprobe is set for the IndexIVFFlat index
        self.index.nprobe = Config.FAISS_NPROBE
        print(f"✅ FAISS index loaded ({self.index.ntotal} vectors, nprobe={self.index.nprobe})")
        
        # Load ID map
        print(f"\n📂 Loading ID map...")
        with open(Config.ID_MAP_PATH, 'rb') as f:
            self.id_map = pickle.load(f)
        print("✅ ID map ready")
        
        print("\n" + "="*80)
        print(" " * 25 + "✅ INITIALIZATION COMPLETE")
        print("="*80)

    def link_entity(self, mention):
        """Links a single mention using the batched method for consistency and caching"""
        # This single-mention method is now a wrapper around the batched method
        predictions, scores = self.link_entities_batched([mention])
        
        predicted_id = predictions[0]
        score = scores[0]
        
        if predicted_id == Config.NIL_ENTITY_ID:
            canonical = Config.NIL_ENTITY_NAME
        elif predicted_id is not None:
            # Find canonical name from ID
            try:
                canonical = self.kb_df[self.kb_df['Entity_ID'] == predicted_id]['Canonical_Name'].iloc[0]
            except IndexError:
                canonical = "UNKNOWN_ENTITY"
        else:
            canonical = None
            
        return predicted_id, canonical, score

    def link_entities_batched(self, mentions):
        """
        Links a list of mentions using the two-stage approach with batching and caching.
        Implements 'Implement batched and asynchronous inference' and 'Add caching mechanisms'.
        """
        
        # 1. Separate cached and uncached mentions
        uncached_mentions = []
        results = []
        
        for mention in mentions:
            if mention in self.cache:
                results.append(self.cache[mention])
            else:
                uncached_mentions.append(mention)
                results.append(None) # Placeholder
                
        if not uncached_mentions:
            return [r[0] for r in results], [r[2] for r in results] # Return only ID and Score
            
        print(f"   - Linking {len(uncached_mentions)} uncached mentions in batch...")
        
        # Stage 1: Generate embeddings in batch
        mention_embeddings = self.bi_encoder.encode(
            uncached_mentions, 
            convert_to_numpy=True, 
            show_progress_bar=False,
            batch_size=Config.BATCH_SIZE
        )
        mention_embeddings = mention_embeddings.astype('float32')
        faiss.normalize_L2(mention_embeddings)
        
        # Stage 2: Retrieve candidates in batch
        scores_batch, indices_batch = self.index.search(mention_embeddings, self.K_CANDIDATES)
        
        # Prepare for Cross-Encoder re-ranking
        ce_inputs = []
        ce_map = [] # Maps CE input index back to the original uncached mention index
        
        for i, (mention, indices) in enumerate(zip(uncached_mentions, indices_batch)):
            for idx in indices:
                if idx != -1:
                    entity_id = self.id_map.get(idx)
                    canonical_name = self.canonical_names[idx]
                    
                    ce_inputs.append([mention, canonical_name])
                    ce_map.append({'mention_idx': i, 'entity_id': entity_id, 'canonical_name': canonical_name})
        
        # Stage 3: Re-rank with cross-encoder in batch
        if ce_inputs:
            ce_scores = self.cross_encoder.predict(ce_inputs, batch_size=Config.BATCH_SIZE)
        else:
            ce_scores = []
            
        # Process re-ranking results
        best_matches = [None] * len(uncached_mentions)
        
        for score, mapping in zip(ce_scores, ce_map):
            mention_idx = mapping['mention_idx']
            
            # Initialize best match for this mention if not set
            if best_matches[mention_idx] is None or score > best_matches[mention_idx]['score']:
                best_matches[mention_idx] = {
                    'entity_id': mapping['entity_id'],
                    'canonical_name': mapping['canonical_name'],
                    'score': float(score)
                }
                
        # Finalize results and update cache
        for i, mention in enumerate(uncached_mentions):
            best_match = best_matches[i]
            
            if best_match is None:
                # No candidates found
                predicted_id, canonical, score = None, None, 0.0
            else:
                # Apply NIL detection (Implements 'Add NIL detection')
                if best_match['score'] < self.confidence_threshold:
                    predicted_id, canonical, score = Config.NIL_ENTITY_ID, Config.NIL_ENTITY_NAME, best_match['score']
                else:
                    predicted_id, canonical, score = best_match['entity_id'], best_match['canonical_name'], best_match['score']
            
            result = (predicted_id, canonical, score)
            self.cache[mention] = result
            
            # Replace placeholder in original results list
            original_idx = results.index(None)
            results[original_idx] = result
            
        # Extract final predictions and scores from the full results list
        all_predictions = [r[0] for r in results]
        all_scores = [r[2] for r in results]
        
        return all_predictions, all_scores

    def evaluate(self, validation_path=Config.VALIDATION_CSV):
        """Evaluates the deep linker with comprehensive KPIs using batched inference"""
        print(f"\n{'='*80}")
        print(" " * 25 + "DEEP LINKER EVALUATION (Batched)")
        print('='*80)
        
        validation_df = pd.read_csv(validation_path)
        
        if not Config.FULL_VALIDATION and len(validation_df) > 10000:
            validation_df = validation_df.sample(n=10000, random_state=42)
            print(f"📊 Sampled 10,000 rows for evaluation")

        total_mentions = len(validation_df)
        mentions = validation_df['Mention'].tolist()
        start_time = time.time()
        
        # Use batched linking
        predictions, scores = self.link_entities_batched(mentions)
        
        validation_df['Predicted_ID'] = predictions
        validation_df['Score'] = scores
            
        # Calculate comprehensive KPIs
        kpis = KPICalculator.calculate_kpis(
            true_labels=validation_df['Entity_ID'].tolist(),
            predicted_labels=predictions,
            predicted_scores=scores,
            threshold=self.confidence_threshold
        )
        
        # Print results
        KPICalculator.print_kpis(kpis, "DEEP LEARNING ENTITY LINKER (Two-Stage)")
        print(f"\n⏱️  Evaluation Time: {time.time() - start_time:.2f}s")
        
        return kpis
