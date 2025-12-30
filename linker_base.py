import pandas as pd
import numpy as np
import os
import random
import faiss
import pickle
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, CrossEncoder
from sentence_transformers.cross_encoder.CrossEncoder import InputExample
from torch.utils.data import DataLoader
from fuzzywuzzy import fuzz, process

# Import Config and KPICalculator
from config import Config
from kpi_calculator import KPICalculator

class CrossEncoderTrainer:
    """Handles cross-encoder fine-tuning with Hard Negative Mining"""
    
    @staticmethod
    def load_data():
        """Loads training data and knowledge base"""
        print("\n📂 Loading training data...")
        
        if not os.path.exists(Config.TRAIN_CSV) or not os.path.exists(Config.KB_PATH):
            print("❌ Required files not found")
            return None, None, None
        
        train_df = pd.read_csv(Config.TRAIN_CSV)
        kb_df = pd.read_csv(Config.KB_PATH)

        kb_map = kb_df.set_index('Entity_ID')['Canonical_Name'].to_dict()
        all_canonical_names = kb_df['Canonical_Name'].tolist()
        
        print(f"✅ Training examples: {len(train_df)}")
        print(f"✅ KB entities: {len(all_canonical_names)}")
        
        return train_df, kb_map, all_canonical_names

    @staticmethod
    def create_training_examples(train_df, kb_map, all_canonical_names):
        """
        Creates positive and hard negative training examples using bi-encoder retrieval.
        This implements the 'Introduce hard negative mining' future work point.
        """
        print("\n🔄 Creating training examples with Hard Negative Mining...")
        
        # Load Bi-Encoder and FAISS index for hard negative mining
        print("   - Loading Bi-Encoder and FAISS index for candidate retrieval...")
        bi_encoder = SentenceTransformer(Config.BI_ENCODER_MODEL)
        index = faiss.read_index(Config.FAISS_INDEX_PATH)
        with open(Config.ID_MAP_PATH, 'rb') as f:
            id_map = pickle.load(f)
        
        train_examples = []
        
        # Prepare mentions for batch encoding
        mentions = train_df['Mention'].tolist()
        mention_embeddings = bi_encoder.encode(
            mentions, 
            convert_to_numpy=True, 
            show_progress_bar=True,
            batch_size=Config.BATCH_SIZE * 4 # Use a larger batch size for faster encoding
        )
        mention_embeddings = mention_embeddings.astype('float32')
        faiss.normalize_L2(mention_embeddings)
        
        # Retrieve candidates in batch
        print("   - Retrieving hard negative candidates in batch...")
        # Retrieve K_CANDIDATES + 1 to ensure we get at least one hard negative even if the true positive is the top hit
        K_HARD_NEGATIVES = Config.NUM_NEGATIVES * 2 
        scores, indices = index.search(mention_embeddings, K_HARD_NEGATIVES + 1)
        
        for i, row in tqdm(train_df.iterrows(), total=len(train_df), desc="Generating"):
            mention = row['Mention']
            true_entity_id = row['Entity_ID']
            true_canonical_name = kb_map.get(true_entity_id)
            
            if true_canonical_name is None:
                continue
            
            # 1. Positive Example
            train_examples.append(InputExample(texts=[mention, true_canonical_name], label=1.0))
            
            # 2. Hard Negative Examples
            retrieved_indices = indices[i]
            negative_count = 0
            
            for idx in retrieved_indices:
                if idx == -1: continue
                
                candidate_entity_id = id_map.get(idx)
                candidate_name = all_canonical_names[idx]
                
                # Check if candidate is NOT the true positive
                if candidate_entity_id != true_entity_id:
                    train_examples.append(InputExample(texts=[mention, candidate_name], label=0.0))
                    negative_count += 1
                
                if negative_count >= Config.NUM_NEGATIVES:
                    break
            
            # Fallback to random negative if not enough hard negatives were found
            if negative_count < Config.NUM_NEGATIVES:
                random_negatives = random.sample(all_canonical_names, Config.NUM_NEGATIVES * 5)
                for neg_name in random_negatives:
                    if neg_name != true_canonical_name and neg_name not in [all_canonical_names[idx] for idx in retrieved_indices if idx != -1]:
                        train_examples.append(InputExample(texts=[mention, neg_name], label=0.0))
                        negative_count += 1
                    if negative_count >= Config.NUM_NEGATIVES:
                        break
                    
        print(f"✅ Generated {len(train_examples)} training examples.")
        return train_examples

    @staticmethod
    def run_full_training():
        """Complete cross-encoder training pipeline"""
        print("\n" + "="*80)
        print(" " * 25 + "STEP 3: CROSS-ENCODER TRAINING")
        print("="*80)
        
        train_df, kb_map, all_canonical_names = CrossEncoderTrainer.load_data()
        
        if train_df is None:
            print("❌ Training data not found. Skipping training.")
            return False
            
        train_examples = CrossEncoderTrainer.create_training_examples(
            train_df, 
            kb_map, 
            all_canonical_names
        )
        
        if not train_examples:
            print("❌ No training examples generated. Skipping training.")
            return False
            
        train_dataloader = DataLoader(
            train_examples, 
            shuffle=True, 
            batch_size=Config.BATCH_SIZE
        )
        
        # Load model
        print(f"\n🤖 Loading Cross-Encoder for fine-tuning: {Config.CROSS_ENCODER_MODEL}")
        model = CrossEncoder(Config.CROSS_ENCODER_MODEL, num_labels=1)
        
        # Train model
        print("\n🚀 Starting fine-tuning...")
        model.fit(
            train_dataloader=train_dataloader,
            epochs=Config.NUM_EPOCHS,
            warmup_steps=100,
            output_path=Config.CROSS_ENCODER_PATH
        )
        
        print(f"\n💾 Saved fine-tuned model to: {Config.CROSS_ENCODER_PATH}")
        print("\n" + "="*80)
        print(" " * 25 + "✅ CROSS-ENCODER TRAINING COMPLETE")
        print("="*80)
        return True

class BaselineEntityLinker:
    """Simple entity linker using fuzzy string matching"""
    
    def __init__(self, confidence_threshold=None):
        self.kb = None
        self.kb_mentions = None
        self.kb_entity_map = None
        self.confidence_threshold = confidence_threshold

    def load_knowledge_base(self, kb_path=Config.KB_PATH):
        """Loads the knowledge base"""
        print(f"\n📚 Loading knowledge base...")
        self.kb = pd.read_csv(kb_path)
        self.kb_mentions = self.kb['Canonical_Name'].tolist()
        self.kb_entity_map = self.kb.set_index('Canonical_Name')['Entity_ID'].to_dict()
        print(f"✅ Loaded {len(self.kb_mentions)} entities")

    def link_entity(self, mention):
        """Links a mention to an entity"""
        candidates = process.extract(
            mention, 
            self.kb_mentions, 
            limit=Config.CANDIDATE_TOP_N, 
            scorer=fuzz.token_sort_ratio
        )
        if candidates:
            best_match_name = candidates[0][0]
            best_match_score = candidates[0][1]
            entity_id = self.kb_entity_map.get(best_match_name)
            
            # Apply confidence threshold if set
            if self.confidence_threshold and best_match_score < self.confidence_threshold:
                return None, best_match_score, None
            
            return entity_id, best_match_score, best_match_name
        return None, 0, None

    def evaluate(self, data_path=Config.VALIDATION_CSV, sample_size=None):
        """Evaluates the linker with comprehensive KPIs"""
        print(f"\n{'='*80}")
        print(" " * 25 + "BASELINE EVALUATION")
        print('='*80)
        
        df = pd.read_csv(data_path)
        
        if sample_size and len(df) > sample_size:
            df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)
            print(f"📊 Sampled {sample_size} rows for evaluation")
        
        predictions = []
        scores = []
        start_time = time.time()
        
        for index, row in tqdm(df.iterrows(), total=len(df), desc="🔗 Linking"):
            mention = row['Mention']
            predicted_id, score, _ = self.link_entity(mention)
            predictions.append(predicted_id)
            scores.append(score)

        df['Predicted_ID'] = predictions
        df['Score'] = scores
        
        # Calculate comprehensive KPIs
        kpis = KPICalculator.calculate_kpis(
            true_labels=df['Entity_ID'].tolist(),
            predicted_labels=predictions,
            predicted_scores=scores,
            threshold=self.confidence_threshold
        )
        
        # Print results
        KPICalculator.print_kpis(kpis, "BASELINE ENTITY LINKER (Fuzzy Matching)")
        print(f"\n⏱️  Evaluation Time: {time.time() - start_time:.2f}s")
        
        return kpis
