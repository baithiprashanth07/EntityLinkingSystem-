import pandas as pd
import numpy as np
import os
import faiss
import pickle
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

# Import Config
from config import Config

class DataProcessor:
    """Handles data preprocessing and knowledge base creation"""
    
    @staticmethod
    def process_entity_linking_data(input_path, output_path):
        """Transforms TSV into standard format"""
        print(f"\n{'='*80}")
        print(f"Processing: {input_path}")
        print('='*80)
        
        if not os.path.exists(input_path):
            print(f"⚠️  File not found: {input_path}")
            return None
        
        try:
            df = pd.read_csv(
                input_path,
                sep='\t',
                header=None,
                names=['Entity_ID', 'Mention_A', 'Mention_B'],
                on_bad_lines='warn',
                engine='python'
            )
        except Exception as e:
            print(f"❌ Error loading file: {e}")
            return None

        print(f"📊 Original shape: {df.shape}")

        # Clean Entity_ID
        df['Entity_ID'] = pd.to_numeric(df['Entity_ID'], errors='coerce')
        df = df.dropna(subset=['Entity_ID'])
        df['Entity_ID'] = df['Entity_ID'].astype(int)

        # Create records for both mentions
        df_a = df[['Mention_A', 'Entity_ID']].rename(columns={'Mention_A': 'Mention'})
        df_b = df[['Mention_B', 'Entity_ID']].rename(columns={'Mention_B': 'Mention'})

        # Combine and clean
        processed_df = pd.concat([df_a, df_b], ignore_index=True)
        processed_df = processed_df.drop_duplicates()
        processed_df = processed_df.dropna(subset=['Mention', 'Entity_ID'])
        processed_df = processed_df[processed_df['Mention'].astype(str).str.strip() != '']

        print(f"✅ Processed shape: {processed_df.shape}")
        print(f"✅ Unique entities: {processed_df['Entity_ID'].nunique()}")

        processed_df.to_csv(output_path, index=False)
        print(f"💾 Saved to: {output_path}")
        
        return processed_df

    @staticmethod
    def create_knowledge_base(train_df, kb_path):
        """Creates knowledge base mapping Entity_ID to Canonical_Name"""
        print("\n" + "="*80)
        print("Creating Knowledge Base")
        print("="*80)
        
        # Use most frequent mention as canonical name
        kb_df = train_df.groupby('Entity_ID')['Mention'].apply(
            lambda x: x.mode()[0]
        ).reset_index()
        kb_df.rename(columns={'Mention': 'Canonical_Name'}, inplace=True)
        kb_df['Entity_ID'] = kb_df['Entity_ID'].astype(int)
        kb_df['Vector'] = np.nan
        
        kb_df.to_csv(kb_path, index=False)
        print(f"✅ Knowledge base: {kb_df.shape[0]} entities")
        print(f"💾 Saved to: {kb_path}")
        
        return kb_df

    @staticmethod
    def run_full_processing():
        """Complete data processing pipeline"""
        print("\n" + "="*80)
        print(" " * 25 + "STEP 1: DATA PROCESSING")
        print("="*80)
        
        # Process training data
        train_df = DataProcessor.process_entity_linking_data(
            Config.TRAIN_TSV, 
            Config.TRAIN_CSV
        )
        
        if train_df is None:
            print("❌ Training data processing failed")
            return False
        
        # Create knowledge base
        kb_df = DataProcessor.create_knowledge_base(train_df, Config.KB_PATH)
        
        # Process validation data
        validation_df = DataProcessor.process_entity_linking_data(
            Config.VALIDATION_TSV,
            Config.VALIDATION_CSV
        )
        
        if validation_df is not None:
            # Filter validation to match KB entities
            valid_entity_ids = set(kb_df['Entity_ID'].unique())
            validation_df_filtered = validation_df[
                validation_df['Entity_ID'].isin(valid_entity_ids)
            ]
            
            print(f"\n✅ Validation filtered: {validation_df_filtered.shape[0]} rows")
            validation_df_filtered.to_csv(Config.VALIDATION_CSV, index=False)
        
        print("\n" + "="*80)
        print(" " * 25 + "✅ DATA PROCESSING COMPLETE")
        print("="*80)
        return True

class EmbeddingGenerator:
    """Generates embeddings and builds FAISS index"""
    
    @staticmethod
    def generate_embeddings_and_index():
        """Complete embedding generation and indexing pipeline"""
        print("\n" + "="*80)
        print(" " * 22 + "STEP 2: EMBEDDING GENERATION")
        print("="*80)
        
        # Load knowledge base
        if not os.path.exists(Config.KB_PATH):
            print(f"❌ {Config.KB_PATH} not found. Run processing first.")
            return False

        kb_df = pd.read_csv(Config.KB_PATH)
        canonical_names = kb_df['Canonical_Name'].tolist()
        entity_ids = kb_df['Entity_ID'].tolist()
        print(f"\n📚 Loaded {len(canonical_names)} canonical names")

        # Load model
        print(f"🤖 Loading Sentence Transformer: {Config.BI_ENCODER_MODEL}")
        model = SentenceTransformer(Config.BI_ENCODER_MODEL)

        # Generate embeddings
        print("\n🔄 Generating embeddings...")
        embeddings = model.encode(
            canonical_names, 
            convert_to_numpy=True, 
            show_progress_bar=True
        )
        embeddings = embeddings.astype('float32')
        print(f"✅ Embeddings shape: {embeddings.shape}")
        
        # Build FAISS index (Upgraded to IndexIVFFlat for scalability)
        d = embeddings.shape[1]
        print(f"\n🔍 Building FAISS IndexIVFFlat (dimension={d}, nlist={Config.FAISS_NLIST})...")
        
        # Normalize for cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Create a Flat index for the quantizer (IndexFlatIP for inner product/cosine similarity)
        quantizer = faiss.IndexFlatIP(d)
        
        # Create IndexIVFFlat
        index = faiss.IndexIVFFlat(quantizer, d, Config.FAISS_NLIST, faiss.METRIC_INNER_PRODUCT)
        
        # Train the index
        print("   - Training IndexIVFFlat...")
        # We use a subset of the data for training if the dataset is very large, but for simplicity, we use all.
        index.train(embeddings)
        
        # Add the vectors to the index
        print("   - Adding vectors to index...")
        index.add(embeddings)
        
        # Set nprobe for search time (can be adjusted later)
        index.nprobe = Config.FAISS_NPROBE
        
        print(f"✅ FAISS index size: {index.ntotal}")
        print(f"✅ Index type: {type(index)}")

        # Save index
        faiss.write_index(index, Config.FAISS_INDEX_PATH)
        print(f"💾 Saved index: {Config.FAISS_INDEX_PATH}")
        
        # Save ID map
        id_map = {i: entity_ids[i] for i in range(len(entity_ids))}
        with open(Config.ID_MAP_PATH, 'wb') as f:
            pickle.dump(id_map, f)
        print(f"💾 Saved ID map: {Config.ID_MAP_PATH}")
        
        print("\n" + "="*80)
        print(" " * 22 + "✅ EMBEDDING GENERATION COMPLETE")
        print("="*80)
        return True
