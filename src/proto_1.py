import whisper
import pandas as pd
import json
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util
from datetime import datetime
import torch
import os
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import warnings

# Logging Setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('elderly_voice.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Suppress warnings from transformers
warnings.filterwarnings('ignore', category=FutureWarning)

# Configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Using device: {DEVICE}")

# GPU memory optimization
if DEVICE == "cuda":
    # Enable optimizations for better GPU performance
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    # Clear cache if needed
    torch.cuda.empty_cache()
    logger.info(f"GPU optimizations enabled. CUDA devices: {torch.cuda.device_count()}")

# Try to use cuML for GPU-accelerated sklearn (optional)
try:
    from cuml.linear_model import LogisticRegression as cuLogisticRegression
    from cuml.preprocessing import StandardScaler as cuStandardScaler
    USE_CUML = (DEVICE == "cuda")
    if USE_CUML:
        logger.info("cuML detected - using GPU-accelerated LogisticRegression")
except ImportError:
    USE_CUML = False
    cuLogisticRegression = None
    cuStandardScaler = None

NER_MODEL = "d4data/biomedical-ner-all"
NER_THRESHOLD = 0.60
NEL_THRESHOLD = 0.65

# Input Validation Utilities
def validate_file_exists(filepath: str, description: str) -> Path:
    """Validate file exists and is readable"""
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"{description} not found: {filepath}")
    if not path.is_file():
        raise ValueError(f"{description} is not a file: {filepath}")
    return path

def validate_audio_format(filepath: str) -> bool:
    """Check if audio file format is supported"""
    supported = ['.ogg', '.mp3', '.wav', '.m4a', '.flac']
    ext = Path(filepath).suffix.lower()
    if ext not in supported:
        logger.warning(f"Unsupported audio format: {ext}. Supported: {supported}")
        return False
    return True

# STT Module
class VoiceToTextPipeline:
    def __init__(self):
        try:
            logger.info("Loading Whisper model...")
            self.model = whisper.load_model("base").to(DEVICE)
            logger.info("[OK] Whisper model loaded")
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}")
            raise

    def transcribe(self, audio_path: str) -> Optional[str]:
        """
        Transcribe audio with error handling
        Returns None if transcription fails
        """
        try:
            # Validate file
            validate_file_exists(audio_path, "Audio file")
            if not validate_audio_format(audio_path):
                return None
            
            logger.info(f"Transcribing: {Path(audio_path).name}")
            result = self.model.transcribe(
                audio_path,
                language="en",
                fp16=(DEVICE == "cuda"),
                verbose=False
            )
            
            text = result["text"].strip()
            if not text:
                logger.warning(f"Empty transcription for {audio_path}")
                return None
                
            logger.info(f"Transcribed: '{text[:50]}...'")
            return text
            
        except Exception as e:
            logger.error(f"Transcription failed for {audio_path}: {e}")
            return None

# Preprocessing Layer
class TextPreprocessor:
    """Clean and normalize transcribed text"""
    
    # Common medical abbreviations
    ABBREVIATIONS = {
        "bp": "blood pressure",
        "hr": "heart rate",
        "temp": "temperature",
        "resp": "respiration",
        "o2": "oxygen",
        "dx": "diagnosis",
        "rx": "prescription",
        "hx": "history",
        "sx": "symptoms",
        "pt": "patient"
    }
    
    # Negation patterns
    NEGATIONS = ["no", "not", "without", "never", "none", "deny", "denies"]
    
    def preprocess(self, text: str) -> Tuple[str, Dict]:
        """
        Preprocess text and return cleaned version + metadata
        Returns: (cleaned_text, metadata_dict)
        """
        if not text:
            return "", {}
        
        original = text
        metadata = {
            "has_negation": False,
            "expanded_abbreviations": [],
            "original_length": len(text)
        }
        
        # Convert to lowercase for processing
        text_lower = text.lower()
        
        # Check for negations
        for neg in self.NEGATIONS:
            if f" {neg} " in f" {text_lower} ":
                metadata["has_negation"] = True
                break
        
        # Expand abbreviations
        words = text_lower.split()
        expanded_words = []
        for word in words:
            clean_word = word.strip('.,!?')
            if clean_word in self.ABBREVIATIONS:
                expanded = self.ABBREVIATIONS[clean_word]
                expanded_words.append(expanded)
                metadata["expanded_abbreviations"].append(f"{clean_word}→{expanded}")
            else:
                expanded_words.append(word)
        
        cleaned = " ".join(expanded_words)
        metadata["processed_length"] = len(cleaned)
        
        if metadata["expanded_abbreviations"]:
            logger.info(f"Expanded: {metadata['expanded_abbreviations']}")
        if metadata["has_negation"]:
            logger.warning("[WARNING] Negation detected in text - review entities carefully")
        
        return cleaned, metadata

# NER Module
class ClinicalNER:
    def __init__(self):
        try:
            logger.info("Loading NER model...")
            # Explicitly set device for GPU acceleration
            device_id = 0 if DEVICE == "cuda" else -1
            self.ner = pipeline(
                "ner",
                model=NER_MODEL,
                tokenizer=NER_MODEL,
                aggregation_strategy="simple",
                device=device_id  # Explicitly use GPU
            )
            logger.info(f"[OK] NER model loaded on {DEVICE}")
        except Exception as e:
            logger.error(f"Failed to load NER model: {e}")
            raise

    def extract(self, text: str) -> List[Dict]:
        """Extract entities with improved filtering"""
        if not text or not text.strip():
            logger.warning("Empty text provided to NER")
            return []
        
        try:
            raw_entities = self.ner(text)
            
            # Filter and clean
            filtered = []
            for r in raw_entities:
                if r["score"] < NER_THRESHOLD:
                    continue
                
                # Clean entity text
                entity_text = r["word"].strip()
                if len(entity_text) < 2:  # Skip single characters
                    continue
                
                filtered.append({
                    "text": entity_text,
                    "label": r["entity_group"],
                    "confidence": round(float(r["score"]), 3),
                    "start": r.get("start", 0),
                    "end": r.get("end", len(text))
                })
            
            logger.info(f"Extracted {len(filtered)} entities (from {len(raw_entities)} raw)")
            return filtered
            
        except Exception as e:
            logger.error(f"NER extraction failed: {e}")
            return []

# HPO Lookup Builder
def build_hpo_lookup(hpo_json: str, output_csv: str) -> bool:
    """Build HPO lookup with validation"""
    try:
        validate_file_exists(hpo_json, "HPO JSON file")
        
        logger.info("Building HPO lookup from JSON...")
        with open(hpo_json, "r", encoding="utf-8") as f:
            data = json.load(f)

        rows = []
        node_count = 0
        synonym_count = 0
        
        for graph in data.get("graphs", []):
            for node in graph.get("nodes", []):
                if "HP_" not in node.get("id", ""):
                    continue
                    
                node_count += 1
                hp_id = "HP:" + node["id"].split("HP_")[-1]

                # Add primary label
                if node.get("lbl"):
                    rows.append({"hp_id": hp_id, "term": node["lbl"].lower()})

                # Add synonyms
                for syn in node.get("meta", {}).get("synonyms", []):
                    synonym_count += 1
                    rows.append({"hp_id": hp_id, "term": syn["val"].lower()})

        # Remove duplicates and save
        df = pd.DataFrame(rows).drop_duplicates()
        df.to_csv(output_csv, index=False)
        
        logger.info(f"[OK] HPO lookup built: {len(df)} terms from {node_count} nodes ({synonym_count} synonyms)")
        return True
        
    except Exception as e:
        logger.error(f"Failed to build HPO lookup: {e}")
        return False

# HPO NEL
class HPONEL:
    def __init__(self, lookup_csv: str):
        try:
            validate_file_exists(lookup_csv, "HPO lookup CSV")
            
            logger.info("Loading HPO lookup...")
            self.df = pd.read_csv(lookup_csv)
            logger.info(f"Loaded {len(self.df)} HPO terms")
            
            logger.info("Loading sentence encoder...")
            # Explicitly set device for GPU acceleration
            self.encoder = SentenceTransformer("all-MiniLM-L6-v2", device=DEVICE)
            
            logger.info("Encoding HPO terms (this may take a moment)...")
            self.embeddings = self.encoder.encode(
                self.df["term"].tolist(),
                convert_to_tensor=True,
                device=DEVICE,  # Ensure encoding happens on GPU
                show_progress_bar=True
            )
            logger.info(f"[OK] HPO NEL ready on {DEVICE}")
            
            # Cache for repeated queries
            self._link_cache = {}
            
        except Exception as e:
            logger.error(f"Failed to initialize HPO NEL: {e}")
            raise

    def link(self, text: str) -> Optional[Dict]:
        """Link text to HPO concept with caching - GPU optimized"""
        if not text or not text.strip():
            return None
        
        # Check cache
        text_lower = text.lower().strip()
        if text_lower in self._link_cache:
            return self._link_cache[text_lower]
        
        try:
            # Encode query on GPU
            q = self.encoder.encode(
                text_lower, 
                convert_to_tensor=True,
                device=DEVICE,  # Explicitly use GPU
                show_progress_bar=False
            )
            
            # Ensure embeddings are on same device as query
            if self.embeddings.device != q.device:
                self.embeddings = self.embeddings.to(q.device)
            
            # Compute similarity on GPU
            scores = util.cos_sim(q, self.embeddings)[0]
            idx = scores.argmax().item()
            score = scores[idx].item()

            if score < NEL_THRESHOLD:
                logger.debug(f"No link found for '{text}' (best score: {score:.3f})")
                self._link_cache[text_lower] = None
                return None

            result = {
                "hp_id": self.df.iloc[idx]["hp_id"],
                "concept": self.df.iloc[idx]["term"],
                "confidence": round(score, 3)
            }
            
            logger.debug(f"Linked '{text}' → {result['hp_id']} ({result['confidence']})")
            self._link_cache[text_lower] = result
            return result
            
        except Exception as e:
            logger.error(f"Linking failed for '{text}': {e}")
            return None
    
    def batch_link(self, texts: List[str], batch_size: int = 32) -> List[Optional[Dict]]:
        """
        Link multiple texts efficiently on GPU with batching
        Faster than individual link() calls for multiple texts
        """
        if not texts:
            return []
        
        results = []
        
        # Filter out cached results first
        texts_to_process = []
        indices_to_process = []
        cached_results = []
        
        for i, text in enumerate(texts):
            if not text or not text.strip():
                results.append(None)
                continue
            
            text_lower = text.lower().strip()
            if text_lower in self._link_cache:
                results.append(self._link_cache[text_lower])
            else:
                texts_to_process.append(text_lower)
                indices_to_process.append(i)
                results.append(None)  # Placeholder
        
        if not texts_to_process:
            return results
        
        try:
            # Process in batches on GPU
            for batch_start in range(0, len(texts_to_process), batch_size):
                batch_texts = texts_to_process[batch_start:batch_start + batch_size]
                batch_indices = indices_to_process[batch_start:batch_start + batch_size]
                
                # Encode entire batch on GPU
                batch_embeddings = self.encoder.encode(
                    batch_texts,
                    convert_to_tensor=True,
                    device=DEVICE,
                    show_progress_bar=False,
                    batch_size=min(batch_size, len(batch_texts))
                )
                
                # Ensure embeddings are on same device
                if self.embeddings.device != batch_embeddings.device:
                    self.embeddings = self.embeddings.to(batch_embeddings.device)
                
                # Compute similarities for entire batch at once (GPU-accelerated)
                batch_scores = util.cos_sim(batch_embeddings, self.embeddings)
                
                # Process each result in batch
                for j, (text_lower, scores) in enumerate(zip(batch_texts, batch_scores)):
                    idx = scores.argmax().item()
                    score = scores[idx].item()
                    
                    original_idx = batch_indices[j]
                    
                    if score >= NEL_THRESHOLD:
                        result = {
                            "hp_id": self.df.iloc[idx]["hp_id"],
                            "concept": self.df.iloc[idx]["term"],
                            "confidence": round(score, 3)
                        }
                        results[original_idx] = result
                        self._link_cache[text_lower] = result
                    else:
                        results[original_idx] = None
                        self._link_cache[text_lower] = None
            
            return results
            
        except Exception as e:
            logger.error(f"Batch linking failed: {e}")
            # Fallback to individual processing
            for i, text in enumerate(texts):
                if results[i] is None:
                    results[i] = self.link(text)
            return results

# Event Creation
def make_event(patient_id: str, timestamp: str, hp_link: Dict, 
               source: str, raw_text: str = "", metadata: Dict = None) -> Dict:
    """Create structured event with metadata"""
    event = {
        "patient_id": patient_id,
        "timestamp": timestamp,
        "hp_id": hp_link["hp_id"],
        "concept": hp_link["concept"],
        "confidence": hp_link["confidence"],
        "source": source,
        "raw_text": raw_text
    }
    
    if metadata:
        event["metadata"] = metadata
    
    return event

# Audio to Events
def events_from_audio(audio_path: str, patient_id: str, 
                     stt: VoiceToTextPipeline, 
                     ner: ClinicalNER, 
                     nel: HPONEL,
                     preprocessor: TextPreprocessor) -> List[Dict]:
    """Process audio with full pipeline"""
    
    # Step 1: Transcribe
    text = stt.transcribe(audio_path)
    if not text:
        logger.error(f"Failed to transcribe {audio_path}")
        return []
    
    # Step 2: Preprocess
    cleaned_text, text_metadata = preprocessor.preprocess(text)
    
    # Step 3: Extract entities
    entities = ner.extract(cleaned_text)
    if not entities:
        logger.warning(f"No entities extracted from {audio_path}")
        return []
    
    # Step 4: Link to HPO
    events = []
    timestamp = datetime.now().isoformat()
    
    for entity in entities:
        link = nel.link(entity["text"])
        if link:
            event = make_event(
                patient_id=patient_id,
                timestamp=timestamp,
                hp_link=link,
                source="audio",
                raw_text=entity["text"],
                metadata={
                    "ner_label": entity["label"],
                    "ner_confidence": entity["confidence"],
                    "has_negation": text_metadata.get("has_negation", False)
                }
            )
            events.append(event)
    
    logger.info(f"Generated {len(events)} events from {audio_path}")
    return events

# MIMIC-IV CSV to Events
def events_from_mimic_csv(diagnoses_csv: str, admissions_csv: str, 
                         nel: HPONEL) -> List[Dict]:
    """Process MIMIC data with validation"""
    try:
        validate_file_exists(diagnoses_csv, "MIMIC diagnoses CSV")
        validate_file_exists(admissions_csv, "MIMIC admissions CSV")
        
        logger.info("Loading MIMIC-IV data...")
        diagnoses = pd.read_csv(diagnoses_csv)
        admissions = pd.read_csv(admissions_csv)
        
        logger.info(f"Loaded {len(diagnoses)} diagnoses, {len(admissions)} admissions")
        
        adm_time = admissions.set_index("hadm_id")["admittime"].to_dict()
        events = []
        
        for idx, row in diagnoses.iterrows():
            timestamp = adm_time.get(row["hadm_id"], None)
            if timestamp is None:
                continue

            link = nel.link(str(row["icd_code"]))
            if link:
                events.append(make_event(
                    patient_id=str(row["subject_id"]),
                    timestamp=timestamp,
                    hp_link=link,
                    source="mimic_icd",
                    raw_text=str(row["icd_code"])
                ))
        
        logger.info(f"Generated {len(events)} events from MIMIC-IV")
        return events
        
    except Exception as e:
        logger.error(f"MIMIC processing failed: {e}")
        return []

# Temporal Features
def extract_temporal_features(event_log: List[Dict]) -> pd.DataFrame:
    """Extract temporal features from event log"""
    df = pd.DataFrame(event_log)
    if df.empty:
        logger.warning("Empty event log")
        return df

    df["timestamp"] = pd.to_datetime(df["timestamp"], format="mixed", errors="coerce")
    df = df.dropna(subset=["timestamp"])

    now = pd.Timestamp.now()

    freq = df.groupby(["patient_id", "hp_id"]).size().reset_index(name="total_count")
    last = df.groupby(["patient_id", "hp_id"])["timestamp"].max().reset_index()
    last["time_since_last_hours"] = (now - last["timestamp"]).dt.total_seconds() / 3600

    recent = df[df["timestamp"] >= now - pd.Timedelta(hours=24)]
    count_24h = recent.groupby(["patient_id", "hp_id"]).size().reset_index(name="count_24h")

    out = freq.merge(last, on=["patient_id", "hp_id"])
    out = out.merge(count_24h, on=["patient_id", "hp_id"], how="left")
    out["count_24h"] = out["count_24h"].fillna(0)

    logger.info(f"Extracted features for {len(out)} unique events")
    return out

# Phase-1 Rule-Based Risk
def temporal_risk_decision_rule(df: pd.DataFrame) -> pd.DataFrame:
    """Apply rule-based risk scoring"""
    def decide(r):
        if r["count_24h"] >= 2 and r["time_since_last_hours"] <= 1:
            return 2
        elif r["count_24h"] >= 1 and r["time_since_last_hours"] <= 6:
            return 1
        return 0

    df["rule_risk"] = df.apply(decide, axis=1)
    return df

# ML Risk Classifior
class TemporalRiskClassifier:
    def __init__(self):
        # Use GPu
        if USE_CUML:
            self.scaler = cuStandardScaler()
            self.model = cuLogisticRegression(max_iter=500, random_state=42)
            logger.info("Using GPU-accelerated cuML for risk classification")
        else:
            self.scaler = StandardScaler()
            self.model = LogisticRegression(max_iter=500, random_state=42)
        self.model_trained = False
        self.single_class_prediction = None

    def fit(self, df: pd.DataFrame) -> bool:
        """Fit model, return True if successful"""
        X = df[["total_count", "count_24h", "time_since_last_hours"]]
        y = df["rule_risk"]

        if len(y.unique()) < 2:
            logger.warning(f"Only one class ({y.unique()[0]}) in training data - cannot train classifier")
            self.model_trained = False
            self.single_class_prediction = y.unique()[0]
            return False
        
        try:
            X_scaled = self.scaler.fit_transform(X)
            self.model.fit(X_scaled, y)
            self.model_trained = True
            logger.info("[OK] Risk classifier trained successfully")
            return True
        except Exception as e:
            logger.error(f"Model training failed: {e}")
            return False

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Predict risk levels"""
        mapping = {0: "LOW", 1: "MODERATE", 2: "HIGH"}
        
        if not self.model_trained:
            df["ml_risk"] = mapping[self.single_class_prediction]
            logger.info(f"Using single-class prediction: {mapping[self.single_class_prediction]}")
            return df

        X = df[["total_count", "count_24h", "time_since_last_hours"]]
        X_scaled = self.scaler.transform(X)
        preds = self.model.predict(X_scaled)

        df["ml_risk"] = [mapping[p] for p in preds]
        return df

    def print_feature_importance(self):
        """Print feature importance"""
        if not self.model_trained:
            print("\nFEATURE IMPORTANCE: Model not trained")
            return

        features = ["total_count", "count_24h", "time_since_last_hours"]
        print("\nFEATURE IMPORTANCE (Logistic Regression)")

        class_mapping = {0: "LOW", 1: "MODERATE", 2: "HIGH"}

        # Handle different cases: single class, binary classification, or multi-class
        if len(self.model.classes_) == 1:
            class_idx = self.model.classes_[0]
            class_name = class_mapping.get(class_idx, f"Class_{class_idx}")
            print(f"\nClass: {class_name} (Only class found in training data)")
            for f, c in zip(features, self.model.coef_[0]):
                print(f"  {f:25s} → {c:.3f}")
        elif len(self.model.classes_) == 2 and self.model.coef_.shape[0] == 1:
            # Binary classification: coef_ has shape (1, n_features)
            positive_class_idx = self.model.classes_[-1]
            negative_class_idx = self.model.classes_[0]

            print(f"\nClass: {class_mapping.get(positive_class_idx, f'Class_{positive_class_idx}')}")
            for f, c in zip(features, self.model.coef_[0]):
                print(f"  {f:25s} → {c:.3f}")

            print(f"\nClass: {class_mapping.get(negative_class_idx, f'Class_{negative_class_idx}')}")
            for f, c in zip(features, -self.model.coef_[0]):
                print(f"  {f:25s} → {c:.3f}")
        elif self.model.coef_.shape[0] == len(self.model.classes_):
            # Multi-class: coef_ has shape (n_classes, n_features)
            for i, class_idx in enumerate(self.model.classes_):
                class_name = class_mapping.get(class_idx, f"Class_{class_idx}")
                print(f"\nClass: {class_name}")
                for f, c in zip(features, self.model.coef_[i]):
                    print(f"  {f:25s} → {c:.3f}")
        else:
            print("\nCould not interpret feature importances due to unexpected model coefficient shape.")
            print(f"self.model.coef_.shape: {self.model.coef_.shape}")
            print(f"self.model.classes_: {self.model.classes_}")

# Final Audio Risk Aggregation
def summarize_audio_risk(df: pd.DataFrame) -> str:
    """Aggregate risk across events"""
    priority = {"LOW": 0, "MODERATE": 1, "HIGH": 2}
    max_risk = df["ml_risk"].map(priority).max()
    reverse = {v: k for k, v in priority.items()}
    return reverse[max_risk]

# Main
if __name__ == "__main__":
    
    # File paths
    AUDIO_FILE = r"C:\Users\kvenk\Documents\Elderly_Voice\data\raw_audio\daddy1.ogg"
    HPO_JSON = r"C:\Users\kvenk\Documents\Elderly_Voice\data\hp.json"
    HPO_LOOKUP = r"C:\Users\kvenk\Documents\Elderly_Voice\data\hpo_lookup.csv"
    MIMIC_DIAG = r"C:\Users\kvenk\Documents\Elderly_Voice\data\diagnoses_icd.csv"  
    MIMIC_ADM = r"C:\Users\kvenk\Documents\Elderly_Voice\data\admissions.csv"

    logger.info("=" * 60)
    logger.info("ELDERLY VOICE MONITORING SYSTEM - STARTING")
    logger.info("=" * 60)

    # Step 1: Build HPO lookup
    if not Path(HPO_LOOKUP).exists():
        if not build_hpo_lookup(HPO_JSON, HPO_LOOKUP):
            logger.error("Failed to build HPO lookup - exiting")
            exit(1)

    # Step 2: Initialize models
    try:
        stt = VoiceToTextPipeline()
        ner = ClinicalNER()
        nel = HPONEL(HPO_LOOKUP)
        preprocessor = TextPreprocessor()
    except Exception as e:
        logger.error(f"Model initialization failed: {e}")
        exit(1)

    # Process events
    event_log = []

  
    if Path(AUDIO_FILE).exists():
        audio_events = events_from_audio(
            AUDIO_FILE, "AUDIO_PATIENT", stt, ner, nel, preprocessor
        )
        event_log.extend(audio_events)
    else:
        logger.warning(f"Audio file not found: {AUDIO_FILE}")

    # MIMIC events
    if Path(MIMIC_DIAG).exists() and Path(MIMIC_ADM).exists():
        mimic_events = events_from_mimic_csv(MIMIC_DIAG, MIMIC_ADM, nel)
        event_log.extend(mimic_events)

    print(f"\nTOTAL EVENTS: {len(event_log)}")

    if not event_log:
        logger.error("No events generated - check your input files")
        exit(1)

    #Temporal analysis
    temporal = extract_temporal_features(event_log)
    
    if temporal.empty:
        logger.error("No temporal features extracted")
        exit(1)
    
    temporal = temporal_risk_decision_rule(temporal)

    # Train and predict
    clf = TemporalRiskClassifier()
    clf.fit(temporal)
    temporal = clf.predict(temporal)

    # Display res
    print("\nEVENT-LEVEL RISK")
    print(temporal[["hp_id", "total_count", "count_24h", "time_since_last_hours", "ml_risk"]])

    final_risk = summarize_audio_risk(temporal)
    print(f"\nFINAL AUDIO RISK -> {final_risk}")

    clf.print_feature_importance()
    
    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 60)