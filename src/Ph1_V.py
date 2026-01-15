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


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")



NER_MODEL = "d4data/biomedical-ner-all"
NER_THRESHOLD = 0.60
NEL_THRESHOLD = 0.65

# STT MODULE

class VoiceToTextPipeline:
    def __init__(self):
        self.model = whisper.load_model("base").to(DEVICE)

    def transcribe(self, audio_path):
        return self.model.transcribe(
            audio_path,
            language="en",
            fp16=(DEVICE == "cuda"),
            verbose=False
        )["text"]

# ============================================================
# NER MODULE
# ============================================================
class ClinicalNER:
    def __init__(self):
        self.ner = pipeline(
            "ner",
            model=NER_MODEL,
            tokenizer=NER_MODEL,
            aggregation_strategy="simple"
            device=0 if DEVICE == "cuda" else -1
        )

    def extract(self, text):
        return [
            {
                "text": r["word"],
                "label": r["entity_group"],
                "confidence": float(r["score"])
            }
            for r in self.ner(text)
            if r["score"] >= NER_THRESHOLD
        ]

# ============================================================
# HPO LOOKUP BUILDER
# ============================================================
def build_hpo_lookup(hpo_json, output_csv):
    with open(hpo_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    rows = []
    for graph in data.get("graphs", []):
        for node in graph.get("nodes", []):
            if "HP_" not in node.get("id", ""):
                continue
            hp_id = "HP:" + node["id"].split("HP_")[-1]

            if node.get("lbl"):
                rows.append({"hp_id": hp_id, "term": node["lbl"].lower()})

            for syn in node.get("meta", {}).get("synonyms", []):
                rows.append({"hp_id": hp_id, "term": syn["val"].lower()})

    pd.DataFrame(rows).drop_duplicates().to_csv(output_csv, index=False)
    print("✅ HPO lookup built")

# ============================================================
# HPO NEL
# ============================================================
class HPONEL:
    def __init__(self, lookup_csv):
        self.df = pd.read_csv(lookup_csv)
        self.encoder = SentenceTransformer("all-MiniLM-L6-v2", device=DEVICE)

        cache_file = "data/hpo_embeddings.npy"

        if os.path.exists(cache_file):
            print("⚡ Loading cached HPO embeddings")
            emb = np.load(cache_file)
        else:
            print("🧠 Computing HPO embeddings (first run only)...")
            emb = self.encoder.encode(
                self.df["term"].tolist(),
                convert_to_tensor=False,
                show_progress_bar=True
            )
            np.save(cache_file, emb)

        self.embeddings = torch.from_numpy(emb).to(DEVICE)

    def link(self, text):
        q = self.encoder.encode(text.lower(), convert_to_tensor=True).to(DEVICE)
        scores = util.cos_sim(q, self.embeddings)[0]
        idx = scores.argmax().item()
        score = scores[idx].item()

        if score < NEL_THRESHOLD:
            return None

        return {
            "hp_id": self.df.iloc[idx]["hp_id"],
            "concept": self.df.iloc[idx]["term"],
            "confidence": round(score, 3)
        }


# ============================================================
# EVENT CREATION
# ============================================================
def make_event(patient_id, timestamp, hp_link, source):
    return {
        "patient_id": patient_id,
        "timestamp": timestamp,
        "hp_id": hp_link["hp_id"],
        "concept": hp_link["concept"],
        "confidence": hp_link["confidence"],
        "source": source
    }

# ============================================================
# AUDIO → EVENTS
# ============================================================
def events_from_audio(audio_path, patient_id, stt, ner, nel):
    text = stt.transcribe(audio_path)
    entities = ner.extract(text)

    events = []
    for e in entities:
        link = nel.link(e["text"])
        if link:
            events.append(
                make_event(
                    patient_id,
                    datetime.now().isoformat(),
                    link,
                    "audio"
                )
            )
    return events

# ============================================================
# MIMIC-IV CSV → EVENTS (ICD → HPO)
# ============================================================
def events_from_mimic_csv(diagnoses_csv, admissions_csv, nel):
    diagnoses = pd.read_csv(diagnoses_csv)
    admissions = pd.read_csv(admissions_csv)

    adm_time = admissions.set_index("hadm_id")["admittime"].to_dict()
    events = []

    for _, row in diagnoses.iterrows():
        timestamp = adm_time.get(row["hadm_id"], None)
        if timestamp is None:
            continue

        link = nel.link(str(row["icd_code"]))
        if link:
            events.append(
                make_event(
                    row["subject_id"],
                    timestamp,
                    link,
                    "mimic_icd"
                )
            )
    return events

# ============================================================
# TEMPORAL FEATURES
# ============================================================
def extract_temporal_features(event_log):
    df = pd.DataFrame(event_log)
    if df.empty:
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

    return out

# ============================================================
# PHASE-1 RULE-BASED RISK
# ============================================================
def temporal_risk_decision_rule(df):
    def decide(r):
        if r["count_24h"] >= 2 and r["time_since_last_hours"] <= 1:
            return 2
        elif r["count_24h"] >= 1 and r["time_since_last_hours"] <= 6:
            return 1
        return 0

    df["rule_risk"] = df.apply(decide, axis=1)
    return df

# ============================================================
# PHASE-2 ML RISK CLASSIFIER
# ============================================================
class TemporalRiskClassifier:
    def __init__(self):
        self.scaler = StandardScaler()
        self.model = LogisticRegression(multi_class="auto", max_iter=500)

    def fit(self, df):
        X = df[["total_count", "count_24h", "time_since_last_hours"]]
        y = df["rule_risk"]

        # Check if there's only one class in the training data
        if len(y.unique()) < 2:
            print(f"Warning: Only one class ({{y.unique()[0]}}) present in training data. Cannot train Logistic Regression model.")
            # Optionally, you could store the single class and create a dummy predict method.
            # For now, we'll avoid fitting a model that will definitely fail.
            self.model_trained = False # Flag to indicate if model was successfully trained
            self.single_class_prediction = y.unique()[0]
            return
        else:
            self.model_trained = True

        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)

    def predict(self, df):
        if not self.model_trained:
            # If model was not trained due to single class, predict that single class
            mapping = {0: "LOW", 1: "MODERATE", 2: "HIGH"}
            df["ml_risk"] = mapping[self.single_class_prediction]
            return df

        X = df[["total_count", "count_24h", "time_since_last_hours"]]
        X_scaled = self.scaler.transform(X)
        preds = self.model.predict(X_scaled)

        mapping = {0: "LOW", 1: "MODERATE", 2: "HIGH"}
        df["ml_risk"] = [mapping[p] for p in preds]
        return df

# ============================================================
# FEATURE IMPORTANCE (FIXED) - Adjusted to handle single-class training gracefully
# ============================================================
def print_feature_importance(clf):
    if not clf.model_trained:
        print("\n📈 FEATURE IMPORTANCE: Model not trained due to single class in data.")
        return

    features = ["total_count", "count_24h", "time_since_last_hours"]
    print("\n📈 FEATURE IMPORTANCE (Logistic Regression)")

    class_mapping = {0: "LOW", 1: "MODERATE", 2: "HIGH"}

    if len(clf.model.classes_) == 1:
        class_idx = clf.model.classes_[0]
        class_name = class_mapping.get(class_idx, f"Class_{class_idx}")
        print(f"\nClass: {class_name} (Only class found in training data)")
        for f, c in zip(features, clf.model.coef_[0]):
            print(f"  {f:25s} → {c:.3f}")
    elif len(clf.model.classes_) == 2 and clf.model.coef_.shape[0] == 1:
        positive_class_idx = clf.model.classes_[-1]
        negative_class_idx = clf.model.classes_[0]

        print(f"\nClass: {class_mapping.get(positive_class_idx, f'Class_{positive_class_idx}')}")
        for f, c in zip(features, clf.model.coef_[0]):
            print(f"  {f:25s} → {c:.3f}")

        print(f"\nClass: {class_mapping.get(negative_class_idx, f'Class_{negative_class_idx}')}")
        for f, c in zip(features, -clf.model.coef_[0]):
            print(f"  {f:25s} → {c:.3f}")
    elif clf.model.coef_.shape[0] == len(clf.model.classes_):
        for i, class_idx in enumerate(clf.model.classes_):
            class_name = class_mapping.get(class_idx, f"Class_{class_idx}")
            print(f"\nClass: {class_name}")
            for f, c in zip(features, clf.model.coef_[i]):
                print(f"  {f:25s} → {c:.3f}")
    else:
        print("\nCould not interpret feature importances due to unexpected model coefficient shape.")
        print(f"clf.model.coef_.shape: {clf.model.coef_.shape}")
        print(f"clf.model.classes_: {clf.model.classes_}")

# ============================================================
# FINAL AUDIO RISK AGGREGATION
# ============================================================
def summarize_audio_risk(df):
    priority = {"LOW": 0, "MODERATE": 1, "HIGH": 2}
    # Ensure df["ml_risk"] contains valid keys for the priority map
    # If the model was not trained, df["ml_risk"] will be a single string
    if isinstance(df["ml_risk"], str):
        return df["ml_risk"]

    max_risk = df["ml_risk"].map(priority).max()
    reverse = {v: k for k, v in priority.items()}
    return reverse[max_risk]

# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":

    AUDIO_FILE = r"C:\Users\kvenk\Documents\Elderly_Voice\data\raw_audio\daddy1.ogg"
    HPO_JSON = r"C:\Users\kvenk\Documents\Elderly_Voice\data\hp.json"
    HPO_LOOKUP = r"C:\Users\kvenk\Documents\Elderly_Voice\data\hpo_lookup.csv"
    MIMIC_DIAG = r"C:\Users\kvenk\Documents\Elderly_Voice\data\diagnoses_icd.csv"  
    MIMIC_ADM = r"C:\Users\kvenk\Documents\Elderly_Voice\data\admissions.csv"      

    build_hpo_lookup(HPO_JSON, HPO_LOOKUP)

    stt = VoiceToTextPipeline()
    ner = ClinicalNER()
    nel = HPONEL(HPO_LOOKUP)

    event_log = [] # Initialize event_log

    # Add audio events
    event_log.extend(events_from_audio(AUDIO_FILE, "AUDIO_PATIENT", stt, ner, nel))

    # Add MIMIC-IV events to diversify the training data
    event_log.extend(events_from_mimic_csv(MIMIC_DIAG, MIMIC_ADM, nel))

    print(f"\n📦 TOTAL EVENTS: {len(event_log)}")
    print(event_log)

    temporal = extract_temporal_features(event_log)
    temporal = temporal_risk_decision_rule(temporal)

    # Check if temporal DataFrame is empty before proceeding
    if temporal.empty:
        print("No temporal events to process after feature extraction. Exiting.")
    else:
        clf = TemporalRiskClassifier()
        clf.fit(temporal)

        temporal = clf.predict(temporal)

        print("\n📊 EVENT-LEVEL RISK")
        print(temporal[["hp_id", "total_count", "count_24h", "time_since_last_hours", "ml_risk"]])

        final_risk = summarize_audio_risk(temporal)
        print(f"\n🚨 FINAL AUDIO RISK → {final_risk}")

        print_feature_importance(clf)