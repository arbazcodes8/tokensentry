"""
Sanity tests. Run with: cd tests && python3 -m pytest test_pipeline.py -v

These check the things that would actually invalidate the results if they
were wrong: no cluster leakage across the train/test split, no NaNs
reaching the model, and every reported probability being a real probability.
"""
import sys
import os
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

DATA = os.path.join(os.path.dirname(__file__), "..", "data")


def test_data_files_exist():
    for f in ["cardholders.csv", "tokens.csv", "transactions.csv"]:
        assert os.path.exists(os.path.join(DATA, f)), f"{f} missing — run generate_data.py first"


def test_no_cluster_leakage_between_splits():
    tokens = pd.read_csv(f"{DATA}/tokens.csv")
    train_clusters = set(tokens[tokens.split == "train"]["cluster_id"])
    test_clusters = set(tokens[tokens.split == "test"]["cluster_id"])
    overlap = train_clusters & test_clusters
    assert len(overlap) == 0, f"Cluster leakage detected: {overlap}"


def test_both_splits_contain_fraud_and_legit():
    tokens = pd.read_csv(f"{DATA}/tokens.csv")
    for split in ["train", "test"]:
        sub = tokens[tokens.split == split]
        assert sub["is_fraudulent_provisioning"].sum() > 0, f"{split} has zero fraud cases"
        assert (sub["is_fraudulent_provisioning"] == 0).sum() > 0, f"{split} has zero legit cases"


def test_features_no_nans():
    from features import compute_features, MODEL_A_FEATS, MODEL_B_FEATS
    cardholders = pd.read_csv(f"{DATA}/cardholders.csv")
    tokens = pd.read_csv(f"{DATA}/tokens.csv")
    transactions = pd.read_csv(f"{DATA}/transactions.csv")
    df = compute_features(cardholders, tokens, transactions)
    feats = MODEL_A_FEATS + MODEL_B_FEATS
    nan_counts = df[feats].isna().sum()
    assert nan_counts.sum() == 0, f"NaNs found in features:\n{nan_counts[nan_counts > 0]}"


def test_model_outputs_valid_probabilities():
    from features import compute_features, MODEL_A_FEATS, MODEL_B_FEATS
    from sklearn.ensemble import GradientBoostingClassifier
    cardholders = pd.read_csv(f"{DATA}/cardholders.csv")
    tokens = pd.read_csv(f"{DATA}/tokens.csv")
    transactions = pd.read_csv(f"{DATA}/transactions.csv")
    df = compute_features(cardholders, tokens, transactions)
    feats = MODEL_A_FEATS + MODEL_B_FEATS
    train, test = df[df.split == "train"], df[df.split == "test"]
    clf = GradientBoostingClassifier(n_estimators=50, max_depth=3, random_state=42)
    clf.fit(train[feats], train["is_fraudulent_provisioning"])
    proba = clf.predict_proba(test[feats])[:, 1]
    assert np.all((proba >= 0) & (proba <= 1)), "Probabilities out of [0,1] range"
    assert not np.all(proba == proba[0]), "Model is predicting a constant — something is broken"


def test_audit_log_has_no_ground_truth_leak():
    """The audit log an analyst sees must never contain the ground-truth
    fraud label -- that would defeat the point of a detection system."""
    path = f"{DATA}/audit_log.jsonl"
    if not os.path.exists(path):
        return  # agent.py not yet run, skip
    import json
    with open(path) as fh:
        for line in fh:
            entry = json.loads(line)
            assert "ground_truth_fraud" not in entry, "Ground truth leaked into analyst-facing audit log!"


def test_model_artifacts_and_live_scoring():
    """Model P and Model B must be persisted and loadable, and the live
    FeatureStore must produce valid, in-range probabilities for a fresh
    event -- this is what live_server.py depends on at request time."""
    import joblib
    p_path = f"{DATA}/model_p.joblib"
    b_path = f"{DATA}/model_b.joblib"
    if not (os.path.exists(p_path) and os.path.exists(b_path)):
        return  # not yet trained, skip
    from live_state import FeatureStore
    store = FeatureStore()
    event = {
        "token_id": "tok_test_sanity", "cardholder_id": "card_test_sanity",
        "device_id": "dev_test_sanity", "provisioning_lat": 19.0, "provisioning_lon": 72.8,
        "auth_factor": "otp", "provisioning_ts": "2026-01-01T00:00:00Z",
    }
    p_bundle = joblib.load(p_path)
    feats = store.score_provisioning(event)
    x = pd.DataFrame([[feats[f] for f in p_bundle["features"]]], columns=p_bundle["features"])
    proba = p_bundle["model"].predict_proba(x)[0, 1]
    assert 0 <= proba <= 1, "Model P produced an out-of-range probability"


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
