from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from time import time
import numpy as np
from scipy.sparse import hstack

def train_and_evaluate_svm(df, seed):
    # Encode category
    le = LabelEncoder()
    df["category_encoded"] = le.fit_transform(df["category"])

    # Split data
    X_text = df["clean_text"]
    X_num = df[["category_encoded", "rating"]]
    y = df["label"]

    X_text_train, X_text_test, X_num_train, X_num_test, y_train, y_test = train_test_split(
        X_text, X_num, y, test_size=0.25, random_state=42, stratify=y
    )

    # TF-IDF
    vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    X_train_tfidf = vectorizer.fit_transform(X_text_train)
    X_test_tfidf = vectorizer.transform(X_text_test)

    # Scale numeric features
    scaler = StandardScaler()
    X_train_num = scaler.fit_transform(X_num_train)
    X_test_num = scaler.transform(X_num_test)

    # Combine features
    X_train_final = hstack([X_train_tfidf, X_train_num])
    X_test_final = hstack([X_test_tfidf, X_test_num])

    # Train (timed)
    start_train = time()
    model = SVC(kernel='linear', probability=True)
    model.fit(X_train_final, y_train)
    train_time = time() - start_train

    # Inference (timed)
    start_inf = time()
    y_pred = model.predict(X_test_final)
    y_probs = model.predict_proba(X_test_final)
    infer_time = time() - start_inf

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")
    auc = roc_auc_score(y_test, y_probs[:, 1])

    return acc, f1, auc, train_time, infer_time, y_test.to_numpy(), y_probs[:,1]
