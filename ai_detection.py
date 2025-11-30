import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc, classification_report
from sklearn.utils import resample
import os
import re

# Preprocess the text data
def clean_text(t):
    t = str(t).lower()
    t = re.sub(r"http\S+", "", t)
    t = re.sub(r"@\w+", "", t)
    t = re.sub(r"#", "", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

# LOAD DATA
data = pd.read_csv("data/AI_Human.csv")

# BALANCE DATA PROPERLY
ai = data[data["generated"] == 1]
human = data[data["generated"] == 0]
human_balanced = resample(human, replace=False, n_samples=len(ai), random_state=42)
data_balanced = pd.concat([ai, human_balanced]).sample(frac=1, random_state=42)

data_balanced["clean_text"] = data_balanced["text"].apply(clean_text)

# TRAIN ON BALANCED DATA
X = data_balanced["clean_text"]
y = data_balanced["generated"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print("Train class counts:\n", y_train.value_counts())
print("Test class counts:\n", y_test.value_counts())

# VECTORIZATION
vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1,3))
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)


model = LogisticRegression(max_iter=1500)
model.fit(X_train_vec, y_train)

y_pred = model.predict(X_test_vec)
print("\nModel accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification report:\n", classification_report(y_test, y_pred))

# VISUALIZATIONS

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt="d", xticklabels=["Human","AI"], yticklabels=["Human","AI"], cmap="Blues")
plt.title("Confusion Matrix")
plt.show()

# ROC Curve
y_prob = model.predict_proba(X_test_vec)[:,1]
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(6,4))
plt.plot(fpr, tpr, label=f"ROC curve (AUC={roc_auc:.2f})")
plt.plot([0,1],[0,1], linestyle="--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()

# Classification Report Heatmap
report = classification_report(y_test, y_pred, output_dict=True)
report_df = pd.DataFrame(report).transpose()
metrics = report_df.iloc[:2, :3]  # precision, recall, f1
plt.figure(figsize=(6,4))
sns.heatmap(metrics, annot=True, cmap="Blues", fmt=".2f")
plt.title("Classification Report Heatmap")
plt.show()


# EXAMPLE TWEETS
example_tweets = [
    "I love spending time with my friends and family",
    "Sabrina never fails to slay in her concerts!",
]

example_clean = [clean_text(t) for t in example_tweets]
example_vec = vectorizer.transform(example_clean)
predictions = model.predict(example_vec)
probabilities = model.predict_proba(example_vec)
for tweet, pred, proba in zip(example_tweets, predictions, probabilities):
    label = "AI" if pred == 1 else "Human"
    confidence = max(proba) * 100
    print(f"'{tweet}' -> {label} ({confidence:.2f}% confidence)")

# Batch processing
file_path = input("\nEnter CSV file with tweets (or press Enter to skip): ").strip()

if file_path != "":
    if os.path.exists(file_path):
        try:
            tweets_df = pd.read_csv(file_path)

            if "text" not in tweets_df.columns:
                print("ERROR: The CSV file must contain a column named 'text'.")
            else:
                print(f"\n✔ Loaded {len(tweets_df)} tweets. Processing...\n")

                tweets_clean = tweets_df["text"].astype(str).apply(clean_text)
                tweets_vec = vectorizer.transform(tweets_clean)

                preds = model.predict(tweets_vec)
                probas = model.predict_proba(tweets_vec)

                tweets_df["prediction"] = ["AI" if p == 1 else "Human" for p in preds]
                tweets_df["confidence"] = probas.max(axis=1)

                print(tweets_df.head())

                tweets_df.to_csv("predicted_output.csv", index=False)
                print("\n✔ Results saved to predicted_output.csv")

        except Exception as e:
            print("Error while reading the file:", e)
    else:
        print("ERROR: File does not exist.")

# Real-time tweet detection
print("\n=== Real-time Tweet Detection ===")
print("Type 'exit' to quit.\n")
while True:
    user_tweet = input("Tweet: ")
    if user_tweet.lower() == "exit": break
    if len(user_tweet.strip()) == 0: 
        print("Please type something.\n")
        continue
    clean = clean_text(user_tweet)
    user_vec = vectorizer.transform([clean])
    pred = model.predict(user_vec)[0]
    proba = model.predict_proba(user_vec)[0]
    label = "AI" if pred == 1 else "Human"
    confidence = max(proba) * 100
    print(f"→ {label} ({confidence:.2f}% confidence)\n")
