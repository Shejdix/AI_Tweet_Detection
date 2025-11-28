import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils import resample
import os
import re

# Preprocess the text data
def clean_text(t):
    t = t.lower()
    t = re.sub(r"http\S+", "", t)      
    t = re.sub(r"@\w+", "", t)         
    t = re.sub(r"#", "", t)            
    t = re.sub(r"[^\w\s']", "", t)     
    t = re.sub(r"\s+", " ", t).strip() 
    return t

# Load the dataset
data = pd.read_csv('data/AI_Human.csv')
extra_df = pd.read_csv('data/extra_human.csv')


data["clean_text"] = data["text"].apply(clean_text)

print("Before balancing:")
print("AI tweets:", sum(data["generated"] == 1))
print("Human tweets:", sum(data["generated"] == 0))


ai = data[data["generated"] == 1]
human = data[data["generated"] == 0]

human_downsampled = resample(human, replace=False, n_samples=len(ai), random_state=42)
data_balanced = pd.concat([ai, human_downsampled]).sample(frac=1, random_state=42)

print("\nAfter balancing:")
print("AI tweets:", sum(data_balanced["generated"] == 1))
print("Human tweets:", sum(data_balanced["generated"] == 0))


X = data["clean_text"]
y = data["generated"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Vectorize the text data using TF-IDF
vectorizer = TfidfVectorizer(max_features=20000, ngram_range=(1,3))
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train a logistic regression model
model = LogisticRegression(max_iter=2000)
model.fit(X_train_vec, y_train)

# Evaluate the model
y_pred = model.predict(X_test_vec)
print("\nModel accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification report:")
print(classification_report(y_test, y_pred))

# Function to predict if new tweets are AI-generated or human-written
example_tweets = [
    "I love spending time with my friends and family",
    "Sabrina never fails to slay in her concerts!",
    "slay",
]

example_clean = [clean_text(t) for t in example_tweets]
example_vec = vectorizer.transform(example_clean)
predictions = model.predict(example_vec)
probabilities = model.predict_proba(example_vec)

# Display predictions for new tweets
for tweet, pred, proba in zip(example_tweets, predictions, probabilities):
    label = "AI" if pred == 1 else "Human"
    confidence = max(proba) * 100
    print(f"'{tweet}' -> {label} ({confidence:.2f}% confidence)")

# Batch processing from CSV file
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
print("Type a tweet to analyze it.")
print("Type 'exit' to quit.\n")

while True:
    user_tweet = input("Tweet: ")

    if user_tweet.lower() == "exit":
        print("Exiting...")
        break

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

