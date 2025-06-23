import pandas as pd
import re
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# 1. Load your data
data = pd.read_csv("../dataset/spam.csv", encoding="latin-1")
data = data.rename(columns={'v1': 'label', 'v2': 'message'})
data = data[['label','message']]

# 2. Preprocess
def clean(text): 
    return re.sub(r'[^a-z\s]', '', text.lower())

data["cleaned"] = data["message"].apply(clean)

# 3. Split
X_train, X_test, y_train, y_test = train_test_split(data["cleaned"], data["label"], test_size=0.2, random_state=42)

# 4. Vectorize
vectorizer = TfidfVectorizer()
X_train_vect = vectorizer.fit_transform(X_train)

# 5. Train Model
model = MultinomialNB()
model.fit(X_train_vect, y_train)

# ✅ Save Model and Vectorizer
with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Model and Vectorizer saved successfully.")