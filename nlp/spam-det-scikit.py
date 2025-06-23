import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
import re

# 1️⃣ Load the dataset
data = pd.read_csv("../dataset/spam.csv", encoding='latin-1')
data = data.rename(columns={'v1': 'label', 'v2': 'message'})
data = data[['label', 'message']]  # Keep only the needed columns

# 2️⃣ Preprocess text
def clean(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)  # remove punctuation
    return text

data["cleaned"] = data["message"].apply(clean)

# 3️⃣ Train-test Split
X_train, X_test, y_train, y_test = train_test_split(data["cleaned"], data["label"],
                                                    test_size=0.2, random_state=42)

# 4️⃣ Vectorization
vectorizer = TfidfVectorizer()
X_train_vect = vectorizer.fit_transform(X_train)
X_test_vect = vectorizer.transform(X_test)

import numpy as np
np.set_printoptions(threshold=np.inf)
xv5 = X_train_vect.toarray()[:5]
non_zero_values = xv5[xv5 != 0]
# print(non_zero_values)
# exit()

# 5️⃣ Model Training
model = MultinomialNB()
model.fit(X_train_vect, y_train)

# 6️⃣ Evaluation
y_pred = model.predict(X_test_vect)
print(classification_report(y_test, y_pred))
