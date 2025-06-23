import re
import pickle

# Load saved files
with open("vectorizer.pkl", "rb") as f:
    loaded_vectorizer = pickle.load(f)

with open("model.pkl", "rb") as f:
    loaded_model = pickle.load(f)

# New prediction
new_message = "Congratulations! You've won a free ticket."
new_message_clean = re.sub(r'[^a-z\s]', '', new_message.lower())
new_vect = loaded_vectorizer.transform([new_message_clean])
prediction = loaded_model.predict(new_vect)

print(f"Result: {new_message} --> {prediction[0]}")
