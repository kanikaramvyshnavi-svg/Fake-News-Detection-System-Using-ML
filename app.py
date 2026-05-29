import streamlit as st
import pickle

# Load model
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Title
st.title("Fake News Detection System")

st.write("This system predicts whether news is REAL or FAKE.")

# Input
news = st.text_area("Enter News Article")

# Button
if st.button("Analyze News"):

    if news.strip() == "":
        st.warning("Please enter news.")

    else:

        # Convert text
        data = vectorizer.transform([news])

        # Predict
        prediction = model.predict(data)[0]

        # Probability
        probability = model.predict_proba(data)

        confidence = max(probability[0]) * 100

        # Result
        st.subheader("Prediction Result")

        if prediction == 1:
            st.success("REAL NEWS")
        else:
            st.error("FAKE NEWS")

        # Confidence
        st.subheader("Confidence Score")

        st.write(f"{confidence:.2f}%")

        # Summary
        st.subheader("News Summary")

        st.write(news[:500])