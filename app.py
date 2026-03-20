import streamlit as st
import pickle

model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

st.title("📧 Spam Email Classifier")
st.write("Check if a message is Spam or Not")

input_msg = st.text_area("Enter your message:")

if st.button("Predict"):
    if input_msg.strip() != "":
        msg_tfidf = vectorizer.transform([input_msg.lower()])
        prediction = model.predict(msg_tfidf)
        prob = model.predict_proba(msg_tfidf)

        confidence = max(prob[0]) * 100

        if prediction[0] == 1:
            st.error("🚨 This is SPAM!")
        else:
            st.success("✅ This is NOT Spam (Ham)")

        st.info(f"Confidence: {confidence:.2f}%")

    else:
        st.warning("Please enter a message")