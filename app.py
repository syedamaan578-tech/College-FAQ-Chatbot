import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Page settings
st.set_page_config(
    page_title="College FAQ Chatbot",
    page_icon="🎓",
    layout="centered"
)

# CSS Styling
st.markdown("""
<style>
.main {
    background-color: #0f1117
}

.title {
    text-align: center;
    color: #4A90E2;
    font-size: 40px;
    font-weight: bold;
}

.subtitle {
    text-align: center;
    color: #aaaaaa;
    margin-bottom: 20px;
}

.chatbox {
    background-color: #1E2A3A;
    padding: 15px;
    border-radius: 10px;
    border-left: 5px solid #1E3A8A;
    margin-top: 20px;
    font-size: 18px;
}
</style>
""", unsafe_allow_html=True)

# Load CSV
faq = pd.read_csv("faq.csv")

# Title
st.markdown('<div class="title">🎓 College FAQ Chatbot</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Ask anything about the college</div>', unsafe_allow_html=True)

# Input
user_question = st.text_input(
    "Ask your question:",
    placeholder="Example: Where is the library?"
)

# Button
if st.button("Get Answer"):

    questions = faq["Question"].tolist()

    vectorizer = CountVectorizer()

    vectors = vectorizer.fit_transform(
        questions + [user_question]
    )

    similarity = cosine_similarity(
        vectors[-1],
        vectors[:-1]
    )

    best_match = similarity.argmax()

    answer = faq.iloc[best_match]["Answer"]

    st.markdown(
        f"""
        <div class="chatbox">
        <b>🤖 Answer:</b><br><br>
        {answer}
        </div>
        """,
        unsafe_allow_html=True
    )