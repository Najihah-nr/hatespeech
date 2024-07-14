import streamlit as st
from streamlit_option_menu import option_menu
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet
from nltk import pos_tag
from joblib import load
import string
import re
import pandas as pd
from PIL import Image

# Download NLTK resources
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
img = Image.open('icon.png')
st.set_page_config(page_title='Hate Speech Detection',page_icon=img)

# Load the model and vectorizer
log_model = load('log_model.pkl')
tfidf_vectorizer = load('tfidf_vectorizer.pkl')

# Load the hate speech lexicon
with open('hatelexicon.txt', 'r') as file:
    hatelexicon = set(file.read().splitlines())

lemmatizer = WordNetLemmatizer()

def get_wordnet_pos(word):
    """Map POS tag to first character lemmatize() accepts."""
    tag = pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)

# Function to preprocess text
def data_preprocess(text):
    text = str(text).lower()  # Convert to lowercase
    text = re.sub(r'https?://\S+|www\.\S+', '', text)  # Remove URLs
    text = re.sub(r'<.*?>+', '', text)  # Remove HTML tags
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)  # Remove punctuation
    text = re.sub(r'\n', '', text)  # Remove newlines
    text = re.sub(r'\w*\d\w*', '', text)  # Remove words containing digits
    text = text.encode('ascii', 'ignore').decode('ascii')  # Remove non-ASCII characters

    # Tokenization
    tokens = word_tokenize(text)

    # POS Tagging and Lemmatization
    lemmatized_tokens = [lemmatizer.lemmatize(word, get_wordnet_pos(word)) for word in tokens if len(word) > 2]

    return " ".join(lemmatized_tokens)

# Unified function for prediction
def predict_hate_speech(text):
    # Preprocess the text
    preprocessed_text = data_preprocess(text)

    # Check if any words from preprocessed input are in the hate speech lexicon
    words_in_lexicon = [word for word in preprocessed_text.split() if word.lower() in hatelexicon]
    if words_in_lexicon:
        return 'Hate_Speech'
    else:
        # Predict using the model
        prediction = log_model.predict(tfidf_vectorizer.transform([preprocessed_text]))[0]
        return prediction

# Function to inject custom CSS for background image
def set_background_image(image_url):
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("{image_url}");
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Sidebar menu
with st.sidebar:
    selected = option_menu(
        menu_title=None,
        options=["Home", "Hate Speech Detector", "Upload File"],
        icons=["house", "hourglass-bottom", "cloud-upload"],
        menu_icon="cast",
        default_index=0,
    )

# Function to display the Home page
def show_home():
    set_background_image("https://wallpapers.com/images/hd/720p-social-background-1280-x-720-deqitd08p8kcwzmt.jpg")
    st.title("Welcome to the Hate Speech Detection Project")
    st.write("""
        This project is dedicated to identifying and mitigating hate speech online.
        The goal is to leverage machine learning and natural language processing
        techniques to detect harmful content and foster a safer online environment.

        However, hate speech is a serious issue that can cause significant harm to individuals and communities.
        It can perpetuate discrimination, incite violence, and create a toxic environment both online and offline.
        Recognizing and addressing hate speech is crucial for fostering a safer and more inclusive society.

        Explore the 'Hate Speech Detector' section to try out the detection tool.
    """)

# Function to display the Hate Speech Detector page
def show_hate_speech_detector():
    # Streamlit CSS styles
    st.markdown("""
    <style>
    .title {
        color: red;
        text-align: left;
        font-family: Arial, Helvetica, sans-serif;
    }
    </style>
    """, unsafe_allow_html=True)

    # Streamlit application title with styled Markdown
    st.markdown('<h1 class="title">Hate Speech Detection</h1>', unsafe_allow_html=True)
    st.write("Enter a text snippet to classify if it's hate speech.")

    # User input
    user_input = st.text_area("Enter your text here:", "")

    # Preprocess button
    if st.button("Preprocess"):
        if user_input.strip():
            # Preprocess the user input
            preprocessed_input = data_preprocess(user_input)
            st.write("Preprocessed Text:", preprocessed_input)
        else:
            st.write("Please enter some text to preprocess.")

    # Predict button
    if st.button("Predict"):
        if user_input.strip():
            # Predict using the unified function
            prediction = predict_hate_speech(user_input)
            if prediction == 'Hate_Speech':
                st.write("This text contains hate speech.")
            else:
                st.write("This text does not contain hate speech.")
        else:
            st.write("Please enter some text to analyze.")

# Function to display the Upload File page
def show_upload_file():
    st.title("Upload a CSV File for Hate Speech Detection")
    st.write("Reminder: The file must contain text data to be analyzed for hate speech.")

    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
    if uploaded_file is not None:
        # Read the CSV file
        df = pd.read_csv(uploaded_file)

        # Check if the file has a 'text' column
        if 'text' in df.columns:
            # Apply the unified prediction function to each row in the 'text' column
            df['prediction'] = df['text'].apply(predict_hate_speech)

            # Display the results
            st.write("Predictions:")
            st.write(df[['text', 'prediction']])
        else:
            st.write("The CSV file does not contain a 'text' column.")

# Function to ask for consent
def ask_for_consent():
    st.title("Data Usage Consent")
    st.write("""
        We need your consent to use the data you provide for system training and evaluation purposes.
        Your data will be used to improve our hate speech detection model and enhance the overall system performance.
    """)
    if st.button("I Agree"):
        st.session_state['consent_given'] = True
        st.experimental_rerun()
    if st.button("I Disagree"):
        st.session_state['consent_given'] = False
        st.experimental_rerun()

# Check if consent is given
if 'consent_given' not in st.session_state:
    st.session_state['consent_given'] = None

if st.session_state['consent_given'] is None:
    ask_for_consent()
elif st.session_state['consent_given'] is False:
    show_home()
else:
    # Conditional rendering based on the selected menu option
    if selected == "Home":
        show_home()
    elif selected == "Hate Speech Detector":
        show_hate_speech_detector()
    elif selected == "Upload File":
        show_upload_file()
