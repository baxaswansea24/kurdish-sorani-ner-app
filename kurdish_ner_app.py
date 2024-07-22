import streamlit as st

# This should be the first Streamlit command
st.set_page_config(layout="wide")

import streamlit as st
import joblib
import nltk
from nltk.tokenize import word_tokenize
import sklearn_crfsuite

# Download necessary NLTK data
nltk.download('punkt', quiet=True)

# Load the trained model
@st.cache_resource
def load_model():
    return joblib.load('kurdish_sorani_ner_model.joblib')

model = load_model()

def word2features(sent, i):
    word = sent[i]
    
    features = {
        'bias': 1.0,
        'word.lower()': word.lower(),
        'word[-3:]': word[-3:],
        'word[-2:]': word[-2:],
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),
        'word.is_kurdish_char': any(ord(char) in range(0x0600, 0x06FF) for char in word),
        'word.has_digit': any(char.isdigit() for char in word),
        'word.prefix-1': word[:1],
        'word.prefix-2': word[:2],
        'word.suffix-1': word[-1:],
        'word.suffix-2': word[-2:],
    }
    
    for offset in [-2, -1, 1, 2]:
        if i + offset >= 0 and i + offset < len(sent):
            features[f'word.{offset}'] = sent[i + offset]
            features[f'word.{offset}.lower()'] = sent[i + offset].lower()
    
    return features

def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]

def predict_entities(text):
    tokens = word_tokenize(text)
    features = sent2features(tokens)
    predictions = model.predict([features])[0]
    return list(zip(tokens, predictions))

# Streamlit app
st.set_page_config(layout="wide")
st.title("Kurdish Sorani Named Entity Recognition")

text_input = st.text_area("Enter Kurdish Sorani text:", height=150)

if st.button("Perform NER"):
    if text_input:
        results = predict_entities(text_input)
        
        st.subheader("Results:")
        html_output = ""
        for token, label in results:
            if label != 'O':
                color = {
                    'B-PERSON': 'lightcoral',
                    'I-PERSON': 'lightcoral',
                    'B-LOCATION': 'lightblue',
                    'I-LOCATION': 'lightblue',
                    'B-ORGANIZATION': 'lightgreen',
                    'I-ORGANIZATION': 'lightgreen'
                }.get(label, 'white')
                html_output += f'<span style="background-color: {color};">{token}</span> '
            else:
                html_output += f'{token} '
        
        st.markdown(html_output, unsafe_allow_html=True)
        
        st.subheader("Entity Counts:")
        entity_counts = {}
        for _, label in results:
            if label != 'O':
                entity_type = label.split('-')[1]
                entity_counts[entity_type] = entity_counts.get(entity_type, 0) + 1
        
        for entity_type, count in entity_counts.items():
            st.write(f"{entity_type}: {count}")
    else:
        st.warning("Please enter some text to analyze.")

st.markdown("""
<style>
    body {
        direction: rtl;
    }
    .stTextArea textarea {
        direction: rtl;
    }
</style>
""", unsafe_allow_html=True)
