import streamlit as st
import pickle
import re
import nltk
import string
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# === CONFIGURATION ===
APP_TITLE = "Fake News Detector"
APP_ICON = "📰"
PRIMARY_COLOR = "#1E88E5"
SECONDARY_COLOR = "#FF5722"
BG_COLOR = "#F5F7F9"

# === SETUP ===
@st.cache_resource
def load_nltk_resources():
    """Load NLTK resources with proper caching"""
    # Explicitly download all required resources first
    try:
        for resource in ['punkt', 'stopwords', 'wordnet', 'omw-1.4']:
            nltk.download(resource, quiet=True)
        
        # Initialize stopwords only after ensuring downloads completed
        stop_words = set(stopwords.words('english'))
        
        # Initialize lemmatizer
        lemmatizer = WordNetLemmatizer()
        
        return {
            'stop_words': stop_words,
            'lemmatizer': lemmatizer
        }
    except Exception as e:
        st.error(f"Error loading NLTK resources: {str(e)}")
        # Provide fallback resources
        return {
            'stop_words': set(),
            'lemmatizer': WordNetLemmatizer()
        }

@st.cache_resource
def load_model_and_vectorizer():
    """Load model and vectorizer with proper caching"""
    try:
        with open('model.pkl', 'rb') as f:
            model = pickle.load(f)
        
        with open('vectorizer.pkl', 'rb') as f:
            vectorizer = pickle.load(f)
        
        return model, vectorizer
    except FileNotFoundError as e:
        st.error(f"Required model files not found: {e}")
        st.stop()

# === TEXT PROCESSING ===
def preprocess_text(text, nlp_resources):
    """Clean and preprocess text for prediction"""
    if not text:
        return ""
        
    # Text cleaning
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    
    # Tokenization and lemmatization
    tokens = text.split()
    
    # Use try/except to handle potential lemmatization errors
    cleaned_tokens = []
    for word in tokens:
        if word not in nlp_resources['stop_words'] and word not in string.punctuation:
            try:
                lemmatized = nlp_resources['lemmatizer'].lemmatize(word)
                cleaned_tokens.append(lemmatized)
            except Exception:
                # Skip problematic words without logging
                continue
    
    return ' '.join(cleaned_tokens)

# Functions
def clear_news():
    st.session_state["news"] = ""

# === PREDICTION FUNCTIONS ===
def get_prediction(text, model, vectorizer, nlp_resources):
    """Get prediction and confidence from the model"""
    try:
        input_clean = preprocess_text(text, nlp_resources)
        
        if not input_clean:
            return None, None, None, []
            
        input_vect = vectorizer.transform([input_clean])
        
        # Prediction
        prediction_proba = model.predict_proba(input_vect)[0]
        prediction = prediction_proba.argmax()
        confidence = prediction_proba[prediction]
        
        # Get feature importance
        feature_names = vectorizer.get_feature_names_out()
        coefficients = model.coef_[0]
        nonzero_indices = input_vect.nonzero()[1]
        
        word_importance = []
        for idx in nonzero_indices:
            word = feature_names[idx]
            weight = coefficients[idx]
            word_importance.append((word, weight))
        
        word_importance = sorted(word_importance, key=lambda x: abs(x[1]), reverse=True)
        
        return prediction, confidence, input_clean, word_importance
    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")
        return None, None, None, []

# === UI COMPONENTS ===
def render_header():
    """Render app header with styling"""
    col1, col2 = st.columns([1, 5])
    with col1:
        st.markdown(f"<h1 style='font-size:3.5rem;margin-bottom:0'>{APP_ICON}</h1>", unsafe_allow_html=True)
    with col2:
        st.markdown(
            f"""
            <h1 style='margin-bottom:0'>{APP_TITLE}</h1>
            <p style='color:#666;margin-top:0'>Analyze news articles to detect potential misinformation</p>
            """, 
            unsafe_allow_html=True
        )
    st.markdown("<hr>", unsafe_allow_html=True)

def render_sidebar(nlp_resources):
    """Render sidebar content"""
    st.sidebar.markdown("## About")
    st.sidebar.info(
        """
        This app uses machine learning to analyze news text and predict 
        whether it might be fake news or legitimate reporting.
        
        **How it works:**
        1. Enter news text in the text area
        2. Click "Analyze Text" 
        3. View the prediction and analysis
        
        **Note:** This tool provides an estimate based on text patterns and 
        is not 100% accurate. Always verify information from multiple reliable sources.
        """
    )
    
    # Add explanation in the sidebar
    st.sidebar.markdown("## How the Detector Works")
    st.sidebar.info(
        """
        This tool uses a machine learning model trained on thousands of real and fake news articles.
        
        **The detection process:**
        1. **Text Preprocessing**: The text is cleaned, lowercased, and transformed to focus on meaningful words.
        2. **Feature Extraction**: The model analyzes patterns in the text that correlate with reliable or unreliable reporting.
        3. **Classification**: Based on these patterns, the model estimates the likelihood that the text represents fake news.
        
        **Common patterns in fake news:**
        - Sensationalist language and excessive punctuation (!!!)
        - Claims about "secret" information or conspiracies
        - Vague attributions ("researchers say", "experts claim")
        - Emotional manipulation rather than factual reporting
        """
    )
            
def render_results(prediction, confidence, word_importance, nlp_resources):
    """Render prediction results with visualization"""
    if prediction is None:
        return
        
    # Create main sections using tabs instead of nested columns
    tab1, tab2 = st.tabs(["Analysis Results", "Text Statistics"])
    
    # Tab 1: Analysis Results
    with tab1:
        # Create columns for layout within results tab
        col1, col2 = st.columns([3, 2])
        
        with col1:
            # Prediction result with styled box
            if prediction == 1:
                st.markdown(
                    f"""
                    <div style="padding:1rem;border-radius:0.5rem;background-color:#E3F2FD;border-left:5px solid #2196F3">
                        <h3 style="color:#1565C0;margin:0">✅ Likely Real News</h3>
                        <p style="margin:0.5rem 0 0 0">The content appears to use language patterns consistent with legitimate news reporting.</p>
                    </div>
                    """, 
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f"""
                    <div style="padding:1rem;border-radius:0.5rem;background-color:#FFEBEE;border-left:5px solid #F44336">
                        <h3 style="color:#C62828;margin:0">❌ Potential Fake News</h3>
                        <p style="margin:0.5rem 0 0 0">The content contains language patterns often associated with misleading information.</p>
                    </div>
                    """, 
                    unsafe_allow_html=True
                )
        
        with col2:
            # Confidence gauge
            confidence_pct = confidence * 100
            fig, ax = plt.subplots(figsize=(4, 0.7))
            
            # Create confidence gauge
            ax.barh([0], [100], color='#E0E0E0', height=0.6)
            if prediction == 1:
                ax.barh([0], [confidence_pct], color='#2196F3', height=0.6)
            else:
                ax.barh([0], [confidence_pct], color='#F44336', height=0.6)
                
            # Add text
            ax.text(50, 0, f"Confidence: {confidence_pct:.1f}%", 
                    ha='center', va='center', fontsize=10, fontweight='bold', color='#333')
            
            # Remove axes
            ax.set_ylim(-0.5, 0.5)
            ax.axis('off')
            
            st.pyplot(fig)
    
    # Tab 2: Text Statistics section
    with tab2:
        if "news" in st.session_state and st.session_state["news"].strip():
            # Calculate text statistics - safer approach
            try:
                original_text = st.session_state["news"]
                original_length = len(original_text)
                original_words = len(original_text.split())
                
                # Add metrics in a better format
                st.subheader("Original Text")
                text_stats_col1, text_stats_col2 = st.columns(2)
                
                with text_stats_col1:
                    st.metric("Characters", original_length)
                with text_stats_col2:
                    st.metric("Words", original_words)
                
                # Process text for cleaned statistics
                cleaned_text = preprocess_text(original_text, nlp_resources)
                if cleaned_text:
                    processed_length = len(cleaned_text)
                    processed_words = len(cleaned_text.split())
                    
                    # Calculate noise reduction
                    if original_length > 0:
                        reduction = ((original_length - processed_length) / original_length * 100)
                        
                        # Display in columns for better formatting
                        st.subheader("After Preprocessing")
                        processed_col1, processed_col2 = st.columns(2)
                        with processed_col1:
                            st.metric("Processed Chars", processed_length)
                        with processed_col2:
                            st.metric("Processed Words", processed_words)
                        
                        st.metric("Noise Reduction", f"{reduction:.1f}%")
            except Exception as e:
                st.error("Error calculating text statistics")

# === MAIN APP ===
def main():
    # Page config
    st.set_page_config(
        page_title=APP_TITLE,
        page_icon=APP_ICON,
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Add custom CSS
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-color: {BG_COLOR};
        }}
        .stButton>button {{
            background-color: {PRIMARY_COLOR};
            color: white;
            border-radius: 4px;
            padding: 0.5rem 1rem;
            border: none;
        }}
        .stButton>button:hover {{
            background-color: #1976D2;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

    # Explicitly ensure NLTK downloads at the very start
    with st.spinner("Loading resources..."):
        # Safe loading of NLTK resources
        try:
            for resource in ['punkt', 'stopwords', 'wordnet', 'omw-1.4']:
                nltk.download(resource, quiet=True)
        except Exception:
            st.warning("Some NLTK resources could not be downloaded. The app will continue with limited functionality.")
    
    # Load resources with error handling
    try:
        nlp_resources = load_nltk_resources()
        model, vectorizer = load_model_and_vectorizer()
    except Exception as e:
        st.error(f"Error loading resources: {str(e)}")
        st.stop()
    
    # Initialize session state for text
    if "news" not in st.session_state:
        st.session_state["news"] = ""
    
    # Render header
    render_header()
    
    # Render sidebar
    render_sidebar(nlp_resources)
    
    # Main area
    st.markdown("### Enter news text for analysis")
    
    # Text area for input
    news_text = st.text_area(
        "Paste or type news content below:",
        value=st.session_state["news"],
        height=200,
        key="news",
        help="Paste the full text of the news article you want to analyze"
    )
    
    # Action buttons
    col1, col2 = st.columns([1, 1])
    
    with col1:
        if st.button("🔍 Analyze Text", type="primary"):
            if not st.session_state["news"].strip():
                st.warning("Please enter some news text first!")
            else:
                # Add a spinner during processing
                with st.spinner("Analyzing text..."):
                    # Small delay for better UX
                    time.sleep(0.5)
                    
                    # Get prediction
                    prediction, confidence, cleaned_text, word_importance = get_prediction(
                        st.session_state["news"], 
                        model, 
                        vectorizer,
                        nlp_resources
                    )
                    
                    # Show results
                    if prediction is not None:
                        render_results(prediction, confidence, word_importance, nlp_resources)
                    else:
                        st.error("Unable to analyze. Please enter more text.")
    
    with col2:
        st.button("🔄 Enter New News", on_click=clear_news)

if __name__ == "__main__":
    main()

# import streamlit as st
# import pickle
# import re
# import nltk
# import string
# import os
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import time
# from nltk.corpus import stopwords
# from nltk.stem import WordNetLemmatizer

# # === CONFIGURATION ===
# APP_TITLE = "Fake News Detector"
# APP_ICON = "📰"
# PRIMARY_COLOR = "#1E88E5"
# SECONDARY_COLOR = "#FF5722"
# BG_COLOR = "#F5F7F9"

# # === SETUP ===
# @st.cache_resource
# def load_nltk_resources():
#     """Load NLTK resources with proper caching"""
#     # Explicitly download all required resources first
#     for resource in ['punkt', 'stopwords', 'wordnet', 'omw-1.4']:
#         try:
#             nltk.download(resource, quiet=True)
#         except LookupError:
#             st.warning(f"Could not download NLTK resource: {resource}")
    
#     # Initialize stopwords only after ensuring downloads completed
#     try:
#         stop_words = set(stopwords.words('english'))
#     except LookupError:
#         st.warning("Could not load stopwords, using empty set instead")
#         stop_words = set()
    
#     # Initialize lemmatizer
#     lemmatizer = WordNetLemmatizer()
    
#     return {
#         'stop_words': stop_words,
#         'lemmatizer': lemmatizer
#     }

# @st.cache_resource
# def load_model_and_vectorizer():
#     """Load model and vectorizer with proper caching"""
#     try:
#         with open('model.pkl', 'rb') as f:
#             model = pickle.load(f)
        
#         with open('vectorizer.pkl', 'rb') as f:
#             vectorizer = pickle.load(f)
        
#         return model, vectorizer
#     except FileNotFoundError as e:
#         st.error(f"Required model files not found: {e}")
#         st.stop()

# # === TEXT PROCESSING ===
# def preprocess_text(text, nlp_resources):
#     """Clean and preprocess text for prediction"""
#     if not text:
#         return ""
        
#     # Text cleaning
#     text = text.lower()
#     text = re.sub(r'\[.*?\]', '', text)
#     text = re.sub(r'https?://\S+|www\.\S+', '', text)
#     text = re.sub(r'<.*?>+', '', text)
#     text = re.sub(r'[^a-zA-Z]', ' ', text)
    
#     # Tokenization and lemmatization
#     tokens = text.split()
    
#     # Use try/except to handle potential lemmatization errors
#     cleaned_tokens = []
#     for word in tokens:
#         if word not in nlp_resources['stop_words'] and word not in string.punctuation:
#             try:
#                 lemmatized = nlp_resources['lemmatizer'].lemmatize(word)
#                 cleaned_tokens.append(lemmatized)
#             except Exception as e:
#                 # Skip problematic words but log the issue - removed st.debug()
#                 st.warning(f"Error lemmatizing word '{word}': {str(e)}")
#                 continue
    
#     return ' '.join(cleaned_tokens)

# # Functions
# def clear_news():
#     st.session_state["news"] = ""

# # === PREDICTION FUNCTIONS ===
# def get_prediction(text, model, vectorizer, nlp_resources):
#     """Get prediction and confidence from the model"""
#     try:
#         input_clean = preprocess_text(text, nlp_resources)
        
#         if not input_clean:
#             return None, None, None, []
            
#         input_vect = vectorizer.transform([input_clean])
        
#         # Prediction
#         prediction_proba = model.predict_proba(input_vect)[0]
#         prediction = prediction_proba.argmax()
#         confidence = prediction_proba[prediction]
        
#         # Get feature importance
#         feature_names = vectorizer.get_feature_names_out()
#         coefficients = model.coef_[0]
#         nonzero_indices = input_vect.nonzero()[1]
        
#         word_importance = []
#         for idx in nonzero_indices:
#             word = feature_names[idx]
#             weight = coefficients[idx]
#             word_importance.append((word, weight))
        
#         word_importance = sorted(word_importance, key=lambda x: abs(x[1]), reverse=True)
        
#         return prediction, confidence, input_clean, word_importance
#     except Exception as e:
#         st.error(f"Error during prediction: {str(e)}")
#         return None, None, None, []

# # === UI COMPONENTS ===
# def render_header():
#     """Render app header with styling"""
#     col1, col2 = st.columns([1, 5])
#     with col1:
#         st.markdown(f"""
#         <div style="background-color:#f0f0f0;border-radius:5px;width:60px;height:60px;display:flex;align-items:center;justify-content:center">
#             <span style="font-size:35px;color:#555">📰</span>
#         </div>
#         """, unsafe_allow_html=True)
#     with col2:
#         st.markdown(
#             f"""
#             <h1 style='margin-bottom:0;color:#333;font-size:32px'>{APP_TITLE}</h1>
#             <p style='color:#666;margin-top:0'>Analyze news articles to detect potential misinformation</p>
#             """, 
#             unsafe_allow_html=True
#         )
#     st.markdown("<hr style='margin-top:0.5rem;margin-bottom:1.5rem'>", unsafe_allow_html=True)

# def render_sidebar():
#     """Render sidebar content"""
#     st.sidebar.markdown("## About")
#     st.sidebar.info(
#         """
#         This app uses machine learning to analyze news text and predict 
#         whether it might be fake news or legitimate reporting.
        
#         **How it works:**
#         1. Enter news text in the text area
#         2. Click "Analyze Text" 
#         3. View the prediction and analysis
        
#         **Note:** This tool provides an estimate based on text patterns and 
#         is not 100% accurate. Always verify information from multiple reliable sources.
#         """
#     )
    
#     # Add explanation in the sidebar
#     st.sidebar.markdown("## How the Detector Works")
#     st.sidebar.info(
#         """
#         This tool uses a machine learning model trained on thousands of real and fake news articles.
        
#         **The detection process:**
#         1. **Text Preprocessing**: The text is cleaned, lowercased, and transformed to focus on meaningful words.
#         2. **Feature Extraction**: The model analyzes patterns in the text that correlate with reliable or unreliable reporting.
#         3. **Classification**: Based on these patterns, the model estimates the likelihood that the text represents fake news.
        
#         **Common patterns in fake news:**
#         - Sensationalist language and excessive punctuation (!!!)
#         - Claims about "secret" information or conspiracies
#         - Vague attributions ("researchers say", "experts claim")
#         - Emotional manipulation rather than factual reporting
#         """
#     )

# def render_text_statistics(nlp_resources):
#     """Render text statistics in a separate component"""
#     if "news" in st.session_state and st.session_state["news"].strip():
#         # Calculate text statistics
#         original_text = st.session_state["news"]
#         original_length = len(original_text)
#         original_words = len(original_text.split())
        
#         # Process text for cleaned statistics - wrapped in try/except
#         try:
#             cleaned_text = preprocess_text(original_text, nlp_resources)
#             processed_length = len(cleaned_text)
#             processed_words = len(cleaned_text.split()) if cleaned_text else 0
            
#             # Calculate noise reduction
#             if original_length > 0:
#                 reduction = ((original_length - processed_length) / original_length * 100)
                
#             # Create the statistics display in a table format with red header
#             st.markdown("""
#             <h3 style="color:#FF0000;margin-bottom:10px">Text Statistics</h3>
#             """, unsafe_allow_html=True)
            
#             # Original Text section
#             st.markdown("<h4>Original Text</h4>", unsafe_allow_html=True)
            
#             # Create two columns for Characters and Words
#             col1, col2 = st.columns(2)
#             with col1:
#                 st.markdown("<b>Characters</b>", unsafe_allow_html=True)
#                 st.markdown(f"<p style='font-size:20px'>{original_length}</p>", unsafe_allow_html=True)
#             with col2:
#                 st.markdown("<b>Words</b>", unsafe_allow_html=True)
#                 st.markdown(f"<p style='font-size:20px'>{original_words}</p>", unsafe_allow_html=True)
            
#             # After Preprocessing section
#             st.markdown("<h4>After Preprocessing</h4>", unsafe_allow_html=True)
            
#             # Create three columns for Characters, Words, and Noise Reduction
#             col1, col2, col3 = st.columns(3)
#             with col1:
#                 st.markdown("<b>Characters</b>", unsafe_allow_html=True)
#                 st.markdown(f"<p style='font-size:20px'>{processed_length}</p>", unsafe_allow_html=True)
#             with col2:
#                 st.markdown("<b>Words</b>", unsafe_allow_html=True)
#                 st.markdown(f"<p style='font-size:20px'>{processed_words}</p>", unsafe_allow_html=True)
#             with col3:
#                 st.markdown("<b>Noise Reduction</b>", unsafe_allow_html=True)
#                 st.markdown(f"<p style='font-size:20px'>{reduction:.1f}%</p>", unsafe_allow_html=True)
                
#         except Exception as e:
#             st.error(f"Error processing text statistics: {str(e)}")

# def render_results(prediction, confidence, word_importance, nlp_resources):
#     """Render prediction results with visualization"""
#     if prediction is None:
#         return
    
#     # Create a header for the results section
#     st.markdown("## Analysis Results")
    
#     # Create columns for the entire layout - Results on left, Stats on right
#     results_col, stats_col = st.columns([2, 1])
    
#     # === RESULTS COLUMN (Left side) ===
#     with results_col:
#         # Create a light blue background container for "Likely Real News" or red for "Potential Fake News"
#         if prediction == 1:
#             st.markdown(
#                 f"""
#                 <div style="padding:1rem;border-radius:0.5rem;background-color:#E3F2FD;margin-bottom:1rem">
#                     <h3 style="color:#1565C0;margin:0">✅ Likely Real News</h3>
#                     <p style="margin:0.5rem 0 0 0">The content appears to use language patterns consistent with legitimate news reporting.</p>
#                 </div>
#                 """, 
#                 unsafe_allow_html=True
#             )
#         else:
#             st.markdown(
#                 f"""
#                 <div style="padding:1rem;border-radius:0.5rem;background-color:#FFEBEE;margin-bottom:1rem">
#                     <h3 style="color:#C62828;margin:0">❌ Potential Fake News</h3>
#                     <p style="margin:0.5rem 0 0 0">The content contains language patterns often associated with misleading information.</p>
#                 </div>
#                 """, 
#                 unsafe_allow_html=True
#             )
        
#         # Confidence gauge - slim blue bar
#         confidence_pct = confidence * 100
        
#         # Create a simple confidence bar with text
#         st.markdown(
#             f"""
#             <div style="margin-bottom:1.5rem">
#                 <div style="display:flex;align-items:center">
#                     <div style="flex-grow:1;height:30px;background-color:#E0E0E0;border-radius:4px">
#                         <div style="width:{confidence_pct}%;height:100%;background-color:#2196F3;border-radius:4px"></div>
#                     </div>
#                     <div style="margin-left:10px;font-weight:bold">Confidence: {confidence_pct:.1f}%</div>
#                 </div>
#             </div>
#             """,
#             unsafe_allow_html=True
#         )
        
#         # The "Key Influential Words" section has been removed as requested
    
#     # === STATS COLUMN (Right side) ===
#     with stats_col:
#         render_text_statistics(nlp_resources)

# # === MAIN APP ===
# def main():
#     # Page config
#     st.set_page_config(
#         page_title=APP_TITLE,
#         page_icon=APP_ICON,
#         layout="wide",
#         initial_sidebar_state="expanded"
#     )
    
#     # Add custom CSS
#     st.markdown(
#         f"""
#         <style>
#         .stApp {{
#             background-color: {BG_COLOR};
#         }}
#         .stButton>button {{
#             background-color: {PRIMARY_COLOR};
#             color: white;
#             border-radius: 4px;
#             padding: 0.5rem 1rem;
#             border: none;
#         }}
#         .stButton>button:hover {{
#             background-color: #1976D2;
#         }}
#         /* Hide the default Streamlit header and footer */
#         header {{
#             visibility: hidden;
#         }}
#         footer {{
#             visibility: hidden;
#         }}
#         /* Reduce white space */
#         .block-container {{
#             padding-top: 1rem;
#             padding-bottom: 1rem;
#         }}
#         </style>
#         """,
#         unsafe_allow_html=True
#     )

#     # Explicitly ensure NLTK downloads at the very start
#     with st.spinner("Loading NLTK resources..."):
#         # Force download these specific resources before anything else
#         nltk.download('punkt', quiet=True)
#         nltk.download('stopwords', quiet=True)
#         nltk.download('wordnet', quiet=True)
#         nltk.download('omw-1.4', quiet=True)
    
#     # Load resources
#     try:
#         nlp_resources = load_nltk_resources()
#         model, vectorizer = load_model_and_vectorizer()
#     except Exception as e:
#         st.error(f"Error loading resources: {str(e)}")
#         st.stop()
    
#     # Initialize session state for text
#     if "news" not in st.session_state:
#         st.session_state["news"] = ""
    
#     # Render header
#     render_header()
    
#     # Render sidebar (without text statistics now)
#     render_sidebar()
    
#     # Main area
#     st.markdown("### Enter news text for analysis")
    
#     # Text area for input
#     news_text = st.text_area(
#         "Paste or type news content below:",
#         value=st.session_state["news"],
#         height=200,
#         key="news",
#         help="Paste the full text of the news article you want to analyze"
#     )
    
#     # Action buttons
#     col1, col2 = st.columns([1, 5])
    
#     with col1:
#         if st.button("🔍 Analyze Text", type="primary"):
#             if not st.session_state["news"].strip():
#                 st.warning("Please enter some news text first!")
#             else:
#                 # Add a spinner during processing
#                 with st.spinner("Analyzing text..."):
#                     # Small delay for better UX
#                     time.sleep(0.5)
                    
#                     # Get prediction
#                     prediction, confidence, cleaned_text, word_importance = get_prediction(
#                         st.session_state["news"], 
#                         model, 
#                         vectorizer,
#                         nlp_resources
#                     )
                    
#                     # Show results
#                     if prediction is not None:
#                         render_results(prediction, confidence, word_importance, nlp_resources)
#                     else:
#                         st.error("Unable to analyze. Please enter more text.")
    
#     with col2:
#         st.button("🔄 Enter New News", on_click=clear_news)

# if __name__ == "__main__":
#     main()