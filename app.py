import streamlit as st
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import nltk
import time
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk import pos_tag, ngrams
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="WebApp",page_icon="📊",layout="wide")
    
headerSection = st.container()
mainSection = st.container()
loginSection = st.container()
logOutSection = st.container()

def add_login_page_style():
    st.markdown("""
        <style>
        /* Full page lavender background */
        .stApp {
            background-color: #E6E6FA;  /* Lavender */
        }

        /* Login box styling */
        .login-box {
            background: white;
            padding: 30px;
            border-radius: 15px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        }

        </style>
    """, unsafe_allow_html=True)

def show_main_page():

    st.markdown("""
    <style>
        /* Background */
        .stApp {
            background-image: url("https://i.ibb.co/qLxcFhdt/delicious-healthy-gmo-free-fruit-copy-space.jpg");
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }

        /* Tabs: bigger, bold, more visible */
        .stTabs [role="tab"] {
            font-size: 22px !important;      
            font-weight: 700 !important;     
            color: #4B0082 !important;       
            padding: 12px 20px !important;   
            border-radius: 8px !important;   
            background-color: rgba(255, 255, 255, 0.7) !important; 
            margin-right: 5px !important;
            border-bottom: none !important;  
            box-shadow: none !important;     /* Remove any shadow */
        }

        /* Highlight active tab in green */
        .stTabs [role="tab"][aria-selected="true"] {
            background-color: #800000 !important; /* Green for active tab */
            color: white !important;
            border-bottom: none !important;      /* Remove underline */
            box-shadow: none !important;         /* Remove underline/shadow */
        }

        /* Import Google Font for cursive headings */
        @import url('https://fonts.googleapis.com/css2?family=Dancing+Script&display=swap');

        /* Headings in cursive */
        h1, h2, h3 {
            font-family: 'Dancing Script', cursive !important;
            color: #800000 !important;          
            letter-spacing: 1px;
        }

        /* Normal text bold and visible */
        p, div.stText, span {
            font-family: 'Poppins', sans-serif !important;
            font-weight: 700 !important;       
            color: #111111 !important;         
        }
    </style>
    """, unsafe_allow_html=True)
    
    home, analysis, chatbot, recommendations, about = st.tabs(
        ["🏠 Home", "🤖 Analysis", "💬 Chatbot","🛍️ recommendations", "📞 About"])

    # Load the dataset
    try:
     df = pd.read_csv("Groceries_dataset.csv")
    #st.success("🥳Congratulations your dataset load successfully !")
    except:
     st.error("😐Dataset not found")
     st.stop()

    # Load Models
    frequent_itemsets = joblib.load("frequent_itemsets.joblib")
    rules = joblib.load("rules.joblib")

    # Load Resources
    with open("corpus.pkl", "rb") as f:
        qa_corpus = pickle.load(f)
    nltk.download('punkt')
    nltk.download('wordnet')
    nltk.download('averaged_perceptron_tagger')
    stemmer = PorterStemmer()
    lemmatizer = WordNetLemmatizer()

    # Sidebar Navigation
    section = st.sidebar.radio("Select from Dataset",
                            ["Dataset Preview", "Dataset Information", "Numerical Summary"])

    if section == "Dataset Preview":
        view_option = st.sidebar.radio("To view the dataset, select show",["Hide","Show"])
        if view_option == 'Show':
            st.sidebar.subheader("✨ Dataset Preview")
            st.sidebar.dataframe(df.head())

    elif section == "Dataset Information":
        st.sidebar.subheader("🌟 Dataset Information")
        col1, col2 = st.sidebar.columns(2)
        col1.metric(label = "Number of Rows", value=df.shape[0])
        col2.metric(label = "Number of Columns", value=df.shape[1])

    elif section == "Numerical Summary":
        with st.sidebar.expander("📊 Summary of the Numerical Columns", expanded=False):
         st.write(df.describe())

    # ------------------ HOME ------------------ #
    with home:
        st.title("🥳 Welcome to the Market Basket Analysis App")
        st.write("""
        This application provides an end-to-end solution for analyzing customer purchase behavior
        using advanced association rule mining techniques.

        **Key Features**
        - Apriori Algorithm  
        - Association Rule Generation  
        - Interactive Visual Insights  
        - Integrated AI Chatbot for Queries  

        Use the navigation tabs above to explore the dataset, perform analysis, generate rules,
        and gain meaningful insights from transactional data.
        """)

    # ------------------ ANALYSIS ------------------ #
    with analysis:
        st.title("Run Apriori Algorithm")

        # User input
        min_support = st.slider("Minimum Support", 0.0, 1.0, 0.1, 0.01)
        min_confidence = st.slider("Minimum Confidence", 0.0, 1.0, 0.1, 0.01)

        if st.button("Show Results"):
            filtered_itemsets = frequent_itemsets[frequent_itemsets["support"] >= min_support]
            if len(filtered_itemsets) > 0:
        
                # Sort by support DESCENDING
                filtered_itemsets = filtered_itemsets.sort_values("support", ascending=False)
                st.subheader("Frequent Itemsets")
                st.dataframe(filtered_itemsets)

                # ---- Plot ----
                plt.figure(figsize=(12, 6))
                sns.barplot(
                    x=filtered_itemsets.index.astype(str),   # KEEP ORIGINAL INDEX
                    y=filtered_itemsets["support"],
                    palette="plasma"
                )
                plt.xlabel("Itemsets (Original Index)")
                plt.ylabel("Support")
                plt.title("Frequent Itemsets Sorted by Support")
                plt.xticks(rotation=90)
                st.pyplot(plt)
            else:
                st.warning("No frequent itemsets found for selected support.")

            # ---------------- Association Rules ----------------
            filtered_rules = rules[
                (rules["support"] >= min_support) &
                (rules["confidence"] >= min_confidence)]

            # Sort by confidence descending
            filtered_rules = filtered_rules.sort_values("confidence", ascending=False)

            st.subheader("Association Rules")
            st.dataframe(filtered_rules)

            if not filtered_rules.empty:
                plt.figure(figsize=(12, 6))
                sns.barplot(
                    x=filtered_rules.index.astype(str),   # keep original index
                    y=filtered_rules["confidence"],
                    palette="viridis"
                )
                plt.xlabel("Rule Index (Original)")
                plt.ylabel("Confidence")
                plt.title("Association Rules Sorted by Confidence")
                plt.xticks(rotation=90)
                st.pyplot(plt)
            else:
                st.warning("No rules found for selected confidence/support.")

    # ------------------ CHATBOT ------------------ #
    with chatbot:
        st.title("AI Chatbot Assistant")

        # --- TF-IDF Setup ---
        questions = [q for q, _ in qa_corpus]
        vectorizer = TfidfVectorizer(stop_words='english', analyzer='char', ngram_range=(3, 5))
        corpus_vectors = vectorizer.fit_transform(questions)

        # --- NLP Preprocessing ---
        def process_text(user_query):
            tokens = [w for w in word_tokenize(user_query.lower()) if w.isalnum()]
            stemmed = [stemmer.stem(w) for w in tokens]
            lemmatized = [lemmatizer.lemmatize(w, pos='v') for w in tokens]
            pos_tags = pos_tag(tokens)
            bigrams = list(ngrams(tokens, 2))
            trigrams = list(ngrams(tokens, 3))
            return stemmed, lemmatized, pos_tags, bigrams, trigrams

        # --- Chatbot Function ---
        def get_response(user_query):
            query_vector = vectorizer.transform([user_query])
            sims = cosine_similarity(query_vector, corpus_vectors).flatten()
            best_idx = np.argmax(sims)
            best_score = sims[best_idx]

            if best_score > 0.1:
                answer = qa_corpus[best_idx][1]
            else:
                answer = "Sorry, I don't have an answer for that. Please ask another question!"
            return answer, best_score

        # --- Chatbot Section ---
        st.subheader("😎 Smart Bot")

        # Dropdown for predefined questions
        dropdown_question = st.selectbox(
            "Select a predefined question (optional):",
            ["-- Select a question --"] + questions
        )

        # Text input for custom question
        user_question = st.text_input("Or type your own question:")

        # Reset dropdown if user starts typing
        if user_question:
            dropdown_question = "-- Select a question --"

        # Determine which question to use
        final_question = None
        if user_question.strip():
            final_question = user_question.strip()
        elif dropdown_question != "-- Select a question --":
            final_question = dropdown_question

        # Get Answer Button
        if st.button("Get Answer"):
            if final_question:
                answer, score = get_response(final_question)
                stemmed, lemmatized, pos_tags, bigrams, trigrams = process_text(final_question)
                st.success(f"Answer: {answer}")
                st.caption(f"🔍 Similarity Score: {score:.2f}")
            else:
                st.warning("Please select or type a question!")

    # ------------------ Recommendations ----------
    with recommendations:
        st.title("🛒 Product Recommendation System")

        # Step 1: collect all unique items from frequent itemsets
        all_items = sorted({item for s in frequent_itemsets["itemsets"] for item in s})

        # Step 2: User selects an item
        selected_item = st.selectbox("Select a product:", all_items)

        # Step 3: Find rules where selected item appears in antecedents
        if selected_item:
            related_rules = rules[
                rules["antecedents"].apply(lambda x: selected_item in x)]

            if not related_rules.empty:
                # sort by confidence DESC
                related_rules = related_rules.sort_values("confidence", ascending=False)
                st.write(f"📦Similar Itemsets")
                for idx, row in related_rules.iterrows():
                    consequents = ", ".join(list(row["consequents"]))
                    st.write(f"- **{consequents}** (Confidence: {row['confidence']:.2f})")

            else:
                st.warning("No recommendations found for this item.")

    # ------------------ ABOUT ------------------ #
    with about:
        st.title("About & Contact") 
        
        st.write("""
        **Market Basket Analysis App** is an interactive platform that helps businesses and analysts understand customer purchasing behavior. 
        By leveraging **Apriori**, the app provides actionable insights and product recommendations to improve sales and customer engagement.

        **Key Features:**
        - Analyze and explore transactional datasets  
        - Generate frequent itemsets and association rules  
        - Run Apriori for predictive insights  
        - Get personalized product recommendations in the 🛍️ Recommendations tab  
        - Ask questions about the data or analysis in the 💬 Chatbot tab  

        **Developer Information:**  
        - **Name:** Sanjana Ramgarhia  
        - **Location:** Chandigarh, India  
        - **Background:** Computer Science Graduate  
        """)
        st.markdown("""
        ### Contact Me  
        - 📧 [Email](mailto:sanjanaramgarhia@gmail.com)
        """)

        st.markdown("---")
        st.button("Log Out", on_click=LoggedOut_Clicked)

def LoggedOut_Clicked():
    """Logout user."""
    st.session_state["loggedIn"] = False

def LoggedIn_Clicked():
    """When login button clicked."""
    st.session_state["loggedIn"] = True
    st.balloons()
    time.sleep(2)
    st.rerun()

def show_login_page():
    st.title("🤝 Welcome to the Portal")
    with loginSection:
        st.subheader("🔐 Login Required")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")

        if st.button("Login"):
            credentials = {
                "manager":"manager123",
                "director":"director123",
                "sanjana":"123"
            }
            if username in credentials and credentials[username] == password:
                st.success(f"😊 Welcome, {username}!")
                LoggedIn_Clicked()
            else:
                st.error("❌ Invalid username or password")

with headerSection:

    add_login_page_style()

    if "loggedIn" not in st.session_state:
        st.session_state["loggedIn"] = False

    if st.session_state["loggedIn"]:
        show_main_page()
    else:
        show_login_page()
