import streamlit as st
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path
import time
import base64

# ----------------------------- PAGE CONFIG -----------------------------
st.set_page_config(
    page_title="UET Taxila AI Chatbot", 
    page_icon="🎓", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------------------- IMAGE LOADING FUNCTION -----------------------------
def get_image_base64(image_path):
    """Convert image to base64 for embedding"""
    try:
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    except:
        return None

# Load images
logo_base64 = get_image_base64("images/logo.png")
about_base64 = get_image_base64("images/download.jpg")
header_base64 = get_image_base64("images/header.jpg")
footer_base64 = get_image_base64("images/images.jpg")

# ----------------------------- CUSTOM STYLING -----------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

* {
    font-family: 'Inter', sans-serif;
}

/* MAIN BACKGROUND */
.stApp {
    background: #F8F9FA;
}

/* REMOVE DEFAULT PADDING */
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

/* HEADER SECTION */
.main-header {
    background: linear-gradient(180deg, #2c5f8d 0%, #3a7bb5 100%);
    padding: 0;
    margin: -2rem -2rem 2rem -2rem;
    border-bottom: 4px solid #d4af37;
}

.header-top {
    background: #ffffff;
    padding: 40px 40px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    border-bottom: 2px solid #e8e8e8;
}

.header-logo {
    display: flex;
    align-items: center;
    gap: 20px;
}

.header-logo img {
    height: 90px;
    width: auto;
}

.header-text {
    text-align: left;
}

.header-title {
    font-size: 26px;
    font-weight: 700;
    color: #1a4d7a;
    margin: 0;
    line-height: 1.2;
}

.header-subtitle {
    font-size: 15px;
    color: #666;
    margin: 5px 0 0 0;
    font-weight: 400;
}

.header-main {
    padding: 35px 40px;
    text-align: center;
}

.chatbot-title {
    font-size: 34px;
    font-weight: 700;
    color: #ffffff;
    margin: 0 0 12px 0;
    text-shadow: 0 2px 4px rgba(0,0,0,0.2);
}

.chatbot-desc {
    font-size: 17px;
    color: #e8f1f8;
    margin: 0;
    font-weight: 300;
}

/* CAMPUS SHOWCASE */
.campus-showcase {
    background: #ffffff;
    padding: 35px;
    margin: 25px 0;
    border-radius: 8px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.08);
}

.showcase-title {
    font-size: 24px;
    font-weight: 600;
    color: #1a4d7a;
    margin: 0 0 25px 0;
    text-align: center;
    padding-bottom: 12px;
    border-bottom: 3px solid #3a7bb5;
}

.campus-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 25px;
    margin-top: 20px;
}

.campus-image {
    border-radius: 10px;
    overflow: hidden;
    box-shadow: 0 4px 15px rgba(0,0,0,0.15);
    transition: transform 0.3s ease, box-shadow 0.3s ease;
    border: 3px solid #e8e8e8;
    background: #f5f5f5;
}

.campus-image:hover {
    transform: translateY(-8px);
    box-shadow: 0 8px 25px rgba(58, 123, 181, 0.3);
    border-color: #3a7bb5;
}

.campus-image img {
    width: 100%;
    height: 240px;
    object-fit: cover;
    display: block;
}

/* STATS SECTION */
.stats-section {
    background: linear-gradient(135deg, #2c5f8d 0%, #3a7bb5 100%);
    padding: 40px 35px;
    margin: 25px 0;
    border-radius: 8px;
    box-shadow: 0 4px 15px rgba(0,0,0,0.15);
}

.stats-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 25px;
}

.stat-item {
    background: rgba(255, 255, 255, 0.98);
    padding: 30px 25px;
    border-radius: 8px;
    text-align: center;
    transition: all 0.3s ease;
    border: 2px solid rgba(212, 175, 55, 0.4);
}

.stat-item:hover {
    transform: translateY(-5px);
    background: #ffffff;
    box-shadow: 0 8px 20px rgba(0,0,0,0.25);
}

.stat-number {
    font-size: 40px;
    font-weight: 700;
    color: #2c5f8d;
    margin: 0;
    line-height: 1;
}

.stat-label {
    font-size: 15px;
    color: #555;
    margin-top: 10px;
    font-weight: 500;
}

/* INPUT SECTION */
.input-section {
    padding-top: 45px;
    
    border-radius: 8px;
}

.section-title {
    font-size: 22px;
    font-weight: 600;
    color: #1a4d7a;
    margin: 0 0 20px 0;
    padding-bottom: 12px;
    border-bottom: 3px solid #3a7bb5;
}

input[type="text"] {
    border: 2px solid #d0d0d0 !important;
    border-radius: 6px !important;
    padding: 15px 20px !important;
    font-size: 15px !important;
    transition: all 0.3s ease !important;
    background: #fafafa !important;
    color: #333 !important;
}

input[type="text"]:focus {
    border-color: #3a7bb5 !important;
    background: #ffffff !important;
    box-shadow: 0 0 0 3px rgba(58, 123, 181, 0.15) !important;
}

/* BUTTONS */
.stButton>button {
    background: linear-gradient(135deg, #2c5f8d 0%, #3a7bb5 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 6px !important;
    padding: 12px 30px !important;
    font-size: 15px !important;
    font-weight: 600 !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 3px 8px rgba(44, 95, 141, 0.3) !important;
}

.stButton>button:hover {
    background: linear-gradient(135deg, #3a7bb5 0%, #2c5f8d 100%) !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 5px 15px rgba(44, 95, 141, 0.4) !important;
}

/* LATEST ANSWER */
.answer-card {
    background: #f0f9ff;
    padding: 25px;
    border-radius: 8px;
    margin: 20px 0;
    border-left: 5px solid #3a7bb5;
    box-shadow: 0 2px 8px rgba(58, 123, 181, 0.15);
}

.answer-header {
    font-size: 16px;
    font-weight: 600;
    color: #1a4d7a;
    margin-bottom: 12px;
    display: flex;
    align-items: center;
    gap: 8px;
}

.answer-content {
    font-size: 15px;
    color: #2c3e50;
    line-height: 1.7;
}

/* CHAT SECTION */
.chat-section {
    background: #ffffff;
    padding: 35px;
    margin: 25px 0;
    border-radius: 8px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.08);
}

.chat-container {
    max-height: 500px;
    overflow-y: auto;
    padding: 20px;
    background: #fafafa;
    border-radius: 6px;
}

.user-message {
    background: #e3f2fd;
    padding: 16px 20px;
    border-radius: 10px 10px 2px 10px;
    margin: 12px 0 12px auto;
    max-width: 70%;
    border: 1px solid #bbdefb;
    box-shadow: 0 2px 6px rgba(0,0,0,0.1);
}

.bot-message {
    background: #ffffff;
    padding: 16px 20px;
    border-radius: 10px 10px 10px 2px;
    margin: 12px auto 12px 0;
    max-width: 70%;
    border: 1px solid #e0e0e0;
    box-shadow: 0 2px 6px rgba(0,0,0,0.1);
}

.message-label {
    font-weight: 600;
    font-size: 14px;
    margin-bottom: 8px;
    display: flex;
    align-items: center;
    gap: 6px;
}

.user-message .message-label {
    color: #1565c0;
}

.bot-message .message-label {
    color: #1a4d7a;
}

.message-text {
    font-size: 14px;
    color: #2c3e50;
    line-height: 1.6;
}

/* SIDEBAR STYLING */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #2c5f8d 0%, #1a4d7a 100%) !important;
}

section[data-testid="stSidebar"] > div:first-child {
    background: linear-gradient(180deg, #2c5f8d 0%, #1a4d7a 100%) !important;
}

/* Sidebar Headers */
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    color: #ffffff !important;
    font-weight: 600 !important;
}

/* Sidebar Text and Labels */
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] div[class*="stMarkdown"] {
    # color: #ffffff !important;
}

/* Sidebar Selectbox */
section[data-testid="stSidebar"] .stSelectbox label {
    color: #ffffff !important;
    font-weight: 500 !important;
}

section[data-testid="stSidebar"] select {
    background: rgba(255, 255, 255, 0.95) !important;
    color: #1a4d7a !important;
    border: 2px solid rgba(255, 255, 255, 0.3) !important;
    font-weight: 500 !important;
}

/* Sidebar Buttons */
section[data-testid="stSidebar"] .stButton>button {
    background: rgba(255, 255, 255, 0.95) !important;
    color: #1a4d7a !important;
    border-radius: 6px !important;
    padding: 5px 7px !important;
    margin: 6px 0 !important;
    text-align: left !important;
    width: 100% !important;
    font-size: 4px !important;
    # font-weight: 500 !important;
    transition: all 0.3s ease !important;
    border: 2px solid rgba(255, 255, 255, 0.3) !important;
    box-shadow: 0 2px 6px rgba(0,0,0,0.15) !important;
            margin: 0px auto !important;
}

section[data-testid="stSidebar"] .stButton>button:hover {
    background: #ffffff !important;
    color: #2c5f8d !important;
    transform: translateX(5px) !important;
    box-shadow: 0 4px 12px rgba(0,0,0,0.25) !important;
}

/* INFO BOX */
.info-banner {
    background: linear-gradient(135deg, #fff8e1 0%, #ffecb3 100%);
    padding: 20px 25px;
    border-radius: 6px;
    border-left: 5px solid #ffa726;
    margin: 20px 0;
    font-size: 15px;
    color: #e65100;
    box-shadow: 0 2px 8px rgba(255, 167, 38, 0.2);
}

.info-banner strong {
    color: #d84315;
    font-weight: 600;
}

/* SCROLLBAR */
.chat-container::-webkit-scrollbar {
    width: 10px;
}

.chat-container::-webkit-scrollbar-track {
    background: #e0e0e0;
    border-radius: 5px;
}

.chat-container::-webkit-scrollbar-thumb {
    background: #3a7bb5;
    border-radius: 5px;
}

.chat-container::-webkit-scrollbar-thumb:hover {
    background: #2c5f8d;
}

/* FOOTER */
.footer {
    background: #ffffff;
    padding: 30px;
    margin: 25px 0 0 0;
    border-radius: 8px;
    text-align: center;
    border-top: 4px solid #3a7bb5;
    box-shadow: 0 -2px 8px rgba(0,0,0,0.05);
}

.footer-title {
    font-size: 20px;
    font-weight: 600;
    color: #1a4d7a;
    margin: 0 0 12px 0;
}

.footer-info {
    font-size: 15px;
    color: #666;
    margin: 8px 0;
}

.footer-link {
    color: #3a7bb5;
    text-decoration: none;
    font-weight: 500;
}

.footer-link:hover {
    text-decoration: underline;
    color: #2c5f8d;
}

.footer-copyright {
    font-size: 13px;
    color: #999;
    margin-top: 20px;
    padding-top: 20px;
    border-top: 1px solid #e0e0e0;
}

/* RESPONSIVE */
@media (max-width: 768px) {
    .campus-grid {
        grid-template-columns: 1fr;
    }
    .stats-grid {
        grid-template-columns: repeat(2, 1fr);
    }
    .header-logo img {
        height: 70px;
    }
    .chatbot-title {
        font-size: 26px;
    }
}
</style>
""", unsafe_allow_html=True)

# ----------------------------- HEADER -----------------------------
logo_html = f'<img src="data:image/png;base64,{logo_base64}" alt="UET Logo">' if logo_base64 else '<div style="width:90px;height:90px;background:#ddd;border-radius:50%;"></div>'

st.markdown(f"""
<div class='main-header'>
    <div class='header-top'>
        <div class='header-logo'>
            {logo_html}
            <div class='header-text'>
                <div class='header-title'>University of Engineering & Technology</div>
                <div class='header-subtitle'>Taxila, Pakistan</div>
            </div>
        </div>
    </div>
    <div class='header-main'>
        <div class='chatbot-title'>🤖 UET Taxila AI Assistant</div>
        <div class='chatbot-desc'>Your Intelligent Guide to University Information</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ----------------------------- CAMPUS SHOWCASE -----------------------------
about_img = f'data:image/png;base64,{about_base64}' if about_base64 else 'https://via.placeholder.com/400x240/2c5f8d/ffffff?text=Campus+View'
header_img = f'data:image/png;base64,{header_base64}' if header_base64 else 'https://via.placeholder.com/400x240/3a7bb5/ffffff?text=Main+Building'
footer_img = f'data:image/png;base64,{footer_base64}' if footer_base64 else 'https://via.placeholder.com/400x240/2c5f8d/ffffff?text=University+Gate'

st.markdown(f"""
<div class='campus-showcase'>
    <div class='showcase-title'>🏛️ Campus Overview</div>
    <div class='campus-grid'>
        <div class='campus-image'>
            <img src='{about_img}' alt='UET Campus'>
        </div>
        <div class='campus-image'>
            <img src='{header_img}' alt='Main Building'>
        </div>
        <div class='campus-image'>
            <img src='{footer_img}' alt='University Gate'>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ----------------------------- STATS SECTION -----------------------------
st.markdown("""
<div class='stats-section'>
    <div class='stats-grid'>
        <div class='stat-item'>
            <div class='stat-number'>50+</div>
            <div class='stat-label'>Academic Programs</div>
        </div>
        <div class='stat-item'>
            <div class='stat-number'>8000+</div>
            <div class='stat-label'>Enrolled Students</div>
        </div>
        <div class='stat-item'>
            <div class='stat-number'>300+</div>
            <div class='stat-label'>Faculty Members</div>
        </div>
        <div class='stat-item'>
            <div class='stat-number'>24/7</div>
            <div class='stat-label'>AI Support</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ----------------------------- LOAD DATA -----------------------------
try:
    data_path = Path("embeddings")
    answers = pickle.load(open(data_path/"faq_answers.pkl", "rb"))
    questions = pickle.load(open(data_path/"faq_questions.pkl", "rb"))
    intents = pickle.load(open(data_path/"faq_intents.pkl", "rb"))
    categories = pickle.load(open(data_path/"faq_categories.pkl", "rb"))

    vectorizer = TfidfVectorizer().fit(questions)
    q_vecs = vectorizer.transform(questions)
except Exception as e:
    st.error(f"⚠️ Error loading data files: {e}")
    st.stop()

manual = {
    "hello": "👋 Hello! Welcome to UET Taxila AI Assistant. How can I help you today?",
    "hi": "👋 Hi there! Feel free to ask anything about UET Taxila.",
    "how are you": "🤖 I'm functioning perfectly and ready to assist you anytime!",
    "thanks": "😊 You're welcome! Feel free to ask more questions.",
    "thank you": "😊 Happy to help! Let me know if you need anything else.",
}

# ----------------------------- SESSION STATES -----------------------------
if "history" not in st.session_state:
    st.session_state.history = []
if "user_q" not in st.session_state:
    st.session_state.user_q = ""
if "sidebar_idx" not in st.session_state:
    st.session_state.sidebar_idx = 0
if "history_count" not in st.session_state:
    st.session_state.history_count = 10
if "latest_answer" not in st.session_state:
    st.session_state.latest_answer = None
if "animate" not in st.session_state:
    st.session_state.animate = False
if "form_key" not in st.session_state:
    st.session_state.form_key = 0

# ----------------------------- ANSWER FUNCTION -----------------------------
def get_answer(q):
    q_low = q.lower().strip()
    
    for k, v in manual.items():
        if k in q_low:
            return v
    
    previous = [x[1].lower().strip() for x in st.session_state.history if x[0] == "You"]
    if q_low in previous:
        return "⚠️ You already asked this question. Please check the chat history below."
    
    vec = vectorizer.transform([q])
    sims = cosine_similarity(vec, q_vecs)[0]
    best = np.argmax(sims)
    
    if sims[best] < 0.35:
        return "❌ Sorry, I don't have an answer for that specific question. Please try rephrasing or ask about admissions, programs, scholarships, hostels, or fee structure."
    
    return answers[best]

# ----------------------------- SIDEBAR -----------------------------
st.sidebar.markdown("<h2 style='text-align: center; margin-bottom: 30px; color: #ffffff;'>FAQ Categories</h2>", unsafe_allow_html=True)

unique_cats = ["All"] + sorted(list(set(categories)))
selected_cat = st.sidebar.selectbox("🔍 Select Category", unique_cats, key="category_select")

if selected_cat == "All":
    cat_ids = list(range(len(questions)))
else:
    cat_ids = [i for i, c in enumerate(categories) if c == selected_cat]

start = st.session_state.sidebar_idx
end = start + 5
visible_q = cat_ids[start:end]

st.sidebar.markdown("<h3 style='margin: 30px 0 15px 0; color: #ffffff;'>💡 Sample Questions</h3>", unsafe_allow_html=True)

for i in visible_q:
    if st.sidebar.button(f"❓ {questions[i][:50]}...", key=f"sidebar_q_{i}"):
        st.session_state.user_q = questions[i]
        st.rerun()

# Navigation buttons
col1, col2 = st.sidebar.columns(2)
with col1:
    if start > 0:
        if st.button("⬅️", key="prev_btn"):
            st.session_state.sidebar_idx = max(0, start - 5)
            st.rerun()

with col2:
    if end < len(cat_ids):
        if st.button("Next ➡️", key="next_btn"):
            st.session_state.sidebar_idx = start + 5
            st.rerun()

st.sidebar.markdown("<div style='margin: 30px 0; border-top: 2px solid rgba(255,255,255,0.3);'></div>", unsafe_allow_html=True)

if st.sidebar.button("🗑️ Clear Chat History", key="clear_btn"):
    st.session_state.history = []
    st.session_state.latest_answer = None
    st.session_state.history_count = 10
    st.rerun()

# ----------------------------- INFO BANNER -----------------------------
st.markdown("""
<div class='info-banner'>
    <strong>💡 How to use:</strong> Type your question below or select from sample questions in the sidebar. Get instant answers about admissions, programs, scholarships, hostels, fees, and more!
</div>
""", unsafe_allow_html=True)

# ----------------------------- INPUT SECTION -----------------------------
st.markdown("<div class='input-section'>", unsafe_allow_html=True)
st.markdown("<div class='section-title'>💬 Ask Your Question</div>", unsafe_allow_html=True)

with st.form(key=f"ask_form_{st.session_state.form_key}"):
    user_q = st.text_input(
        "Question", 
        value=st.session_state.user_q,
        placeholder="e.g., What are the admission requirements for Computer Science?",
        label_visibility="collapsed"
    )
    
    col1, col2 = st.columns([1, 5])
    with col1:
        submitted = st.form_submit_button("🤖 Ask Bot")
    with col2:
        clear_input = st.form_submit_button("🔄 Clear")

st.markdown("</div>", unsafe_allow_html=True)

# ----------------------------- PROCESS QUESTION -----------------------------
if clear_input:
    st.session_state.user_q = ""
    st.session_state.form_key += 1
    st.rerun()

if submitted and user_q.strip():
    reply = get_answer(user_q.strip())
    st.session_state.latest_answer = reply
    st.session_state.history.append(("You", user_q.strip()))
    st.session_state.history.append(("Bot", reply))
    st.session_state.user_q = ""
    st.session_state.animate = True
    st.session_state.form_key += 1
    st.rerun()

# ----------------------------- LATEST ANSWER -----------------------------
if st.session_state.latest_answer:
    latest_text = st.session_state.latest_answer
    placeholder = st.empty()
    
    if st.session_state.animate:
        displayed = ""
        for char in latest_text:
            displayed += char
            placeholder.markdown(f"""
            <div class='answer-card'>
                <div class='answer-header'>🤖 AI Assistant Response</div>
                <div class='answer-content'>{displayed}</div>
            </div>
            """, unsafe_allow_html=True)
            time.sleep(0.008)
        st.session_state.animate = False
    else:
        placeholder.markdown(f"""
        <div class='answer-card'>
            <div class='answer-header'>🤖 AI Assistant Response</div>
            <div class='answer-content'>{latest_text}</div>
        </div>
        """, unsafe_allow_html=True)

# ----------------------------- CHAT HISTORY -----------------------------
if st.session_state.history:
    st.markdown("<div class='chat-section'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>💬 Conversation History</div>", unsafe_allow_html=True)
    
    st.markdown("<div class='chat-container'>", unsafe_allow_html=True)
    
    show = st.session_state.history[-st.session_state.history_count:]
    
    for sender, msg in show:
        if sender == "You":
            st.markdown(f"""
            <div class='user-message'>
                <div class='message-label'>👤 You</div>
                <div class='message-text'>{msg}</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class='bot-message'>
                <div class='message-label'>🤖 Assistant</div>
                <div class='message-text'>{msg}</div>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    if len(st.session_state.history) > st.session_state.history_count:
        if st.button("📜 Load More Messages", key="load_more_btn"):
            st.session_state.history_count += 10
            st.rerun()
    
    st.markdown("</div>", unsafe_allow_html=True)

# ----------------------------- FOOTER -----------------------------
st.markdown("""
<div class='footer'>
    <div class='footer-title'>University of Engineering & Technology, Taxila</div>
    <div class='footer-info'>
        🌐 <a href='https://www.uettaxila.edu.pk' target='_blank' class='footer-link'>www.uettaxila.edu.pk</a>
    </div>
    <div class='footer-info'>
        📧 info@uettaxila.edu.pk | ☎️ +92-51-9047000
    </div>
    <div class='footer-info'>
        📍 G.T Road, Taxila, Punjab, Pakistan
    </div>
    <div class='footer-copyright'>
        © 2026 UET Taxila | Powered by AI Technology
    </div>
</div>
""", unsafe_allow_html=True)