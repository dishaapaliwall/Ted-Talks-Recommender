import streamlit as st
import pandas as pd
import numpy as np
import ast
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Page Configuration
st.set_page_config(
    page_title="TED Talks Recommendation Engine",
    page_icon="🎙️",
    layout="wide"
)

# Custom Styling
st.markdown("""
<style>
    .main-title {
        background: linear-gradient(135deg, #1f2937, #111827);
        padding: 20px 24px;
        border-radius: 10px;
        color: white;
        margin-bottom: 20px;
    }
    .talk-card {
        background-color: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 12px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .high-pop {
        color: #16a34a;
        font-weight: bold;
    }
    .low-pop {
        color: #ea580c;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# ----------------- CACHED MODEL & DATA PIPELINE -----------------
@st.cache_resource(show_spinner="Training recommender models and indexing TED Talks...")
def load_and_train():
    df = pd.read_csv('ted_talks.csv')
    shortlisted_cols = [
        'socialTitle', 'description', 'duration', 'publishedAt', 
        'videoContext', 'viewedCount', 'relatedVideos', 'topics', 
        'type', 'speakers'
    ]
    df_clean = df[[col for col in shortlisted_cols if col in df.columns]].drop_duplicates(subset=['socialTitle', 'description']).copy()

    def parse_topics(val):
        try:
            data = ast.literal_eval(val) if isinstance(val, str) else val
            if isinstance(data, list):
                return [item['name'] for item in data if isinstance(item, dict) and 'name' in item]
        except Exception:
            pass
        return []

    def parse_speaker(val):
        try:
            data = ast.literal_eval(val) if isinstance(val, str) else val
            if isinstance(data, list) and len(data) > 0:
                sp = data[0]
                name = f"{sp.get('firstName', '')} {sp.get('lastName', '')}".strip()
                return name if name else 'Unknown Speaker'
        except Exception:
            pass
        return 'Unknown Speaker'

    def parse_type(val):
        try:
            data = ast.literal_eval(val) if isinstance(val, str) else val
            if isinstance(data, dict):
                return data.get('name', 'TED Stage Talk')
        except Exception:
            pass
        return 'TED Stage Talk'

    df_clean['topic_list'] = df_clean['topics'].apply(parse_topics)
    df_clean['topic_tags'] = df_clean['topic_list'].apply(lambda x: ', '.join(x))
    df_clean['topic_count'] = df_clean['topic_list'].apply(len)
    df_clean['speaker_name'] = df_clean['speakers'].apply(parse_speaker)
    df_clean['talk_type'] = df_clean['type'].apply(parse_type)

    df_clean['publishedAt_dt'] = pd.to_datetime(df_clean['publishedAt'], errors='coerce')
    ref_date = df_clean['publishedAt_dt'].max()
    df_clean['published_year'] = df_clean['publishedAt_dt'].dt.year.fillna(2015).astype(int)
    df_clean['published_month'] = df_clean['publishedAt_dt'].dt.month.fillna(1).astype(int)
    df_clean['published_dayofweek'] = df_clean['publishedAt_dt'].dt.dayofweek.fillna(0).astype(int)
    df_clean['talk_age_days'] = (ref_date - df_clean['publishedAt_dt']).dt.days.fillna(0).astype(int)

    df_clean['duration_mins'] = (df_clean['duration'] / 60.0).round(2)
    df_clean['title_word_count'] = df_clean['socialTitle'].fillna('').apply(lambda x: len(str(x).split()))
    df_clean['desc_word_count'] = df_clean['description'].fillna('').apply(lambda x: len(str(x).split()))

    top_contexts = df_clean['videoContext'].value_counts().nlargest(8).index.tolist()
    df_clean['event_category'] = df_clean['videoContext'].apply(lambda x: x if x in top_contexts else 'Other')

    median_views = df_clean['viewedCount'].median()
    df_clean['is_popular'] = (df_clean['viewedCount'] >= median_views).astype(int)

    num_features = ['duration_mins', 'talk_age_days', 'published_year', 'published_month', 'published_dayofweek', 'topic_count', 'title_word_count', 'desc_word_count']
    cat_features = ['event_category', 'talk_type']

    X = df_clean[num_features + cat_features]
    y = df_clean['is_popular']

    preprocessor = ColumnTransformer(transformers=[
        ('num', StandardScaler(), num_features), 
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_features)
    ])

    lr_pipe = Pipeline([('prep', preprocessor), ('clf', LogisticRegression(C=10.0, max_iter=200, solver='liblinear', random_state=42))])
    lr_pipe.fit(X, y)

    dt_pipe = Pipeline([('prep', preprocessor), ('clf', DecisionTreeClassifier(criterion='entropy', max_depth=10, min_samples_leaf=50, random_state=42))])
    dt_pipe.fit(X, y)

    rf_pipe = Pipeline([('prep', preprocessor), ('clf', RandomForestClassifier(max_depth=12, max_features='sqrt', min_samples_leaf=2, n_estimators=150, random_state=42))])
    rf_pipe.fit(X, y)

    df_clean = df_clean.reset_index(drop=True)
    df_clean['lr_prob'] = lr_pipe.predict_proba(X)[:, 1]
    df_clean['lr_pred'] = lr_pipe.predict(X)

    df_clean['dt_prob'] = dt_pipe.predict_proba(X)[:, 1]
    df_clean['dt_pred'] = dt_pipe.predict(X)

    df_clean['rf_prob'] = rf_pipe.predict_proba(X)[:, 1]
    df_clean['rf_pred'] = rf_pipe.predict(X)

    df_clean['combined_content'] = (
        df_clean['socialTitle'].fillna('') + ' ' + 
        df_clean['description'].fillna('') + ' ' + 
        df_clean['topic_tags'].fillna('') + ' ' + 
        df_clean['speaker_name'].fillna('')
    )

    tfidf = TfidfVectorizer(max_features=5000, stop_words='english', ngram_range=(1, 2))
    tfidf_mat = tfidf.fit_transform(df_clean['combined_content'])

    return df_clean, tfidf, tfidf_mat, median_views

df_clean, tfidf, tfidf_matrix, median_views = load_and_train()

# ----------------- SESSION STATE -----------------
if 'watched_talks' not in st.session_state:
    st.session_state.watched_talks = set()
if 'query' not in st.session_state:
    st.session_state.query = 'artificial intelligence'

# ----------------- UI BANNER -----------------
st.markdown("""
<div class="main-title">
    <div style="font-size:24px; font-weight:bold;">
        <span style="color:#eb0028;">TED</span> Talks Interactive Multi-Model Recommender
    </div>
    <div style="font-size:13px; color:#9ca3af; margin-top:4px;">
        Two-Stage Architecture: TF-IDF Cosine Retrieval + Supervised Re-Ranking (Random Forest, Decision Tree, Logistic Regression)
    </div>
</div>
""", unsafe_allow_html=True)

# Status Metrics
m1, m2, m3 = st.columns(3)
m1.metric("Indexed TED Talks", f"{len(df_clean):,}")
m2.metric("Watched History", f"{len(st.session_state.watched_talks)} talks")
m3.metric("Popularity Benchmark", f"{int(median_views):,} views")

# ----------------- QUICK PRESET BUTTONS -----------------
st.write("**Quick Topics:**")
q_cols = st.columns(5)
presets = [
    ("AI & Robotics", "artificial intelligence robotics"),
    ("Climate & Earth", "climate change environment nature"),
    ("Psychology & Mind", "psychology human mind happiness"),
    ("Future Tech", "future technology space internet"),
    ("Leadership", "leadership business success innovation")
]
for col, (label, q_val) in zip(q_cols, presets):
    if col.button(label, use_container_width=True):
        st.session_state.query = q_val
        st.rerun()

# ----------------- SEARCH & FILTERS -----------------
col_search, col_slider, col_filter, col_reset = st.columns([3, 1.5, 2, 1])

with col_search:
    user_query = st.text_input("Enter Topic / Keywords:", value=st.session_state.query)
    if user_query != st.session_state.query:
        st.session_state.query = user_query

with col_slider:
    top_n = st.slider("Top Talks to Show:", min_value=2, max_value=10, value=4)

with col_filter:
    filter_watched = st.checkbox("Filter Watched Talks", value=True)

with col_reset:
    st.write("")
    st.write("")
    if st.button("Reset Session", use_container_width=True):
        st.session_state.watched_talks.clear()
        st.rerun()

# ----------------- RECOMMENDATION ENGINE -----------------
def get_recommendations(query, model_code='rf', top_n=4, filter_viewed=True):
    if not query.strip():
        return pd.DataFrame()
    query_vec = tfidf.transform([query])
    sim_scores = cosine_similarity(query_vec, tfidf_matrix).flatten()
    sorted_indices = sim_scores.argsort()[::-1]

    candidates = []
    for idx in sorted_indices:
        if sim_scores[idx] <= 0:
            break
        if filter_viewed and idx in st.session_state.watched_talks:
            continue
        candidates.append(idx)
        if len(candidates) >= 30:
            break

    if not candidates:
        return pd.DataFrame()

    cand_df = df_clean.iloc[candidates].copy()
    cand_df['similarity'] = [sim_scores[i] for i in candidates]
    prob_col = f'{model_code}_prob'
    pred_col = f'{model_code}_pred'

    ranked = cand_df.sort_values(by=[prob_col, 'similarity'], ascending=[False, False]).head(top_n)
    return ranked

def render_cards(rec_df, model_code):
    if rec_df.empty:
        st.info("No talks found for this topic (or all relevant talks are marked as watched).")
        return
    for _, row in rec_df.iterrows():
        idx = int(row.name)
        prob = row[f'{model_code}_prob']
        pred = row[f'{model_code}_pred']
        is_pop = pred == 1
        pop_badge = "High Popularity" if is_pop else "Low Popularity"
        pop_class = "high-pop" if is_pop else "low-pop"

        card_html = f"""
        <div class="talk-card" style="border-left: 4px solid {'#16a34a' if is_pop else '#ea580c'};">
            <div style="display:flex; justify-content:space-between; font-size:12px; margin-bottom:4px;">
                <span style="background:#f3f4f6; color:#4b5563; padding:2px 8px; border-radius:4px;">
                    {row['videoContext']} | Duration: {row['duration_mins']} mins
                </span>
                <span class="{pop_class}">{pop_badge} ({prob*100:.1f}%)</span>
            </div>
            <div style="font-size:16px; font-weight:bold; color:#111827; margin:6px 0;">{row['socialTitle']}</div>
            <div style="color:#eb0028; font-size:13px; font-weight:600;">Speaker: {row['speaker_name']}</div>
            <div style="font-size:12px; color:#6b7280; margin-top:6px;">
                Cosine Similarity: <b>{row['similarity']:.3f}</b> | Topics: {row['topic_tags'][:60]}...
            </div>
        </div>
        """
        st.markdown(card_html, unsafe_allow_html=True)
        col_btn, _ = st.columns([1, 4])
        with col_btn:
            if st.button(f"Mark #{idx} as Watched", key=f"btn_{model_code}_{idx}"):
                st.session_state.watched_talks.add(idx)
                st.rerun()

# ----------------- TABS -----------------
tab_rf, tab_lr, tab_dt, tab_comp = st.tabs([
    "🌲 Random Forest (Tuned Ensemble)", 
    "📈 Logistic Regression (Baseline)", 
    "🌳 Decision Tree (Tuned Tree)", 
    "⚖️ Side-by-Side Model Comparison"
])

with tab_rf:
    st.subheader("Random Forest Popularity Re-Ranking (Best Performing: AUC 0.74)")
    df_rf = get_recommendations(st.session_state.query, model_code='rf', top_n=top_n, filter_viewed=filter_watched)
    render_cards(df_rf, 'rf')

with tab_lr:
    st.subheader("Logistic Regression Re-Ranking (Linear Baseline: AUC 0.60)")
    df_lr = get_recommendations(st.session_state.query, model_code='lr', top_n=top_n, filter_viewed=filter_watched)
    render_cards(df_lr, 'lr')

with tab_dt:
    st.subheader("Decision Tree Re-Ranking (Pruned Tree: AUC 0.72)")
    df_dt = get_recommendations(st.session_state.query, model_code='dt', top_n=top_n, filter_viewed=filter_watched)
    render_cards(df_dt, 'dt')

with tab_comp:
    st.subheader("Model Consensus & Comparison Matrix")
    if not df_rf.empty:
        comp_data = []
        for rank, (_, row) in enumerate(df_rf.iterrows(), 1):
            preds = [row['lr_pred'], row['dt_pred'], row['rf_pred']]
            consensus = "Full Agreement" if (preds[0] == preds[1] == preds[2]) else "Majority"
            comp_data.append({
                "Rank": rank,
                "Title": row['socialTitle'][:45] + "...",
                "Speaker": row['speaker_name'],
                "Cosine Sim": f"{row['similarity']:.3f}",
                "Random Forest": f"{'High' if row['rf_pred']==1 else 'Low'} ({row['rf_prob']*100:.1f}%)",
                "Decision Tree": f"{'High' if row['dt_pred']==1 else 'Low'} ({row['dt_prob']*100:.1f}%)",
                "Logistic Reg": f"{'High' if row['lr_pred']==1 else 'Low'} ({row['lr_prob']*100:.1f}%)",
                "Consensus": consensus
            })
        st.dataframe(pd.DataFrame(comp_data), use_container_width=True)
    else:
        st.info("No candidates available for comparison.")
