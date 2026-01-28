import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from textblob import TextBlob
import re
import nltk
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt

# --- 0. CONFIGURATION & SETUP ---
st.set_page_config(
    page_title="CCTV Market Intelligence",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="👁️"
)

# Force Download NLTK Data Quietly
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon', quiet=True)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
    
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# --- 1. CLEAN LIGHT THEME UI (CSS) ---
st.markdown("""
<style>
    /* Global Font & Background */
    .stApp {background-color: #F8F9FA; font-family: 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;}
    
    /* Card Styling */
    .css-1r6slb0, .css-keje6w, .stDataFrame, .stPlotlyChart {
        background-color: #FFFFFF;
        border-radius: 8px;
        box-shadow: 0 1px 2px rgba(0,0,0,0.05);
        padding: 15px;
        border: 1px solid #E5E7EB;
        margin-bottom: 10px;
    }
    
    /* Metrics Styling */
    div[data-testid="stMetric"] {
        background-color: #FFFFFF;
        padding: 15px;
        border-radius: 8px;
        border-left: 5px solid #2563EB;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    /* Headers */
    h1, h2, h3 {color: #111827 !important; font-weight: 700; letter-spacing: -0.02em;}
    h4, h5 {color: #374151 !important; font-weight: 600;}
    hr {margin: 2em 0;}
    
    /* Sidebar */
    section[data-testid="stSidebar"] {background-color: #FFFFFF; border-right: 1px solid #E5E7EB;}
</style>
""", unsafe_allow_html=True)

# --- 2. DATA ENGINE ---
@st.cache_data
def load_data():
    filename = 'final_dataframe.xlsx'
    try:
        df = pd.read_excel(filename, engine='openpyxl')
    except:
        try:
            df = pd.read_csv('final_dataframe.xlsx - Sheet1.csv', encoding='utf-8')
        except:
            st.error("❌ Data file not found. Please ensure 'final_dataframe.xlsx' is in the folder.")
            return pd.DataFrame()

    # Cleaning
    df['Rating_Num'] = df['Rating'].astype(str).apply(
        lambda x: float(x.split()[0]) if 'out of' in x else float(x) if x.replace('.','',1).isdigit() else None
    )
    df['Date_Clean'] = pd.to_datetime(df['Date'].str.extract(r'on (.*)')[0], errors='coerce')
    df['Full_Text'] = df['Title'].fillna('') + ' ' + df['Text'].fillna('')
    df['Full_Text'] = df['Full_Text'].str.lower()
    
    # Company Logic
    def get_company(style):
        s = str(style).upper()
        if 'CP' in s: return 'CP Plus'
        if 'EZ' in s: return 'EZVIZ'
        if 'TP' in s: return 'TP-Link'
        if 'MI' in s: return 'Xiaomi'
        return 'Other'
    df['Company'] = df['Style'].apply(get_company)
    
    # VADER Sentiment
    sid = SentimentIntensityAnalyzer()
    df['Vader_Score'] = df['Full_Text'].apply(lambda x: sid.polarity_scores(str(x))['compound'])
    
    # Label
    def get_label(s):
        if s >= 0.05: return 'Positive'
        if s <= -0.05: return 'Negative'
        return 'Neutral'
    df['Sentiment_Label'] = df['Vader_Score'].apply(get_label)
    
    return df

df = load_data()
if df.empty: st.stop()

# --- 3. SIDEBAR CONTROLS ---
st.sidebar.title("🎛️ Analytics Controls")

# Toggles
view_mode = st.sidebar.radio("Analysis Scope:", ["Market Overview", "Brand Deep Dive", "Model Inspector"])

df_filtered = df.copy()

if view_mode == "Brand Deep Dive":
    brand = st.sidebar.selectbox("Select Brand:", sorted(df['Company'].unique()))
    df_filtered = df[df['Company'] == brand]
elif view_mode == "Model Inspector":
    brand = st.sidebar.selectbox("Filter Brand:", sorted(df['Company'].unique()))
    models = sorted(df[df['Company'] == brand]['Style'].dropna().unique())
    model = st.sidebar.selectbox("Select Model:", models)
    df_filtered = df[df['Style'] == model]

# Date Filter
dates = st.sidebar.date_input("Date Range", [df['Date_Clean'].min(), df['Date_Clean'].max()])
if len(dates) == 2:
    df_filtered = df_filtered[(df_filtered['Date_Clean'].dt.date >= dates[0]) & (df_filtered['Date_Clean'].dt.date <= dates[1])]

# --- 4. DASHBOARD HEADER & KPI ---
st.title("👁️ CCTV Strategic Command Center")
st.caption(f"**Scope:** {view_mode} | **Data Points:** {len(df_filtered)}")

k1, k2, k3, k4 = st.columns(4)
vol = len(df_filtered)
rating = df_filtered['Rating_Num'].mean()
nss = df_filtered['Vader_Score'].mean() * 100
crit = len(df_filtered[df_filtered['Rating_Num']<=2])

k1.metric("Total Reviews", f"{vol:,}")
k2.metric("Avg Rating", f"{rating:.2f} ⭐")
k3.metric("Net Sentiment Score", f"{nss:.1f}", help="Target > 20")
k4.metric("Critical Complaints", f"{crit}", help="1-2 Star Reviews")

st.markdown("---")

# =========================================================
# SECTION 1: STRATEGY & OPPORTUNITY
# =========================================================
st.header("⚔️ Strategy: Where to Win")

c1, c2 = st.columns([2, 1])

with c1:
    st.subheader("1. The Quadrant Attack Map")
    st.markdown("**Action:** :red_circle: **Attack** the Top-Left (High Hate/High Vol). :large_green_circle: **Copy** the Top-Right.")
    
    features = {
        'Connectivity': ['wifi', 'connect', 'offline', '5g', 'signal'],
        'App Quality': ['app', 'login', 'ui', 'interface', 'slow'],
        'Video Quality': ['picture', 'video', 'hd', 'resolution', 'clear'],
        'Night Vision': ['night', 'dark', 'ir', 'black'],
        'Motion Detect': ['motion', 'alert', 'notification'],
        'Audio': ['sound', 'mic', 'audio', 'talk'],
        'Reliability': ['stopped', 'dead', 'warranty', 'service'],
        'Installation': ['install', 'setup', 'mount']
    }
    
    matrix = []
    for f, k in features.items():
        sub = df_filtered[df_filtered['Full_Text'].apply(lambda x: any(kw in x for kw in k))]
        if len(sub) > 0:
            matrix.append({'Feature': f, 'Volume': len(sub), 'Sentiment': sub['Vader_Score'].mean()})
            
    df_matrix = pd.DataFrame(matrix)
    if not df_matrix.empty:
        fig_map = px.scatter(df_matrix, x='Sentiment', y='Volume', 
                             size='Volume', color='Sentiment', text='Feature',
                             color_continuous_scale='RdYlGn', range_x=[-0.6, 0.8], size_max=60)
        # Quadrant Backgrounds
        fig_map.add_shape(type="rect", x0=-0.6, y0=df_matrix['Volume'].mean(), x1=0, y1=df_matrix['Volume'].max()*1.1,
                          fillcolor="red", opacity=0.1, layer="below", line_width=0)
        fig_map.add_shape(type="rect", x0=0, y0=df_matrix['Volume'].mean(), x1=0.8, y1=df_matrix['Volume'].max()*1.1,
                          fillcolor="green", opacity=0.1, layer="below", line_width=0)
        
        fig_map.update_traces(textposition='top center')
        fig_map.update_layout(height=400, margin=dict(t=20, b=20, l=20, r=20))
        st.plotly_chart(fig_map, use_container_width=True)

with c2:
    st.subheader("2. Best Received Specs")
    st.caption("Specs correlated with 5-Star Reviews")
    
    pos_df = df_filtered[df_filtered['Rating_Num'] == 5]
    specs = {
        '3MP Resolution': pos_df['Full_Text'].str.contains('3mp').sum(),
        '4MP/5MP Res': pos_df['Full_Text'].str.contains('4mp|5mp').sum(),
        '360 Rotation': pos_df['Full_Text'].str.contains('360|rotate').sum(),
        'Color Night Vision': pos_df['Full_Text'].str.contains('color night').sum(),
        'Human Detect': pos_df['Full_Text'].str.contains('human|person').sum()
    }
    df_specs = pd.DataFrame(list(specs.items()), columns=['Spec', 'Mentions']).sort_values('Mentions', ascending=True)
    fig_specs = px.bar(df_specs, x='Mentions', y='Spec', orientation='h', color='Mentions', color_continuous_scale='Teal')
    fig_specs.update_layout(height=400, margin=dict(t=20, b=20))
    st.plotly_chart(fig_specs, use_container_width=True)

# =========================================================
# SECTION 2: ENGINEERING & RELIABILITY
# =========================================================
st.markdown("---")
st.header("🛠️ Engineering: What to Fix")

c3, c4 = st.columns(2)

with c3:
    st.subheader("3. Engineering Debt Matrix")
    st.caption("Top Bugs by Severity (Volume × Negativity)")
    
    bugs = {
        'Offline Randomly': r'offline|disconnect',
        'No 5G Support': r'5g|2.4',
        'App Crash/Lag': r'crash|slow|hang',
        'SD Card Fail': r'sd card|memory|format',
        'QR/Pairing Fail': r'scan|qr|code'
    }
    debt = []
    for bug, pattern in bugs.items():
        matches = df_filtered[df_filtered['Full_Text'].str.contains(pattern)]
        count = len(matches)
        if count > 0:
            avg_r = matches['Rating_Num'].mean()
            severity = count * (5 - (avg_r if pd.notnull(avg_r) else 3))
            debt.append({'Issue': bug, 'Volume': count, 'Avg Rating': avg_r, 'Severity': severity})
    
    df_debt = pd.DataFrame(debt).sort_values('Severity', ascending=False)
    fig_debt = px.bar(df_debt, x='Severity', y='Issue', orientation='h', color='Severity', color_continuous_scale='Reds')
    st.plotly_chart(fig_debt, use_container_width=True)

with c4:
    st.subheader("4. The 'App' Problem")
    st.caption("Specific Connectivity Pain Points")
    conn_issues = {
        'Goes Offline': df_filtered['Full_Text'].str.contains('offline').sum(),
        'Login Failed': df_filtered['Full_Text'].str.contains('login|sign in').sum(),
        'WiFi Range': df_filtered['Full_Text'].str.contains('range|signal|weak').sum(),
        'Cloud Cost': df_filtered['Full_Text'].str.contains('cloud|subscription').sum()
    }
    df_conn = pd.DataFrame(list(conn_issues.items()), columns=['Issue', 'Count']).sort_values('Count', ascending=True)
    fig_conn = px.bar(df_conn, x='Count', y='Issue', orientation='h', color_discrete_sequence=['#9333EA'])
    st.plotly_chart(fig_conn, use_container_width=True)

# Death Timeline (Full Width)
st.subheader("5. The Mortality Curve (Strict Logic)")
st.caption("Counts only explicit hardware failure reports (e.g., 'died in 2 months').")

def extract_fail_strict(text):
    match = re.search(r'(stopped working|died|failed|broke) (after|in|within) (\d+) (month|day|week)', str(text))
    if match:
        num = int(match.group(3))
        unit = match.group(4)
        if 'month' in unit: return num * 30
        if 'week' in unit: return num * 7
        return num
    return None

bad_reviews = df_filtered[df_filtered['Rating_Num'] <= 2].copy()
bad_reviews['Fail_Day'] = bad_reviews['Full_Text'].apply(extract_fail_strict)
fail_data = bad_reviews['Fail_Day'].dropna()

if not fail_data.empty:
    fig_death = px.histogram(fail_data, nbins=30, labels={'value': 'Days after purchase'}, 
                             color_discrete_sequence=['#DC2626'])
    median_fail = fail_data.median()
    fig_death.add_vline(x=median_fail, line_dash="dash", annotation_text=f"Median: {int(median_fail)} Days")
    st.plotly_chart(fig_death, use_container_width=True)
else:
    st.info("No significant hardware failure patterns found in current selection.")

# =========================================================
# SECTION 3: GROWTH & SENTIMENT
# =========================================================
st.markdown("---")
st.header("🚀 Growth: How to Sell")

c5, c6 = st.columns(2)

with c5:
    st.subheader("6. Purchase Drivers")
    st.caption("Why customers buy (Marketing Hooks)")
    drivers = {
        'Easy Installation': pos_df['Full_Text'].str.contains('easy|install').sum(),
        'Value for Money': pos_df['Full_Text'].str.contains('price|value').sum(),
        'Picture Quality': pos_df['Full_Text'].str.contains('picture|quality|clear').sum(),
        'Remote View': pos_df['Full_Text'].str.contains('remote|view').sum()
    }
    df_drivers = pd.DataFrame(list(drivers.items()), columns=['Driver', 'Mentions']).sort_values('Mentions', ascending=True)
    fig_drivers = px.bar(df_drivers, x='Mentions', y='Driver', orientation='h', color_discrete_sequence=['#10B981'])
    st.plotly_chart(fig_drivers, use_container_width=True)

with c6:
    st.subheader("7. Sentiment Trend")
    trend = df_filtered.set_index('Date_Clean').resample('M')['Vader_Score'].mean().reset_index()
    fig_trend = px.line(trend, x='Date_Clean', y='Vader_Score', markers=True, color_discrete_sequence=['#2563EB'])
    fig_trend.add_hline(y=0, line_dash="dash", line_color="red")
    st.plotly_chart(fig_trend, use_container_width=True)

# Word Clouds
st.subheader("8. Voice of Customer")
wc1, wc2 = st.columns(2)
stop_words = set(STOPWORDS).union(['camera', 'product', 'good', 'bad', 'cctv', 'use'])

def plot_cloud(text, cmap):
    if not text: return None
    wc = WordCloud(width=600, height=300, background_color='white', stopwords=stop_words, colormap=cmap).generate(text)
    fig, ax = plt.subplots(); ax.imshow(wc); ax.axis('off'); return fig

with wc1:
    st.markdown("**Positive Themes**")
    pos_text = ' '.join(pos_df['Full_Text'])
    if pos_text: st.pyplot(plot_cloud(pos_text, 'summer'))

with wc2:
    st.markdown("**Negative Themes**")
    neg_text = ' '.join(df_filtered[df_filtered['Rating_Num'] <= 2]['Full_Text'])
    if neg_text: st.pyplot(plot_cloud(neg_text, 'magma'))
