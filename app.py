import streamlit as st
import pandas as pd
import plotly.express as px
from textblob import TextBlob
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import re

# Page Config
st.set_page_config(page_title="Amazon Review Analysis", layout="wide")
st.title("📹 CCTV Product Review Analysis Dashboard")

# --- 1. Data Loading & Cleaning ---
@st.cache_data
def load_data():
    file_path = 'final_dataframe.xlsx'

    try:
        # Only change: read Excel directly instead of CSV
        df = pd.read_excel(file_path)
    except FileNotFoundError:
        st.error(f"File not found! Please ensure your data file is named '{file_path}' and is in the same folder.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error loading Excel file: {e}")
        return pd.DataFrame()

    # --- Clean Rating ---
    df['Rating_Num'] = df['Rating'].astype(str).apply(lambda x: float(x.split()[0]) if 'out of' in x else None)
    
    # --- Clean Date ---
    df['Date_Clean'] = pd.to_datetime(df['Date'].str.extract(r'on (.*)')[0], errors='coerce')
    
    # --- Clean Helpful ---
    def parse_helpful(x):
        if pd.isna(x): return 0
        if 'One' in str(x): return 1
        num = re.findall(r'\d+', str(x))
        return int(num[0]) if num else 0
    df['Helpful_Num'] = df['Helpful'].apply(parse_helpful)
    
    # --- Verified Status ---
    df['Is_Verified'] = df['Verified'] == 'Yes'
    
    # --- Text & Sentiment ---
    df['Full_Text'] = df['Title'].fillna('') + ' ' + df['Text'].fillna('')
    df['Sentiment_Score'] = df['Full_Text'].apply(lambda x: TextBlob(str(x)).sentiment.polarity)
    
    # --- Sentiment Label ---
    def get_sentiment_label(score):
        if score > 0.1: return 'Positive'
        elif score < -0.1: return 'Negative'
        else: return 'Neutral'
    df['Sentiment_Label'] = df['Sentiment_Score'].apply(get_sentiment_label)
    
    return df

df = load_data()

if not df.empty:
    # --- 2. Sidebar Filters ---
    st.sidebar.header("Filters")
    
    # Handle missing dates
    if df['Date_Clean'].isnull().all():
        st.warning("No valid dates found in data.")
        min_date, max_date = pd.Timestamp.now(), pd.Timestamp.now()
    else:
        min_date = df['Date_Clean'].min()
        max_date = df['Date_Clean'].max()
        
    date_range = st.sidebar.date_input("Select Date Range", [min_date, max_date])
    
    all_styles = ['All'] + sorted(list(df['Style'].dropna().unique()))
    selected_style = st.sidebar.selectbox("Select Model (Style)", all_styles)
    
    # Filter Data
    # Ensure date_range has two values (start and end)
    if isinstance(date_range, tuple) and len(date_range) == 2:
        mask = (df['Date_Clean'].dt.date >= date_range[0]) & (df['Date_Clean'].dt.date <= date_range[1])
    else:
        mask = pd.Series([True] * len(df))

    if selected_style != 'All':
        mask = mask & (df['Style'] == selected_style)
    df_filtered = df[mask]

    # --- 3. KPI Metrics ---
    st.markdown("### 📊 Executive Summary")
    col1, col2, col3, col4 = st.columns(4)
    
    total_reviews = len(df_filtered)
    avg_rating = df_filtered['Rating_Num'].mean()
    nss = ((df_filtered['Sentiment_Label'] == 'Positive').sum() - (df_filtered['Sentiment_Label'] == 'Negative').sum()) / total_reviews * 100 if total_reviews > 0 else 0
    
    col1.metric("Total Reviews", f"{total_reviews}")
    col2.metric("Average Rating", f"{avg_rating:.2f} ⭐")
    col3.metric("Net Sentiment Score", f"{nss:.1f}%")
    col4.metric("Avg Helpfulness", f"{df_filtered['Helpful_Num'].mean():.2f}")
    
    st.markdown("---")

    # --- 4. Charts Row 1 (Trend & Verified Analysis) ---
    c1, c2 = st.columns(2)
    
    with c1:
        st.subheader("📈 Rating Trend Over Time")
        if not df_filtered.empty:
            trend_data = df_filtered.set_index('Date_Clean').resample('M')['Rating_Num'].mean().reset_index()
            fig_trend = px.line(trend_data, x='Date_Clean', y='Rating_Num', markers=True, 
                                title='Monthly Average Rating')
            fig_trend.update_yaxes(range=[0, 5.5])
            st.plotly_chart(fig_trend, use_container_width=True)
        else:
            st.info("No data available for trend chart.")

    with c2:
        st.subheader("✅ Verified vs. Unverified Ratings")
        if not df_filtered.empty:
            fig_box = px.box(df_filtered, x='Is_Verified', y='Rating_Num', color='Is_Verified',
                             title='Do Verified Buyers Rate Differently?',
                             labels={'Is_Verified': 'Verified Purchase', 'Rating_Num': 'Rating'})
            st.plotly_chart(fig_box, use_container_width=True)

    # --- 5. Charts Row 2 (Models & Sentiment Dist) ---
    c3, c4 = st.columns(2)
    
    with c3:
        st.subheader("🏆 Model Leaderboard")
        if not df_filtered.empty:
            model_perf = df_filtered.groupby('Style')['Rating_Num'].agg(['mean', 'count']).reset_index()
            model_perf = model_perf[model_perf['count'] > 5].sort_values(by='mean', ascending=True)
            if not model_perf.empty:
                fig_bar = px.bar(model_perf, x='mean', y='Style', orientation='h', 
                                 color='mean', title="Top Models (Avg Rating)", color_continuous_scale='Viridis')
                fig_bar.update_layout(xaxis_title="Avg Rating", yaxis_title="Model")
                st.plotly_chart(fig_bar, use_container_width=True)
            else:
                st.info("Not enough data per model to display leaderboard.")

    with c4:
        st.subheader("📊 Sentiment Score Distribution")
        if not df_filtered.empty:
            fig_hist = px.histogram(df_filtered, x='Sentiment_Score', nbins=30, 
                                    title='Sentiment Polarity (-1 to +1)', color_discrete_sequence=['purple'])
            fig_hist.update_layout(bargap=0.1)
            st.plotly_chart(fig_hist, use_container_width=True)

    # --- 6. Word Clouds ---
    st.markdown("---")
    st.subheader("🗣️ Voice of the Customer")
    
    wc_col1, wc_col2 = st.columns(2)
    stopwords = set(STOPWORDS)
    stopwords.update(['camera', 'product', 'good', 'cctv', 'one', 'use', 'will', 'quality', 'app'])

    def plot_wordcloud(text, title, cmap):
        if not text: return None
        wc = WordCloud(width=800, height=400, background_color='white', stopwords=stopwords, colormap=cmap).generate(text)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wc, interpolation='bilinear')
        ax.axis('off')
        ax.set_title(title)
        return fig

    with wc_col1:
        pos_reviews = df_filtered[df_filtered['Sentiment_Label'] == 'Positive']['Full_Text']
        if not pos_reviews.empty:
            pos_text = ' '.join(pos_reviews)
            fig_pos = plot_wordcloud(pos_text, "✅ Positive Themes (4-5 Star)", 'summer')
            if fig_pos: st.pyplot(fig_pos)
        else:
            st.warning("No positive reviews in selection.")

    with wc_col2:
        neg_reviews = df_filtered[df_filtered['Sentiment_Label'] == 'Negative']['Full_Text']
        if not neg_reviews.empty:
            neg_text = ' '.join(neg_reviews)
            fig_neg = plot_wordcloud(neg_text, "❌ Negative Themes (1-2 Star)", 'magma')
            if fig_neg: st.pyplot(fig_neg)
        else:
            st.warning("No negative reviews in selection.")

    # --- 7. Helpful Negative Reviews ---
    st.markdown("---")
    st.subheader("⚠️ Top 'Helpful' Negative Reviews")
    neg_helpful = df_filtered[df_filtered['Rating_Num'] <= 2].sort_values(by='Helpful_Num', ascending=False).head(5)
    
    if not neg_helpful.empty:
        for i, row in neg_helpful.iterrows():
            with st.expander(f"{row['Rating_Num']}⭐: {row['Title']} (Helpful Votes: {row['Helpful_Num']})"):
                st.write(f"**Model:** {row['Style']} | **Date:** {row['Date_Clean'].date()}")
                st.error(row['Text'])
    else:
        st.info("No negative reviews found with helpful votes.")
