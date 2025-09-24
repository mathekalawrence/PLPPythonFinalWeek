"""
CORD-19 Data Explorer Streamlit App
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from data_analysis1 import CORDAnalyzer
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="CORD-19 Data Explorer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.5rem;
        color: #2e86ab;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

def main():
    # Header
    st.markdown('<h1 class="main-header">📚 CORD-19 Research Data Explorer</h1>', 
                unsafe_allow_html=True)
    
    st.write("""
    This application provides an interactive exploration of the CORD-19 dataset, 
    which contains metadata about COVID-19 research papers. Analyze publication trends, 
    top journals, and research patterns.
    """)
    
    # Initializing the analyzer
    analyzer = CORDAnalyzer('data/metadata.csv')
    
    # Sidebar
    st.sidebar.title("Navigation")
    section = st.sidebar.radio(
        "Go to:",
        ["Dataset Overview", "Publication Trends", "Journal Analysis", 
         "Word Frequency", "Data Samples", "About"]
    )
    
    # Loading data with caching
    @st.cache_data
    def load_and_analyze_data():
        if analyzer.load_data():
            analyzer.clean_data()
            return analyzer.cleaned_df, analyzer
        return None, None
    
    df, analyzer_obj = load_and_analyze_data()
    
    if df is None:
        st.error(" Could not load the dataset. Please check if 'data/metadata.csv' exists.")
        return
    
    # Section 1: Dataset Overview
    if section == "Dataset Overview":
        st.markdown('<h2 class="section-header">📋 Dataset Overview</h2>', 
                    unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Papers", len(df))
        
        with col2:
            st.metric("Columns", len(df.columns))
        
        with col3:
            years = df['publication_year'].unique()
            st.metric("Publication Years", len(years))
        
        # Data summary
        st.subheader("Data Summary")
        st.write(f"**Date Range:** {df['publish_time'].min().strftime('%Y-%m-%d')} to "
                f"{df['publish_time'].max().strftime('%Y-%m-%d')}")
        
        # Displaying sample data
        st.subheader("Sample Data")
        st.dataframe(df.head(10), use_container_width=True)
        
        # Missing data visualization
        st.subheader("Data Completeness")
        missing_data = df.isnull().sum()
        missing_percent = (missing_data / len(df)) * 100
        
        fig, ax = plt.subplots(figsize=(10, 6))
        missing_percent[missing_percent > 0].plot(kind='bar', ax=ax, color='salmon')
        ax.set_title('Missing Data Percentage by Column')
        ax.set_ylabel('Percentage Missing')
        ax.tick_params(axis='x', rotation=45)
        st.pyplot(fig)
    
    # Section 2: Publication Trends
    elif section == "Publication Trends":
        st.markdown('<h2 class="section-header">📈 Publication Trends</h2>', 
                    unsafe_allow_html=True)
        
        # Yearly publications
        yearly_counts = df['publication_year'].value_counts().sort_index()
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Publications by Year")
            fig, ax = plt.subplots(figsize=(10, 6))
            yearly_counts.plot(kind='bar', ax=ax, color='lightblue')
            ax.set_xlabel('Year')
            ax.set_ylabel('Number of Publications')
            ax.grid(axis='y', alpha=0.3)
            st.pyplot(fig)
        
        with col2:
            st.subheader("Statistics")
            st.write(f"**Peak Year:** {yearly_counts.idxmax()}")
            st.write(f"**Peak Publications:** {yearly_counts.max()}")
            st.write(f"**Total Publications:** {yearly_counts.sum()}")
        
        # Monthly trends
        st.subheader("Monthly Publication Trends")
        df['publication_month'] = df['publish_time'].dt.to_period('M')
        monthly_counts = df['publication_month'].value_counts().sort_index()
        
        fig, ax = plt.subplots(figsize=(12, 6))
        monthly_counts.plot(ax=ax, color='darkblue', linewidth=2)
        ax.set_title('Monthly Publication Trends')
        ax.set_xlabel('Month')
        ax.set_ylabel('Number of Publications')
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
    
    # Section 3: Journal Analysis
    elif section == "Journal Analysis":
        st.markdown('<h2 class="section-header">🏥 Journal Analysis</h2>', 
                    unsafe_allow_html=True)
        
        top_n = st.slider("Number of top journals to show:", 5, 20, 10)
        
        journal_counts = df['journal'].value_counts().head(top_n)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader(f"Top {top_n} Journals")
            fig, ax = plt.subplots(figsize=(10, 8))
            journal_counts.plot(kind='barh', ax=ax, color='lightgreen')
            ax.set_xlabel('Number of Papers')
            ax.invert_yaxis()
            ax.grid(axis='x', alpha=0.3)
            st.pyplot(fig)
        
        with col2:
            st.subheader("Journal Statistics")
            st.write(f"**Total Unique Journals:** {df['journal'].nunique()}")
            st.write(f"**Top Journal:** {journal_counts.index[0]}")
            st.write(f"**Papers in Top Journal:** {journal_counts.iloc[0]}")
    
    # Section 4: Word Frequency
    elif section == "Word Frequency":
        st.markdown('<h2 class="section-header">🔤 Word Frequency Analysis</h2>', 
                    unsafe_allow_html=True)
        
        # Simple word frequency from titles
        st.subheader("Common Words in Paper Titles")
        
        # Get word frequencies
        all_titles = ' '.join(df['title'].dropna().astype(str))
        words = all_titles.lower().split()
        word_freq = pd.Series(words).value_counts().head(20)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        word_freq.plot(kind='barh', ax=ax, color='orange')
        ax.set_xlabel('Frequency')
        ax.invert_yaxis()
        ax.grid(axis='x', alpha=0.3)
        st.pyplot(fig)
        
        # Abstracting length distribution
        st.subheader("Abstract Length Distribution")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(df['abstract_word_count'], bins=50, color='purple', alpha=0.7)
        ax.set_xlabel('Word Count')
        ax.set_ylabel('Number of Papers')
        ax.grid(alpha=0.3)
        st.pyplot(fig)
        
        st.write(f"**Average abstract length:** {df['abstract_word_count'].mean():.1f} words")
        st.write(f"**Longest abstract:** {df['abstract_word_count'].max()} words")
    
    # Section 5: Data Samples
    elif section == "Data Samples":
        st.markdown('<h2 class="section-header">🔍 Data Exploration</h2>', 
                    unsafe_allow_html=True)
        
        # Filtering options
        col1, col2 = st.columns(2)
        
        with col1:
            year_filter = st.selectbox(
                "Filter by year:",
                options=['All'] + sorted(df['publication_year'].unique().tolist())
            )
        
        with col2:
            journal_filter = st.selectbox(
                "Filter by journal:",
                options=['All'] + df['journal'].value_counts().head(20).index.tolist()
            )
        
        # Applying filters
        filtered_df = df.copy()
        if year_filter != 'All':
            filtered_df = filtered_df[filtered_df['publication_year'] == year_filter]
        if journal_filter != 'All':
            filtered_df = filtered_df[filtered_df['journal'] == journal_filter]
        
        st.write(f"**Filtered results:** {len(filtered_df)} papers")
        
        # Displaying filtered results
        for idx, row in filtered_df.head(10).iterrows():
            with st.expander(f"{row['title']}"):
                st.write(f"**Journal:** {row['journal']}")
                st.write(f"**Publication Date:** {row['publish_time'].strftime('%Y-%m-%d')}")
                st.write(f"**Abstract:** {row['abstract'][:500]}...")
    
    # Section 6: About
    else:
        st.markdown('<h2 class="section-header">ℹ️ About This Project</h2>', 
                    unsafe_allow_html=True)
        
        st.write("""
        ### CORD-19 Data Explorer
        
        This interactive application explores the CORD-19 dataset, which contains metadata 
        about COVID-19 research papers from various sources.
        
        **Dataset Source:** [Allen Institute for AI](https://www.kaggle.com/allen-institute-for-ai/CORD-19-research-challenge)
        
        **Features:**
        -  Publication trends analysis
        -  Journal productivity rankings
        -  Word frequency analysis
        -  Interactive data filtering
        
        **Technologies Used:**
        - Python
        - Pandas for data manipulation
        - Matplotlib/Seaborn for visualization
        - Streamlit for web application
        
        """)

if __name__ == "__main__":
    main()