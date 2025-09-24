"""
CORD-19 Data Analysis Module

"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
#from wordcloud import WordCloud
from collections import Counter
import re
import warnings
warnings.filterwarnings('ignore')

# Setting style for better visualizations
plt.style.use('default')
sns.set_palette("husl")

class CORDAnalyzer:
    def __init__(self, file_path='data/metadata.csv'):
        """
        Initializing the CORD-19 data analyzer
        """
        self.file_path = file_path
        self.df = None
        self.cleaned_df = None
        
    def load_data(self):
        """
        Loading and initial exploration of the dataset
        """
        try:
            print("Loading CORD-19 metadata...")
            self.df = pd.read_csv(self.file_path)
            print(f" Dataset loaded successfully! Shape: {self.df.shape}")
            return True
        except FileNotFoundError:
            print(" File not found. Please check the file path.")
            return False
        except Exception as e:
            print(f" Error loading file: {e}")
            return False
    
    def basic_exploration(self):
        """
        Performing basic data exploration
        """
        print("\n" + "="*50)
        print("BASIC DATA EXPLORATION")
        print("="*50)
        
        print(f"Dataset dimensions: {self.df.shape}")
        print(f"Number of columns: {len(self.df.columns)}")
        
        print("\nFirst 5 rows:")
        print(self.df.head())
        
        print("\nData types:")
        print(self.df.dtypes)
        
        print("\nMissing values per column:")
        missing_data = self.df.isnull().sum()
        missing_percent = (missing_data / len(self.df)) * 100
        missing_info = pd.DataFrame({
            'Missing Count': missing_data,
            'Missing Percentage': missing_percent
        })
        print(missing_info[missing_info['Missing Count'] > 0])
        
        print("\nBasic statistics for numerical columns:")
        print(self.df.describe())
    
    def clean_data(self):
        """
        Cleaning and preparing the data for analysis
        """
        print("\n" + "="*50)
        print("DATA CLEANING")
        print("="*50)
        
        # Creating a copy for cleaning
        self.cleaned_df = self.df.copy()
        
        # Handling publication dates
        self.cleaned_df['publish_time'] = pd.to_datetime(
            self.cleaned_df['publish_time'], errors='coerce'
        )
        
        # Extracting year from publication date
        self.cleaned_df['publication_year'] = self.cleaned_df['publish_time'].dt.year
        
        # Creating abstract word count
        self.cleaned_df['abstract_word_count'] = self.cleaned_df['abstract'].apply(
            lambda x: len(str(x).split()) if pd.notnull(x) else 0
        )
        
        # Filling missing values
        self.cleaned_df['journal'] = self.cleaned_df['journal'].fillna('Unknown Journal')
        self.cleaned_df['abstract'] = self.cleaned_df['abstract'].fillna('No abstract available')
        
        # Removing rows with critical missing data
        initial_count = len(self.cleaned_df)
        self.cleaned_df = self.cleaned_df.dropna(subset=['title', 'publish_time'])
        final_count = len(self.cleaned_df)
        
        print(f"Rows removed due to missing critical data: {initial_count - final_count}")
        print(f"Final dataset shape: {self.cleaned_df.shape}")
        
        return self.cleaned_df
    
    def analyze_publications_over_time(self):
        """
        Analyzing publication trends over time
        """
        yearly_counts = self.cleaned_df['publication_year'].value_counts().sort_index()
        
        plt.figure(figsize=(12, 6))
        yearly_counts.plot(kind='bar', color='skyblue')
        plt.title('Number of COVID-19 Publications by Year', fontsize=16, fontweight='bold')
        plt.xlabel('Publication Year')
        plt.ylabel('Number of Papers')
        plt.xticks(rotation=45)
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig('publications_by_year.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        return yearly_counts
    
    def analyze_top_journals(self, top_n=15):
        """
        Analyzing top publishing journals
        """
        journal_counts = self.cleaned_df['journal'].value_counts().head(top_n)
        
        plt.figure(figsize=(12, 8))
        journal_counts.plot(kind='barh', color='lightgreen')
        plt.title(f'Top {top_n} Journals Publishing COVID-19 Research', fontsize=16, fontweight='bold')
        plt.xlabel('Number of Papers')
        plt.gca().invert_yaxis()
        plt.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        plt.savefig('top_journals.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        return journal_counts
    
    def create_title_wordcloud(self):
        """
        Creating a word cloud from paper titles
        """
        # Combining all titles
        all_titles = ' '.join(self.cleaned_df['title'].dropna().astype(str))
        
        # Clean the text
        words = re.findall(r'\b[a-zA-Z]{4,}\b', all_titles.lower())
        word_freq = Counter(words)
        
        # Removing common stop words
        stop_words = {'study', 'using', 'based', 'results', 'method', 'analysis', 
                     'research', 'paper', 'article', 'review', 'covid', 'sars', 'cov'}
        filtered_words = {word: count for word, count in word_freq.items() 
                         if word not in stop_words and count > 10}
        
        if filtered_words:
            wordcloud = WordCloud(
                width=800, 
                height=400, 
                background_color='white',
                colormap='viridis'
            ).generate_from_frequencies(filtered_words)
            
            plt.figure(figsize=(12, 6))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.title('Most Frequent Words in Paper Titles', fontsize=16, fontweight='bold')
            plt.axis('off')
            plt.tight_layout()
            plt.savefig('title_wordcloud.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            return dict(sorted(filtered_words.items(), key=lambda x: x[1], reverse=True)[:20])
        
        return {}
    
    def analyze_abstract_lengths(self):
        """
        Analyzing distribution of abstract lengths
        """
        plt.figure(figsize=(10, 6))
        plt.hist(self.cleaned_df['abstract_word_count'], bins=50, color='lightcoral', alpha=0.7)
        plt.title('Distribution of Abstract Word Counts', fontsize=16, fontweight='bold')
        plt.xlabel('Word Count')
        plt.ylabel('Number of Papers')
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig('abstract_lengths.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        return self.cleaned_df['abstract_word_count'].describe()
    
    def run_complete_analysis(self):
        """
        Running the complete analysis pipeline
        """
        if not self.load_data():
            return None
        
        self.basic_exploration()
        self.clean_data()
        
        results = {}
        
        # Performing the various analyses
        results['yearly_publications'] = self.analyze_publications_over_time()
        results['top_journals'] = self.analyze_top_journals()
        results['word_frequencies'] = self.create_title_wordcloud()
        results['abstract_stats'] = self.analyze_abstract_lengths()
        
        print("\n" + "="*50)
        print("ANALYSIS COMPLETE")
        print("="*50)
        
        # Printing summary findings
        print("\nKEY FINDINGS:")
        print(f"- Total papers analyzed: {len(self.cleaned_df)}")
        print(f"- Publication years: {sorted(self.cleaned_df['publication_year'].unique())}")
        print(f"- Most productive year: {results['yearly_publications'].idxmax()} "
              f"({results['yearly_publications'].max()} papers)")
        print(f"- Top journal: {results['top_journals'].index[0]} "
              f"({results['top_journals'].iloc[0]} papers)")
        print(f"- Average abstract length: {results['abstract_stats']['mean']:.1f} words")
        
        return results

# Example illustration of usage
if __name__ == "__main__":
    analyzer = CORDAnalyzer()
    results = analyzer.run_complete_analysis()