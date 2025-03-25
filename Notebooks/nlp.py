import pandas as pd
import numpy as np
import re
import os
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from datetime import datetime
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LinearRegression
import nltk
from wordcloud import WordCloud
import warnings
warnings.filterwarnings('ignore')

# Create visualizations directory if it doesn't exist
os.makedirs('nlp_viz', exist_ok=True)

# Download NLTK resources if not already downloaded
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')

# Set plot style
plt.style.use('ggplot')
sns.set(font_scale=1.2)

# 1. Load and clean the data
def load_and_clean_data(file_path):
    print(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path, low_memory=False)
    
    if 'Match.Length' in df.columns:
        df.rename(columns={'Match.Length': 'Match Length'}, inplace=True)
    
    if 'Completion.Date' in df.columns:
        df.rename(columns={'Completion.Date': 'Completion Date'}, inplace=True)
    
    notes_col = None
    for col in ['Match Support Contact Notes', 'answer']:
        if col in df.columns:
            notes_col = col
            break
    
    if notes_col is None:
        raise ValueError("Could not find notes column in the dataset")
    
    print(f"Using '{notes_col}' column for text analysis")
    
    # Clean data
    print("Cleaning data...")
    df['cleaned_notes'] = df[notes_col].apply(clean_text)
    
    sample_df = df[[notes_col, 'cleaned_notes']].head(50)
    sample_df.to_csv('cleaning_samples.csv', index=False)
    print("Saved 50 sample cleanings to cleaning_samples.csv for inspection")
    
    df['Completion Date'] = pd.to_datetime(df['Completion Date'], errors='coerce')
    df['time_period'] = df['Completion Date'].apply(get_time_period)
    
    return df

def clean_text(text):
    if not isinstance(text, str):
        return ""
    
    # Replace special characters and symbols
    cleaned = re.sub(r'[¬?\\]', ' ', text)
    
    # Replace placeholder names with generic terms
    cleaned = re.sub(r'L_first_name', 'Little', cleaned)
    cleaned = re.sub(r'L_last_name', '', cleaned)
    cleaned = re.sub(r'B_first_name', 'Big', cleaned)
    cleaned = re.sub(r'B_last_name', '', cleaned)
    
    # Remove form elements
    cleaned = re.sub(r'\*\*This form counts as.*?BBBS', '', cleaned, flags=re.DOTALL)
    cleaned = re.sub(r'Match Engagement Coordinator.*?\)', '', cleaned, flags=re.DOTALL)
    
    # Extract answers from Question/Answer format
    answers = []
    qa_pairs = re.findall(r'Question:\s*(.*?)\s*Answer:\s*(.*?)(?=Question:|$)', cleaned, re.DOTALL)
    
    if qa_pairs:
        for _, answer in qa_pairs:
            answer = answer.strip()
            if answer and answer != '.' and answer != '..':
                answers.append(answer)
        
        if answers:
            return ' '.join(answers)
    
    return cleaned

def get_time_period(date):
    if pd.isna(date):
        return 'unknown'
    
    year = date.year
    half_year = 1 if date.month < 7 else 2
    
    return f"{year}-H{half_year}"

# 2. Split data by time period
def split_by_time_period(df):
    print("Splitting data by time period...")
    time_periods = df['time_period'].unique()
    period_dfs = {}
    
    for period in time_periods:
        if period == 'unknown':
            continue
        period_df = df[df['time_period'] == period].copy()
        period_dfs[period] = period_df
        print(f"  {period}: {len(period_df)} records")
    
    return period_dfs

# 3. NLP Analysis
def prepare_nlp_stopwords():
    # Combine NLTK stopwords with domain-specific ones
    stop_words = set(stopwords.words('english'))
    domain_stopwords = {
        'l_first_name', 'l_last_name', 'b_first_name', 'b_last_name',
        'said', 'asked', 'names', 'little', 'big', 'match', 'bbs',
        'question', 'answer', 'msc', 'bs', 'ls', 'will', 'can', 'just',
        'like', 'get', 'got', 'going', 'went', 'go', 'see', 'day', 'time',
        'week', 'month', 'has', 'have', 'had', 'does', 'did', 'would',
        'could', 'should', 'says', 'said', 'told', 'talk', 'talked',
        'discussed', 'met', 'meeting', 'called', 'things', 'thing'
    }
    
    return stop_words.union(domain_stopwords)

def tokenize_text(text, stop_words):
    if not isinstance(text, str):
        return []
    
    # Tokenize and filter
    tokens = word_tokenize(text.lower())
    return [word for word in tokens if word.isalpha() and len(word) > 2 and word not in stop_words]

def analyze_word_importance(df, stop_words, min_occurrences=20):
    print("Analyzing word importance...")
    # Create a document-term matrix
    all_tokens = []
    doc_tokens = []
    
    for _, row in df.iterrows():
        tokens = tokenize_text(row['cleaned_notes'], stop_words)
        doc_tokens.append(tokens)
        all_tokens.extend(tokens)
    
    # Get word counts
    word_counts = Counter(all_tokens)
    
    # Filter words that appear less than min_occurrences times
    frequent_words = {word for word, count in word_counts.items() if count >= min_occurrences}
    
    word_stats = {}
    
    for word in frequent_words:
        rows_with_word = df[[word in tokens for tokens in doc_tokens]].copy()
        rows_without_word = df[[word not in tokens for tokens in doc_tokens]].copy()
        
        if len(rows_with_word) < min_occurrences:
            continue
        
        avg_length_with_word = rows_with_word['Match Length'].mean()
        avg_length_without_word = rows_without_word['Match Length'].mean()
        
        difference = avg_length_with_word - avg_length_without_word
        
        word_stats[word] = {
            'occurrences': len(rows_with_word),
            'avg_length_with_word': round(avg_length_with_word, 2),
            'avg_length_without_word': round(avg_length_without_word, 2),
            'difference': round(difference, 2)
        }
    
    sorted_stats = sorted(word_stats.items(), key=lambda x: abs(x[1]['difference']), reverse=True)
    
    return sorted_stats, word_stats

def categorize_words(word_stats):
    categories = {
        "Activities": ["activities", "activity", "together", "fun", "games", "played", "playing", "play",
                       "movies", "movie", "outside", "park", "sports", "basketball", "football", "craft",
                       "crafts", "draw", "drawing", "paint", "painting", "read", "reading", "book", "books"],
        "Communication": ["talk", "talking", "talked", "communicate", "communication", "conversation", 
                          "texting", "text", "call", "calls", "called", "phone", "email", "emails", "contact",
                          "discuss", "discussed", "discussion", "sharing", "shared", "share"],
        "Relationship": ["relationship", "bond", "bonding", "close", "closer", "comfortable", "trust",
                         "friendship", "friend", "friends", "happy", "enjoy", "enjoying", "enjoyed", 
                         "connect", "connecting", "connection", "support", "supporting", "supported"],
        "Challenges": ["challenge", "challenges", "difficult", "difficulty", "hard", "struggle", "struggles",
                       "struggling", "issue", "issues", "problem", "problems", "concern", "concerns",
                       "worried", "worry", "anxious", "anxiety", "stress", "stressed"],
        "School": ["school", "college", "university", "class", "classes", "grade", "grades", "homework",
                   "study", "studying", "studied", "academic", "academics", "teacher", "teachers", 
                   "education", "educational", "learn", "learning", "learned", "student"],
        "Family": ["family", "parent", "parents", "mom", "mother", "dad", "father", "brother", "brothers",
                   "sister", "sisters", "sibling", "siblings", "relative", "relatives", "grandparent",
                   "grandparents", "grandmother", "grandfather", "home"]
    }
    
    categorized_words = {}
    for category, category_words in categories.items():
        categorized_words[category] = []
        
        for word, stats in word_stats.items():
            if word in category_words:
                categorized_words[category].append({
                    'word': word,
                    **stats
                })
        
        categorized_words[category] = sorted(categorized_words[category], 
                                             key=lambda x: abs(x['difference']), 
                                             reverse=True)
    
    category_impact = {}
    for category, words in categorized_words.items():
        if words:
            avg_impact = sum(item['difference'] for item in words) / len(words)
            top_positive = [w for w in words if w['difference'] > 0]
            top_negative = [w for w in words if w['difference'] < 0]
            
            category_impact[category] = {
                'avg_impact': round(avg_impact, 2),
                'word_count': len(words),
                'top_positive_word': top_positive[0] if top_positive else None,
                'top_negative_word': top_negative[0] if top_negative else None
            }
    
    return categorized_words, category_impact

# 4. Create visualizations
def create_visualizations(word_stats, categorized_words, category_impact, time_period=None):
    period_str = f"_{time_period}" if time_period else ""
    
    top_positive = [(word, stats['difference']) 
                    for word, stats in word_stats.items() 
                    if stats['difference'] > 0][:15]
    
    top_negative = [(word, stats['difference']) 
                    for word, stats in word_stats.items() 
                    if stats['difference'] < 0][:15]
    
    top_positive.sort(key=lambda x: x[1], reverse=True)
    top_negative.sort(key=lambda x: x[1])
    
    plt.figure(figsize=(12, 8))
    sns.barplot(x=[x[1] for x in top_positive], y=[x[0] for x in top_positive], palette='Blues_d')
    plt.title(f'Top 15 Words Associated with Longer Match Length{" - " + time_period if time_period else ""}', fontsize=16)
    plt.xlabel('Difference in Average Match Length (months)', fontsize=14)
    plt.ylabel('Word', fontsize=14)
    plt.tight_layout()
    plt.savefig(f'nlp_viz/positive_words{period_str}.png', dpi=300)
    plt.close()
    
    # Negative words chart
    plt.figure(figsize=(12, 8))
    sns.barplot(x=[x[1] for x in top_negative], y=[x[0] for x in top_negative], palette='Reds_d')
    plt.title(f'Top 15 Words Associated with Shorter Match Length{" - " + time_period if time_period else ""}', fontsize=16)
    plt.xlabel('Difference in Average Match Length (months)', fontsize=14)
    plt.ylabel('Word', fontsize=14)
    plt.tight_layout()
    plt.savefig(f'nlp_viz/negative_words{period_str}.png', dpi=300)
    plt.close()
    
    categories = list(category_impact.keys())
    impacts = [category_impact[cat]['avg_impact'] for cat in categories]
    word_counts = [category_impact[cat]['word_count'] for cat in categories]
    
    sorted_indices = sorted(range(len(impacts)), key=lambda i: abs(impacts[i]), reverse=True)
    categories = [categories[i] for i in sorted_indices]
    impacts = [impacts[i] for i in sorted_indices]
    word_counts = [word_counts[i] for i in sorted_indices]
    
    colors = ['green' if imp > 0 else 'red' for imp in impacts]
    
    plt.figure(figsize=(12, 8))
    bars = plt.bar(categories, impacts, color=colors)
    plt.title(f'Average Impact of Word Categories on Match Length{" - " + time_period if time_period else ""}', fontsize=16)
    plt.xlabel('Category', fontsize=14)
    plt.ylabel('Avg. Difference in Match Length (months)', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    
    # Add word count as text on bars
    for bar, count in zip(bars, word_counts):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., 
                 height + (0.1 if height > 0 else -0.1),
                 f'{count} words',
                 ha='center', va='bottom' if height > 0 else 'top')
    
    plt.tight_layout()
    plt.savefig(f'nlp_viz/category_impact{period_str}.png', dpi=300)
    plt.close()
    
    word_impact_dict = {word: stats['difference'] for word, stats in word_stats.items()}
    
    max_abs_impact = max([abs(val) for val in word_impact_dict.values()])
    norm_word_impact = {word: impact/max_abs_impact for word, impact in word_impact_dict.items()}
    
    def color_func(word, **kwargs):
        impact = norm_word_impact.get(word, 0)
        if impact > 0:
            intensity = int(255 * min(impact, 1))
            return f"rgb({255-intensity}, {255-intensity}, 255)"
        else:
            intensity = int(255 * min(abs(impact), 1))
            return f"rgb(255, {255-intensity}, {255-intensity})"
    
    wc = WordCloud(
        background_color='white',
        max_words=100,
        width=800,
        height=400,
        color_func=color_func
    )
    
    abs_word_impact = {word: abs(impact) for word, impact in word_impact_dict.items()}
    wc.generate_from_frequencies(abs_word_impact)
    
    plt.figure(figsize=(16, 8))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    plt.title(f'Word Cloud of Important Words{" - " + time_period if time_period else ""}\n(Blue = Positive Impact, Red = Negative Impact)', fontsize=16)
    plt.tight_layout()
    plt.savefig(f'nlp_viz/wordcloud{period_str}.png', dpi=300)
    plt.close()

def main():
    df = load_and_clean_data('../Data/Training.csv')
    
    print("Saving cleaned data to cleaned_training.csv...")
    df.to_csv('cleaned_training.csv', index=False)
    print("Cleaned data saved successfully.")
    
    period_dfs = split_by_time_period(df)
    stop_words = prepare_nlp_stopwords()

    print("\nAnalyzing all data combined...")
    sorted_stats_all, word_stats_all = analyze_word_importance(df, stop_words)
    
    print("\nTop 30 words by impact on Match Length (all periods):")
    for word, stats in sorted_stats_all[:30]:
        direction = 'increases' if stats['difference'] > 0 else 'decreases'
        print(f"{word}: {direction} length by {abs(stats['difference'])} months (occurs {stats['occurrences']} times)")
    
    print("\nCategorizing words...")
    categorized_words, category_impact = categorize_words(dict(sorted_stats_all))

    print("\nCategory impact on match length:")
    for category, impact in category_impact.items():
        print(f"{category}: Average impact {impact['avg_impact']} months ({impact['word_count']} words identified)")
        if impact['top_positive_word']:
            print(f"  Top positive word: {impact['top_positive_word']['word']} ({impact['top_positive_word']['difference']})")
        if impact['top_negative_word']:
            print(f"  Top negative word: {impact['top_negative_word']['word']} ({impact['top_negative_word']['difference']})")
    
    print("\nCreating visualizations for all data...")
    create_visualizations(dict(sorted_stats_all), categorized_words, category_impact)
    
    for period, period_df in period_dfs.items():
        print(f"\nAnalyzing time period: {period}")
        sorted_stats_period, word_stats_period = analyze_word_importance(period_df, stop_words)
        
        if len(sorted_stats_period) > 10:
            categorized_words_period, category_impact_period = categorize_words(dict(sorted_stats_period))
            print(f"Creating visualizations for {period}...")
            create_visualizations(dict(sorted_stats_period), categorized_words_period, category_impact_period, period)
    
    print("\nAnalysis complete. Visualizations saved to nlp_viz/ directory.")

if __name__ == "__main__":
    main()