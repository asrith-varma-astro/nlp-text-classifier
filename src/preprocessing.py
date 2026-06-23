"""
preprocessing.py
----------------
Text cleaning utilities for the NLP text classification pipeline.
"""

import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download required NLTK data on first use
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)

_stop_words = set(stopwords.words('english'))
_lemmatizer = WordNetLemmatizer()


def clean_sentence(sentence: str, remove_stopwords: bool = False, lemmatize: bool = False) -> str:
    """
    Clean and normalise a raw sentence.

    Steps:
        1. Strip HTML tags
        2. Remove non-alphabetic characters
        3. Lowercase
        4. Tokenize
        5. (Optional) Remove stopwords
        6. (Optional) Lemmatize tokens

    Args:
        sentence: Raw input text.
        remove_stopwords: If True, stopwords are filtered out.
        lemmatize: If True, tokens are lemmatized.

    Returns:
        Cleaned sentence as a single string.
    """
    # Remove HTML tags
    sentence = re.sub(r'<[^>]+>', '', sentence)
    # Keep only alphabetic characters and whitespace
    sentence = re.sub(r'[^a-zA-Z\s]', '', sentence)
    # Lowercase
    sentence = sentence.lower()
    # Tokenize
    tokens = word_tokenize(sentence)

    if remove_stopwords:
        tokens = [t for t in tokens if t not in _stop_words]

    if lemmatize:
        tokens = [_lemmatizer.lemmatize(t) for t in tokens]

    return " ".join(tokens)


def preprocess_dataframe(df, text_col: str = 'sentence', label_col: str = 'label',
                          remove_stopwords: bool = False, lemmatize: bool = False):
    """
    Apply cleaning pipeline to a DataFrame and encode labels.

    Args:
        df: Input DataFrame with raw text and labels.
        text_col: Name of the column containing raw sentences.
        label_col: Name of the column containing string labels ('question' / 'sentence').
        remove_stopwords: Passed to clean_sentence.
        lemmatize: Passed to clean_sentence.

    Returns:
        DataFrame with 'cleaned_sentence' column added and labels encoded as int (1=question, 0=sentence).
    """
    df = df.copy()
    df['cleaned_sentence'] = df[text_col].apply(
        lambda s: clean_sentence(s, remove_stopwords=remove_stopwords, lemmatize=lemmatize)
    )
    label_map = {'question': 1, 'sentence': 0}
    df[label_col] = df[label_col].map(label_map)
    return df
