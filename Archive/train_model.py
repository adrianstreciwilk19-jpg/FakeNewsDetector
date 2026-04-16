import re
import joblib
import pandas as pd

from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import ComplementNB
from sklearn.model_selection import cross_validate
from sklearn.model_selection import GridSearchCV

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / 'data'
MODELS_DIR = BASE_DIR / 'models'
MODELS_DIR.mkdir(exist_ok=True)

FAKE_PATH = DATA_DIR / 'Fake.csv'
TRUE_PATH = DATA_DIR / 'True.csv'
MODEL_PATH = MODELS_DIR / 'fake_news_detector_2.joblib'

def _clean_text(text: str) -> str:
    """Proste czyszczenie tekstu"""
    if pd.isna(text):
        return ''
    
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+',' ', text) #usunięcie linków
    text = re.sub(r'[^a-zA-Z\s]', ' ', text) #tylko litery i spacje
    text = re.sub(r'\s+', ' ', text).strip() #redukcja wielokrotnych spacji
    
    return text

def _load_data(fake_path: Path, true_path: Path) -> pd.DataFrame:
    """Wczytanie i połączenie danych"""
    if not fake_path.exists():
        raise FileNotFoundError(f'Brak pliku: {fake_path}')
    if not true_path.exists():
        raise FileNotFoundError(f'Brak pliku: {true_path}')
    
    fake_df = pd.read_csv(fake_path)
    true_df = pd.read_csv(true_path)

    fake_df['label'] = 0 #fake
    true_df['label'] = 1 #true

    df = pd.concat([fake_df, true_df], ignore_index = True)

    req_col = ['title', 'text', 'label']
    for col in req_col:
        if col not in df.columns:
            raise ValueError(f'Brakuje kolumny {col}')
        
    #połączenie tytułu i treści w jeden tekst
    df['title'] = df['title'].fillna('')
    df['text'] = df['text'].fillna('')
    df['content'] = (df['title'] + ' ' + df['text']).apply(_clean_text)

    #usunięcie pustych rekordów
    df = df[df['content'].str.len() > 0].copy()

    return df[['content', 'label']]

def _build_model() -> Pipeline:
    """Budowa pipelin'u ML"""

    model = Pipeline(
        steps=[
            (
                'tfidf',
                TfidfVectorizer(
                    stop_words='english',
                    max_df=0.7,
                    min_df=2,
                    ngram_range=(1,2),
                ),
            ),
            (
                'clf',
                LogisticRegression(
                    max_iter=1000,
                    class_weight='balanced',
                    random_state=42,
                ),
            ),
        ]
    )
    return model

def _main() -> None:
    print('Wczytywanie danych...')
    df = _load_data(FAKE_PATH, TRUE_PATH)

    x = df['content']
    y = df['label']

    print(f'Liczba rekordów: {len(df)}')
    print('Podział na zbiór treningowy i testowy...')

    X_train, X_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    print('Budowa i trening modelu...')
    model = _build_model()
    model.fit(X_train, y_train)

    print('Predykcja na zboiorze testowym...')
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print('\n===WYNIKI====')
    print(f'Accuracy: {acc:.4f}')
    print('\nConfusion matrix:')
    print(cm)

    print('\nClassification report:')
    print(classification_report(y_test, y_pred, target_names=['FAKE', 'REAL']))

    print(f'\nZapisywanie modelu do: {MODEL_PATH}')
    joblib.dump(model, MODEL_PATH)

    print('Gotowe. Model został wytrenowany i zapisany.')

if __name__ == '__main__':
    _main()