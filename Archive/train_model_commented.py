import re  # moduł do pracy z wyrażeniami regularnymi, używany tutaj do czyszczenia tekstu
import joblib  # biblioteka do zapisywania i wczytywania wytrenowanych modeli ML
import pandas as pd  # pandas do operacji na danych tabelarycznych

from pathlib import Path  # wygodna obsługa ścieżek do plików i katalogów

#from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer  # zamienia tekst na liczby z użyciem statystyki TF-IDF
from sklearn.linear_model import LogisticRegression  # klasyfikator logistyczny do rozpoznawania FAKE / REAL
from sklearn.metrics import (
    accuracy_score,  # oblicza dokładność modelu
    classification_report,  # generuje szczegółowy raport jakości klasyfikacji
    confusion_matrix,  # tworzy macierz pomyłek modelu
)
from sklearn.model_selection import train_test_split  # dzieli dane na część treningową i testową
from sklearn.pipeline import Pipeline  # pozwala połączyć kilka etapów ML w jeden spójny pipeline
#from sklearn.preprocessing import FunctionTransformer

# Ustalenie katalogu, w którym znajduje się aktualnie uruchamiany plik .py
BASE_DIR = Path(__file__).resolve().parent

# Ścieżka do katalogu z danymi wejściowymi
DATA_DIR = BASE_DIR / 'data'

# Ścieżka do katalogu, w którym będzie zapisany wytrenowany model
MODELS_DIR = BASE_DIR / 'models'

# Utworzenie katalogu models, jeśli jeszcze nie istnieje
MODELS_DIR.mkdir(exist_ok=True)

# Ścieżka do pliku zawierającego fake newsy
FAKE_PATH = DATA_DIR / 'Fake.csv'

# Ścieżka do pliku zawierającego prawdziwe newsy
TRUE_PATH = DATA_DIR / 'True.csv'

# Docelowa ścieżka zapisania gotowego modelu po treningu
MODEL_PATH = MODELS_DIR / 'fake_news_detector_2.joblib'


def _clean_text(text: str) -> str:
    """Proste czyszczenie tekstu"""
    # Sprawdzenie, czy przekazana wartość jest pusta / brakująca (NaN)
    if pd.isna(text):
        return ''
    
    # Zamiana danych na string i sprowadzenie wszystkich liter do małych znaków
    text = str(text).lower()

    # Usunięcie linków zaczynających się np. od http... albo www...
    text = re.sub(r'http\S+|www\S+',' ', text) #usunięcie linków

    # Usunięcie wszystkiego poza literami alfabetu angielskiego i spacjami
    # Dzięki temu model dostaje czystszy tekst bez cyfr i znaków specjalnych
    text = re.sub(r'[^a-zA-Z\s]', ' ', text) #tylko litery i spacje

    # Zamiana wielu spacji na jedną oraz usunięcie spacji z początku i końca tekstu
    text = re.sub(r'\s+', ' ', text).strip() #redukcja wielokrotnych spacji
    
    # Zwrócenie oczyszczonej wersji tekstu
    return text


def _load_data(fake_path: Path, true_path: Path) -> pd.DataFrame:
    """Wczytanie i połączenie danych"""
    # Sprawdzenie, czy plik z fake newsami istnieje
    if not fake_path.exists():
        raise FileNotFoundError(f'Brak pliku: {fake_path}')

    # Sprawdzenie, czy plik z prawdziwymi newsami istnieje
    if not true_path.exists():
        raise FileNotFoundError(f'Brak pliku: {true_path}')
    
    # Wczytanie obu plików CSV do osobnych DataFrame'ów
    fake_df = pd.read_csv(fake_path)
    true_df = pd.read_csv(true_path)

    # Dodanie etykiety 0 dla fake newsów
    fake_df['label'] = 0 #fake

    # Dodanie etykiety 1 dla prawdziwych newsów
    true_df['label'] = 1 #true

    # Połączenie obu tabel w jedną dużą tabelę z pełnym zbiorem danych
    df = pd.concat([fake_df, true_df], ignore_index = True)

    # Lista kolumn, które muszą istnieć w danych wejściowych,
    # aby dalsza część programu mogła działać poprawnie
    req_col = ['title', 'text', 'label']

    # Sprawdzenie obecności każdej wymaganej kolumny
    for col in req_col:
        if col not in df.columns:
            raise ValueError(f'Brakuje kolumny {col}')
        
    # Połączenie tytułu i treści artykułu w jeden ciąg tekstowy,
    # bo model będzie analizował jedną wspólną kolumnę tekstową
    df['title'] = df['title'].fillna('')
    df['text'] = df['text'].fillna('')
    df['content'] = (df['title'] + ' ' + df['text']).apply(_clean_text)

    # Usunięcie rekordów, które po czyszczeniu są puste
    # Takie wiersze nie wnoszą żadnej wartości do treningu modelu
    df = df[df['content'].str.len() > 0].copy()

    # Zwrócenie tylko tych dwóch kolumn, które są potrzebne dalej:
    # treści wejściowej oraz etykiety klasy
    return df[['content', 'label']]


def _build_model() -> Pipeline:
    """Budowa pipelin'u ML"""

    # Zbudowanie pipeline'u, czyli połączenia kolejnych etapów przetwarzania danych:
    # 1. zamiana tekstu na cechy liczbowe (TF-IDF)
    # 2. klasyfikacja za pomocą regresji logistycznej
    model = Pipeline(
        steps=[
            (
                'tfidf',
                TfidfVectorizer(
                    stop_words='english',  # usunięcie najczęstszych angielskich słów typu "the", "is", "and"
                    max_df=0.7,  # odrzucenie słów występujących w ponad 70% dokumentów
                    min_df=2,  # uwzględnienie tylko słów pojawiających się co najmniej w 2 dokumentach
                    ngram_range=(1,2),  # analiza pojedynczych słów oraz par słów
                ),
            ),
            (
                'clf',
                LogisticRegression(
                    max_iter=1000,  # zwiększona liczba iteracji, aby algorytm zdążył się zbiec
                    class_weight='balanced',  # wyrównanie wag klas przy ewentualnie niezbalansowanych danych
                    random_state=42,  # stałe ziarno losowości dla powtarzalnych wyników
                ),
            ),
        ]
    )

    # Zwrócenie gotowego pipeline'u
    return model


def _main() -> None:
    # Informacja dla użytkownika o rozpoczęciu wczytywania danych
    print('Wczytywanie danych...')

    # Załadowanie i przygotowanie zbioru danych z plików CSV
    df = _load_data(FAKE_PATH, TRUE_PATH)

    # Pobranie tekstów wejściowych do zmiennej x
    x = df['content']

    # Pobranie etykiet klas do zmiennej y
    y = df['label']

    # Wyświetlenie liczby wszystkich rekordów użytych do trenowania i testu
    print(f'Liczba rekordów: {len(df)}')
    print('Podział na zbiór treningowy i testowy...')

    # Podział danych:
    # - 80% na trening
    # - 20% na test
    # stratify=y pilnuje, aby proporcje klas FAKE/REAL były podobne w obu zbiorach
    X_train, X_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    # Komunikat o rozpoczęciu budowy i trenowania modelu
    print('Budowa i trening modelu...')

    # Utworzenie pipeline'u modelu
    model = _build_model()

    # Trenowanie modelu na zbiorze treningowym
    model.fit(X_train, y_train)

    # Komunikat o uruchomieniu predykcji na danych testowych
    print('Predykcja na zboiorze testowym...')

    # Wygenerowanie przewidywań modelu dla danych testowych
    y_pred = model.predict(X_test)

    # Obliczenie dokładności modelu
    acc = accuracy_score(y_test, y_pred)

    # Wyliczenie macierzy pomyłek,
    # która pokazuje ile przykładów model zaklasyfikował poprawnie lub błędnie
    cm = confusion_matrix(y_test, y_pred)

    # Wyświetlenie sekcji z wynikami
    print('\n===WYNIKI====')
    print(f'Accuracy: {acc:.4f}')

    # Wyświetlenie macierzy pomyłek
    print('\nConfusion matrix:')
    print(cm)

    # Wyświetlenie szczegółowego raportu klasyfikacji
    # zawierającego precision, recall, f1-score i support dla każdej klasy
    print('\nClassification report:')
    print(classification_report(y_test, y_pred, target_names=['FAKE', 'REAL']))

    # Poinformowanie użytkownika, gdzie model zostanie zapisany
    print(f'\nZapisywanie modelu do: {MODEL_PATH}')

    # Zapis gotowego modelu do pliku .joblib
    joblib.dump(model, MODEL_PATH)

    # Końcowy komunikat potwierdzający zakończenie treningu i zapisu modelu
    print('Gotowe. Model został wytrenowany i zapisany.')


# Standardowy punkt startowy programu.
# Ten warunek sprawia, że funkcja _main() uruchomi się tylko wtedy,
# gdy plik zostanie odpalony bezpośrednio, a nie gdy zostanie zaimportowany jako moduł.
if __name__ == '__main__':
    _main()
