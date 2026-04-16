import re  # Moduł do operacji na wyrażeniach regularnych, używany do czyszczenia tekstu.
import joblib  # Biblioteka do zapisu i odczytu wytrenowanych obiektów Pythona, np. modelu ML.
import pandas as pd  # Pandas służy do pracy z danymi tabelarycznymi w postaci DataFrame.

from pathlib import Path  # Ułatwia budowanie ścieżek w sposób przenośny między systemami operacyjnymi.

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

# Wyznaczenie katalogu, w którym znajduje się aktualnie uruchomiony plik.
# Dzięki temu dalsze ścieżki są liczone względem lokalizacji skryptu,
# a nie względem bieżącego katalogu roboczego terminala.
BASE_DIR = Path(__file__).resolve().parent

# Katalog z danymi wejściowymi wykorzystywanymi do treningu.
DATA_DIR = BASE_DIR / 'data'

# Katalog docelowy, w którym zostanie zapisany wytrenowany model.
MODELS_DIR = BASE_DIR / 'models'

# Utworzenie katalogu models, jeśli jeszcze nie istnieje.
# Parametr exist_ok=True zapobiega błędowi przy ponownym uruchomieniu skryptu.
MODELS_DIR.mkdir(exist_ok=True)

# Ścieżka do pliku zawierającego przykłady fałszywych wiadomości.
FAKE_PATH = DATA_DIR / 'Fake.csv'

# Ścieżka do pliku zawierającego przykłady prawdziwych wiadomości.
TRUE_PATH = DATA_DIR / 'True.csv'

# Pełna ścieżka docelowa dla zapisanego modelu.
# Rozszerzenie .joblib jest standardowo używane do serializacji modeli sklearn.
MODEL_PATH = MODELS_DIR / 'fake_news_detector_2.joblib'


def _clean_text(text: str) -> str:
    """Proste czyszczenie tekstu"""

    # Sprawdzenie, czy wejściowa wartość jest pusta z punktu widzenia pandas,
    # np. NaN. W takim przypadku funkcja zwraca pusty string,
    # żeby dalsze przetwarzanie nie kończyło się błędem.
    if pd.isna(text):
        return ''

    # Zamiana danych wejściowych na string oraz sprowadzenie całego tekstu
    # do małych liter. To redukuje liczbę wariantów tego samego słowa,
    # np. "News" i "news" będą traktowane identycznie.
    text = str(text).lower()

    # Usunięcie adresów URL zaczynających się od http... lub www...
    # Linki zwykle nie niosą istotnej informacji semantycznej dla tego typu modelu,
    # a często tylko zwiększają szum w danych.
    text = re.sub(r'http\S+|www\S+',' ', text) #usunięcie linków

    # Pozostawienie wyłącznie liter alfabetu angielskiego oraz spacji.
    # Wszystkie cyfry, znaki specjalne i interpunkcja są zastępowane spacją,
    # co upraszcza słownik cech wejściowych dla wektoryzatora TF-IDF.
    text = re.sub(r'[^a-zA-Z\s]', ' ', text) #tylko litery i spacje

    # Redukcja wielu kolejnych spacji do jednej oraz usunięcie spacji
    # z początku i końca tekstu. To domyka etap normalizacji wejścia.
    text = re.sub(r'\s+', ' ', text).strip() #redukcja wielokrotnych spacji

    # Zwrócenie oczyszczonego tekstu, gotowego do dalszej obróbki.
    return text


def _load_data(fake_path: Path, true_path: Path) -> pd.DataFrame:
    """Wczytanie i połączenie danych"""

    # Walidacja istnienia pliku z fałszywymi wiadomościami.
    # Jawny wyjątek daje czytelniejszy komunikat niż późniejszy błąd read_csv.
    if not fake_path.exists():
        raise FileNotFoundError(f'Brak pliku: {fake_path}')

    # Walidacja istnienia pliku z prawdziwymi wiadomościami.
    if not true_path.exists():
        raise FileNotFoundError(f'Brak pliku: {true_path}')

    # Wczytanie danych CSV do struktur DataFrame.
    # Każdy plik staje się osobnym zbiorem rekordów.
    fake_df = pd.read_csv(fake_path)
    true_df = pd.read_csv(true_path)

    # Dodanie etykiety klasy do zbioru fake.
    # W tym projekcie 0 oznacza wiadomość fałszywą.
    fake_df['label'] = 0 #fake

    # Dodanie etykiety klasy do zbioru true.
    # W tym projekcie 1 oznacza wiadomość prawdziwą.
    true_df['label'] = 1 #true

    # Połączenie obu zbiorów w jeden wspólny DataFrame.
    # ignore_index=True nadaje nowy, spójny indeks po scaleniu.
    df = pd.concat([fake_df, true_df], ignore_index = True)

    # Lista kolumn wymaganych do dalszego przetwarzania.
    # Skrypt zakłada istnienie tytułu, treści i etykiety klasy.
    req_col = ['title', 'text', 'label']

    # Kontrola poprawności struktury danych wejściowych.
    # Jeśli brakuje którejkolwiek wymaganej kolumny, trening nie ma sensu
    # i należy zakończyć działanie z czytelnym komunikatem.
    for col in req_col:
        if col not in df.columns:
            raise ValueError(f'Brakuje kolumny {col}')

    # Uzupełnienie pustych wartości w kolumnie title pustym stringiem,
    # aby możliwe było bezpieczne łączenie tekstu bez wartości NaN.
    df['title'] = df['title'].fillna('')

    # Analogiczne uzupełnienie braków w kolumnie text.
    df['text'] = df['text'].fillna('')

    # Zbudowanie nowej kolumny content jako połączenia tytułu i treści.
    # Następnie na każdym rekordzie wykonywane jest czyszczenie tekstu.
    # To właśnie kolumna content będzie finalnym wejściem do modelu.
    df['content'] = (df['title'] + ' ' + df['text']).apply(_clean_text)

    # Usunięcie rekordów, które po czyszczeniu są puste.
    # Takie przykłady nie wnoszą żadnej informacji do treningu,
    # a mogłyby pogarszać jakość modelu.
    df = df[df['content'].str.len() > 0].copy()

    # Zwrócenie wyłącznie dwóch kolumn potrzebnych dalej:
    # content jako cechy wejściowe i label jako oczekiwana klasa.
    return df[['content', 'label']]


def _build_model() -> Pipeline:
    """Budowa pipelin'u ML"""

    # Utworzenie pipeline'u, czyli łańcucha przetwarzania danych.
    # Dzięki temu tekst wejściowy najpierw zostanie zamieniony na reprezentację
    # numeryczną, a następnie przekazany do klasyfikatora.
    # Pipeline upraszcza trening, predykcję i późniejszy zapis modelu do pliku.
    model = Pipeline(
        steps=[
            (
                'tfidf',
                TfidfVectorizer(
                    # Usuwanie angielskich słów bardzo częstych i mało informacyjnych,
                    # takich jak "the", "is", "and".
                    stop_words='english',

                    # Pominięcie tokenów występujących w ponad 70% dokumentów.
                    # Takie wyrazy zwykle słabo odróżniają klasy i zwiększają szum.
                    max_df=0.7,

                    # Pominięcie tokenów bardzo rzadkich, które pojawiły się
                    # w mniej niż 2 dokumentach. To ogranicza przeuczenie i rozmiar cech.
                    min_df=2,

                    # Użycie zarówno pojedynczych słów, jak i par słów.
                    # Pozwala to modelowi wychwycić nie tylko słownictwo,
                    # ale również krótkie frazy charakterystyczne dla klas.
                    ngram_range=(1,2),
                ),
            ),
            (
                'clf',
                LogisticRegression(
                    # Zwiększenie limitu iteracji optymalizatora.
                    # Przy większej liczbie cech tekstowych domyślny limit
                    # bywa niewystarczający do zbieżności.
                    max_iter=1000,

                    # Wyrównanie wpływu klas podczas uczenia.
                    # Ma to znaczenie szczególnie wtedy, gdy dane nie są idealnie zbalansowane.
                    class_weight='balanced',

                    # Ustalenie ziarna losowości dla powtarzalności wyników.
                    random_state=42,
                ),
            ),
        ]
    )

    # Zwrócenie kompletnego pipeline'u gotowego do treningu.
    return model


def _main() -> None:
    # Komunikat informujący użytkownika o rozpoczęciu etapu przygotowania danych.
    print('Wczytywanie danych...')

    # Załadowanie i przygotowanie zbioru treningowego z dwóch plików CSV.
    df = _load_data(FAKE_PATH, TRUE_PATH)

    # Wydzielenie kolumny wejściowej zawierającej tekst dokumentu.
    x = df['content']

    # Wydzielenie kolumny docelowej zawierającej etykietę klasy.
    y = df['label']

    # Informacja diagnostyczna o liczbie rekordów po scaleniu i czyszczeniu danych.
    print(f'Liczba rekordów: {len(df)}')

    # Komunikat o przejściu do podziału danych na trening i test.
    print('Podział na zbiór treningowy i testowy...')

    # Podział danych na część treningową i testową.
    # test_size=0.2 oznacza, że 20% rekordów trafia do testu.
    # random_state=42 zapewnia powtarzalność podziału.
    # stratify=y utrzymuje podobny rozkład klas w obu zbiorach.
    X_train, X_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    # Komunikat o rozpoczęciu budowy i uczenia modelu.
    print('Budowa i trening modelu...')

    # Utworzenie pipeline'u z wektoryzatorem TF-IDF i klasyfikatorem logistycznym.
    model = _build_model()

    # Trening modelu na zbiorze uczącym.
    # Na tym etapie pipeline sam wykona wektoryzację tekstu,
    # a następnie dopasuje klasyfikator do danych.
    model.fit(X_train, y_train)

    # Komunikat o uruchomieniu predykcji na zbiorze testowym.
    print('Predykcja na zboiorze testowym...')

    # Wygenerowanie przewidywanych etykiet dla danych testowych.
    y_pred = model.predict(X_test)

    # Obliczenie accuracy, czyli udziału poprawnych predykcji.
    acc = accuracy_score(y_test, y_pred)

    # Obliczenie macierzy pomyłek pokazującej liczbę trafień i błędów
    # w rozbiciu na klasy rzeczywiste i przewidziane.
    cm = confusion_matrix(y_test, y_pred)

    # Sekcja prezentująca wyniki ewaluacji modelu.
    print('\n===WYNIKI====')
    print(f'Accuracy: {acc:.4f}')

    print('\nConfusion matrix:')
    print(cm)

    # classification_report zwraca bardziej szczegółowe metryki,
    # m.in. precision, recall i f1-score dla każdej klasy.
    # target_names definiuje przyjazne nazwy klas w raporcie tekstowym.
    print('\nClassification report:')
    print(classification_report(y_test, y_pred, target_names=['FAKE', 'REAL']))

    # Informacja o lokalizacji zapisu gotowego modelu.
    print(f'\nZapisywanie modelu do: {MODEL_PATH}')

    # Zapis całego pipeline'u do pliku.
    # Dzięki temu w przyszłości można odczytać model i używać go do predykcji
    # bez ponownego treningu.
    joblib.dump(model, MODEL_PATH)

    # Końcowy komunikat potwierdzający zakończenie procesu treningu i zapisu.
    print('Gotowe. Model został wytrenowany i zapisany.')


# Standardowy punkt wejścia programu.
# Ten warunek sprawia, że funkcja _main() wykona się tylko wtedy,
# gdy plik jest uruchamiany bezpośrednio, a nie importowany jako moduł.
if __name__ == '__main__':
    _main()
