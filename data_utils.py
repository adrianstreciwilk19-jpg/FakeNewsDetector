# Import klasy Path z pathlib do wygodnej i przenośnej obsługi ścieżek plików.
from pathlib import Path

# Import modułu re do operacji na wyrażeniach regularnych.
import re

# Pandas służy tu do pracy na danych tabelarycznych (DataFrame).
import pandas as pd

# Funkcja do podziału danych na zbiór treningowy i testowy.
from sklearn.model_selection import train_test_split


# Stała definiująca ziarno losowości.
# Dzięki temu podział danych będzie powtarzalny między uruchomieniami.
RANDOM_STATE = 42

# Domyślny udział zbioru testowego: 20% wszystkich danych.
TEST_SIZE = 0.2


def clean_text(text: str) -> str:
    """
    Proste czyszczenie tekstu:
    - zamiana na małe litery
    - usunięcie linków
    - usunięcie znaków specjalnych
    - redukcja wielokrotnych spacji
    """

    # Jeżeli wartość jest pusta / NaN, zwracany jest pusty string.
    # To zabezpiecza dalsze operacje tekstowe przed błędami.
    if pd.isna(text):
        return ""

    # Konwersja do stringa i normalizacja do małych liter.
    # Ujednolica zapis i zmniejsza liczbę wariantów tych samych słów.
    text = str(text).lower()

    # Usunięcie linków zaczynających się od http... lub www...
    # Zastępujemy je spacją, aby nie skleić sąsiednich słów.
    text = re.sub(r"http\S+|www\S+", " ", text)

    # Usunięcie wszystkich znaków innych niż litery A-Z / a-z i białe znaki.
    # To upraszcza tekst przed dalszym przetwarzaniem ML.
    text = re.sub(r"[^a-zA-Z\s]", " ", text)

    # Zamiana wielu spacji/tabów/enterów na pojedynczą spację
    # oraz usunięcie spacji z początku i końca tekstu.
    text = re.sub(r"\s+", " ", text).strip()

    # Zwrócenie oczyszczonego tekstu.
    return text


def validate_columns(df: pd.DataFrame, required_columns: list[str]) -> None:
    """
    Sprawdza, czy DataFrame zawiera wymagane kolumny.
    """

    # Budowa listy brakujących kolumn poprzez iterację po wymaganej liście.
    missing_columns = [col for col in required_columns if col not in df.columns]

    # Jeżeli choć jednej wymaganej kolumny brakuje,
    # rzucany jest wyjątek z czytelną informacją diagnostyczną.
    if missing_columns:
        raise ValueError(
            f"Brakuje wymaganych kolumn w danych: {missing_columns}"
        )


def prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Przygotowuje finalny DataFrame do ML:
    - uzupełnia braki
    - łączy title i text do jednej kolumny content
    - czyści tekst
    - zostawia tylko content i label
    """

    # Walidacja wejścia: funkcja zakłada obecność trzech kolumn źródłowych.
    validate_columns(df, ["title", "text", "label"])

    # Tworzenie kopii DataFrame, aby nie modyfikować obiektu przekazanego z zewnątrz.
    # To dobra praktyka ograniczająca skutki uboczne.
    df = df.copy()

    # Uzupełnienie braków w tytule pustym stringiem.
    # Dzięki temu operacja konkatenacji nie zwróci NaN.
    df["title"] = df["title"].fillna("")

    # Uzupełnienie braków w treści pustym stringiem.
    df["text"] = df["text"].fillna("")

    # Zbudowanie finalnej kolumny content:
    # 1. połączenie title i text w jeden string,
    # 2. przepuszczenie wyniku przez funkcję czyszczącą clean_text.
    df["content"] = (df["title"] + " " + df["text"]).apply(clean_text)

    # Usunięcie rekordów, które po czyszczeniu nie zawierają żadnej treści.
    # copy() ponownie ogranicza ryzyko problemów typu SettingWithCopyWarning.
    df = df[df["content"].str.len() > 0].copy()

    # Zwracany jest tylko minimalny zestaw kolumn potrzebny do uczenia modelu:
    # content jako cecha wejściowa i label jako etykieta klasy.
    return df[["content", "label"]]


def load_data(fake_path: str | Path, true_path: str | Path) -> pd.DataFrame:
    """
    Wczytuje Fake.csv i True.csv, dodaje etykiety i zwraca gotowy DataFrame.
    label:
    - 0 = fake
    - 1 = real
    """

    # Konwersja przekazanych ścieżek do obiektów Path.
    # Pozwala to obsługiwać zarówno stringi, jak i już gotowe Path.
    fake_path = Path(fake_path)
    true_path = Path(true_path)

    # Jawna walidacja istnienia pliku z danymi fake.
    # W razie problemu użytkownik dostaje czytelny wyjątek.
    if not fake_path.exists():
        raise FileNotFoundError(f"Nie znaleziono pliku: {fake_path}")

    # Jawna walidacja istnienia pliku z danymi true.
    if not true_path.exists():
        raise FileNotFoundError(f"Nie znaleziono pliku: {true_path}")

    # Wczytanie obu plików CSV do osobnych DataFrame.
    fake_df = pd.read_csv(fake_path)
    true_df = pd.read_csv(true_path)

    # Nadanie etykiety klasie fake.
    fake_df["label"] = 0

    # Nadanie etykiety klasie real/true.
    true_df["label"] = 1

    # Połączenie obu zbiorów w jeden wspólny DataFrame.
    # ignore_index=True resetuje indeks po konkatenacji.
    df = pd.concat([fake_df, true_df], ignore_index=True)

    # Przygotowanie danych do dalszego użycia w ML:
    # czyszczenie, łączenie pól i wybór finalnych kolumn.
    df = prepare_dataframe(df)

    # Zwrócenie gotowego DataFrame.
    return df


def split_data(
    df: pd.DataFrame,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
):
    """
    Dzieli dane na train/test i zwraca:
    X_train, X_test, y_train, y_test
    """

    # Walidacja wejścia: funkcja wymaga już danych po preprocessingu,
    # czyli z kolumnami content i label.
    validate_columns(df, ["content", "label"])

    # Wyodrębnienie cech wejściowych.
    X = df["content"]

    # Wyodrębnienie etykiet klas.
    y = df["label"]

    # Podział danych na część treningową i testową.
    # test_size określa proporcję testu,
    # random_state zapewnia powtarzalność,
    # stratify=y zachowuje proporcje klas w obu zbiorach.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    # Zwrócenie czterech obiektów gotowych do trenowania i ewaluacji modelu.
    return X_train, X_test, y_train, y_test


def load_and_split_data(
    fake_path: str | Path,
    true_path: str | Path,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
):
    """
    Skrót: wczytuje dane i od razu robi split.
    """

    # Wczytanie i przygotowanie pełnego zbioru danych.
    df = load_data(fake_path, true_path)

    # Natychmiastowy podział na train/test.
    # Funkcja stanowi wygodny wrapper upraszczający użycie modułu.
    return split_data(
        df=df,
        test_size=test_size,
        random_state=random_state,
    )


if __name__ == "__main__":
    # Ustalenie katalogu bazowego jako folderu, w którym znajduje się aktualny plik.
    BASE_DIR = Path(__file__).resolve().parent

    # Zbudowanie ścieżki do katalogu z danymi.
    DATA_DIR = BASE_DIR / "data"

    # Zbudowanie pełnej ścieżki do pliku Fake.csv.
    fake_path = DATA_DIR / "Fake.csv"

    # Zbudowanie pełnej ścieżki do pliku True.csv.
    true_path = DATA_DIR / "True.csv"

    # Wczytanie i przygotowanie danych.
    df = load_data(fake_path, true_path)

    # Podział danych na zbiory treningowe i testowe.
    X_train, X_test, y_train, y_test = split_data(df)

    # Wyświetlenie kilku pierwszych rekordów przygotowanego zbioru.
    print("=== PODGLĄD DANYCH ===")
    print(df.head())

    # Wyświetlenie informacji o rozmiarach zbiorów.
    print("\n=== ROZMIARY ===")
    print(f"Liczba wszystkich rekordów: {len(df)}")
    print(f"X_train: {len(X_train)}")
    print(f"X_test: {len(X_test)}")
    print(f"y_train: {len(y_train)}")
    print(f"y_test: {len(y_test)}")