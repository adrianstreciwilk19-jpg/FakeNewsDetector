# Import klasy Path z modułu pathlib.
# Dzięki temu można wygodnie operować na ścieżkach do plików i folderów
# w sposób bardziej czytelny niż na zwykłych stringach.
from pathlib import Path

# Import biblioteki joblib.
# Służy ona tutaj do wczytania wcześniej zapisanego modelu uczenia maszynowego z pliku .joblib.
import joblib

# BASE_DIR ustala katalog, w którym znajduje się aktualnie uruchamiany plik predict.py.
# __file__ oznacza ścieżkę do tego pliku,
# resolve() zamienia ją na pełną ścieżkę absolutną,
# parent pobiera katalog nadrzędny, czyli folder z tym skryptem.
BASE_DIR = Path(__file__).resolve().parent

# MODEL_PATH buduje pełną ścieżkę do pliku modelu.
# Program zakłada, że model znajduje się w folderze "models"
# i ma nazwę "fake_news_detector.joblib".
MODEL_PATH = BASE_DIR / 'models' / 'fake_news_detector.joblib'

# Główna funkcja programu.
# Została nazwana _main(), a adnotacja -> None oznacza,
# że funkcja nic nie zwraca.
def _main() -> None:
    # Sprawdzenie, czy plik modelu istnieje pod wskazaną ścieżką.
    # Jeśli nie istnieje, program przerywa działanie i zgłasza błąd FileNotFoundError.
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            # Komunikat błędu zawiera dokładną ścieżkę, gdzie program próbował znaleźć model.
            f'Nie znaleziono modelu: {MODEL_PATH}\n'
            # Dodatkowa informacja dla użytkownika,
            # że najpierw trzeba wytrenować model przez uruchomienie train_model.py.
            'Najpierw uruchom train_model.py'
        )
    
    # Wczytanie wytrenowanego modelu z pliku .joblib do zmiennej model.
    model = joblib.load(MODEL_PATH)

    # Wyświetlenie użytkownikowi informacji, że może wpisać tekst do analizy.
    print('Wpisz tekst do analizy.')

    # Pobranie tekstu od użytkownika z konsoli.
    # input('Tekst: ') pokazuje napis "Tekst: ",
    # a strip() usuwa zbędne spacje i znaki białe z początku i końca wpisu.
    user_text = input('Tekst: ').strip()

    # Sprawdzenie, czy użytkownik rzeczywiście coś wpisał.
    # Jeśli po usunięciu spacji tekst jest pusty, program wyświetla komunikat
    # i kończy działanie funkcji przez return.
    if not user_text:
        print('Nie podano tekstu')
        return
    
    # Wykonanie predykcji na podstawie wpisanego tekstu.
    # Model oczekuje listy tekstów, dlatego przekazujemy [user_text].
    # Wynik predict() również zwraca listę/tablicę, więc [0] pobiera pierwszy element.
    pred = model.predict([user_text])[0]

    # Sprawdzenie, czy model posiada metodę predict_proba.
    # Nie każdy model ją ma, więc ten warunek zabezpiecza program przed błędem.
    if hasattr(model, 'predict_proba'):
        # Obliczenie prawdopodobieństw klas dla wpisanego tekstu.
        # Wynik dla jednego tekstu to tablica dwóch wartości,
        # np. [prawdopodobieństwo_FAKE, prawdopodobieństwo_REAL].
        prob = model.predict_proba([user_text])[0]

        # Pobranie pierwszego elementu tablicy jako prawdopodobieństwo klasy FAKE.
        fake_prob = prob[0]

        # Pobranie drugiego elementu tablicy jako prawdopodobieństwo klasy REAL.
        real_prob = prob[1]
    else:
        # Jeśli model nie obsługuje predict_proba,
        # ustawiamy wartości prawdopodobieństw na None.
        fake_prob = None
        real_prob = None

    # Zamiana numerycznej predykcji modelu na etykietę tekstową.
    # Jeśli model zwróci 1, wynik interpretowany jest jako REAL.
    # W przeciwnym razie jako FAKE.
    label = 'REAL' if pred == 1 else 'FAKE'

    # Wyświetlenie nagłówka sekcji z wynikiem.
    print('\n===WYNIK===')

    # Wyświetlenie końcowej klasyfikacji tekstu.
    print(f'Klasyfikacja: {label}')

    # Sprawdzenie, czy prawdopodobieństwa zostały wyznaczone.
    # Jeśli tak, zostaną wypisane z dokładnością do 4 miejsc po przecinku.
    if fake_prob is not None and real_prob is not None:
        print(f'Prawdopodobieństwo FAKE: {fake_prob:.4f}')
        print(f'Prawdopodobieństwo REAL: {real_prob:.4f}')

# Ten warunek sprawia, że funkcja _main() uruchomi się tylko wtedy,
# gdy plik zostanie uruchomiony bezpośrednio jako skrypt.
# Jeśli ten plik zostałby zaimportowany do innego programu,
# _main() nie wykona się automatycznie.
if __name__ =='__main__':
    _main()
