from pathlib import Path  # Pathlib zapewnia przenośną, obiektową obsługę ścieżek plików i katalogów.
import joblib  # joblib służy tutaj do deserializacji wcześniej zapisanego modelu ML z pliku .joblib.

# Wyznaczenie katalogu bazowego skryptu.
# __file__ -> ścieżka do aktualnie wykonywanego pliku,
# .resolve() -> zamiana na ścieżkę bezwzględną,
# .parent -> katalog zawierający ten plik.
# Dzięki temu dalsze odwołania do plików modelu nie zależą od bieżącego katalogu uruchomienia terminala.
BASE_DIR = Path(__file__).resolve().parent

# Złożenie docelowej ścieżki do wytrenowanego modelu.
# Operator / w Path tworzy kolejne segmenty ścieżki w sposób niezależny od systemu operacyjnego.
MODEL_PATH = BASE_DIR / 'models' / 'fake_news_detector.joblib'

# Główna logika programu została zamknięta w osobnej funkcji,
# co poprawia czytelność kodu i pozwala kontrolować moment jej wywołania.
def _main() -> None:
    # Walidacja istnienia pliku modelu przed próbą jego wczytania.
    # Taki warunek pozwala przerwać działanie programu w kontrolowany sposób,
    # zamiast dopuścić do mniej czytelnego błędu podczas joblib.load(...).
    if not MODEL_PATH.exists():
        # Rzucenie wyjątku z precyzyjnym komunikatem diagnostycznym.
        # Użytkownik od razu dostaje informację, którego pliku brakuje
        # i jaki krok powinien wykonać wcześniej.
        raise FileNotFoundError(
            f'Nie znaleziono modelu: {MODEL_PATH}\n'
            'Najpierw uruchom train_model.py'
        )
    
    # Deserializacja modelu z pliku do pamięci operacyjnej.
    # Od tego momentu obiekt "model" udostępnia metody inferencyjne,
    # np. predict() oraz w niektórych przypadkach predict_proba().
    model = joblib.load(MODEL_PATH)

    # Komunikat informujący użytkownika, że program oczekuje na dane wejściowe.
    print('Wpisz tekst do analizy.')

    # Pobranie tekstu od użytkownika ze standardowego wejścia.
    # .strip() usuwa białe znaki z początku i końca łańcucha,
    # dzięki czemu np. samo wciśnięcie spacji nie zostanie potraktowane jako poprawny tekst.
    user_text = input('Tekst: ').strip()

    # Walidacja pustego wejścia.
    # Jeśli po oczyszczeniu tekst jest pusty, program kończy działanie tej funkcji wcześniej,
    # żeby nie przekazywać niepoprawnych danych do modelu.
    if not user_text:
        print('Nie podano tekstu')
        return
    
    # Wykonanie predykcji klasy dla pojedynczego rekordu wejściowego.
    # Model oczekuje danych wejściowych w postaci iterowalnej kolekcji tekstów,
    # dlatego pojedynczy string jest opakowany w listę [user_text].
    # Wynik predict(...) jest zwykle tablicą / listą etykiet,
    # więc [0] pobiera predykcję dla pierwszego i jedynego elementu wejściowego.
    pred = model.predict([user_text])[0]

    # Nie każdy model implementuje metodę predict_proba().
    # hasattr(...) zabezpiecza kod przed błędem AttributeError
    # i pozwala warunkowo wyliczyć prawdopodobieństwa klas tylko wtedy,
    # gdy dany estymator faktycznie to wspiera.
    if hasattr(model, 'predict_proba'):
        # Zwrócone prawdopodobieństwa dotyczą pojedynczej próbki,
        # więc ponownie pobierany jest pierwszy wiersz wyniku.
        prob = model.predict_proba([user_text])[0]

        # Założenie mapowania klas:
        # indeks 0 -> FAKE,
        # indeks 1 -> REAL.
        # To działa poprawnie tylko wtedy, gdy model został wytrenowany
        # z taką właśnie kolejnością klas.
        fake_prob = prob[0]
        real_prob = prob[1]
    else:
        # Gdy model nie udostępnia prawdopodobieństw,
        # zmienne są jawnie ustawiane na None,
        # co upraszcza późniejszą logikę warunkowego wyświetlania wyniku.
        fake_prob = None
        real_prob = None

    # Mapowanie numerycznej etykiety modelu na czytelną etykietę tekstową.
    # Przyjęta logika:
    # 1 -> REAL,
    # każda inna wartość -> FAKE.
    # To oznacza, że poprawność tej linii zależy od sposobu zakodowania klas na etapie treningu.
    label = 'REAL' if pred == 1 else 'FAKE'

    # Sekcja prezentacji wyników użytkownikowi.
    print('\n===WYNIK===')
    print(f'Klasyfikacja: {label}')

    # Warunkowe wyświetlenie prawdopodobieństw tylko wtedy,
    # gdy zostały wcześniej faktycznie wyznaczone.
    if fake_prob is not None and real_prob is not None:
        # Formatowanie :.4f ogranicza liczbę miejsc po przecinku do czterech,
        # poprawiając czytelność wyniku bez utraty podstawowej informacji.
        print(f'Prawdopodobieństwo FAKE: {fake_prob:.4f}')
        print(f'Prawdopodobieństwo REAL: {real_prob:.4f}')

# Klasyczny punkt wejścia dla skryptu uruchamianego bezpośrednio.
# Warunek zabezpiecza przed automatycznym wykonaniem _main()
# w sytuacji, gdy plik zostałby zaimportowany jako moduł do innego programu.
if __name__ =='__main__':
    _main()
