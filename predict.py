# Import klasy Path do wygodnej i przenośnej obsługi ścieżek plików.
from pathlib import Path

# joblib służy do wczytywania wcześniej zapisanego modelu ML z pliku .joblib.
import joblib

# Import funkcji czyszczącej tekst, aby preprocessing wejścia użytkownika
# był zgodny z preprocessingiem użytym podczas trenowania modelu.
from data_utils import clean_text


# Ustalenie katalogu bazowego jako folderu, w którym znajduje się aktualny plik.
BASE_DIR = Path(__file__).resolve().parent

# Zbudowanie pełnej ścieżki do finalnego modelu zapisanego po tuningu.
MODEL_PATH = BASE_DIR / "models" / "fake_news_model.joblib"

# Próg pewności predykcji.
# Jeśli którakolwiek z klas osiągnie co najmniej tę wartość, model uznaje predykcję za pewną.
CONFIDENCE_THRESHOLD = 0.70

# Jawne mapowanie etykiet liczbowych na nazwy klas.
LABEL_MAP = {
    0: "FAKE",
    1: "REAL",
}


def _extract_probabilities(model, processed_text: str) -> tuple[float | None, float | None]:
    """
    Zwraca prawdopodobieństwa dla klas FAKE i REAL.
    Jeśli model nie wspiera predict_proba, zwraca (None, None).
    """

    # Nie każdy model scikit-learn wspiera predict_proba.
    # Jeśli metoda nie istnieje, nie próbujemy liczyć prawdopodobieństw.
    if not hasattr(model, "predict_proba"):
        return None, None

    # Pobranie wektora prawdopodobieństw dla jednej próbki tekstowej.
    probabilities = model.predict_proba([processed_text])[0]

    # classes_ przechowuje kolejność klas używaną przez model.
    # Nie należy zakładać "na sztywno", że indeks 0 to FAKE, a 1 to REAL.
    if not hasattr(model, "classes_"):
        return None, None

    # Zbudowanie mapowania: klasa -> prawdopodobieństwo.
    class_to_prob = dict(zip(model.classes_, probabilities))

    # Pobranie prawdopodobieństw dla klas 0 i 1.
    fake_prob = class_to_prob.get(0)
    real_prob = class_to_prob.get(1)

    # Zwrócenie obu wartości.
    return fake_prob, real_prob


def main() -> None:
    # Walidacja istnienia pliku modelu przed próbą jego wczytania.
    # Dzięki temu użytkownik dostaje czytelny błąd zamiast mało zrozumiałego wyjątku z joblib.
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Nie znaleziono modelu: {MODEL_PATH}\n"
            "Najpierw uruchom tune_and_save.py"
        )

    # Wczytanie gotowego modelu z dysku.
    model = joblib.load(MODEL_PATH)

    # Komunikat dla użytkownika rozpoczynający interakcję w trybie CLI.
    print("Wpisz tekst do analizy.")

    # Pobranie tekstu od użytkownika i usunięcie białych znaków z początku i końca.
    user_text = input("Tekst: ").strip()

    # Zabezpieczenie przed pustym wejściem.
    if not user_text:
        print("Nie podano tekstu.")
        return

    # Czyszczenie wejścia użytkownika tak samo jak podczas przygotowania danych treningowych.
    # To bardzo ważne dla spójności działania modelu.
    processed_text = clean_text(user_text)

    # Jeżeli po czyszczeniu tekst stał się pusty, predykcja nie ma sensu.
    if not processed_text:
        print("Tekst po czyszczeniu jest pusty i nie może zostać przeanalizowany.")
        return

    # Predykcja klasy dla pojedynczego tekstu.
    # Model oczekuje wejścia iterowalnego, więc tekst jest opakowany w listę.
    pred = model.predict([processed_text])[0]

    # Zamiana etykiety liczbowej na czytelną nazwę klasy.
    # Jeśli z jakiegoś powodu pojawi się nieznana etykieta, zostanie pokazana jako string.
    label = LABEL_MAP.get(pred, str(pred))

    # Próba pobrania prawdopodobieństw dla obu klas.
    fake_prob, real_prob = _extract_probabilities(model, processed_text)

    # Nagłówek sekcji wynikowej.
    print("\n=== WYNIK ===")

    # Jeśli model nie zwraca prawdopodobieństw, wypisujemy tylko klasę.
    if fake_prob is None or real_prob is None:
        print(f"Klasyfikacja: {label}")
        print("Ten model nie udostępnia predict_proba, więc brak wartości pewności.")
        return

    # Sprawdzenie, czy model osiągnął wymagany próg pewności
    # dla którejkolwiek z dwóch klas.
    is_confident = (
        fake_prob >= CONFIDENCE_THRESHOLD
        or real_prob >= CONFIDENCE_THRESHOLD
    )

    # Jeśli model jest wystarczająco pewny, pokazujemy końcową klasyfikację.
    if is_confident:
        print(f"Klasyfikacja: {label}")
    else:
        # Jeśli nie osiągnięto progu pewności, sygnalizujemy niepewną predykcję.
        print("Klasyfikacja: Model nie jest pewny predykcji")

    # Wyświetlenie prawdopodobieństw w dwóch formatach:
    # - jako wartość z zakresu 0-1,
    # - jako procent.
    print(f"Prawdopodobieństwo FAKE: {fake_prob:.4f} ({fake_prob * 100:.2f}%)")
    print(f"Prawdopodobieństwo REAL: {real_prob:.4f} ({real_prob * 100:.2f}%)")


# Uruchomienie funkcji głównej tylko wtedy,
# gdy plik wykonywany jest bezpośrednio jako skrypt.
if __name__ == "__main__":
    main()