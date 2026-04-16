# Import klasy Path do wygodnej, przenośnej obsługi ścieżek plików i katalogów.
from pathlib import Path

# Import typu Any do dokładniejszych adnotacji typów w słownikach parametrów i wyników.
from typing import Any

# Import modułu json do odczytu i zapisu danych w formacie JSON.
import json

# joblib służy do serializacji gotowego modelu ML na dysk.
import joblib

# Pandas jest używany do pracy na tabeli wyników GridSearchCV.
import pandas as pd

# Kalibracja klasyfikatora - przydaje się przy modelach, które mają dawać sensowne prawdopodobieństwa.
from sklearn.calibration import CalibratedClassifierCV

# Wektoryzator TF-IDF zamienia tekst na reprezentację numeryczną.
from sklearn.feature_extraction.text import TfidfVectorizer

# Klasyfikator regresji logistycznej.
from sklearn.linear_model import LogisticRegression

# Narzędzia do oceny jakości modelu na zbiorze testowym.
from sklearn.metrics import classification_report, confusion_matrix

# GridSearchCV wykonuje strojenie hiperparametrów metodą przeszukiwania siatki.
# StratifiedKFold daje kontrolowany, powtarzalny podział z zachowaniem proporcji klas.
from sklearn.model_selection import GridSearchCV, StratifiedKFold

# Klasyfikator Complement Naive Bayes.
from sklearn.naive_bayes import ComplementNB

# Pipeline pozwala połączyć preprocessing tekstu i klasyfikator w jeden obiekt.
from sklearn.pipeline import Pipeline

# Liniowy klasyfikator SVM.
from sklearn.svm import LinearSVC

# Import funkcji pomocniczej, która wczytuje dane i od razu wykonuje train/test split.
from data_utils import load_and_split_data


# Stała zapewniająca powtarzalność wyników wszędzie tam, gdzie występuje losowość.
RANDOM_STATE = 42

# Jawnie zdefiniowane etykiety klas używane przy raportowaniu wyników.
CLASS_LABELS = [0, 1]

# Czytelne nazwy klas odpowiadające wartościom 0 i 1.
CLASS_NAMES = ["FAKE", "REAL"]


# Katalog bazowy projektu, czyli folder, w którym znajduje się aktualny plik.
BASE_DIR = Path(__file__).resolve().parent

# Katalog z danymi wejściowymi.
DATA_DIR = BASE_DIR / "data"

# Katalog na raporty tekstowe / wyniki eksperymentów.
REPORTS_DIR = BASE_DIR / "reports"

# Katalog na zapisane modele.
MODELS_DIR = BASE_DIR / "models"

# Utworzenie katalogów wynikowych, jeśli jeszcze nie istnieją.
REPORTS_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

# Ścieżki do plików z danymi wejściowymi.
FAKE_PATH = DATA_DIR / "Fake.csv"
TRUE_PATH = DATA_DIR / "True.csv"

# Ścieżka do pliku JSON z nazwą najlepszego modelu bazowego.
# Zgodnie z twoim założeniem plik wynikowy jest przechowywany w katalogu reports.
BEST_BASELINE_JSON_PATH = REPORTS_DIR / "baseline_results.json"

# Ścieżka do CSV z pełnymi wynikami GridSearchCV.
TUNING_RESULTS_CSV_PATH = REPORTS_DIR / "gridsearch_results.csv"

# Ścieżka do JSON z finalnymi metrykami modelu.
FINAL_METRICS_JSON_PATH = REPORTS_DIR / "final_metrics.json"

# Ścieżka do zapisu finalnego modelu.
MODEL_PATH = MODELS_DIR / "fake_news_model.joblib"


def load_best_baseline_name(json_path: Path) -> str:
    """
    Odczytuje nazwę najlepszego baseline'u z pliku JSON.
    """

    # Walidacja istnienia pliku wejściowego.
    # Jeśli plik nie istnieje, użytkownik dostaje czytelny komunikat, co uruchomić najpierw.
    if not json_path.exists():
        raise FileNotFoundError(
            f"Nie znaleziono pliku: {json_path}\n"
            f"Najpierw uruchom train_baselines.py"
        )

    # Odczyt pliku JSON z kodowaniem UTF-8.
    with open(json_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    # Pobranie wartości pola "best_model".
    best_model = payload.get("best_model")

    # Walidacja zawartości JSON-a.
    if not best_model:
        raise ValueError(
            f"Plik {json_path} nie zawiera pola 'best_model'."
        )

    # Zwrócenie nazwy zwycięskiego baseline'u.
    return best_model


def build_pipeline(model_name: str) -> Pipeline:
    """
    Buduje pipeline dla zwycięskiego modelu.
    """

    # Bazowa konfiguracja TF-IDF ustawiona jawnie dla czytelności.
    # Część z tych parametrów może zostać później nadpisana przez GridSearchCV.
    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 1),
        min_df=2,
        max_df=0.9,
    )

    # Wybór klasyfikatora zależnie od nazwy zwycięskiego baseline'u.
    if model_name == "logreg":
        clf = LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        )

    elif model_name == "linearsvc":
        # LinearSVC opakowany kalibracją, aby model dawał lepiej skalibrowane przewidywania.
        base_svc = LinearSVC(
            class_weight="balanced",
            random_state=RANDOM_STATE,
        )
        clf = CalibratedClassifierCV(
            estimator=base_svc,
            method="sigmoid",
            cv=3,
        )

    elif model_name == "complementnb":
        # ComplementNB również opakowany kalibracją.
        base_nb = ComplementNB()
        clf = CalibratedClassifierCV(
            estimator=base_nb,
            method="sigmoid",
            cv=3,
        )

    # Zabezpieczenie przed nieobsługiwaną nazwą modelu.
    else:
        raise ValueError(f"Nieobsługiwany model: {model_name}")

    # Złożenie pełnego pipeline'u:
    # 1. wektoryzacja tekstu,
    # 2. klasyfikacja.
    return Pipeline([
        ("vect", vectorizer),
        ("clf", clf),
    ])


def get_param_grid(model_name: str) -> dict[str, list[Any]]:
    """
    Zwraca siatkę parametrów pod GridSearchCV.
    Parametry są małe i sensowne na start.
    """

    # Część wspólna siatki hiperparametrów dla TF-IDF.
    common_vect_grid: dict[str, list[Any]] = {
        # Lista stop words dla języka angielskiego.
        "vect__stop_words": ["english"],

        # Sprawdzane są unigramy i układ unigramy + bigramy.
        "vect__ngram_range": [(1, 1), (1, 2)],

        # Minimalna liczba wystąpień tokena w korpusie.
        "vect__min_df": [2, 5],

        # Maksymalny udział dokumentów, w których może wystąpić token.
        "vect__max_df": [0.7, 0.9],
    }

    # Dla regresji logistycznej strojony jest parametr regularyzacji C.
    if model_name == "logreg":
        return {
            **common_vect_grid,
            "clf__C": [0.5, 1.0, 2.0, 5.0],
        }

    # Dla skalibrowanego LinearSVC trzeba zejść do estymatora wewnętrznego.
    if model_name == "linearsvc":
        return {
            **common_vect_grid,
            "clf__estimator__C": [0.5, 1.0, 2.0, 5.0],
        }

    # Dla skalibrowanego ComplementNB strojony jest parametr alpha.
    if model_name == "complementnb":
        return {
            **common_vect_grid,
            "clf__estimator__alpha": [0.1, 0.5, 1.0, 2.0],
        }

    # Zabezpieczenie dla nieznanej nazwy modelu.
    raise ValueError(f"Nieobsługiwany model: {model_name}")


def run_grid_search(
    model_name: str,
    X_train: pd.Series,
    y_train: pd.Series,
    cv: int = 5,
) -> GridSearchCV:
    """
    Strojenie modelu przez GridSearchCV.
    """

    # Budowa pipeline'u dla wskazanego modelu.
    pipeline = build_pipeline(model_name)

    # Pobranie siatki parametrów odpowiedniej dla wybranego modelu.
    param_grid = get_param_grid(model_name)

    # Zestaw metryk liczonych w walidacji krzyżowej.
    scoring = {
        "accuracy": "accuracy",
        "precision_macro": "precision_macro",
        "recall_macro": "recall_macro",
        "f1_macro": "f1_macro",
    }

    # Strategia CV z zachowaniem proporcji klas i pełną powtarzalnością.
    cv_strategy = StratifiedKFold(
        n_splits=cv,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    # Konfiguracja GridSearchCV:
    # - estimator: pipeline,
    # - param_grid: zestaw kombinacji parametrów,
    # - scoring: wiele metryk jednocześnie,
    # - refit="f1_macro": po zakończeniu wybierz i przetrenuj najlepszy wariant wg F1 macro,
    # - cv: kontrolowany podział walidacyjny,
    # - n_jobs=-1: użycie wszystkich rdzeni CPU,
    # - verbose=2: bardziej szczegółowy log postępu,
    # - return_train_score=False: bez metryk treningowych,
    # - error_score="raise": błędy mają zatrzymać wykonanie.
    grid = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring=scoring,
        refit="f1_macro",
        cv=cv_strategy,
        n_jobs=-1,
        verbose=2,
        return_train_score=False,
        error_score="raise",
    )

    # Właściwe uruchomienie strojenia.
    grid.fit(X_train, y_train)

    # Zwrócenie obiektu GridSearchCV zawierającego wszystkie wyniki i najlepszy model.
    return grid


def evaluate_on_test(
    best_model: Pipeline,
    X_test: pd.Series,
    y_test: pd.Series,
) -> tuple[dict[str, Any], str, list[list[int]]]:
    """
    Końcowa ocena najlepszego modelu na zbiorze testowym.
    """

    # Predykcja klas na zbiorze testowym.
    y_pred = best_model.predict(X_test)

    # Raport metryk w formacie słownikowym - wygodny do późniejszego zapisu do JSON.
    report_dict = classification_report(
        y_test,
        y_pred,
        labels=CLASS_LABELS,
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )

    # Ten sam raport w formie tekstowej - wygodny do wydruku w konsoli.
    report_text = classification_report(
        y_test,
        y_pred,
        labels=CLASS_LABELS,
        target_names=CLASS_NAMES,
        zero_division=0,
    )

    # Macierz pomyłek pokazująca liczbę trafień i błędnych klasyfikacji.
    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=CLASS_LABELS,
    ).tolist()

    # Zwrócenie wszystkich artefaktów ewaluacji potrzebnych dalej w pipeline.
    return report_dict, report_text, cm


def save_grid_results(grid: GridSearchCV, output_path: Path) -> None:
    """
    Zapisuje pełne wyniki GridSearchCV do CSV.
    """

    # cv_results_ to słownik zawierający szczegółowe wyniki wszystkich kombinacji parametrów.
    results_df = pd.DataFrame(grid.cv_results_)

    # Sortowanie po randze dla metryki refit, czyli tutaj rank_test_f1_macro.
    results_df = results_df.sort_values(by="rank_test_f1_macro")

    # Zapis wyników do pliku CSV.
    results_df.to_csv(output_path, index=False)


def save_final_metrics(
    model_name: str,
    grid: GridSearchCV,
    report_dict: dict[str, Any],
    cm: list[list[int]],
    output_path: Path,
    train_size: int,
    test_size: int,
) -> None:
    """
    Zapisuje najważniejsze metadane i metryki końcowe do JSON.
    """

    # Złożenie końcowego payloadu z najważniejszymi informacjami o modelu.
    payload = {
        "selected_baseline": model_name,
        "selection_metric": "f1_macro",
        "best_params": grid.best_params_,
        "best_cv_score_f1_macro": grid.best_score_,
        "test_accuracy": report_dict["accuracy"],
        "test_accuracy_percent": round(report_dict["accuracy"] * 100, 2),
        "test_report": report_dict,
        "confusion_matrix": cm,
        "label_mapping": {
            "0": "FAKE",
            "1": "REAL",
        },
        "train_size": train_size,
        "test_size": test_size,
        "saved_model_path": MODEL_PATH.name,
    }

    # Zapis wyników końcowych do JSON.
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=4, ensure_ascii=False)


def main() -> None:
    # Odczyt informacji o najlepszym baseline'ie z poprzedniego etapu pipeline'u.
    print("Odczyt zwycięskiego baseline'u...")
    best_model_name = load_best_baseline_name(BEST_BASELINE_JSON_PATH)
    print(f"Najlepszy baseline: {best_model_name}")

    # Wczytanie danych i podział na train/test.
    print("Wczytywanie danych i split...")
    X_train, X_test, y_train, y_test = load_and_split_data(
        fake_path=FAKE_PATH,
        true_path=TRUE_PATH,
    )

    # Uruchomienie strojenia hiperparametrów dla wybranego modelu.
    print("Uruchamianie GridSearchCV...")
    grid = run_grid_search(
        model_name=best_model_name,
        X_train=X_train,
        y_train=y_train,
        cv=5,
    )

    # Wyświetlenie najlepszego zestawu parametrów znalezionego przez GridSearchCV.
    print("\n=== BEST PARAMS ===")
    print(grid.best_params_)

    # Wyświetlenie najlepszego wyniku walidacji krzyżowej.
    print("\n=== BEST CV SCORE (f1_macro) ===")
    print(f"{grid.best_score_:.4f}")

    # Pobranie najlepszego, już przetrenowanego modelu.
    best_model = grid.best_estimator_

    # Ocena końcowa na odłożonym zbiorze testowym.
    print("\nOcena na zbiorze testowym...")
    report_dict, report_text, cm = evaluate_on_test(
        best_model=best_model,
        X_test=X_test,
        y_test=y_test,
    )

    # Wydruk raportu klasyfikacji.
    print("\n=== CLASSIFICATION REPORT ===")
    print(report_text)

    # Wydruk macierzy pomyłek.
    print("\n=== CONFUSION MATRIX ===")
    print(cm)

    # Wydruk accuracy zarówno jako liczby z zakresu 0-1, jak i w procentach.
    print("\n=== TEST ACCURACY ===")
    print(f"{report_dict['accuracy']:.4f}")
    print(f"{report_dict['accuracy'] * 100:.2f}%")

    # Zapis pełnych wyników strojenia do CSV.
    print("\nZapisywanie wyników GridSearchCV...")
    save_grid_results(grid, TUNING_RESULTS_CSV_PATH)

    # Zapis najważniejszych metryk końcowych do JSON.
    print("Zapisywanie metryk końcowych...")
    save_final_metrics(
        model_name=best_model_name,
        grid=grid,
        report_dict=report_dict,
        cm=cm,
        output_path=FINAL_METRICS_JSON_PATH,
        train_size=len(X_train),
        test_size=len(X_test),
    )

    # Zapis finalnego modelu na dysk.
    print(f"Zapisywanie modelu do: {MODEL_PATH}")
    joblib.dump(best_model, MODEL_PATH)

    # Komunikaty końcowe.
    print("\nGotowe.")
    print(f"Wyniki grid search: {TUNING_RESULTS_CSV_PATH}")
    print(f"Metryki końcowe:   {FINAL_METRICS_JSON_PATH}")
    print(f"Model:             {MODEL_PATH}")


if __name__ == "__main__":
    # Uruchomienie głównego przepływu programu tylko przy bezpośrednim odpaleniu pliku.
    main()