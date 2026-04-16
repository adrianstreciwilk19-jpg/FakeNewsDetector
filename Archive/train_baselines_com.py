# Import klasy Path do wygodnej, obiektowej obsługi ścieżek plików i katalogów.
from pathlib import Path

# Import modułu json do zapisu danych wynikowych w formacie JSON.
import json

# Pandas służy tutaj do tworzenia i obróbki tabelarycznych wyników ewaluacji modeli.
import pandas as pd

# Wektoryzator TF-IDF zamienia tekst na reprezentację numeryczną zrozumiałą dla modeli ML.
from sklearn.feature_extraction.text import TfidfVectorizer

# Model regresji logistycznej do klasyfikacji tekstu.
from sklearn.linear_model import LogisticRegression

# Funkcja do walidacji krzyżowej i jednoczesnego liczenia wielu metryk.
from sklearn.model_selection import cross_validate

# Wariant Naive Bayes dobrze radzący sobie m.in. z niezbalansowanymi cechami tekstowymi.
from sklearn.naive_bayes import ComplementNB

# Pipeline pozwala połączyć kolejne etapy przetwarzania w jeden spójny obiekt.
from sklearn.pipeline import Pipeline

# Liniowy Support Vector Classifier – często bardzo mocny baseline dla klasyfikacji tekstu.
from sklearn.svm import LinearSVC

# Import własnych funkcji pomocniczych odpowiedzialnych za wczytanie i podział danych.
from data_utils_com import load_data, split_data


# Ustalenie katalogu bazowego jako folderu, w którym znajduje się aktualny plik.
BASE_DIR = Path(__file__).resolve().parent

# Ścieżka do katalogu z danymi wejściowymi.
DATA_DIR = BASE_DIR / 'data'

# Ścieżka do katalogu, w którym będą zapisywane raporty i wyniki modeli.
REPORTS_DIR = BASE_DIR / 'models'

# Utworzenie katalogu wynikowego, jeśli jeszcze nie istnieje.
# exist_ok=True zapobiega błędowi przy ponownym uruchomieniu programu.
REPORTS_DIR.mkdir(exist_ok=True)

# Pełna ścieżka do pliku z fake newsami.
FAKE_PATH = DATA_DIR / 'Fake.csv'

# Pełna ścieżka do pliku z prawdziwymi newsami.
TRUE_PATH = DATA_DIR / 'True.csv'

# Plik CSV z pełnym rankingiem wyników modeli.
RESULT_CSV_PATH = REPORTS_DIR / 'baseline_results.csv'

# Plik JSON z nazwą najlepszego modelu i podstawowymi metadanymi selekcji.
BEST_MODEL_JSON_PATH = REPORTS_DIR / 'baseline_results.json'


def get_training_split():
    """Pobiera dane z data_utils.py i zwraca tylko split treningowy.
    X_test i y_test są tu celowo ignorowane"""

    # Wczytanie pełnego zbioru danych z obu plików CSV.
    df = load_data(FAKE_PATH, TRUE_PATH)

    # Podział danych na część treningową i testową.
    #X_train, X_test, y_train, y_test = split_data(df)

    # Zwracana jest wyłącznie część treningowa.
    # Test jest świadomie pomijany, ponieważ ten skrypt służy do porównania baseline’ów
    # za pomocą cross-validation na treningu.
    return split_data(df)


def build_bs_models():
    """Tworzy słownik pipelinów do porównania"""

    # Wspólna konfiguracja wektoryzatora TF-IDF dla wszystkich modeli.
    # Dzięki temu porównanie modeli jest bardziej uczciwe,
    # bo różni się tylko klasyfikator, a nie preprocessing.
    tfidf_params = {
        # Usuwanie angielskich stop words.
        # Dla danych anglojęzycznych może poprawić jakość, dla polskich byłoby nietrafione.
        'stop_words': 'english',

        # Ignorowanie słów występujących w więcej niż 70% dokumentów.
        # Pozwala odsiać bardzo częste, mało informacyjne tokeny.
        'max_df': 0.7,

        # Ignorowanie słów bardzo rzadkich – takich, które pojawiły się mniej niż 2 razy.
        'min_df': 2,

        # Użycie zarówno unigramów, jak i bigramów.
        # Model widzi więc nie tylko pojedyncze słowa, ale też pary słów.
        'ngram_range': (1, 2),
    }

    # Słownik modeli bazowych.
    # Każdy wpis to kompletny pipeline: wektoryzacja tekstu + klasyfikator.
    models = {
        'logreg': Pipeline(
            steps=[
                # Krok 1: zamiana tekstu na macierz TF-IDF.
                ('vect', TfidfVectorizer(**tfidf_params)),

                # Krok 2: klasyfikacja przy użyciu regresji logistycznej.
                (
                    'clf',
                    LogisticRegression(
                        # Zwiększony limit iteracji, aby zmniejszyć ryzyko braku zbieżności.
                        max_iter=2000,

                        # Wyrównanie wpływu klas w przypadku niezbalansowanego datasetu.
                        class_weight="balanced",

                        # Ziarno losowości zapewniające powtarzalność działania modelu.
                        random_state=32,
                    ),
                ),
            ]
        ),

        'linearsvc': Pipeline(
            steps=[
                # Krok 1: identyczna wektoryzacja TF-IDF.
                ('vect', TfidfVectorizer(**tfidf_params)),

                # Krok 2: klasyfikacja przy użyciu liniowego SVM.
                (
                    'clf',
                    LinearSVC(
                        # Balansowanie klas – podobnie jak wyżej.
                        class_weight="balanced",

                        # Ustawienie ziarna dla powtarzalności tam, gdzie ma zastosowanie.
                        random_state=32,
                    ),
                ),
            ]
        ),

        'complementnb': Pipeline(
            steps=[
                # Krok 1: wektoryzacja TF-IDF.
                ('vect', TfidfVectorizer(**tfidf_params)),

                # Krok 2: klasyfikacja przy użyciu Complement Naive Bayes.
                # Parametr alpha kontroluje wygładzanie.
                ('clf', ComplementNB(alpha=1.1)),
            ]
        ),
    }

    # Zwrócenie kompletnego słownika gotowych pipeline’ów.
    return models


def evaluate_bs_models(models, X_train, y_train, cv=5):
    """Liczy cross-validation dla każdego modelu i zwraca DataFrame z wynikami"""

    # Definicja metryk, które będą liczone w walidacji krzyżowej.
    scoring = {
        "accuracy": "accuracy",
        "precision_macro": "precision_macro",
        "recall_macro": "recall_macro",
        "f1_macro": "f1_macro",
    }

    # Lista, do której będą trafiały wiersze z wynikami kolejnych modeli.
    rows = []

    # Iteracja po wszystkich modelach przekazanych w słowniku.
    for model_name, pipeline in models.items():
        print(f'\nTrwa ocena modelu: {model_name}')

        # Walidacja krzyżowa:
        # - estimator: pełny pipeline
        # - X, y: dane treningowe
        # - cv=5: 5-fold cross-validation
        # - scoring: zestaw metryk
        # - n_jobs=-1: użycie wszystkich dostępnych rdzeni CPU
        # - return_train_score=False: nie liczymy metryk na train, tylko na walidacji
        # - error_score='raise': w razie błędu przerwij działanie i pokaż wyjątek
        scores = cross_validate(
            estimator=pipeline,
            X=X_train,
            y=y_train,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
            return_train_score=False,
            error_score='raise',
        )

        # Inicjalizacja słownika wynikowego dla jednego modelu.
        row = {'model': model_name}

        # Dla każdej metryki obliczana jest średnia i odchylenie standardowe
        # z wyników uzyskanych na kolejnych foldach.
        for metric_name in scoring:
            values = scores[f'test_{metric_name}']
            row[f'{metric_name}_mean'] = values.mean()
            row[f'{metric_name}_std'] = values.std()

        # Dodatkowo zapisywany jest średni czas trenowania i scoringu.
        row['fit_time_mean'] = scores['fit_time'].mean()
        row['score_time_mean'] = scores['score_time'].mean()

        # Dodanie gotowego rekordu do listy.
        rows.append(row)

    # Zamiana listy słowników na DataFrame.
    results_df = pd.DataFrame(rows)

    # Posortowanie wyników malejąco według średniego F1 macro.
    # To domyślna metryka wyboru najlepszego baseline'u.
    results_df = results_df.sort_values(
        by='f1_macro_mean',
        ascending=False,
    ).reset_index(drop=True)

    # Zwrócenie uporządkowanej tabeli z wynikami.
    return results_df


def select_best_model(results_df, metric='f1_macro_mean'):
    """Wybiera najlepszy model na podstawie wskazanej metryki"""

    # Ponowne sortowanie po wskazanej metryce i pobranie pierwszego wiersza.
    best_row = results_df.sort_values(by=metric, ascending=False).iloc[0]

    # Zwrócenie nazwy zwycięskiego modelu.
    return best_row['model']


def save_results(results_df, best_model_name):
    """
    Zapisuje ranking modeli i nazwę zwycięzcy.
    """

    # Zapis pełnej tabeli wyników do pliku CSV.
    results_df.to_csv(RESULT_CSV_PATH, index=False)

    # Przygotowanie uproszczonego payloadu JSON z informacją o zwycięzcy.
    payload = {
        "best_model": best_model_name,
        "selection_metric": "f1_macro_mean",
        "results_file": str(RESULT_CSV_PATH.name),
    }

    # Zapis danych JSON do pliku w kodowaniu UTF-8.
    # ensure_ascii=False pozwala zachować polskie znaki bez unicode escape.
    with open(BEST_MODEL_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=4, ensure_ascii=False)


def main():
    # Komunikat informacyjny o rozpoczęciu pobierania danych treningowych.
    print("Wczytywanie splitu treningowego...")
    X_train, X_test, y_train, y_test = get_training_split()

    # Budowa zestawu modeli bazowych.
    print("Budowa modeli bazowych...")
    models = build_bs_models()

    # Uruchomienie porównania modeli metodą cross-validation.
    print("Porównywanie modeli przez cross-validation...")
    results_df = evaluate_bs_models(
        models=models,
        X_train=X_train,
        y_train=y_train,
        cv=5,
    )

    # Wybór najlepszego modelu na podstawie domyślnej metryki.
    best_model_name = select_best_model(results_df)

    best_pipeline = models[best_model_name]
    best_pipeline.fit(X_train, y_train)

    test_score = best_pipeline.score(X_test, y_test)
    print(f"Test accuracy najlepszego modelu: {test_score:.4f}")

    # Zapis wyników na dysk.
    save_results(results_df, best_model_name)

    # Wyświetlenie tabeli wyników w konsoli.
    print("\n=== WYNIKI BASELINE ===")
    print(results_df.to_string(index=False))

    # Wyświetlenie nazwy najlepszego baseline’u i lokalizacji zapisanych plików.
    print(f"\nNajlepszy baseline: {best_model_name}")
    print(f"Wyniki zapisane do: {RESULT_CSV_PATH}")
    print(f"Nazwa zwycięzcy zapisana do: {BEST_MODEL_JSON_PATH}")


if __name__ == "__main__":
    # Uruchomienie głównego przepływu programu tylko wtedy,
    # gdy plik wykonywany jest bezpośrednio jako skrypt.
    main()