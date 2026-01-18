# Quantum Machine Learning - SGH Final Project

**Projekt końcowy:** Wprowadzenie do Kwantowego Uczenia Maszynowego – Semestr Zimowy 2025/26

## 📋 Opis projektu

Porównanie modelu **kwantowego (QTSA)** z **klasycznym (MLP)** w zadaniu **binarnej klasyfikacji kierunku rynku** (UP/DOWN) na danych finansowych.

### Główne cechy:
- **Problem:** Klasyfikacja binarna (predykcja kierunku zmian cen akcji)
- **Architektura kwantowa:** Serial Data Re-uploading na 1 kubicie
- **Dane:** Log-returns (stacjonarne cechy) z normalizacją RobustScaler
- **Ticker domyślny:** NVDA (Nvidia)
- **Framework:** PennyLane + PyTorch
- **Metryki:** Accuracy, Confusion Matrix, BCE Loss

## 🚀 Szybki start

### 1. Instalacja zależności

```bash
uv sync
```

### 2. Uruchomienie projektu

**Opcja A: Jako skrypt Python**
```bash
uv run python qtsa_comparison.py
```


## 📊 Struktura projektu

```
Quantum-Machine-Learning-SGH/
├── qtsa_comparison.py          # Główny plik projektu (Python Percent Format)
├── qtsa_comparison.ipynb       # Notebook (automatycznie generowany z .py)
├── pyproject.toml              # Zależności i konfiguracja
├── uv.lock                     # Lock file dla uv
├── README.md                   # Ten plik
└── LICENSE                     # Licencja projektu
```

## 🔧 Konfiguracja

Wszystkie parametry znajdują się na początku pliku `qtsa_comparison.py`:

```python
TICKER = "NVDA"              # Ticker (NVDA, ^GSPC, BTC-USD, etc.)
START_DATE = "2020-01-01"    # Data początkowa
END_DATE = "2025-01-01"      # Data końcowa
WINDOW_SIZE = 20             # Długość okna czasowego (dni)
EPOCHS = 100                 # Liczba epok treningu
BATCH_SIZE = 32              # Rozmiar batcha
LR = 0.01                    # Learning rate
```

## 🎯 Architektura modeli

### QTSA (Quantum Time Series Analysis)
- **1 kubit** z serial data re-uploading
- **63 parametry** (dla window_size=20)
- Obwód: Rot → RX → Rot → RX → ... → Rot → ⟨Z⟩
- Output: P(market UP) = (⟨Z⟩ + 1) / 2

### MLP (Classical Baseline)
- **3 warstwy** fully-connected z Dropout
- **~1,100 parametrów**
- Architektura: 20 → 64 → 32 → 1 (Sigmoid)
- Output: P(market UP) przez Sigmoid

## 📈 Wyniki

Model generuje:
1. **Confusion matrices** dla QTSA i MLP
2. **Wykres porównawczy accuracy**
3. **Historia treningu** (loss i accuracy)
4. **Metryki finalne** wypisane w konsoli

## 🔬 Podejście Financial ML

Projekt stosuje profesjonalne praktyki z finansowego uczenia maszynowego:

- ✅ **Stacjonarne cechy:** Log-returns zamiast surowych cen
- ✅ **RobustScaler:** Odporny na outliers (kryzisy rynkowe)
- ✅ **Klasyfikacja binarna:** Przewidywanie kierunku (nie ceny)
- ✅ **Chronologiczny split:** 80/20 train/test bez shuffle
- ✅ **Metryki biznesowe:** Accuracy, Confusion Matrix

## 📚 Zależności

Główne pakiety (pełna lista w `pyproject.toml`):
- `pennylane` - framework kwantowy
- `torch` - PyTorch dla treningu
- `yfinance` - pobieranie danych finansowych
- `scikit-learn` - preprocessing i metryki
- `matplotlib`, `seaborn` - wizualizacje
- `pandas`, `numpy` - manipulacja danymi
- `tqdm` - progress bars
- `jupytext` - synchronizacja .py ↔ .ipynb

## 💡 Rozszerzenia (opcjonalne)

1. Dodanie RSI jako drugiej cechy
2. Klasyfikacja wieloklasowa (STRONG_UP, UP, NEUTRAL, DOWN, STRONG_DOWN)
3. Symulacja strategii tradingowej
4. Akceleracja GPU (`lightning.gpu`)
5. Testy na różnych tickerach (S&P500, Forex, Crypto)

## 👨‍💻 Autor

Projekt końcowy z kursu **Wprowadzenie do Kwantowego Uczenia Maszynowego**  
SGH - Szkoła Główna Handlowa  
Semestr Zimowy 2025/26

## 📄 Licencja

Zobacz plik [LICENSE](LICENSE)
