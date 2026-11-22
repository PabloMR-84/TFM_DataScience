"""
scripts/predict.py

Orquestador de inferencia batch para generar predicciones de ocupación (+1h, +2h, +3h).
Uso:
    python scripts/predict.py
    python scripts/predict.py --latest-only
    python scripts/predict.py --last-hours 168
"""

from modeling.predict import main

if __name__ == "__main__":
    main()