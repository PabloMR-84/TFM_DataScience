"""
scripts/predict_model.py

Orquestador de inferencia batch para generar predicciones de ocupación (+1h, +2h, +3h)
y banderas de alerta de vaciado/saturación calibradas.
Uso:
    python scripts/predict_model.py
    python scripts/predict_model.py --latest-only
    python scripts/predict_model.py --last-hours 168
"""

from modeling.predict import main

if __name__ == "__main__":
    main()