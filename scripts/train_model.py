"""
scripts/train_model.py

Orquestador de entrenamiento del modelo predictivo LightGBM multi-horizonte.
Uso:
    python scripts/train_model.py
    python scripts/train_model.py --gold-dir data/gold --output-dir models/lgbm_v1
"""

from modeling.train import main

if __name__ == "__main__":
    main()