"""
main_BBDD.py
------------
Orquestador del pipeline ETL / Medallion Architecture (SmartRebalance ZGZ).
Ejecuta la ingesta, limpieza, y el modelado dimensional unificando las capas Bronze, Silver y Gold.
"""

from src.config import RAW_DIR, SILVER_DIR, GOLD_DIR
from src.io import (
    fetch_all_raw_data,
    load_historical_parquets,
    save_silver_df,
    save_gold_df,
)
from src.cleaning import (
    clean_stations,
    clean_rebalancing,
    clean_rankings,
    clean_patterns,
    clean_history,
    clean_historical_snapshots,
)
from src.models_gold import (
    build_dim_stations,
    build_dim_patterns,
    build_dim_calendar,
    build_fact_snapshots,
    build_gold_dataset,
)

def main():
    print("=" * 70)
    print("INICIANDO PIPELINE MODULAR SMARTREBALANCE ZGZ")
    print("=" * 70)

    # 1. INGESTA BRONZE
    print("\n--- 1. Ingesta Capa Bronze ---")
    raw_data = fetch_all_raw_data()
    df_historical_raw = load_historical_parquets()
    print(f" [OK] Histórico Parquet cargado ({len(df_historical_raw)} registros).")

    # 2. TRANSFORMACIÓN SILVER
    print("\n--- 2. Procesamiento Capa Silver ---")
    df_stations_silver = clean_stations(raw_data["stations"])
    df_rebalancing_silver = clean_rebalancing(raw_data["rebalancing"])
    df_rankings_silver = clean_rankings(raw_data["rankings"])
    df_patterns_silver = clean_patterns(raw_data["patterns"])
    df_history_silver = clean_history(raw_data["history"])
    df_historical_silver = clean_historical_snapshots(df_historical_raw)

    save_silver_df(df_stations_silver, "stations_silver.csv")
    save_silver_df(df_rebalancing_silver, "rebalancing_silver.csv")
    save_silver_df(df_rankings_silver, "rankings_silver.csv")
    save_silver_df(df_patterns_silver, "patterns_silver.csv")
    save_silver_df(df_history_silver, "bizi_history_silver.csv")
    save_silver_df(df_historical_silver, "historical_snapshots_silver.parquet")
    print(" [OK] Tablas de la Capa Silver procesadas y guardadas.")

    # 3. MODELADO GOLD
    print("\n--- 3. Construcción Modelo Estrella y Extrapolación Capa Gold ---")
    df_dim_stations = build_dim_stations(df_stations_silver)
    df_dim_patterns = build_dim_patterns(df_patterns_silver)
    df_fact_snapshots = build_fact_snapshots(df_historical_silver)
    df_dim_calendar = build_dim_calendar(df_fact_snapshots, df_history_silver)

    df_gold = build_gold_dataset(
        df_snapshots_gold=df_fact_snapshots,
        df_calendar=df_dim_calendar,
        df_stations_gold=df_dim_stations,
        df_patterns_gold=df_dim_patterns,
        df_rankings_silver=df_rankings_silver,
    )

    save_gold_df(df_dim_stations, "dim_stations.parquet")
    save_gold_df(df_dim_patterns, "dim_patterns.parquet")
    save_gold_df(df_dim_calendar, "dim_calendar.parquet")
    save_gold_df(df_fact_snapshots, "fact_station_snapshots.parquet")
    save_gold_df(df_gold, "df_gold.parquet")

    print(f" [OK] Dataset 'df_gold' exportado: {df_gold.shape}")
    print("\n============================================================")
    print("PIPELINE COMPLETADO CON ÉXITO")
    print("============================================================")

if __name__ == "__main__":
    main()