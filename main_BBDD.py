"""
main_BBDD.py
------------
Orquestador principal del pipeline ETL / Arquitectura Medallón (SmartRebalance ZGZ).
Ejecuta la ingesta en Bronze, las transformaciones y validaciones de rango en Silver,
los controles de calidad y cobertura de capturas, y el modelado dimensional unificado en Gold
previniendo la fuga de información temporal (Data Leakage) e identificando datos estimaciones (cold start).
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
from src.utils import (
    auditar_merge,
    evaluar_cobertura_scheduler,
)


def main():
    print("=" * 70)
    print("INICIANDO PIPELINE MODULAR SMARTREBALANCE ZGZ (BRONZE - SILVER - GOLD)")
    print("=" * 70)

    # ==========================================================================
    # 1. INGESTA CAPA BRONZE
    # ==========================================================================
    print("\n--- 1. Ingesta Capa Bronze ---")
    raw_data = fetch_all_raw_data()
    df_historical_raw = load_historical_parquets()
    print(f" [OK] Histórico Parquet cargado ({len(df_historical_raw):,} registros).")

    # ==========================================================================
    # 2. TRANSFORMACIÓN Y VALIDACIONES CAPA SILVER
    # ==========================================================================
    print("\n--- 2. Procesamiento y Limpieza Capa Silver ---")
    df_stations_silver = clean_stations(raw_data["stations"])
    df_rebalancing_silver = clean_rebalancing(raw_data["rebalancing"])
    df_rankings_silver = clean_rankings(raw_data["rankings"])
    df_patterns_silver = clean_patterns(raw_data["patterns"])
    df_history_silver = clean_history(raw_data["history"])
    df_historical_silver = clean_historical_snapshots(df_historical_raw)

    # Persistencia en data/silver/
    save_silver_df(df_stations_silver, "stations_silver.csv")
    save_silver_df(df_rebalancing_silver, "rebalancing_silver.csv")
    save_silver_df(df_rankings_silver, "rankings_silver.csv")
    save_silver_df(df_patterns_silver, "patterns_silver.csv")
    save_silver_df(df_history_silver, "bizi_history_silver.csv")
    save_silver_df(df_historical_silver, "historical_snapshots_silver.parquet")
    print(" [OK] Tablas de la Capa Silver procesadas, validadas y guardadas.")

    # ==========================================================================
    # 3. CONTROLES DE CALIDAD Y COBERTURA DE CAPTURAS
    # ==========================================================================
    print("\n--- 3. Auditoría de Cobertura del Scheduler ---")
    if not df_historical_silver.empty:
        evaluar_cobertura_scheduler(
            df_snapshots=df_historical_silver,
            frecuencia_minutos=10,
            timestamp_col="timestamp",
            station_id_col="station_id",
        )

    # ==========================================================================
    # 4. MODELADO DIMENSIONAL Y EXTRAPOLACIÓN CAPA GOLD
    # ==========================================================================
    print("\n--- 4. Construcción Modelo Estrella y Extrapolación Capa Gold ---")
    df_dim_stations = build_dim_stations(df_stations_silver)
    df_dim_patterns = build_dim_patterns(df_patterns_silver)
    df_fact_snapshots = build_fact_snapshots(df_historical_silver)
    df_dim_calendar = build_dim_calendar(df_fact_snapshots, df_history_silver)

    # Generar dataset Gold integrado con bandera is_estimated y filtro temporal pre-leakage
    df_gold = build_gold_dataset(
        df_snapshots_gold=df_fact_snapshots,
        df_calendar=df_dim_calendar,
        df_stations_gold=df_dim_stations,
        df_patterns_gold=df_dim_patterns,
        df_rankings_silver=df_rankings_silver,
    )

    # Auditoría de cardinalidad del dataset final
    auditar_merge(
        df_izq=df_fact_snapshots,
        df_der=df_dim_calendar,
        df_resultado=df_gold,
        nombre_union="Integración Final Capa Gold",
    )

    # Persistencia en data/gold/
    save_gold_df(df_dim_stations, "dim_stations.parquet")
    save_gold_df(df_dim_patterns, "dim_patterns.parquet")
    save_gold_df(df_dim_calendar, "dim_calendar.parquet")
    save_gold_df(df_fact_snapshots, "fact_station_snapshots.parquet")
    save_gold_df(df_gold, "df_gold.parquet")

    print(f"\n [OK] Dataset unificado 'df_gold' exportado exitosamente.")
    print(f"      - Dimensiones finales: {df_gold.shape[0]:,} filas x {df_gold.shape[1]} columnas")
    if "is_estimated" in df_gold.columns:
        n_obs = (df_gold["is_estimated"] == 0).sum()
        n_est = (df_gold["is_estimated"] == 1).sum()
        print(f"      - Desglose de observaciones: {n_obs:,} reales (0) / {n_est:,} estimadas (1)")

    print("\n============================================================")
    print("PIPELINE COMPLETADO CON ÉXITO")
    print("============================================================")


if __name__ == "__main__":
    main()