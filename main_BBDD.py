"""
main_BBDD.py
------------
Orquestador principal del pipeline ETL / Arquitectura Medallón (SmartRebalance ZGZ).
Ejecuta la ingesta en Bronze, las transformaciones y validaciones de rango en Silver,
el modelado dimensional unificado en Gold (con diagnóstico explícito de Cold Start)
y los controles de cardinalidad y cobertura operativa de capturas.
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

    save_silver_df(df_stations_silver, "stations_silver.csv")
    save_silver_df(df_rebalancing_silver, "rebalancing_silver.csv")
    save_silver_df(df_rankings_silver, "rankings_silver.csv")
    save_silver_df(df_patterns_silver, "patterns_silver.csv")
    save_silver_df(df_history_silver, "bizi_history_silver.csv")
    save_silver_df(df_historical_silver, "historical_snapshots_silver.parquet")
    print(" [OK] Tablas de la Capa Silver procesadas, validadas y guardadas.")

    # ==========================================================================
    # 3. MODELADO DIMENSIONAL Y EXTRAPOLACIÓN CAPA GOLD
    # ==========================================================================
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

    auditar_merge(
        df_izq=df_fact_snapshots,
        df_der=df_dim_calendar,
        df_resultado=df_gold,
        nombre_union="Dataset Gold Final Integration",
        strict=True,
    )

    # ==========================================================================
    # 4. AUDITORÍA DE COBERTURA OPERATIVA DEL SCHEDULER
    # ==========================================================================
    print("\n--- 4. Auditoría de Cobertura del Scheduler ---")
    if not df_fact_snapshots.empty:
        df_cobertura = evaluar_cobertura_scheduler(
            df_snapshots=df_fact_snapshots,
            df_stations=df_dim_stations,
            df_calendar=df_dim_calendar,
            frecuencia_minutos=10,
            timestamp_col="timestamp",
            station_id_col="station_id",
        )
        save_gold_df(df_cobertura, "coverage_report.parquet")
        save_gold_df(df_cobertura, "coverage_report.csv")

    # ==========================================================================
    # 5. PERSISTENCIA FINAL EN CAPA GOLD
    # ==========================================================================
    save_gold_df(df_dim_stations, "dim_stations.parquet")
    save_gold_df(df_dim_patterns, "dim_patterns.parquet")
    save_gold_df(df_dim_calendar, "dim_calendar.parquet")
    save_gold_df(df_fact_snapshots, "fact_station_snapshots.parquet")
    save_gold_df(df_gold, "df_gold.parquet")
    print(" [OK] Tablas dimensionales y dataset Gold guardados exitosamente en data/gold/.")

    print("=" * 70)
    print("PIPELINE ETL COMPLETADO EXITOSAMENTE")
    print("=" * 70)


if __name__ == "__main__":
    main()