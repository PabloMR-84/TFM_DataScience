"""
src/models_gold.py
-------------------
Módulo de modelado dimensional (Capa Gold) y construcción del dataset analítico.

==============================================================================
ESPECIFICACIÓN DEL OBJETIVO PREDICTIVO, HORIZONTE Y MATRIZ DE VARIABLES
==============================================================================
1. Objetivo Predictivo:
   - Predecir la ocupación física real (occupancy_pct) y el riesgo de
     desabastecimiento/saturación por estación y hora para optimizar la
     logística de redistribución de Bizi Zaragoza.
   - Las recomendaciones de la fuente externa (rebalancing_report / bikesToMove)
     NO equivalen a la verdad observada (ground truth) y se mantienen al margen
     de la variable objetivo.

2. Horizonte de Anticipación (Unificado con docs/entregas/02_datos_y_preparacion.md):
   - Granularidad base: Capturas cada 10 minutos agrupadas en franjas horarias (t).
   - Corto plazo (h = 1 hora): Detección de desequilibrio inminente para
     intervenciones de emergencia.
   - Táctico (h = 2 a 3 horas, ej. evaluación a las 7:00 AM para las 9:00 AM):
     Planificación previa de rutas de furgonetas antes de las horas punta.

3. Matriz de Variables (Features y Targets):
   - Identificadores y Tiempo: station_id, timestamp, date_key, hour, is_estimated,
     cold_start_status.
   - Verdad Observada (Targets): bikes_available, anchors_free, occupancy_pct.
   - Extrapolación Cold Start (Separada): occupancy_estimated (nunca sustituye
     a occupancy_pct como verdad de evaluación).
   - Contexto Temporal / Macro (Disponible <= t): day_type_effective,
     is_holiday, is_weekend, demand_score, city_avg_occupancy, balance_index.
   - Perfiles Basales (Dim_Patterns, disponible <= t): occupancy_avg, bikes_avg.
   - Indicadores Operativos (Rankings, disponible <= t): turnover_score,
     empty_hours, full_hours, problem_hours (sin fallbacks ni relleno con ceros).
==============================================================================
"""

from typing import List, Optional, Set, Tuple
import numpy as np
import pandas as pd
import requests

from etl.utils import safe_local_timestamp


def _compute_available_at(df: pd.DataFrame, timezone: str = "Europe/Madrid") -> pd.Series:
    """
    Calcula el instante exacto en que un registro pasa a estar disponible,
    exigiendo que haya finalizado tanto su periodo (window_end) como su
    publicación (generated_at). Si ambos existen, toma el máximo (el más tardío).
    Si ninguno existe, devuelve NaT para evitar asumir disponibilidad futura.
    """
    if df.empty:
        return pd.Series(dtype=f"datetime64[ns, {timezone}]")

    candidates = []
    for col in ["window_end", "generated_at"]:
        if col in df.columns:
            candidates.append(safe_local_timestamp(df[col], timezone=timezone))

    if not candidates:
        return pd.Series(pd.NaT, index=df.index, dtype=f"datetime64[ns, {timezone}]")

    return pd.concat(candidates, axis=1).max(axis=1)


def build_dim_stations(df_stations_silver: pd.DataFrame) -> pd.DataFrame:
    """Construye la dimensión geográfica y física de estaciones (Dim_Stations)."""
    if df_stations_silver.empty:
        return pd.DataFrame()

    cols = ["station_id", "name", "latitude", "longitude", "capacity"]
    existing_cols = [c for c in cols if c in df_stations_silver.columns]

    df_dim = (
        df_stations_silver[existing_cols]
        .dropna(subset=["station_id"])
        .drop_duplicates(subset=["station_id"], keep="last")
        .sort_values("station_id")
        .reset_index(drop=True)
    )
    df_dim["station_id"] = df_dim["station_id"].astype("Int64")
    return df_dim


def build_dim_patterns(df_patterns_silver: pd.DataFrame) -> pd.DataFrame:
    """
    Construye la dimensión de patrones horarios basales (Dim_Patterns),
    conservando generated_at, window_end y available_at para filtrar por disponibilidad.
    """
    if df_patterns_silver.empty:
        return pd.DataFrame()

    cols = [
        "station_id",
        "day_type",
        "hour",
        "occupancy_avg",
        "bikes_avg",
        "generated_at",
        "window_end",
    ]
    existing_cols = [c for c in cols if c in df_patterns_silver.columns]

    df_dim = (
        df_patterns_silver[existing_cols]
        .dropna(subset=["station_id", "day_type", "hour"])
        .copy()
    )
    df_dim["station_id"] = df_dim["station_id"].astype("Int64")
    df_dim["hour"] = df_dim["hour"].astype("Int64")
    df_dim["day_type"] = df_dim["day_type"].astype(str).str.upper().str.strip()
    df_dim["available_at"] = _compute_available_at(df_dim)

    if df_dim["available_at"].notna().any():
        df_dim.sort_values("available_at", inplace=True)

    df_dim = df_dim.drop_duplicates(
        subset=["station_id", "day_type", "hour", "available_at"], keep="last"
    ).reset_index(drop=True)
    return df_dim


def fetch_public_holidays(years: List[int]) -> Set[str]:
    """Obtiene los festivos nacionales y de Aragón (ES-AR) desde Nager.Date."""
    holiday_dates: Set[str] = set()

    for yr in years:
        try:
            url = f"https://date.nager.at/api/v3/PublicHolidays/{yr}/ES"
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                for item in response.json():
                    counties = item.get("counties") or []
                    if item.get("global") or ("ES-AR" in counties):
                        holiday_dates.add(item["date"])
        except Exception as e:
            print(f"[AVISO] Fallo al consultar la API de festivos para el año {yr}: {e}")

    return holiday_dates


def generate_recurring_holidays(
    years: List[int],
    custom_md_list: Optional[List[Tuple[int, int]]] = None,
) -> Set[str]:
    """Genera fechas YYYY-MM-DD para festivos fijos nacionales y locales de Zaragoza."""
    if custom_md_list is None:
        custom_md_list = [
            (1, 1),    # Año Nuevo
            (1, 6),    # Reyes Magos
            (1, 29),   # San Valero (Local Zaragoza)
            (3, 5),    # Cincomarzada (Local Zaragoza)
            (10, 12),  # El Pilar / Fiesta Nacional
            (12, 24),  # Nochebuena
            (12, 25),  # Navidad
            (12, 31),  # Nochevieja
        ]

    generated_dates: Set[str] = set()
    for yr in years:
        for month, day in custom_md_list:
            generated_dates.add(f"{yr:04d}-{month:02d}-{day:02d}")

    return generated_dates


def build_dim_calendar(
    df_snapshots: pd.DataFrame,
    df_history_silver: Optional[pd.DataFrame] = None,
    custom_local_holidays_md: Optional[List[Tuple[int, int]]] = None,
) -> pd.DataFrame:
    """
    Construye Dim_Calendar homogeneizando todas las fechas en Europe/Madrid.
    Para evitar fuga de información (data leakage), el contexto macro diario
    (demand_score, city_avg_occupancy, balance_index) se une comprobando tanto
    el fin de la jornada diaria (window_end = date + 1 día) como su publicación
    (generated_at) mediante available_at <= date.
    """
    candidate_dates = []
    if not df_snapshots.empty and "date" in df_snapshots.columns:
        s_dates = safe_local_timestamp(df_snapshots["date"]).dropna()
        if not s_dates.empty:
            candidate_dates.extend([s_dates.min(), s_dates.max()])

    if df_history_silver is not None and not df_history_silver.empty and "date" in df_history_silver.columns:
        h_dates = safe_local_timestamp(df_history_silver["date"]).dropna()
        if not h_dates.empty:
            candidate_dates.extend([h_dates.min(), h_dates.max()])

    if not candidate_dates:
        return pd.DataFrame()

    min_date = min(candidate_dates).floor("D")
    max_date = max(candidate_dates).floor("D")

    date_range = pd.date_range(start=min_date, end=max_date, freq="D", tz="Europe/Madrid")

    df_calendar = pd.DataFrame({"date": date_range})
    df_calendar["date_key"] = df_calendar["date"].dt.strftime("%Y%m%d").astype("Int64")
    df_calendar["year"] = df_calendar["date"].dt.year.astype("Int64")
    df_calendar["month"] = df_calendar["date"].dt.month.astype("Int64")
    df_calendar["day_of_week"] = df_calendar["date"].dt.dayofweek.astype("Int64")
    df_calendar["is_weekend"] = df_calendar["day_of_week"].isin([5, 6]).astype("Int64")

    years = df_calendar["year"].dropna().unique().astype(int).tolist()
    api_holidays = fetch_public_holidays(years)
    recurring_holidays = generate_recurring_holidays(years, custom_local_holidays_md)
    all_holidays = api_holidays.union(recurring_holidays)

    df_calendar["is_holiday"] = (
        df_calendar["date"].dt.strftime("%Y-%m-%d").isin(all_holidays).astype("Int64")
    )

    df_calendar["day_type_effective"] = "WEEKDAY"
    df_calendar.loc[
        (df_calendar["is_weekend"] == 1) | (df_calendar["is_holiday"] == 1),
        "day_type_effective",
    ] = "WEEKEND"

    if df_history_silver is not None and not df_history_silver.empty and "date" in df_history_silver.columns:
        history_cols = [
            "date",
            "window_end",
            "generated_at",
            "demand_score",
            "city_avg_occupancy",
            "balance_index",
        ]
        existing_hist = [c for c in history_cols if c in df_history_silver.columns]

        df_hist_sorted = df_history_silver[existing_hist].dropna(subset=["date"]).copy()
        df_hist_sorted["date"] = safe_local_timestamp(df_hist_sorted["date"]).dt.floor("D")
        if "window_end" not in df_hist_sorted.columns:
            df_hist_sorted["window_end"] = df_hist_sorted["date"] + pd.Timedelta(days=1)
        else:
            df_hist_sorted["window_end"] = safe_local_timestamp(df_hist_sorted["window_end"]).fillna(
                df_hist_sorted["date"] + pd.Timedelta(days=1)
            )

        df_hist_sorted["history_available_at"] = _compute_available_at(df_hist_sorted)
        df_hist_sorted = (
            df_hist_sorted.dropna(subset=["history_available_at"])
            .sort_values("history_available_at")
            .drop_duplicates(subset=["date"], keep="last")
            .reset_index(drop=True)
        )

        merge_hist_cols = [
            c
            for c in ["history_available_at", "demand_score", "city_avg_occupancy", "balance_index"]
            if c in df_hist_sorted.columns
        ]

        df_calendar = pd.merge_asof(
            df_calendar.sort_values("date"),
            df_hist_sorted[merge_hist_cols],
            left_on="date",
            right_on="history_available_at",
            direction="backward",
            allow_exact_matches=True,
        )

    return df_calendar


def build_fact_snapshots(df_historical_silver: pd.DataFrame) -> pd.DataFrame:
    """
    Construye la tabla de hechos de capturas (Fact_StationSnapshots) normalizando
    las marcas temporales a Europe/Madrid.
    """
    if df_historical_silver.empty:
        return pd.DataFrame()

    df = df_historical_silver.copy()
    time_col = "timestamp" if "timestamp" in df.columns else "recordedAt"

    df["timestamp"] = safe_local_timestamp(df[time_col])
    df["date"] = df["timestamp"].dt.floor("D")
    df["date_key"] = df["timestamp"].dt.strftime("%Y%m%d").astype("Int64")
    df["hour"] = df["timestamp"].dt.hour.astype("Int64")

    cols_map = {
        "id": "station_id",
        "bikesAvailable": "bikes_available",
        "anchorsFree": "anchors_free",
    }
    df.rename(columns=cols_map, inplace=True, errors="ignore")

    keep_cols = [
        "station_id",
        "date_key",
        "date",
        "timestamp",
        "hour",
        "bikes_available",
        "anchors_free",
    ]
    existing_keep = [c for c in keep_cols if c in df.columns]

    df_fact = df[existing_keep].dropna(subset=["station_id", "timestamp"]).copy()
    df_fact["station_id"] = df_fact["station_id"].astype("Int64")
    return df_fact


def _log_cold_start_diagnostics(
    df_gold: pd.DataFrame,
    df_stations_gold: pd.DataFrame,
    df_patterns_gold: pd.DataFrame,
) -> None:
    """
    Emite un informe claro por consola sobre la cobertura de Dim_Patterns y el
    estado de extrapolación Cold Start en df_gold.
    """
    print("\n--- Diagnóstico de Cobertura de Patrones y Cold Start ---")

    if not df_stations_gold.empty and "station_id" in df_stations_gold.columns:
        all_st = set(df_stations_gold["station_id"].dropna().astype(int).tolist())
        pat_st = (
            set(df_patterns_gold["station_id"].dropna().astype(int).tolist())
            if not df_patterns_gold.empty and "station_id" in df_patterns_gold.columns
            else set()
        )
        missing_pat_ids = sorted(all_st - pat_st)

        print(f"  - Estaciones en catálogo (Dim_Stations)     : {len(all_st)}")
        print(f"  - Estaciones con perfil en Dim_Patterns     : {len(pat_st)}")
        print(f"  - Estaciones SIN perfil en Dim_Patterns     : {len(missing_pat_ids)}")

        if missing_pat_ids:
            if "name" in df_stations_gold.columns:
                name_map = (
                    df_stations_gold.dropna(subset=["station_id"])
                    .set_index("station_id")["name"]
                    .to_dict()
                )
                preview = [f"{sid} ({name_map.get(sid, 'N/D')})" for sid in missing_pat_ids[:15]]
            else:
                preview = [str(sid) for sid in missing_pat_ids[:15]]

            suffix = f" ... (+{len(missing_pat_ids) - 15} más)" if len(missing_pat_ids) > 15 else ""
            print(f"  - IDs sin patrón basal (no extrapolables)   : {', '.join(preview)}{suffix}")

    total_rows = len(df_gold)
    obs_rows = int(df_gold["occupancy_pct"].notna().sum())
    est_rows = int((df_gold["is_estimated"] == 1).sum())
    unavail_rows = int((df_gold["occupancy_pct"].isna() & df_gold["occupancy_estimated"].isna()).sum())

    print(f"  - Registros reales observados (OBSERVED)    : {obs_rows:,} / {total_rows:,} ({obs_rows / total_rows * 100:.2f}%)")
    print(f"  - Registros extrapolados (COLD START)       : {est_rows:,} / {total_rows:,} ({est_rows / total_rows * 100:.2f}%)")
    print(f"  - Registros sin dato ni patrón (UNAVAILABLE): {unavail_rows:,} / {total_rows:,} ({unavail_rows / total_rows * 100:.2f}%)")


def build_gold_dataset(
    df_snapshots_gold: pd.DataFrame,
    df_calendar: pd.DataFrame,
    df_stations_gold: pd.DataFrame,
    df_patterns_gold: pd.DataFrame,
    df_rankings_silver: Optional[pd.DataFrame] = None,
    df_history_silver: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Unifica las dimensiones y la tabla de hechos en el dataset Gold final (df_gold):
    - Impone validate='m:1' en los cruces dimensionales estáticos para impedir duplicaciones.
    - Exige disponibilidad temporal estricta (max(window_end, generated_at) <= timestamp)
      SIN fallbacks ni rellenos con ceros en rankings, patrones y datos diarios de ciudad.
    - Mantiene occupancy_pct (observada) separada de occupancy_estimated (extrapolada).
    """
    if df_snapshots_gold.empty:
        return pd.DataFrame()

    df_gold = df_snapshots_gold.copy()

    # 1. Cruce con Dim_Calendar (m:1 estricto)
    if not df_calendar.empty:
        cal_cols = [
            "date_key",
            "day_type_effective",
            "is_holiday",
            "demand_score",
            "city_avg_occupancy",
            "balance_index",
        ]
        exist_cal = [c for c in cal_cols if c in df_calendar.columns]
        df_gold = df_gold.merge(
            df_calendar[exist_cal],
            on="date_key",
            how="left",
            validate="m:1",
        )

    # 1b. Refuerzo de disponibilidad temporal exacta (timestamp >= available_at) en datos diarios de ciudad
    city_metrics = ["demand_score", "city_avg_occupancy", "balance_index"]
    if df_history_silver is not None and not df_history_silver.empty and "date" in df_history_silver.columns:
        exist_city = [c for c in city_metrics if c in df_history_silver.columns]
        if exist_city and "timestamp" in df_gold.columns:
            df_gold.drop(columns=exist_city, errors="ignore", inplace=True)
            hist_tmp = df_history_silver.copy()
            hist_tmp["date"] = safe_local_timestamp(hist_tmp["date"]).dt.floor("D")
            if "window_end" not in hist_tmp.columns:
                hist_tmp["window_end"] = hist_tmp["date"] + pd.Timedelta(days=1)
            else:
                hist_tmp["window_end"] = safe_local_timestamp(hist_tmp["window_end"]).fillna(
                    hist_tmp["date"] + pd.Timedelta(days=1)
                )
            hist_tmp["available_at"] = _compute_available_at(hist_tmp)
            valid_hist = (
                hist_tmp[["available_at"] + exist_city]
                .dropna(subset=["available_at"])
                .sort_values("available_at")
                .drop_duplicates(subset=["available_at"], keep="last")
                .reset_index(drop=True)
            )
            if not valid_hist.empty:
                df_gold = df_gold.sort_values("timestamp").reset_index(drop=True)
                df_gold = pd.merge_asof(
                    df_gold,
                    valid_hist,
                    left_on="timestamp",
                    right_on="available_at",
                    direction="backward",
                    allow_exact_matches=True,
                )
                df_gold.drop(columns=["available_at"], errors="ignore", inplace=True)
            else:
                for col in exist_city:
                    df_gold[col] = np.nan

    # 2. Cruce con Dim_Stations (m:1 estricto)
    if not df_stations_gold.empty:
        st_cols = ["station_id", "name", "latitude", "longitude", "capacity"]
        exist_st = [c for c in st_cols if c in df_stations_gold.columns]
        df_gold = df_gold.merge(
            df_stations_gold[exist_st],
            on="station_id",
            how="left",
            validate="m:1",
        )

    # 3. Ocupación real observada (Ground Truth) con blindaje de tipos float64
    if "bikes_available" in df_gold.columns and "capacity" in df_gold.columns:
        bikes_f = df_gold["bikes_available"].astype("float64")
        cap_f = df_gold["capacity"].astype("float64")
        occ = np.where(cap_f > 0, (bikes_f / cap_f).round(4), np.nan)
        occ = np.where((occ < 0.0) | (occ > 1.0), np.nan, occ)
        df_gold["occupancy_pct"] = pd.Series(occ, index=df_gold.index, dtype="float64").replace(-0.0, 0.0)
        df_gold.drop(columns=["capacity"], inplace=True, errors="ignore")
    else:
        df_gold["occupancy_pct"] = np.nan

    # 4. Cruce con Dim_Patterns con disponibilidad temporal estricta (available_at <= timestamp, sin fallbacks)
    pat_metrics = ["occupancy_avg", "bikes_avg"]
    for col in pat_metrics:
        df_gold[col] = np.nan

    if (
        not df_patterns_gold.empty
        and "day_type_effective" in df_gold.columns
        and "timestamp" in df_gold.columns
    ):
        pat = df_patterns_gold.copy()
        avail_pat_metrics = [c for c in pat_metrics if c in pat.columns]
        pat["available_at"] = (
            safe_local_timestamp(pat["available_at"])
            if "available_at" in pat.columns
            else _compute_available_at(pat)
        )

        valid_pat = (
            pat[["station_id", "day_type", "hour", "available_at"] + avail_pat_metrics]
            .dropna(subset=["station_id", "day_type", "hour", "available_at"])
            .copy()
        )

        if not valid_pat.empty:
            df_gold.drop(columns=avail_pat_metrics, errors="ignore", inplace=True)
            df_gold["_st_id_key"] = df_gold["station_id"].astype("int64")
            df_gold["_day_type_key"] = df_gold["day_type_effective"].astype(str).str.upper().str.strip()
            df_gold["_hour_key"] = df_gold["hour"].astype("int64")

            valid_pat["_st_id_key"] = valid_pat["station_id"].astype("int64")
            valid_pat["_day_type_key"] = valid_pat["day_type"].astype(str).str.upper().str.strip()
            valid_pat["_hour_key"] = valid_pat["hour"].astype("int64")

            valid_pat = (
                valid_pat.sort_values("available_at")
                .drop_duplicates(subset=["_st_id_key", "_day_type_key", "_hour_key", "available_at"], keep="last")
                .reset_index(drop=True)
            )
            df_gold = df_gold.sort_values("timestamp").reset_index(drop=True)

            df_gold = pd.merge_asof(
                df_gold,
                valid_pat[["_st_id_key", "_day_type_key", "_hour_key", "available_at"] + avail_pat_metrics],
                left_on="timestamp",
                right_on="available_at",
                by=["_st_id_key", "_day_type_key", "_hour_key"],
                direction="backward",
                allow_exact_matches=True,
            )
            df_gold.drop(
                columns=["_st_id_key", "_day_type_key", "_hour_key", "available_at"],
                errors="ignore",
                inplace=True,
            )

    # 5. Cruce con Rankings con disponibilidad temporal estricta (available_at <= timestamp, SIN fallback)
    rk_metrics = ["turnover_score", "empty_hours", "full_hours", "problem_hours"]
    for col in rk_metrics:
        if col not in df_gold.columns:
            df_gold[col] = np.nan

    if df_rankings_silver is not None and not df_rankings_silver.empty and "timestamp" in df_gold.columns:
        rk = df_rankings_silver.copy()
        avail_rk_metrics = [c for c in rk_metrics if c in rk.columns]
        rk["available_at"] = _compute_available_at(rk)

        valid_rk = (
            rk[["station_id", "available_at"] + avail_rk_metrics]
            .dropna(subset=["station_id", "available_at"])
            .copy()
        )

        if not valid_rk.empty and avail_rk_metrics:
            df_gold.drop(columns=avail_rk_metrics, errors="ignore", inplace=True)
            df_gold["_st_id_key"] = df_gold["station_id"].astype("int64")
            valid_rk["_st_id_key"] = valid_rk["station_id"].astype("int64")

            valid_rk = (
                valid_rk.sort_values("available_at")
                .drop_duplicates(subset=["_st_id_key", "available_at"], keep="last")
                .reset_index(drop=True)
            )
            df_gold = df_gold.sort_values("timestamp").reset_index(drop=True)

            df_gold = pd.merge_asof(
                df_gold,
                valid_rk[["_st_id_key", "available_at"] + avail_rk_metrics],
                left_on="timestamp",
                right_on="available_at",
                by="_st_id_key",
                direction="backward",
                allow_exact_matches=True,
            )
            df_gold.drop(columns=["_st_id_key", "available_at"], errors="ignore", inplace=True)

    # 6. Extrapolación Cold Start en columna SEPARADA (occupancy_estimated) y trazabilidad
    if "occupancy_avg" in df_gold.columns:
        if "city_avg_occupancy" in df_gold.columns:
            city_scale = (df_gold["city_avg_occupancy"] / 0.5).fillna(1.0)
        else:
            city_scale = 1.0

        est_vals = np.where(
            df_gold["occupancy_avg"].notna(),
            (df_gold["occupancy_avg"] * city_scale).clip(0.0, 1.0).round(4),
            np.nan,
        )
        df_gold["occupancy_estimated"] = pd.Series(est_vals, index=df_gold.index, dtype="float64").replace(-0.0, 0.0)
    else:
        df_gold["occupancy_estimated"] = np.nan

    # is_estimated = 1 solo cuando falta lectura real Y existe estimación válida
    df_gold["is_estimated"] = (
        df_gold["occupancy_pct"].isna() & df_gold["occupancy_estimated"].notna()
    ).astype("Int64")

    df_gold["cold_start_status"] = np.select(
        [
            df_gold["occupancy_pct"].notna(),
            df_gold["occupancy_pct"].isna() & df_gold["occupancy_estimated"].notna(),
        ],
        [
            "OBSERVED",
            "ESTIMATED_FROM_PATTERN",
        ],
        default="UNAVAILABLE_NO_PATTERN",
    )

    if "station_id" in df_gold.columns and "timestamp" in df_gold.columns:
        df_gold.sort_values(by=["station_id", "timestamp"], inplace=True)
        df_gold.reset_index(drop=True, inplace=True)

    _log_cold_start_diagnostics(df_gold, df_stations_gold, df_patterns_gold)

    return df_gold