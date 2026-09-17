"""
Generate the synthetic datasets used by every cheat-sheet notebook.

Run once:  python data/make_data.py

Files produced (all in this folder):

  hourly_power_clean.csv   2 years of hourly GB-style power data, tidy.
  hourly_power_raw.csv     same data but deliberately messy (the "as received" version):
                           - missing hours (a whole day + scattered gaps)
                           - duplicated rows
                           - NaNs in temp/price
                           - -999 sentinels in temperature
                           - price stored as text with a few "missing" strings (object dtype)
                           - unsorted rows
  meters.csv               ~300 customer meters (tabular data for merge/groupby practice)
  meter_readings_daily.csv daily kWh per meter, long format, with some gaps
  weather_forecasts.csv    temperature forecasts issued twice a day for the next 48h
                           (origin_datetime vs forecast_datetime -> leakage exercises)
  meter_halfhourly_2023.csv.gz
                           UK-style half-hourly settlement data for 20 meters in 2023:
                           settlement_date (local), settlement_period (1..48; 46 on the
                           March DST day, 50 on the October one), kwh. Planted faults:
                           - one meter stuck at a constant value for a week (M100003, June)
                           - one meter reporting Wh instead of kWh for a month (M100007, Sept)
                           - one meter with a whole missing fortnight (M100011, April)
                           - scattered missing periods, 40 duplicated rows
                           - solar meters export (negative kwh) at midday in summer

Everything is deterministic (seed=42).
"""
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
rng = np.random.default_rng(42)


# ----------------------------------------------------------------------------
# 1. Hourly power system data
# ----------------------------------------------------------------------------
def make_hourly() -> pd.DataFrame:
    idx = pd.date_range("2022-01-01", "2023-12-31 23:00", freq="h", tz="UTC")
    n = len(idx)
    hours = idx.hour.values
    doy = idx.dayofyear.values
    dow = idx.dayofweek.values

    # --- weather ---
    seasonal_temp = 10 - 8 * np.cos(2 * np.pi * (doy - 15) / 365.25)
    diurnal_temp = 3 * np.sin(2 * np.pi * (hours - 9) / 24)
    temp_noise = np.zeros(n)
    for i in range(1, n):  # AR(1) weather noise
        temp_noise[i] = 0.97 * temp_noise[i - 1] + rng.normal(0, 0.6)
    temp = seasonal_temp + diurnal_temp + temp_noise

    wind = np.zeros(n)
    for i in range(1, n):
        wind[i] = 0.95 * wind[i - 1] + rng.normal(0, 1.0)
    wind = np.clip(7 + 2.5 * wind / wind.std(), 0, None)

    daylight = np.clip(np.sin(np.pi * (hours - 6) / 12), 0, None)
    season_solar = 0.55 + 0.45 * -np.cos(2 * np.pi * (doy - 172) / 365.25 + np.pi)
    solar = 800 * daylight * season_solar * rng.uniform(0.4, 1.0, n)

    # --- consumption (MWh) ---
    profile = np.array([
        .78, .74, .72, .71, .72, .76, .84, .94, 1.0, 1.02, 1.02, 1.01,
        1.0, .98, .97, .98, 1.04, 1.12, 1.15, 1.1, 1.03, .96, .88, .82,
    ])
    base = 30_000 * profile[hours]
    weekend = np.where(dow >= 5, -2_200, 0)
    heating = 380 * np.clip(15 - temp, 0, None)
    cooling = 120 * np.clip(temp - 22, 0, None)
    trend = -0.02 * np.arange(n)  # slow decline in demand
    cons_noise = np.zeros(n)
    for i in range(1, n):
        cons_noise[i] = 0.8 * cons_noise[i - 1] + rng.normal(0, 450)
    consumption = base + weekend + heating + cooling + trend + cons_noise

    # --- day-ahead price (EUR/MWh) ---
    gas = 80 + 60 * np.exp(-((np.arange(n) - 6000) / 3000) ** 2)  # 2022 gas spike
    residual_load = consumption - 900 * wind - 12 * solar
    price = (
        gas
        + 0.006 * (residual_load - residual_load.mean())
        + rng.normal(0, 10, n)
    )
    spikes = rng.random(n) < 0.004
    price = price + spikes * rng.uniform(80, 250, n)
    price = np.where(rng.random(n) < 0.002, rng.uniform(-20, 0, n), price)  # negative prices

    df = pd.DataFrame({
        "time": idx,
        "consumption_mwh": consumption.round(1),
        "temp_c": temp.round(2),
        "wind_ms": wind.round(2),
        "solar_wm2": solar.round(1),
        "price_eur_mwh": price.round(2),
    })
    return df


def mess_up(df: pd.DataFrame) -> pd.DataFrame:
    raw = df.copy()
    raw["time"] = raw["time"].dt.strftime("%Y-%m-%d %H:%M:%S")  # strings, tz dropped

    # whole missing day + scattered missing hours
    missing_day = raw["time"].str.startswith("2022-03-27")  # DST-change day in Europe
    scattered = rng.random(len(raw)) < 0.003
    raw = raw.loc[~(missing_day | scattered)].copy()

    # NaNs and sentinels
    raw.loc[rng.random(len(raw)) < 0.01, "temp_c"] = np.nan
    raw.loc[rng.random(len(raw)) < 0.003, "temp_c"] = -999.0
    raw.loc[rng.random(len(raw)) < 0.005, "price_eur_mwh"] = np.nan

    # price as text with "N/A"
    price_txt = raw["price_eur_mwh"].map(lambda v: "missing" if pd.isna(v) else f"{v:.2f}")
    raw["price_eur_mwh"] = price_txt

    # duplicated rows
    dups = raw.sample(15, random_state=1)
    raw = pd.concat([raw, dups])

    # a region column that is constant (useless but realistic)
    raw["region"] = "GB"

    # shuffle order
    raw = raw.sample(frac=1, random_state=2).reset_index(drop=True)
    return raw


# ----------------------------------------------------------------------------
# 2. Tabular customer / meter data
# ----------------------------------------------------------------------------
def make_meters(n_meters: int = 300) -> tuple[pd.DataFrame, pd.DataFrame]:
    regions = ["London", "North", "Midlands", "Scotland", "Wales"]
    tariffs = ["Fixed", "Variable", "TOU"]
    ctype = rng.choice(["residential", "sme"], n_meters, p=[0.85, 0.15])

    meters = pd.DataFrame({
        "meter_id": [f"M{100000 + i}" for i in range(n_meters)],
        "region": rng.choice(regions, n_meters, p=[.3, .2, .2, .2, .1]),
        "tariff": rng.choice(tariffs, n_meters, p=[.5, .3, .2]),
        "customer_type": ctype,
        "annual_kwh_estimate": np.where(
            ctype == "sme", rng.normal(25_000, 8_000, n_meters), rng.normal(3_200, 900, n_meters)
        ).round(0),
        "signup_date": pd.Timestamp("2021-01-01")
        + pd.to_timedelta(rng.integers(0, 700, n_meters), unit="D"),
        "has_solar": rng.random(n_meters) < 0.12,
    })
    meters.loc[rng.random(n_meters) < 0.04, "tariff"] = np.nan
    meters.loc[rng.random(n_meters) < 0.02, "annual_kwh_estimate"] = np.nan
    # inconsistent casing in region for a few rows
    meters.loc[meters.sample(6, random_state=3).index, "region"] = (
        meters["region"].str.lower()
    )

    days = pd.date_range("2023-01-01", "2023-12-31", freq="D")
    doy = days.dayofyear.values
    season = 1 + 0.35 * np.cos(2 * np.pi * (doy - 15) / 365.25)
    rows = []
    for _, m in meters.iterrows():
        daily_mean = (m["annual_kwh_estimate"] if not np.isnan(m["annual_kwh_estimate"]) else 3200) / 365
        kwh = daily_mean * season * rng.lognormal(0, 0.25, len(days))
        if m["has_solar"]:
            kwh = kwh - 4 * (1 - season) * rng.uniform(0.5, 1.0, len(days))
        rows.append(pd.DataFrame({"meter_id": m["meter_id"], "date": days, "kwh": kwh.round(3)}))
    readings = pd.concat(rows, ignore_index=True)
    # some meters have gaps; a handful of meters in readings do not exist in meters table
    readings = readings.loc[rng.random(len(readings)) > 0.02].copy()
    orphan = readings.sample(200, random_state=4).copy()
    orphan["meter_id"] = "M999999"
    readings = pd.concat([readings, orphan], ignore_index=True)
    return meters, readings


# ----------------------------------------------------------------------------
# 3. Weather forecasts with origin timestamps (Enefit-style)
# ----------------------------------------------------------------------------
def make_forecasts(clean: pd.DataFrame) -> pd.DataFrame:
    actual = clean.set_index("time")["temp_c"]
    origins = pd.date_range("2022-01-01", "2023-12-31 12:00", freq="12h", tz="UTC")
    horizons = np.arange(1, 49)
    o = origins.repeat(len(horizons))            # keeps tz-awareness
    h = np.tile(horizons, len(origins))
    fc_time = o + pd.to_timedelta(h, unit="h")
    truth = actual.reindex(fc_time).values
    err_sd = 0.4 + 0.06 * h  # forecast error grows with horizon
    fc = truth + rng.normal(0, 1, len(h)) * err_sd + 0.3  # small warm bias
    out = pd.DataFrame({
        "origin_datetime": o,
        "forecast_datetime": fc_time,
        "horizon_h": h,
        "temp_forecast_c": np.round(fc, 2),
    })
    return out.dropna(subset=["temp_forecast_c"]).reset_index(drop=True)


# ----------------------------------------------------------------------------
# 4. Half-hourly settlement-period panel for 20 meters (UK style, local time)
# ----------------------------------------------------------------------------
def make_halfhourly(meters: pd.DataFrame, n_meters: int = 20) -> pd.DataFrame:
    # UTC grid covering local 2023, then settlement date/period in Europe/London
    utc = pd.date_range("2022-12-31 22:00", "2024-01-01 02:00", freq="30min", tz="UTC")
    local = utc.tz_convert("Europe/London")
    grid = pd.DataFrame({"utc": utc, "settlement_date": local.date})
    grid = grid[(grid["settlement_date"] >= pd.Timestamp("2023-01-01").date())
                & (grid["settlement_date"] <= pd.Timestamp("2023-12-31").date())].copy()
    grid["settlement_period"] = grid.groupby("settlement_date").cumcount() + 1
    grid["local_hour"] = local[grid.index].hour + local[grid.index].minute / 60
    grid["doy"] = pd.DatetimeIndex(grid["settlement_date"]).dayofyear
    grid["dow"] = pd.DatetimeIndex(grid["settlement_date"]).dayofweek
    grid = grid.reset_index(drop=True)

    hh = grid["local_hour"].values
    res_profile = (0.6 + 0.25 * np.exp(-((hh - 7.5) / 1.2) ** 2)
                   + 0.9 * np.exp(-((hh - 18.5) / 2.0) ** 2))
    sme_profile = 0.3 + 1.2 * ((hh >= 8) & (hh < 18)) * (1 - 0.3 * (grid["dow"].values >= 5))
    season = 1 + 0.35 * np.cos(2 * np.pi * (grid["doy"].values - 15) / 365.25)
    solar_shape = np.clip(np.sin(np.pi * (hh - 6) / 12), 0, None) ** 2
    solar_season = 0.5 + 0.5 * -np.cos(2 * np.pi * (grid["doy"].values - 172) / 365.25 + np.pi)

    sel = meters.head(n_meters)
    frames = []
    for _, m in sel.iterrows():
        est = m["annual_kwh_estimate"] if not np.isnan(m["annual_kwh_estimate"]) else 3200
        prof = sme_profile if m["customer_type"] == "sme" else res_profile
        daily = est / 365 * season
        kwh = daily * prof / prof.mean() / 48 * rng.lognormal(0, 0.3, len(grid))
        if m["has_solar"]:
            kwh = kwh - 0.9 * solar_shape * solar_season * rng.uniform(0.6, 1.0, len(grid))
        frames.append(pd.DataFrame({
            "meter_id": m["meter_id"],
            "settlement_date": grid["settlement_date"].astype(str),
            "settlement_period": grid["settlement_period"],
            "kwh": kwh.round(3),
        }))
    panel = pd.concat(frames, ignore_index=True)
    sd = pd.to_datetime(panel["settlement_date"])

    # faults
    stuck = (panel["meter_id"] == "M100003") & (sd >= "2023-06-05") & (sd < "2023-06-12")
    panel.loc[stuck, "kwh"] = 0.187
    wh = (panel["meter_id"] == "M100007") & (sd.dt.month == 9)
    panel.loc[wh, "kwh"] = (panel.loc[wh, "kwh"] * 1000).round(0)
    gap = (panel["meter_id"] == "M100011") & (sd >= "2023-04-10") & (sd < "2023-04-24")
    panel = panel.loc[~gap]
    panel = panel.loc[rng.random(len(panel)) > 0.001]
    dups = panel.sample(40, random_state=5)
    panel = pd.concat([panel, dups], ignore_index=True)
    return panel.sample(frac=1, random_state=6).reset_index(drop=True)


if __name__ == "__main__":
    clean = make_hourly()
    clean.to_csv(HERE / "hourly_power_clean.csv", index=False)
    mess_up(clean).to_csv(HERE / "hourly_power_raw.csv", index=False)

    meters, readings = make_meters()
    meters.to_csv(HERE / "meters.csv", index=False)
    readings.to_csv(HERE / "meter_readings_daily.csv", index=False)

    make_forecasts(clean).to_csv(HERE / "weather_forecasts.csv", index=False)
    make_halfhourly(meters).to_csv(HERE / "meter_halfhourly_2023.csv.gz", index=False, compression="gzip")
    for f in sorted(list(HERE.glob("*.csv")) + list(HERE.glob("*.csv.gz"))):
        print(f"{f.name:28s} {f.stat().st_size/1e6:6.1f} MB")
