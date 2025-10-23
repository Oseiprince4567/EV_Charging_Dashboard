# helpers.py — Hamburg EV charging (SensorThings API), Streamlit-cached helpers
# Keeps your original vibe: same imports, comments, geodesic/Nominatim, Folium.

import urllib  # Import module for working with URLs
import urllib.request
import json  # Import module for working with JSON data
#import sqlite3  # Import module for SQLite database interaction
import pandas as pd  # Import pandas for data manipulation
import folium  # Import folium for creating interactive maps
from folium.plugins import MarkerCluster  # clustering for small markers
import datetime as dt  # Import datetime for working with dates and times
from geopy.distance import geodesic  # Import geodesic for calculating distances
from geopy.geocoders import Nominatim  # Import Nominatim for geocoding
import streamlit as st  # Import Streamlit for creating web app
from typing import List
import requests
from tqdm import tqdm  # Added for progress bar
# ------------------------------------------------------------------------------
# PostgreSQL connection for Analytics
# ------------------------------------------------------------------------------
from sqlalchemy import create_engine, text
import logging

# --- Database engine setup ---
from sqlalchemy import create_engine
import streamlit as st

# Use the database URL stored in Streamlit Secrets
engine = create_engine(st.secrets["database"]["url"], pool_pre_ping=True)


# Configure logging once at the top
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

#PG_URI = "postgresql://evuser@localhost:5432/evcharging"
#engine = create_engine(PG_URI, pool_pre_ping=True)
BASE_URL = "https://iot.hamburg.de/v1.0/Datastreams?$expand=Thing($expand=Locations)"

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

# -------------------------------------------------------------------
# Update Datastream–Station Mapping Table
# -------------------------------------------------------------------
def update_datastream_station_table():
    """
    Fetch the latest Datastream → Station mapping from Hamburg IoT API
    and update the datastream_station table in PostgreSQL.
    Includes all Datastreams, not just those with 'Lade' in the name.
    """
    
    log.info("🌍 Fetching latest Datastream–Station mapping from Hamburg IoT API...")
    all_rows = []
    url = BASE_URL
    page = 0

    while url:
        page += 1
        log.info(f"📦 Fetching page {page}...")
        try:
            r = requests.get(url, timeout=60)
            r.raise_for_status()
            data = r.json()
        except Exception as e:
            log.warning(f"⚠️ Failed page {page}: {e}")
            break

        for ds in data.get("value", []):
            ds_id = ds.get("@iot.id")
            thing = ds.get("Thing", {})
            locs = thing.get("Locations", [])
            lat, lon, addr = None, None, None

            if locs:
                try:
                    loc0 = locs[0]
                    addr = loc0.get("name") or loc0.get("description")
                    coords = loc0.get("location", {}).get("coordinates", [])
                    if len(coords) == 2:
                        lon, lat = coords
                except Exception:
                    pass

            all_rows.append({
                "datastream_id": ds_id,
                "station_name": thing.get("name", "Unknown"),
                "description": thing.get("description", ""),
                "latitude": lat,
                "longitude": lon,
                "address": addr,
            })

        url = data.get("@iot.nextLink")

    df = pd.DataFrame(all_rows)
    log.info(f"✅ Retrieved {len(df)} datastream entries with metadata.")

    if df.empty:
        log.warning("⚠️ No datastreams found — skipping update.")
        return

    # Drop duplicates and push to DB
    df = df.drop_duplicates(subset="datastream_id")

    with engine.begin() as conn:
        conn.execute(text("DROP TABLE IF EXISTS datastream_station"))
        df.to_sql("datastream_station", conn, if_exists="replace", index=False)

    log.info(f"🗄️ datastream_station table updated successfully with {len(df)} entries.")
# ------------------------------------------------------------------------------
# Constants / configuration
# ------------------------------------------------------------------------------

# Base endpoint: “Lade*” Things, expand Locations and each Datastream's latest Observation
HAMBURG_URL = (
    "https://iot.hamburg.de/v1.0/Things?"
    "$filter=substringof('Lade',name)"
    "&$expand=Locations,Datastreams/Observations($orderby=phenomenonTime%20desc;$top=1)"
    "&$count=true"
)

# Rough Hamburg bounding box: (min_lon, min_lat, max_lon, max_lat)
HAMBURG_BBOX = (9.5, 53.35, 10.35, 53.75)

# Normalize raw observation -> (normalized status, folium color)
STATUS_MAP = {
    # English
    "AVAILABLE": ("free", "green"),
    "OCCUPIED": ("busy", "red"),
    "CHARGING": ("busy", "red"),
    "RESERVED": ("reserved", "orange"),
    "OUTOFORDER": ("out_of_order", "gray"),
    "INOPERATIVE": ("out_of_order", "gray"),
    "SUSPENDED": ("suspended", "cadetblue"),
    "SUSPENDEDEV": ("suspended", "cadetblue"),  # appears in Hamburg feed
    "UNKNOWN": ("unknown", "blue"),
    # Defensive German keys (just in case)
    "FREI": ("free", "green"),
    "BELEGT": ("busy", "red"),
    "RESERVIERT": ("reserved", "orange"),
    "AUSSER_BETRIEB": ("out_of_order", "gray"),
    None: ("unknown", "blue"),
}

# Nominal power estimates (tune as you verify sites)
KW_ESTIMATE = {
    ("AC", "S_TYPE_2_CEE_7_7"): 22,
    ("AC", "S_TYPE_2"): 22,
    ("AC", "S_TYPE_2_CABLE_ATTACHED"): 22,
    ("DC", "C_CCS_2"): 50,       # change to 100/150 where known
    ("DC", "C_CHADEMO"): 50,
}

# Labels / examples for popup detail enrichment
MODE_LABEL = {"MODE3": "Mode 3", "MODE4": "Mode 4"}
CURRENT_LABEL = {"AC": "Standard AC", "DC": "Fast DC"}
CONNECTOR_EXAMPLES = {
    "S_TYPE_2_CEE_7_7": ["VW e-Golf", "Renault Zoe"],
    "S_TYPE_2": ["VW e-Golf", "Renault Zoe"],
    "S_TYPE_2_CABLE_ATTACHED": ["VW e-Golf", "Renault Zoe"],
    "C_CCS_2": ["VW ID.4", "Hyundai Ioniq 5"],
    "C_CHADEMO": ["Nissan Leaf"],
}

def format_updated(ts_utc):
    """Return 'HH:MM:SS today' if same local day, else 'YYYY-MM-DD HH:MM'."""
    try:
        ts = pd.to_datetime(ts_utc, utc=True)
        local = ts.tz_convert("Europe/Berlin")
        now_local = pd.Timestamp.now(tz="Europe/Berlin")
        if local.date() == now_local.date():
            return f"{local.strftime('%H:%M:%S')} today"
        return local.strftime("%Y-%m-%d %H:%M")
    except Exception:
        return "—"

def connector_lines_for_station(station_row, connectors_df, max_lines=2):
    """Return up to `max_lines` formatted connector lines for the station popup."""
    out = []
    try:
        thing_id = station_row["thing_id"]
        sub = connectors_df.reset_index()
        sub = sub[sub["thing_id"] == thing_id].sort_values("connector_name")

        for idx, (_, r) in enumerate(sub.head(max_lines).iterrows(), 1):
            mode = MODE_LABEL.get(str(r.get("charging_protocol") or "").upper(), "")
            ctype = str(r.get("connector_type") or "")
            examples = CONNECTOR_EXAMPLES.get(ctype, [])
            example = f", e.g., {examples[0]}" if examples else ""
            cur = CURRENT_LABEL.get(str(r.get("current") or "").upper(), "")
            kw = r.get("kw_nominal")
            if not (isinstance(kw, (int, float)) and kw > 0):
                cur_raw = str(r.get("current") or "").upper()
                kw = 22 if cur_raw == "AC" else (50 if cur_raw == "DC" else None)
            kwpart = f" — ~{int(kw)} kW" if kw else ""

            line = f"Connector {idx:02d}: {mode} ({ctype}{example}), {cur}{kwpart}"
            out.append(line)

        extra = max(len(sub) - max_lines, 0)
        if extra > 0:
            out.append(f"+{extra} more…")
    except Exception as e:
        out.append(f"(error building connector info: {e})")
    return out


# ------------------------------------------------------------------------------
# Internal utilities (parsing + normalization)
# ------------------------------------------------------------------------------

def _parse_phenomenon_time(val):
    """OGC phenomenonTime can be an instant or 'start/end'. Return right side as UTC Timestamp or NaT."""
    if not val:
        return pd.NaT
    try:
        if isinstance(val, str) and "/" in val:
            val = val.split("/")[-1]
        return pd.to_datetime(val, utc=True, errors="coerce")
    except Exception:
        return pd.NaT

def _normalize_status(raw):
    key = raw.strip().upper() if isinstance(raw, str) else None
    return STATUS_MAP.get(key, STATUS_MAP[None])[0]

def _status_color(norm_status):
    for _, (norm, color) in STATUS_MAP.items():
        if norm == norm_status:
            return color
    return "blue"

def _estimate_kw(current, connector_type):
    key = (str(current or "").upper(), str(connector_type or "").upper())
    return KW_ESTIMATE.get(key)

# ------------------------------------------------------------------------------
# Data access + wrangling (Hamburg SensorThings)
# ------------------------------------------------------------------------------

@st.cache_data(show_spinner=True)
def fetch_hamburg_things(url: str = HAMBURG_URL) -> list:
    """Iterate through Hamburg SensorThings Things with pagination. Returns list of dicts."""
    things = []
    next_url = url
    while next_url:
        with urllib.request.urlopen(next_url) as resp:
            payload = json.loads(resp.read().decode())
        things.extend(payload.get("value", []))
        next_url = payload.get("@iot.nextLink")
    return things

@st.cache_data(show_spinner=True)
def normalize_connectors(things: list) -> pd.DataFrame:
    """
    Connector-level DataFrame:
      thing_id, asset_id, station_name, address, latitude, longitude,
      datastream_id, connector_name, connector_type, current, charging_protocol,
      raw_status, status, marker_color, observed_at, kw_nominal
    """
    rows = []
    for thing in things:
        thing_id = thing.get("@iot.id")
        station_name = thing.get("name", "Unknown")
        props = thing.get("properties") or {}
        asset_id = props.get("assetID")

        # Location
        address = None
        lat, lon = None, None
        try:
            loc0 = (thing.get("Locations") or [{}])[0]
            address = loc0.get("description")
            coords = (((loc0.get("location") or {}).get("geometry") or {}).get("coordinates")
                      or [None, None])
            lon, lat = coords[0], coords[1]
        except Exception:
            pass

        # Each Datastream is a connector with latest Observation
        for ds in thing.get("Datastreams", []):
            datastream_id = ds.get("@iot.id")
            connector_name = ds.get("name")
            ds_props = ds.get("properties") or {}
            connector_type = ds_props.get("connectorType")
            current = ds_props.get("current")
            charging_protocol = ds_props.get("chargingProtocol")

            obs = (ds.get("Observations") or [])
            obs = obs[0] if obs else None

            raw_status = obs.get("result") if obs else None
            status = _normalize_status(raw_status)
            marker_color = _status_color(status)
            observed_at = _parse_phenomenon_time(obs.get("phenomenonTime") if obs else None)

            kw_nominal = _estimate_kw(current, connector_type)

            rows.append({
                "thing_id": thing_id,
                "asset_id": asset_id,
                "station_name": station_name,
                "address": address,
                "latitude": float(lat) if lat is not None else None,
                "longitude": float(lon) if lon is not None else None,
                "datastream_id": datastream_id,
                "connector_name": connector_name,
                "connector_type": connector_type,
                "current": current,
                "charging_protocol": charging_protocol,
                "raw_status": raw_status,
                "status": status,
                "marker_color": marker_color,
                "observed_at": observed_at,
                "kw_nominal": kw_nominal,
            })

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # Cleanup
    df = df.dropna(subset=["latitude", "longitude", "observed_at"])
    df = df.drop_duplicates(subset=["datastream_id", "observed_at"])
    df = df.sort_values("observed_at").reset_index(drop=True)

    # Column order
    cols = [
        "thing_id", "asset_id", "station_name", "address",
        "latitude", "longitude",
        "datastream_id", "connector_name", "connector_type",
        "current", "charging_protocol",
        "raw_status", "status", "marker_color", "observed_at",
        "kw_nominal",
    ]
    df = df[cols]
    return df

@st.cache_data(show_spinner=True)
def aggregate_stations(connectors_df: pd.DataFrame) -> pd.DataFrame:
    """
    Station-level metrics incl. estimated kW:
      total_connectors, n_free, n_busy, n_reserved, n_out_of_order,
      n_suspended, n_unknown, last_seen,
      kw_available_now, kw_in_use, kw_total_nominal
    """
    if connectors_df.empty:
        return connectors_df.copy()

    base = ["thing_id", "asset_id", "station_name", "address", "latitude", "longitude"]

    counts = (
        connectors_df.assign(count=1)
        .pivot_table(index=base, columns="status", values="count", aggfunc="sum", fill_value=0)
        .reset_index()
    )
    for s in ["free", "busy", "reserved", "out_of_order", "suspended", "unknown"]:
        if s not in counts.columns:
            counts[s] = 0

    counts["total_connectors"] = (
        counts["free"] + counts["busy"] + counts["reserved"] +
        counts["out_of_order"] + counts["suspended"] + counts["unknown"]
    )

    tmp = connectors_df.reset_index()
    time_col = "observed_at" if "observed_at" in tmp.columns else ("time" if "time" in tmp.columns else None)
    if time_col is None:
        tmp["time"] = connectors_df.index
        time_col = "time"
    last_seen = tmp.groupby(base, as_index=False)[time_col].max().rename(columns={time_col: "last_seen"})

    # kW aggregates
    kw = tmp.copy()
    if "kw_nominal" not in kw.columns:
        kw["kw_nominal"] = 0

    kw_free = (
        kw[kw["status"] == "free"]
        .groupby(base, as_index=False)["kw_nominal"]
        .sum()
        .rename(columns={"kw_nominal": "kw_available_now"})
    )

    kw_in_use = (
        kw[kw["status"].isin(["busy", "reserved", "suspended"])]
        .groupby(base, as_index=False)["kw_nominal"]
        .sum()
        .rename(columns={"kw_nominal": "kw_in_use"})
    )

    kw_total = (
        kw.groupby(base, as_index=False)["kw_nominal"]
        .sum().rename(columns={"kw_nominal": "kw_total_nominal"})
    )

    stations = counts.merge(last_seen, on=base, how="left") \
                     .merge(kw_free, on=base, how="left") \
                     .merge(kw_in_use, on=base, how="left") \
                     .merge(kw_total, on=base, how="left")

    for c in ["kw_available_now", "kw_in_use", "kw_total_nominal"]:
        if c not in stations.columns:
            stations[c] = 0
        stations[c] = stations[c].fillna(0).astype(int)

    stations = stations.rename(columns={
        "free": "n_free",
        "busy": "n_busy",
        "reserved": "n_reserved",
        "out_of_order": "n_out_of_order",
        "suspended": "n_suspended",
        "unknown": "n_unknown",
    }).sort_values(["station_name", "thing_id"]).reset_index(drop=True)

    return stations

# ------------------------------------------------------------------------------
# Public API helpers
# ------------------------------------------------------------------------------

@st.cache_data(show_spinner=True)
def load_hamburg_data(url: str = HAMBURG_URL):
    """Fetch live Hamburg data and return (connectors_df, stations_df)."""
    things = fetch_hamburg_things(url)
    connectors = normalize_connectors(things)
    stations = aggregate_stations(connectors)

    if not connectors.empty:
        df = connectors.copy().rename(columns={"observed_at": "time"})
        df = df.set_index("time")
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        else:
            df.index = df.index.tz_convert("UTC")
        connectors = df

    return connectors, stations

# ------------------------------------------------------------------------------
# Geocoding & proximity
# ------------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def geocode_address(address: str):
    """Geocode within Hamburg (DE) only; supports German characters."""
    if not address:
        return None
    geolocator = Nominatim(user_agent="hamburg_ev_app")

    left, bottom, right, top = HAMBURG_BBOX
    # geopy expects viewbox as [(left, top), (right, bottom)]
    viewbox = [(left, top), (right, bottom)]

    loc = geolocator.geocode(
        address,
        exactly_one=True,
        addressdetails=True,
        timeout=10,
        country_codes="de",
        viewbox=viewbox,
        bounded=True,
        language="de",
    )
    return (loc.latitude, loc.longitude) if loc else None

@st.cache_data(show_spinner=False)
def suggest_addresses(query: str, limit: int = 8):
    """Return up to `limit` suggestions (label, lat, lon) within Hamburg."""
    if not query or len(query.strip()) < 3:
        return []
    geolocator = Nominatim(user_agent="hamburg_ev_app")

    left, bottom, right, top = HAMBURG_BBOX
    viewbox = [(left, top), (right, bottom)]

    results = geolocator.geocode(
        query,
        exactly_one=False,
        addressdetails=True,
        timeout=10,
        country_codes="de",
        viewbox=viewbox,
        bounded=True,
        language="de",
        limit=limit,
    )
    out = []
    if results:
        for r in results:
            out.append({"label": r.address, "lat": r.latitude, "lon": r.longitude})
    return out

def find_nearest_station(user_location, stations_df: pd.DataFrame, require_free: bool = False):
    """Find nearest station to (lat, lon). If require_free=True, require n_free > 0."""
    if user_location is None or len(user_location) != 2 or stations_df is None or stations_df.empty:
        return None

    base = stations_df if not require_free else stations_df.loc[stations_df["n_free"] > 0]
    if base.empty:
        return None

    # Work on an explicit copy to avoid SettingWithCopyWarning
    work = base.copy()

    def _dist(row):
        if pd.notna(row.get("latitude")) and pd.notna(row.get("longitude")):
            try:
                return geodesic(user_location, (row["latitude"], row["longitude"])).meters
            except Exception:
                return float("inf")
        return float("inf")

    # Use .assign instead of chained assignment
    work = work.assign(distance_m=work.apply(_dist, axis=1))

    if work["distance_m"].empty:
        return None

    return work.loc[work["distance_m"].idxmin()]


def find_nearest_connector(user_location, connectors_df: pd.DataFrame, require_free: bool = True):
    """Find nearest connector to user_location. If require_free=True, only status=='free'."""
    if user_location is None or len(user_location) != 2 or connectors_df is None or connectors_df.empty:
        return None

    # Start from a clean copy of a flat frame
    df = connectors_df.reset_index().copy()

    work = df.loc[df["status"].str.lower() == "free"].copy() if require_free else df.copy()
    if work.empty:
        return None

    def _dist(row):
        if pd.notna(row.get("latitude")) and pd.notna(row.get("longitude")):
            try:
                return geodesic(user_location, (row["latitude"], row["longitude"])).meters
            except Exception:
                return float("inf")
        return float("inf")

    work = work.assign(distance_m=work.apply(_dist, axis=1))

    if work["distance_m"].empty:
        return None

    return work.loc[work["distance_m"].idxmin()]


# ------------------------------------------------------------------------------
# Map helpers (smaller clustered markers + color legend)
# ------------------------------------------------------------------------------

def _add_map_legend(m):
    """Floating color legend with real colored dots."""
    legend_html = """
    <div style="
        position: fixed; bottom: 18px; left: 18px; z-index: 9999;
        background-color: white; padding: 10px 12px; border: 1px solid #ddd;
        border-radius: 10px; box-shadow: 0 1px 3px rgba(0,0,0,.15); font-size: 13px;">
      <div style="margin-bottom:6px;"><b>Legend</b></div>
      <div><span style="background: green; width:12px; height:12px; display:inline-block; border-radius:50%; margin-right:6px;"></span>free</div>
      <div><span style="background: red; width:12px; height:12px; display:inline-block; border-radius:50%; margin-right:6px;"></span>busy</div>
      <div><span style="background: orange; width:12px; height:12px; display:inline-block; border-radius:50%; margin-right:6px;"></span>reserved</div>
      <div><span style="background: gray; width:12px; height:12px; display:inline-block; border-radius:50%; margin-right:6px;"></span>out_of_order</div>
      <div><span style="background: cadetblue; width:12px; height:12px; display:inline-block; border-radius:50%; margin-right:6px;"></span>suspended</div>
      <div><span style="background: blue; width:12px; height:12px; display:inline-block; border-radius:50%; margin-right:6px;"></span>unknown</div>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))
    return m


def create_charging_map_stations(stations_df: pd.DataFrame, user_location=None):
    """
    Create a Folium map at station level with small CircleMarkers + clustering.
    Color rule: green if n_free>0, else red, else gray/orange/cadetblue/blue.
    """
    m = folium.Map(location=[53.5511, 9.9937], zoom_start=12, control_scale=True)

    if user_location:
        folium.CircleMarker(
            location=user_location, radius=6, color="blue",
            fill=True, fillColor="blue", fillOpacity=0.9,
            popup="You are here", weight=2,
        ).add_to(m)

    cluster = MarkerCluster(name="Stations").add_to(m)

    for _, r in stations_df.iterrows():
        lat, lon = r["latitude"], r["longitude"]
        if pd.isna(lat) or pd.isna(lon):
            continue

        if r["n_free"] > 0:
            color = "green"
        elif r["n_busy"] > 0:
            color = "red"
        elif r["n_out_of_order"] > 0:
            color = "gray"
        elif r["n_reserved"] > 0:
            color = "orange"
        elif r["n_suspended"] > 0:
            color = "cadetblue"
        else:
            color = "blue"

        popup_html = (
            f"<b>{r.get('station_name','(no name)')}</b><br>"
            f"{r.get('address','') or ''}<br>"
            f"Free: {int(r['n_free'])}/{int(r['total_connectors'])}<br>"
            f"Busy: {int(r['n_busy'])} • Reserved: {int(r['n_reserved'])} • "
            f"Out: {int(r['n_out_of_order'])} • Susp: {int(r['n_suspended'])}"
        )

        folium.CircleMarker(
            location=[lat, lon],
            radius=6, color=color, weight=2,
            fill=True, fillColor=color, fillOpacity=0.9,
            popup=folium.Popup(popup_html, max_width=320),
        ).add_to(cluster)

    return _add_map_legend(m)


def create_charging_map_connectors(connectors_df: pd.DataFrame, user_location=None):
    """
    Create a Folium map at connector level (one marker per Datastream)
    with small CircleMarkers + clustering.
    """
    m = folium.Map(location=[53.5511, 9.9937], zoom_start=12, control_scale=True)

    if user_location:
        folium.CircleMarker(
            location=user_location, radius=6, color="blue",
            fill=True, fillColor="blue", fillOpacity=0.9,
            popup="You are here", weight=2,
        ).add_to(m)

    # Ensure we have a plain DataFrame (connectors may be time-indexed)
    df = connectors_df.reset_index().copy()
    cluster = MarkerCluster(name="Connectors").add_to(m)

    for _, r in df.iterrows():
        lat, lon = r.get("latitude"), r.get("longitude")
        if pd.isna(lat) or pd.isna(lon):
            continue
        color = r.get("marker_color", "blue")

        # best-effort local time string
        observed = r.get("observed_at") if "observed_at" in r else r.get("time")
        try:
            obs_local = pd.to_datetime(observed, utc=True).tz_convert("Europe/Berlin").strftime("%Y-%m-%d %H:%M:%S %Z")
        except Exception:
            obs_local = ""

        popup_html = (
            f"<b>{r.get('station_name','(no name)')}</b><br>"
            f"{r.get('connector_name','')}"
            f"{' • ' + str(r.get('connector_type')) if pd.notna(r.get('connector_type')) else ''}<br>"
            f"Status: <b>{r.get('status','unknown')}</b>"
            f"{'<br>Observed: ' + obs_local if obs_local else ''}"
        )

        folium.CircleMarker(
            location=[lat, lon],
            radius=5, color=color, weight=2,
            fill=True, fillColor=color, fillOpacity=0.9,
            popup=folium.Popup(popup_html, max_width=320),
        ).add_to(cluster)

    return _add_map_legend(m)

##Weekly Station Usage Analytics

def compute_weekly_station_usage():
    """
    Compute weekly charging usage (last 7 days) aggregated by unique datastream/station.
    Ensures grouping by both datastream_id and station_id to avoid collapsing.
    """
    with engine.connect() as conn:
        query = text("""
            WITH obs AS (
                SELECT o.datastream_id, 
                       o.station_id, 
                       o.phenomenon_time, 
                       o.result
                FROM observations o
                WHERE o.phenomenon_time >= NOW() - INTERVAL '7 days'
            ),
            charging AS (
                SELECT *,
                       LEAD(phenomenon_time) OVER (
                           PARTITION BY datastream_id, station_id 
                           ORDER BY phenomenon_time
                       ) AS next_time
                FROM obs
                WHERE result = 'CHARGING'
            )
            SELECT 
                ds.datastream_id,
                COALESCE(ds.station_name, 'Unknown') AS station_name,
                COALESCE(ds.address, '—') AS address,
                ds.latitude,
                ds.longitude,
                SUM(EXTRACT(EPOCH FROM (COALESCE(next_time, NOW()) - phenomenon_time)) / 60) AS charging_minutes,
                COUNT(*) AS sessions
            FROM charging c
            JOIN datastream_station ds 
              ON c.datastream_id = ds.datastream_id
            GROUP BY 
                ds.datastream_id, 
                ds.station_name, 
                ds.address, 
                ds.latitude, 
                ds.longitude
            ORDER BY charging_minutes DESC;
        """)
        df = pd.read_sql(query, conn)

    df["charging_hours"] = df["charging_minutes"] / 60
    return df


# ... (keep existing imports and code)

# ===============================
# Analytics Helpers (Daily + Yesterday)
# ===============================
from datetime import date, timedelta
from sqlalchemy import text

# ----------------------------------------------------------------
# ⚡ Realistic Top-10 busiest stations (aggregated by station_name)
# ----------------------------------------------------------------
def compute_daily_top10():
    """Return today's top 10 busiest EV stations (aggregated by station_name)."""
    query = text("""
        WITH ordered AS (
            SELECT
                o.station_id,
                ds.station_name,
                ds.address,
                ds.latitude,
                ds.longitude,
                o.phenomenon_time,
                o.result,
                LEAD(o.phenomenon_time)
                    OVER (PARTITION BY o.station_id ORDER BY o.phenomenon_time)
                    AS next_time
            FROM observations o
            JOIN datastream_station ds ON o.station_id = ds.station_id
            WHERE o.phenomenon_time::date = CURRENT_DATE
        ),
        charging_intervals AS (
            SELECT
                station_name,
                address,
                latitude,
                longitude,
                EXTRACT(EPOCH FROM (next_time - phenomenon_time)) / 3600.0 AS duration_hrs
            FROM ordered
            WHERE result = 'CHARGING' AND next_time IS NOT NULL
        ),
        aggregated AS (
            SELECT
                station_name,
                address,
                latitude,
                longitude,
                SUM(duration_hrs) AS charging_hours,
                COUNT(*) AS sessions
            FROM charging_intervals
            GROUP BY station_name, address, latitude, longitude
        )
        SELECT *
        FROM aggregated
        WHERE charging_hours > 0
        ORDER BY charging_hours DESC
        LIMIT 10;
    """)
    with engine.begin() as conn:
        return pd.read_sql(query, conn)


def compute_yesterday_top10():
    """Return yesterday's top 10 busiest EV stations (aggregated by station_name)."""
    query = text("""
        WITH ordered AS (
            SELECT
                o.station_id,
                ds.station_name,
                ds.address,
                ds.latitude,
                ds.longitude,
                o.phenomenon_time,
                o.result,
                LEAD(o.phenomenon_time)
                    OVER (PARTITION BY o.station_id ORDER BY o.phenomenon_time)
                    AS next_time
            FROM observations o
            JOIN datastream_station ds ON o.station_id = ds.station_id
            WHERE o.phenomenon_time::date = CURRENT_DATE - INTERVAL '1 day'
        ),
        charging_intervals AS (
            SELECT
                station_name,
                address,
                latitude,
                longitude,
                EXTRACT(EPOCH FROM (next_time - phenomenon_time)) / 3600.0 AS duration_hrs
            FROM ordered
            WHERE result = 'CHARGING' AND next_time IS NOT NULL
        ),
        aggregated AS (
            SELECT
                station_name,
                address,
                latitude,
                longitude,
                SUM(duration_hrs) AS charging_hours,
                COUNT(*) AS sessions
            FROM charging_intervals
            GROUP BY station_name, address, latitude, longitude
        )
        SELECT *
        FROM aggregated
        WHERE charging_hours > 0
        ORDER BY charging_hours DESC
        LIMIT 10;
    """)
    with engine.begin() as conn:
        return pd.read_sql(query, conn)



def compute_daily_usage(top_station_names: List[str]) -> pd.DataFrame:
    """
    Compute daily charging hours for selected top stations by station_name.
    """
    if not top_station_names:
        return pd.DataFrame(columns=["station_name", "date", "charging_hours"])

    placeholders = ",".join([f":s{i}" for i in range(len(top_station_names))])
    params = {f"s{i}": name for i, name in enumerate(top_station_names)}

    with engine.connect() as conn:
        query = text(f"""
            SELECT ds.station_name,
                   DATE(o.phenomenon_time) AS date,
                   COUNT(*) AS sessions
            FROM observations o
            JOIN datastream_station ds ON o.datastream_id = ds.datastream_id
            WHERE o.result = 'CHARGING'
              AND ds.station_name IN ({placeholders})
            GROUP BY ds.station_name, DATE(o.phenomenon_time)
            ORDER BY date;
        """)
        df = pd.read_sql(query, conn, params=params)

    # Convert sessions to hours (1 session ≈ 1 hr assumption)
    df["charging_hours"] = df["sessions"] * 1.0
    return df



# ------------------------------------------------------------------------------
# 📊 Daily Analytics Readers (for Streamlit Analytics tab)
# ------------------------------------------------------------------------------

def read_daily_summary(limit_days: int = 7):
    """
    Return recent rows from the daily_station_usage table
    for summary charts and KPIs.
    """
    try:
        with engine.connect() as conn:
            q = text("""
                SELECT date, station_name, sessions, charging_hours
                FROM daily_station_usage
                WHERE date >= CURRENT_DATE - :n
                ORDER BY date, station_name;
            """)
            df = pd.read_sql(q, conn, params={"n": limit_days})
        return df
    except Exception as e:
        log.error(f"Error reading daily summary: {e}")
        return pd.DataFrame(columns=["date", "station_name", "sessions", "charging_hours"])


def read_top_today(top_n: int = 10):
    """
    Return top stations for today by charging_hours from daily_station_usage.
    """
    try:
        with engine.connect() as conn:
            q = text("""
                SELECT station_name, sessions, charging_hours, address
                FROM daily_station_usage
                WHERE date = CURRENT_DATE
                ORDER BY charging_hours DESC, sessions DESC
                LIMIT :n;
            """)
            df = pd.read_sql(q, conn, params={"n": top_n})
        return df
    except Exception as e:
        log.error(f"Error reading top today: {e}")
        return pd.DataFrame(columns=["station_name", "sessions", "charging_hours", "address"])
