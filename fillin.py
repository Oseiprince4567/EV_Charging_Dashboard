import os
import time
from sqlalchemy import create_engine, text

# --- Handle missing requests gracefully ---
try:
    import requests
except ModuleNotFoundError:
    import streamlit as st
    st.error("⚠️ 'requests' module missing — please rebuild dependencies.")
    raise

# --- Database connection (uses GitHub Secret or local env var) ---
PG_URI = os.getenv("DATABASE_URL")
if not PG_URI:
    raise ValueError("❌ DATABASE_URL environment variable not set. Did you add it as a GitHub Secret?")

engine = create_engine(PG_URI, pool_pre_ping=True)

# ============================================================
# Base endpoint — no filter, fetch all pages
# ============================================================
BASE_URL = (
    "https://iot.hamburg.de/v1.0/Things"
    "?$expand=Locations,Datastreams($expand=Observations($top=5;$orderby=phenomenonTime desc))"
)

def fetch_page(url, attempt=1, max_attempts=5):
    """Fetch one API page with retry/backoff."""
    try:
        r = requests.get(url, timeout=30)
        if r.status_code in (429, 500, 502, 503, 504):
            if attempt >= max_attempts:
                r.raise_for_status()
            sleep_s = min(2 ** attempt, 30)
            print(f"⚠️ Retry {attempt} — sleeping {sleep_s}s …")
            time.sleep(sleep_s)
            return fetch_page(url, attempt + 1, max_attempts)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        raise RuntimeError(f"HTTP error on {url}: {e}")

def iter_all_things(first_url):
    """Iterate through all pages."""
    url = first_url
    total = 0
    while url:
        payload = fetch_page(url)
        value = payload.get("value", [])
        total += len(value)
        yield from value
        url = payload.get("@iot.nextLink")
        if url:
            print("  ↪️ Fetching next page…")
    print(f"✅ Retrieved {total} Things in total.")

def is_ev_station(thing):
    """Client-side EV filter."""
    textfields = (
        (thing.get("name") or "") +
        " " +
        (thing.get("description") or "")
    ).lower()
    return "ladestation" in textfields

def upsert_station(conn, thing):
    if not thing.get("Locations"):
        return None
    loc = thing["Locations"][0]
    try:
        coords = loc["location"]["geometry"]["coordinates"]
        lat, lon = float(coords[1]), float(coords[0])
    except Exception:
        return None

    addr = (loc.get("description") or "").strip() or "Unspecified"
    name = thing.get("name", "Unnamed Station").strip()
    thing_id = str(thing.get("@iot.id"))

    connector = "Unknown"
    power_kw = None
    for ds in thing.get("Datastreams", []):
        props = ds.get("properties") or {}
        if props.get("connectorType"):
            connector = props["connectorType"]
        break

    conn.execute(
        text("""
            INSERT INTO datastream_station
                (datastream_id, station_name, address, latitude, longitude,
                 connector_type, power_kw, city, country)
            VALUES
                (:dsid, :name, :addr, :lat, :lon, :conn, :pkw, 'Hamburg', 'Germany')
            ON CONFLICT (datastream_id) DO UPDATE
            SET station_name = EXCLUDED.station_name,
                address      = EXCLUDED.address,
                latitude     = EXCLUDED.latitude,
                longitude    = EXCLUDED.longitude;
        """),
        dict(dsid=thing_id, name=name, addr=addr, lat=lat, lon=lon, conn=connector, pkw=power_kw),
    )
    return thing_id

def insert_latest_obs(conn, dsid, datastreams):
    for ds in datastreams:
        obs_list = ds.get("Observations") or []
        if not obs_list:
            continue
        obs = obs_list[0]
        result = obs.get("result")
        t = obs.get("phenomenonTime")
        if not (result and t):
            continue
        conn.execute(
            text("""
                INSERT INTO observations (station_id, phenomenon_time, result)
                SELECT station_id, :t, :r
                FROM datastream_station WHERE datastream_id=:dsid;
            """),
            dict(t=t, r=result, dsid=dsid),
        )

def create_ev_view(conn):
    conn.execute(
        text("""
            CREATE OR REPLACE VIEW ev_stations AS
            SELECT *
            FROM datastream_station
            WHERE position('ladestation' in lower(station_name)) > 0
               OR position('ladestation' in lower(address)) > 0;
        """)
    )

def main():
    print("🔄 Fetching all Hamburg IoT Things (client-side EV filter)...")
    inserted = 0
    with engine.begin() as conn:
        for thing in iter_all_things(BASE_URL):
            if not is_ev_station(thing):
                continue  # skip non-EV
            try:
                dsid = upsert_station(conn, thing)
                if dsid:
                    insert_latest_obs(conn, dsid, thing.get("Datastreams", []))
                    inserted += 1
            except Exception as e:
                print(f"⚠️ Skipped {thing.get('name','<no name>')}: {e}")
        create_ev_view(conn)
    print(f"🚀 Sync complete — {inserted} EV stations processed.")

if __name__ == "__main__":
    main()
