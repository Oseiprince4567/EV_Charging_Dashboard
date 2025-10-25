# evhamapp.py — Hamburg EV Charging (LIVE + Analytics via PostgreSQL)
# Full app with two pages:
# 1) ⚡ Live Dashboard (SensorThings)
# 2) 📈 Analytics (PostgreSQL weekly/daily usage)

import re
import math
import pandas as pd
import streamlit as st
import sys
# --- Ensure folium is installed at runtime ---
try:
    import folium
except ModuleNotFoundError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "folium==0.18.0"])
    import folium
from folium.plugins import MarkerCluster, HeatMap
from streamlit_folium import st_folium
from geopy.distance import geodesic
from hamburghelpers import *
import plotly.express as px
import subprocess
from datetime import datetime

# -------------------------------------------------------------------
# Page config MUST be the first Streamlit call
# -------------------------------------------------------------------
st.set_page_config(page_title="Hamburg EV Charging", page_icon="⚡", layout="wide")

# -------------------------------------------------------------------
# Helpers from our library
# -------------------------------------------------------------------
try:
    from hamburghelpers import (
        load_hamburg_data,
        geocode_address,
        suggest_addresses,
        find_nearest_station,
        find_nearest_connector,
        HAMBURG_URL,
        HAMBURG_BBOX,
        compute_weekly_station_usage,
        compute_daily_usage,
    )
except Exception:
    # Optional fallback if you kept an alternate helpers file name
    from helpers import (
        load_hamburg_data,
        geocode_address,
        suggest_addresses,
        find_nearest_station,
        find_nearest_connector,
        HAMBURG_URL,
        HAMBURG_BBOX,
        compute_weekly_station_usage,
        compute_daily_usage,
    )

HAMBURG_CENTER = (53.5511, 9.9937)

# -------------------------------------------------------------------
# Session state: map center + source
# -------------------------------------------------------------------
if "center" not in st.session_state:
    st.session_state.center = HAMBURG_CENTER
if "center_source" not in st.session_state:
    st.session_state.center_source = "Hamburg center"

def in_bbox(latlon, bbox):
    lat, lon = latlon
    return (bbox[1] <= lat <= bbox[3]) and (bbox[0] <= lon <= bbox[2])

# -------------------------------------------------------------------
# Small utilities used in popups / legends
# -------------------------------------------------------------------
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

def extract_street(address: str) -> str:
    if not isinstance(address, str) or not address.strip():
        return None
    first = address.split(",")[0].strip()
    return re.sub(r"\s+\d+[A-Za-z]?$", "", first)

def connector_lines_for_station(station_row, connectors_df, max_lines=3):
    """Return up to `max_lines` nicely formatted connector lines for the station popup."""
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
            out.append(f"Connector {idx:02d}: {mode} ({ctype}{example}), {cur}{kwpart}")

        extra = max(len(sub) - max_lines, 0)
        if extra > 0:
            out.append(f"+{extra} more…")
    except Exception as e:
        out.append(f"(error building connector info: {e})")
    return out

def add_map_legend(m):
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

# -------------------------------------------------------------------
# Sidebar: page switcher
# -------------------------------------------------------------------
page = st.sidebar.radio("Select View", ["⚡ Live Dashboard", "📈 Analytics"], index=0)

# ===================================================================
# ⚡ LIVE DASHBOARD
# ===================================================================
if page == "⚡ Live Dashboard":
    st.title("⚡ Hamburg EV Charging — Live Status")

    # ---- Source + refresh
    st.sidebar.header("Data source")
    st.sidebar.write("**Live:** Hamburg SensorThings API")
    st.sidebar.caption(HAMBURG_URL)
    if st.sidebar.button("Refresh data"):
        st.cache_data.clear()

    # ---- Live view & filters
    view = st.sidebar.radio("View", ["Stations (recommended)", "Connectors (detailed)"])
    if view.startswith("Stations"):
        min_free = st.sidebar.number_input("Minimum free connectors", min_value=0, value=0, step=1)
    else:
        status_filter = st.sidebar.multiselect(
            "Connector status",
            ["free", "busy", "reserved", "out_of_order", "suspended", "unknown"],
            default=["free", "busy"],
        )

    # ---- Load live data
    with st.spinner("Loading live data from Hamburg …"):
        connectors_df, stations_df = load_hamburg_data()

    # ---- Last update + KPIs
    last_utc = None
    try:
        if connectors_df is not None and not connectors_df.empty:
            last_utc = pd.to_datetime(connectors_df.index.max(), utc=True)
        elif stations_df is not None and not stations_df.empty and "last_seen" in stations_df.columns:
            last_utc = pd.to_datetime(stations_df["last_seen"].max(), utc=True)
    except Exception:
        last_utc = None
    if last_utc is not None:
        last_local = last_utc.tz_convert("Europe/Berlin")
        st.caption(f"Last update (local): **{last_local.strftime('%Y-%m-%d %H:%M:%S %Z')}**")

    k1, k2, k3, k4 = st.columns(4)
    stations_count   = len(stations_df) if stations_df is not None else 0
    connectors_count = int(stations_df["total_connectors"].sum()) if stations_df is not None and "total_connectors" in stations_df else 0
    free_count       = int(stations_df["n_free"].sum()) if stations_df is not None and "n_free" in stations_df else 0
    busy_count       = int(stations_df["n_busy"].sum()) if stations_df is not None and "n_busy" in stations_df else 0
    k1.metric("Stations", f"{stations_count:,}")
    k2.metric("Connectors", f"{connectors_count:,}")
    k3.metric("Free connectors", f"{free_count:,}")
    k4.metric("Busy connectors", f"{busy_count:,}")

    # ---- Street dropdown (pre-geocoded)
    st.sidebar.header("Find nearest (Hamburg)")
    # Build street list once
    street_options = []
    if stations_df is not None and not stations_df.empty and "address" in stations_df.columns:
        streets = (
            stations_df["address"]
            .dropna()
            .map(extract_street)
            .dropna()
            .unique()
            .tolist()
        )
        street_options = sorted(set(streets), key=lambda s: s.casefold())

    mode = st.sidebar.radio("Location input", ["Street list (recommended)", "Type address", "Coordinates"], index=0)
    addr_query = ""

    if mode == "Street list (recommended)":
        if street_options:
            selected_street = st.sidebar.selectbox("Street (type to search)", options=street_options, index=0, key="street_select")
            def _street(addr: str) -> str:
                if not isinstance(addr, str) or not addr.strip():
                    return ""
                first = addr.split(",")[0].strip()
                return re.sub(r"\s+\d+[A-Za-z]?$", "", first)
            subset = stations_df[stations_df["address"].map(_street) == selected_street]
            if not subset.empty:
                ctr_lat = float(subset["latitude"].median())
                ctr_lon = float(subset["longitude"].median())
                st.session_state.center = (ctr_lat, ctr_lon)
                st.session_state.center_source = f"street: {selected_street}"
        else:
            st.sidebar.info("No street list available yet — switch to 'Type address'.")

    elif mode == "Type address":
        addr_query = st.sidebar.text_input("Adresse / Ort (Vorschläge ab 3 Zeichen)")
        suggestions = []
        if len(addr_query.strip()) >= 3:
            try:
                suggestions = suggest_addresses(addr_query, limit=10) or []
            except Exception:
                suggestions = []
        if suggestions:
            labels = ["⟪Select suggestion⟫"] + [s["label"] for s in suggestions]
            sel = st.sidebar.selectbox("Suggestions", labels, index=0, key="addr_suggestion")
            if sel != labels[0]:
                chosen = next(s for s in suggestions if s["label"] == sel)
                st.session_state.center = (float(chosen["lat"]), float(chosen["lon"]))
                st.session_state.center_source = "suggestion"

    elif mode == "Coordinates":
        lat = st.sidebar.text_input("Lat")
        lon = st.sidebar.text_input("Lon")
        if lat and lon:
            try:
                candidate = (float(lat), float(lon))
                if in_bbox(candidate, HAMBURG_BBOX):
                    st.session_state.center = candidate
                    st.session_state.center_source = "coordinates"
                else:
                    st.sidebar.warning("Outside Hamburg — using city center.")
                    st.session_state.center = HAMBURG_CENTER
                    st.session_state.center_source = "Hamburg center"
            except ValueError:
                st.sidebar.warning("Invalid coordinates.")

    require_free    = st.sidebar.checkbox("Require free connector", value=True)
    compute_nearest = st.sidebar.button("Compute nearest")
    if compute_nearest and mode == "Type address":
        addr_query = addr_query.strip()
        if addr_query:
            try:
                geocoded = geocode_address(addr_query)
            except Exception:
                geocoded = None
            if geocoded and in_bbox(geocoded, HAMBURG_BBOX):
                st.session_state.center = geocoded
                st.session_state.center_source = "geocoded"
            else:
                st.sidebar.warning("Address not found in Hamburg — using city center.")
                st.session_state.center = HAMBURG_CENTER
                st.session_state.center_source = "Hamburg center"

    # ---- Center info
    user_loc = st.session_state.center
    st.caption(f"Center: {user_loc[0]:.5f}, {user_loc[1]:.5f} ({st.session_state.center_source})")

    # ---------------- Map ----------------
    st.subheader("Interactive map")

    # Base map
    m = folium.Map(location=user_loc, zoom_start=15, tiles="cartodbpositron", control_scale=True)
    # "You are here"
    folium.CircleMarker(location=user_loc, radius=6, color="blue",
                        fill=True, fillColor="blue", fillOpacity=0.9,
                        popup="You are here", weight=2).add_to(m)
    cluster = MarkerCluster(name="Locations").add_to(m)

    def add_distance_str(latlon):
        try:
            return f"{int(geodesic(user_loc, latlon).meters)} m"
        except Exception:
            return "—"

    # Markers
    if view.startswith("Stations"):
        df_map = stations_df.copy()
        if min_free > 0:
            df_map = df_map[df_map["n_free"] >= min_free]

        for _, r in df_map.iterrows():
            lat_s, lon_s = r["latitude"], r["longitude"]
            if pd.isna(lat_s) or pd.isna(lon_s):
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

            dist_str = add_distance_str((lat_s, lon_s))
            conn_lines = connector_lines_for_station(r, connectors_df, max_lines=3)
            if not conn_lines:
                inferred_cur = "AC/DC Mix" if r["total_connectors"] > 0 else "Unknown"
                inferred_kw = r.get("kw_available_now", 0)
                conn_lines = [f"Station: {inferred_cur} — ~{inferred_kw} kW available"]

            kw_free_est = r.get("kw_available_now", 0)
            counts_line = (
                f"Free: {int(r['n_free'])}/{int(r['total_connectors'])}"
                f"{f' (~{kw_free_est} kW free)' if kw_free_est > 0 else ''}<br>"
                f"Busy: {int(r['n_busy'])} • Reserved: {int(r['n_reserved'])}"
            )
            updated_line = f"Distance: {dist_str} | Updated: {format_updated(r.get('last_seen'))}"

            popup_html = (
                f"<b>{r.get('station_name','(no name)')}</b><br>"
                f"{r.get('address','') or ''}<br>"
                f"{counts_line}<br>"
                + "<br>".join(conn_lines) + "<br>"
                + updated_line
            )

            folium.CircleMarker(
                location=[lat_s, lon_s],
                radius=3, color=color, weight=1.5,
                fill=True, fillColor=color, fillOpacity=0.9,
                popup=folium.Popup(popup_html, max_width=350),
            ).add_to(cluster)

    else:
        df_map = connectors_df.reset_index().copy()
        if "status" in df_map.columns and 'status_filter' in locals() and status_filter:
            df_map = df_map[df_map["status"].isin(status_filter)]

        for _, r in df_map.iterrows():
            lat_c, lon_c = r.get("latitude"), r.get("longitude")
            if pd.isna(lat_c) or pd.isna(lon_c):
                continue

            color = r.get("marker_color", "blue")
            observed = r.get("observed_at") if "observed_at" in r else r.get("time")
            try:
                obs_local = pd.to_datetime(observed, utc=True).tz_convert("Europe/Berlin").strftime("%Y-%m-%d %H:%M:%S %Z")
            except Exception:
                obs_local = ""
            dist_str = add_distance_str((lat_c, lon_c))

            ctype = str(r.get("connector_type") or "")
            examples = CONNECTOR_EXAMPLES.get(ctype, [])
            example_str = f", e.g., {examples[0]}" if examples else ""
            cur = CURRENT_LABEL.get(str(r.get("current") or "").upper(), "")
            kw = r.get("kw_nominal", "")
            kwpart = f" — ~{int(kw)} kW" if isinstance(kw, (int, float)) and kw > 0 else ""

            popup_html = (
                f"<b>{r.get('station_name','(no name)')}</b><br>"
                f"{r.get('connector_name','')}{' • ' + ctype if ctype else ''}{example_str}<br>"
                f"Status: <b>{r.get('status','unknown')}</b><br>"
                f"{cur}{kwpart}<br>"
                f"{'<br>Observed: ' + obs_local if obs_local else ''}<br>"
                f"Distance: <b>{dist_str}</b>"
            )

            folium.CircleMarker(
                location=[lat_c, lon_c],
                radius=2.5, color=color, weight=1.5,
                fill=True, fillColor=color, fillOpacity=0.9,
                popup=folium.Popup(popup_html, max_width=320),
            ).add_to(cluster)

    m = add_map_legend(m)

    # ---- Nearest (connector first, then station) + draw line
    st.subheader("Nearest station / connector")
    visible_connectors = connectors_df.reset_index()
    if view.startswith("Connectors") and 'status_filter' in locals() and status_filter:
        visible_connectors = visible_connectors[visible_connectors["status"].isin(status_filter)]

    nearest_conn = find_nearest_connector(st.session_state.center, visible_connectors, require_free=require_free)
    if nearest_conn is None and require_free:
        nearest_conn = find_nearest_connector(st.session_state.center, visible_connectors, require_free=False)

    if nearest_conn is not None:
        dist_m = int(nearest_conn["distance_m"])
        loc_conn = (float(nearest_conn["latitude"]), float(nearest_conn["longitude"]))
        if all(math.isfinite(v) for v in [st.session_state.center[0], st.session_state.center[1], loc_conn[0], loc_conn[1]]):
            folium.PolyLine([st.session_state.center, loc_conn], color="blue", weight=3, opacity=0.6).add_to(m)
            m.fit_bounds([st.session_state.center, loc_conn])

            ctype = str(nearest_conn.get("connector_type") or "")
            examples = CONNECTOR_EXAMPLES.get(ctype, [])
            example_str = f", e.g., {examples[0]}" if examples else ""
            cur = CURRENT_LABEL.get(str(nearest_conn.get("current") or "").upper(), "")
            kw = nearest_conn.get("kw_nominal")
            if not (isinstance(kw, (int, float)) and kw > 0):
                cur_raw = str(nearest_conn.get("current") or "").upper()
                kw = 22 if cur_raw == "AC" else (50 if cur_raw == "DC" else None)
            kwpart = f" — ~{int(kw)} kW" if kw else ""

            popup_html = (
                f"<b>Nearest connector</b><br>"
                f"{nearest_conn.get('station_name','(no name)')}<br>"
                f"Status: <b>{nearest_conn.get('status','?')}</b><br>"
                f"{cur}{kwpart}{example_str}<br>"
                f"Distance: <b>{dist_m} m</b>"
            )

            folium.CircleMarker(
                location=loc_conn, radius=5, color="black",
                fill=True, fillColor="yellow", fillOpacity=0.9,
                popup=folium.Popup(popup_html, max_width=280),
                weight=2,
            ).add_to(m)

        try:
            obs = nearest_conn.get("observed_at") or nearest_conn.get("time")
            obs_local = pd.to_datetime(obs, utc=True).tz_convert("Europe/Berlin").strftime("%Y-%m-%d %H:%M:%S %Z")
        except Exception:
            obs_local = "—"

        msg = f"**Nearest connector:** {nearest_conn.get('station_name','(no name)')} — {dist_m} m away"
        if kw:
            msg += f" · at ~{int(kw)} kW"
        msg += f" • status **{nearest_conn.get('status','?')}**"
        if obs_local:
            msg += f" • observed {obs_local}"
        st.success(msg)

    else:
        nearest = find_nearest_station(st.session_state.center, stations_df, require_free=require_free)
        if nearest is None:
            st.info("No station matched the criteria.")
        else:
            dist_m = int(geodesic(st.session_state.center, (nearest['latitude'], nearest['longitude'])).meters)
            loc_sta = (float(nearest["latitude"]), float(nearest["longitude"]))
            if all(math.isfinite(v) for v in [st.session_state.center[0], st.session_state.center[1], loc_sta[0], loc_sta[1]]):
                folium.PolyLine([st.session_state.center, loc_sta], color="blue", weight=3, opacity=0.6).add_to(m)
                m.fit_bounds([st.session_state.center, loc_sta])

                folium.CircleMarker(
                    location=loc_sta, radius=5, color="black",
                    fill=True, fillColor="yellow", fillOpacity=0.9,
                    popup=folium.Popup(
                        f"<b>Nearest station</b><br>{nearest['station_name']}<br>"
                        f"Free: {int(nearest['n_free'])}/{int(nearest['total_connectors'])}<br>"
                        f"Distance: <b>{dist_m} m</b>", max_width=260),
                    weight=2,
                ).add_to(m)

            last_seen_local = pd.to_datetime(nearest["last_seen"], utc=True, errors="coerce")
            last_seen_str = last_seen_local.tz_convert("Europe/Berlin").strftime("%Y-%m-%d %H:%M:%S %Z") if pd.notna(last_seen_local) else "—"
            kw_free_est = nearest.get("kw_available_now", 0)

            msg = (
                f"**Nearest station:** {nearest['station_name']} — {dist_m} m away · "
                f"Free: {int(nearest['n_free'])}/{int(nearest['total_connectors'])}"
            )
            if kw_free_est and kw_free_est > 0:
                msg += f" (~{kw_free_est} kW)"
            msg += f" · Last seen: {last_seen_str}"
            st.success(msg)

    # Color chips
    st.markdown(
        """
        <div style="margin-top:6px; margin-bottom:8px; font-size:13px;">
          <b>Color codes:</b>
          <span style="display:inline-block; background:green; width:12px; height:12px; border-radius:50%; margin:0 6px 0 10px;"></span>free
          <span style="display:inline-block; background:red; width:12px; height:12px; border-radius:50%; margin:0 6px 0 12px;"></span>busy
          <span style="display:inline-block; background:orange; width:12px; height:12px; border-radius:50%; margin:0 6px 0 12px;"></span>reserved
          <span style="display:inline-block; background:gray; width:12px; height:12px; border-radius:50%; margin:0 6px 0 12px;"></span>out_of_order
          <span style="display:inline-block; background:cadetblue; width:12px; height:12px; border-radius:50%; margin:0 6px 0 12px;"></span>suspended
          <span style="display:inline-block; background:blue; width:12px; height:12px; border-radius:50%; margin:0 6px 0 12px;"></span>unknown
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Render map (key forces re-center on change)
    ret = st_folium(m, width=1100, height=650, key=f"map-{round(user_loc[0],5)}-{round(user_loc[1],5)}-{view}")

    # Click-to-recenter
    if isinstance(ret, dict) and ret.get("last_clicked"):
        lat = ret["last_clicked"].get("lat")
        lon = ret["last_clicked"].get("lng")
        if lat is not None and lon is not None:
            candidate = (float(lat), float(lon))
            if in_bbox(candidate, HAMBURG_BBOX):
                st.session_state.center = candidate
                st.session_state.center_source = "map click"

    # Data previews
    with st.expander("Preview: stations dataframe"):
        st.dataframe(stations_df.head(50))
    with st.expander("Preview: connectors dataframe"):
        if connectors_df is not None and not connectors_df.empty:
            st.dataframe(connectors_df.tail(50).reset_index().rename(columns={"index": "time"}))
        else:
            st.write("No connector data.")



# ===================================================================
# 📈 ANALYTICS (PostgreSQL — Combined: Today & Yesterday)
# ===================================================================
elif page == "📈 Analytics":
    from hamburghelpers import compute_daily_top10, compute_yesterday_top10
    from datetime import date, timedelta
    

    st.title("📈 EV Charging Activity Dashboard")

    # ===========================================================
    # 🔄 Refresh IoT Data Button
    # ===========================================================
    st.markdown("### 🔄 Live Data Update Status")
    st.info(
    """
    This dashboard is powered by the **Hamburg SensorThings API**, which is
    automatically updated every **six hours** via **GitHub Actions.  
    This automation connects to a **Neon PostgreSQL** database to ensure that all maps
    and analytics reflect the **latest IoT observations.
    """
    )

    tab_today, tab_yesterday = st.tabs(["⚡ Today (Live Analytics)", "📅 Yesterday (Summary)"])

    # -----------------------------------------------------------
    # 🟢 TODAY TAB
    # -----------------------------------------------------------
    with tab_today:
        st.subheader("⚡ Top 10 Active Stations (Today)")

        usage_today = compute_daily_top10()

        if usage_today.empty:
            st.warning("No live data found for today.")
        else:
            total_hours = usage_today["charging_hours"].sum()
            total_sessions = usage_today["sessions"].sum()

            c1, c2, c3 = st.columns(3)
            c1.metric("Total charging hours (today)", f"{total_hours:.1f} h")
            c2.metric("Total sessions (today)", f"{int(total_sessions)}")
            c3.metric("Active stations", len(usage_today))

            # Data table
            st.dataframe(
                usage_today[["station_name", "address", "charging_hours", "sessions"]]
                .sort_values("charging_hours", ascending=False)
                .reset_index(drop=True),
                use_container_width=True,
            )

            # Tabs inside today's analytics
            t1, t2, t3 = st.tabs(["🌍 Heatmap", "🕸️ Radar Chart", "🏆 Top Station"])

            # Heatmap
            with t1:
                from streamlit_folium import st_folium
                from folium.plugins import HeatMap
                import folium
                import numpy as np

                st.subheader("Heatmap of Today’s Top 10 Stations")
                m = folium.Map(location=[53.55, 9.99], zoom_start=12, tiles="CartoDB positron")
                # --- Clean & validate coordinates ---
                usage_today["latitude"] = pd.to_numeric(usage_today["latitude"], errors="coerce")
                usage_today["longitude"] = pd.to_numeric(usage_today["longitude"], errors="coerce")
                usage_today["charging_hours"] = pd.to_numeric(usage_today["charging_hours"], errors="coerce").fillna(0)
                usage_today["sessions"] = pd.to_numeric(usage_today["sessions"], errors="coerce").fillna(0)

                valid = usage_today.dropna(subset=["latitude", "longitude", "charging_hours"])
                valid = valid[
                    (valid["latitude"].between(-90, 90))
                    & (valid["longitude"].between(-180, 180))
                    & (valid["charging_hours"] > 0)
                ]
                
                max_hours = valid["charging_hours"].max()
                if pd.isna(max_hours) or max_hours <= 0:
                    max_hours = 1.0  # prevent divide-by-zero
                    
                heat_points = []        
                for r in valid.itertuples():
                    try:
                        lat = float(r.latitude)
                        lon = float(r.longitude)
                        weight = float(r.charging_hours / max_hours)
                        if (
                            np.isfinite(lat)
                            and np.isfinite(lon)
                            and np.isfinite(weight)
                            and -90 <= lat <= 90
                            and -180 <= lon <= 180
                        ):
                            heat_points.append([lat, lon, weight])
                    except Exception:
                        continue  # Skip bad rows safely
                
                # --- Add HeatMap layer safely ---
                if len(heat_points) > 0:
                   try:
                       #Force every coordinate to be numeric float32 before Folium check
                       heat_points = np.array(heat_points, dtype="float32").tolist()
                       HeatMap(heat_points, radius=25, blur=15, max_zoom=14).add_to(m)
                   except TypeError as e:
                       st.warning("⚠️ Some invalid coordinate values were skipped in the heatmap.")
                   except Exception as e:
                       st.error(f"⚠️ Heatmap rendering error: {e}")
                else:
                    st.warning("No valid coordinates found for today's heatmap.")
                    
                # --- Add station markers ---
                for r in valid.itertuples():
                    try:
                        popup = (
                            f"<b>{r.station_name}</b><br>"
                            f"📍 {r.address}<br>"
                            f"⏱ {r.charging_hours:.1f} h<br>"
                            f"🔌 {int(r.sessions)} sessions"
                        )
                        folium.CircleMarker(
                            location=[float(r.latitude), float(r.longitude)],
                            radius=6,
                            color="red",
                            fill=True,
                            fill_color="orange",
                            fill_opacity=0.85,
                            popup=popup,
                        ).add_to(m)
                    except Exception:
                        continue

            st_folium(m, width=1000, height=520)

            # 🕸️ Radar chart (Plotly — Today)
            with t2:
                import plotly.express as px
                st.subheader("Top 10 Station Performance (Today – Radar Chart)")

                fig = px.line_polar(
                    usage_today,
                    r="charging_hours",
                    theta="station_name",
                    line_close=True,
                    color="station_name",
                    template="plotly_dark",
                    markers=True,
                )

                fig.update_traces(fill="toself", opacity=0.7)
                fig.update_layout(
                    height=520,
                    legend_title_text="Station",
                    legend=dict(
                        orientation="v",
                        yanchor="top",
                        y=1,
                        xanchor="left",
                        x=1.25,
                        bgcolor="rgba(0,0,0,0)",
                        bordercolor="gray",
                        borderwidth=0.5,
                        font=dict(size=10)
                    ),
                    polar=dict(
                        radialaxis=dict(
                            showline=True,
                            linewidth=1,
                            gridcolor="gray",
                            gridwidth=0.3,
                            tickfont=dict(size=12, color="white", family="Arial Black")
                        ),
                        angularaxis=dict(
                            tickfont=dict(size=9)
                        ),
                    ),
                    margin=dict(l=20, r=250, t=50, b=20),
                )
                st.plotly_chart(fig, use_container_width=True)

            # Top Station Summary
            with t3:
                top = usage_today.iloc[0]
                st.success(
                    f"**{top['station_name']}** at **{top['address']}** "
                    f"is today’s busiest with **{top['charging_hours']:.1f} h** "
                    f"and **{int(top['sessions'])} sessions.**"
                )
                st.dataframe(usage_today, use_container_width=True)

    # -----------------------------------------------------------
    # 🔵 YESTERDAY TAB
    # -----------------------------------------------------------
    with tab_yesterday:
        st.subheader("📅 Top 10 Busiest Stations (Yesterday)")
        usage_yday = compute_yesterday_top10()

        if usage_yday.empty:
            st.warning("No data found for yesterday.")
        else:
            total_hours_y = usage_yday["charging_hours"].sum()
            total_sessions_y = usage_yday["sessions"].sum()

            c1, c2, c3 = st.columns(3)
            c1.metric("Total charging hours", f"{total_hours_y:.1f} h")
            c2.metric("Total sessions", f"{int(total_sessions_y)}")
            c3.metric("Active stations", len(usage_yday))

            st.dataframe(
                usage_yday[["station_name", "address", "charging_hours", "sessions"]]
                .sort_values("charging_hours", ascending=False)
                .reset_index(drop=True),
                use_container_width=True,
            )

            y1, y2, y3 = st.tabs(["🌍 Heatmap", "🕸️ Radar Chart", "🏆 Top Station"])

            # Heatmap
            with y1:
                from streamlit_folium import st_folium
                from folium.plugins import HeatMap
                import folium

                st.subheader("Heatmap of Yesterday’s Top 10 Stations")
                m = folium.Map(location=[53.55, 9.99], zoom_start=12, tiles="CartoDB positron")
                max_hours = usage_yday["charging_hours"].max() or 1
                heat_points = [
                    [r.latitude, r.longitude, r.charging_hours / max_hours]
                    for r in usage_yday.itertuples()
                    if pd.notna(r.latitude) and pd.notna(r.longitude)
                ]
                if heat_points:
                    HeatMap(heat_points, radius=25, blur=15, max_zoom=14).add_to(m)
                for r in usage_yday.itertuples():
                    if pd.isna(r.latitude) or pd.isna(r.longitude):
                        continue
                    popup = f"<b>{r.station_name}</b><br>📍 {r.address}<br>⏱ {r.charging_hours:.1f} h<br>🔌 {r.sessions} sessions"
                    folium.CircleMarker(
                        location=[r.latitude, r.longitude],
                        radius=6,
                        color="red",
                        fill=True,
                        fill_color="orange",
                        fill_opacity=0.85,
                        popup=popup,
                    ).add_to(m)
                st_folium(m, width=1000, height=520)

            # 🕸️ Radar chart (Plotly — Yesterday)
            with y2:
                st.subheader("Top 10 Station Performance (Yesterday – Radar Chart)")

                fig = px.line_polar(
                    usage_yday,
                    r="charging_hours",
                    theta="station_name",
                    line_close=True,
                    color="station_name",
                    template="plotly_dark",
                    markers=True,
                )

                fig.update_traces(fill="toself", opacity=0.7)
                fig.update_layout(
                    height=520,
                    legend_title_text="Station",
                    legend=dict(
                        orientation="v",
                        yanchor="top",
                        y=1,
                        xanchor="left",
                        x=1.05,
                        bgcolor="rgba(0,0,0,0)",
                        bordercolor="gray",
                        borderwidth=0.5
                    ),
                    polar=dict(
                        radialaxis=dict(
                            showline=True,
                            linewidth=1,
                            gridcolor="gray",
                            gridwidth=0.3,
                            tickfont=dict(size=10)
                        ),
                        angularaxis=dict(
                            tickfont=dict(size=9)
                        ),
                    ),
                    margin=dict(l=20, r=200, t=50, b=20),
                )
                st.plotly_chart(fig, use_container_width=True)

            # Top Station Summary
            with y3:
                top_y = usage_yday.iloc[0]
                st.success(
                    f"**{top_y['station_name']}** at **{top_y['address']}** "
                    f"was yesterday’s busiest with **{top_y['charging_hours']:.1f} h** "
                    f"and **{int(top_y['sessions'])} sessions.**"
                )
                st.dataframe(usage_yday, use_container_width=True)
