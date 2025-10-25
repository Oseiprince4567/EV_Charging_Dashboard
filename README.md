⚡ EV Charging Activity Dashboard – Hamburg

About project
The EV Charging Activity Dashboard is a real-time, data-driven web application that visualizes the spatial and temporal dynamics of electric vehicle (EV) charging stations across Hamburg, Germany. It integrates open IoT data from the Hamburg SensorThings API with automated data ingestion, cloud database synchronization, and interactive analytics built in Streamlit.

Rationale 
The project was developed to:
* Demonstrate real-time data integration using open urban IoT streams.
* Provide decision-support tools for EV infrastructure planning and energy management.
* Showcase a serverless automation pipeline (GitHub Actions → Neon PostgreSQL → Streamlit Cloud).
* Support research on urban sustainability, smart mobility, and renewable-energy transitions.

Key features
Feature	Description
🔄 Automated Data Pipeline	The dashboard fetches EV-charging observations every 6 hours through a scheduled GitHub Actions workflow that writes to a Neon PostgreSQL database.
🌐 Live IoT Integration	Directly interfaces with Hamburg’s SensorThings API, providing near-real-time data from city-wide charging stations.
🗺️ Interactive Spatial Analytics	Built with Folium and Streamlit-Folium, offering heatmaps, markers, and dynamic spatial insights.
📊 Temporal Performance Analysis	Includes radar and time-series plots powered by Plotly and Pandas, summarizing session durations and utilization intensity.
🧩 Modular Codebase	Separate modules for data ingestion (fillin.py), analytics (hamburghelpers.py), and UI (evhamapp.py) for maintainability.

Technical Stack
* Frontend & Visualization: Streamlit, Plotly, Folium
* Data Handling: Pandas, NumPy
* Database: Neon PostgreSQL (hosted)
* Backend Automation: GitHub Actions (cron job every 6 hours)
* APIs: Hamburg SensorThings API (IoT)
* Language: Python 3.12

Dashboard Deployment
1. Streamlit Cloud hosts the dashboard:
    * evhamapp.py → main Streamlit app
    * .streamlit/secrets.toml → Neon DB credentials
2. GitHub Actions automatically runs fillin.py every 6 hours:
    * Fetches latest IoT data
    * Updates Neon database
3. Neon Database serves as persistent backend storage for analytics.

🧩 Repository Structure

ev_charging_dashboard/
│
├── evhamapp.py              # Main Streamlit dashboard app
├── hamburghelpers.py        # Data analytics & helper functions
├── fillin.py                # IoT data ingestion script
├── requirements.txt         # Python dependencies
├── .github/workflows/       # GitHub Actions automation
│   └── fillin.yml
├── .streamlit/
│   └── secrets.toml         # (excluded from Git for security)
└── README.md                # Project documentation


Contact
Author: Prince Osei Boateng 📍 M.Sc. Photogrammetry & Geoinformatics, HFT Stuttgart (https://www.linkedin.com/in/oseiprince/)| Portfolio: https://oseiprince4567.github.io/Portfolio/profile.html
