import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
import cv2

# Import custom Crater-X modules
from lunar_data import LunarCraterSimulator
from glitch_eraser import SelfHealingGlitchEraser
from subsurface import SubsurfaceIceEstimator
from pathfinding import LunarPathPlanner

# Set page config
st.set_page_config(
    page_title="Crater-X Lunar Terminal",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Reusable component for synchronized slider + number input
def synced_slider_input(label, min_val, max_val, default_val, key_prefix, step=1):
    slider_key = f"{key_prefix}_slider"
    num_key = f"{key_prefix}_number"
    
    # Initialize state
    if slider_key not in st.session_state:
        st.session_state[slider_key] = default_val
    if num_key not in st.session_state:
        st.session_state[num_key] = default_val
        
    def sync_slide_to_num():
        st.session_state[num_key] = st.session_state[slider_key]
        
    def sync_num_to_slide():
        val = st.session_state[num_key]
        clamped = int(max(min_val, min(max_val, val)))
        st.session_state[slider_key] = clamped
        st.session_state[num_key] = clamped
        
    # Render layout side-by-side with wider column (30%) to ensure the input field is wide and easy to type in
    col_slide, col_num = st.columns([7, 3])
    with col_slide:
        val = st.slider(
            label,
            min_value=min_val,
            max_value=max_val,
            key=slider_key,
            on_change=sync_slide_to_num,
            step=step
        )
    with col_num:
        # Invisible spacer to align number box with the slider track
        st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
        st.number_input(
            "Value Input",
            min_value=min_val,
            max_value=max_val,
            key=num_key,
            on_change=sync_num_to_slide,
            step=step,
            label_visibility="collapsed"
        )
    return val

## Custom Cyberpunk / NASA Control Room Theme Styling - Deep Space Navy & Sky Blue Theme
st.markdown("""
<style>
/* Font import */
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;500;700;900&family=Rajdhani:wght@500;700&family=Outfit:wght@300;400;500;700&display=swap');

/* Main app containers */
html, body, [data-testid="stAppViewContainer"] {
    background-color: #081120 !important;
    color: #CBD5E1 !important;
    font-family: 'Outfit', sans-serif !important;
}

[data-testid="stHeader"] {
    background: rgba(8, 17, 32, 0.85) !important;
    backdrop-filter: blur(12px);
    border-bottom: 1px solid #1E3A8A;
}

[data-testid="stSidebar"] {
    background-color: #040914 !important;
    border-right: 1.5px solid #1E3A8A !important;
}

/* Force soft light gray text on specific text tags inside the sidebar */
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] .stWidgetLabel,
[data-testid="stSidebar"] .stMarkdown {
    color: #CBD5E1 !important;
}

/* Force selectbox value text in the sidebar to be dark slate/black so it is visible in the white box */
[data-testid="stSidebar"] [data-baseweb="select"] div,
[data-testid="stSidebar"] [data-baseweb="select"] span {
    color: #081120 !important;
}

/* Sidebar Title and Text */
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3, [data-testid="stSidebar"] h4 {
    font-family: 'Orbitron', sans-serif !important;
    color: #38BDF8 !important;
    text-shadow: 0 0 5px rgba(56, 189, 248, 0.3) !important;
}

/* Standard Headers (Main Heading is off-white) */
h1, h2, h3, h4 {
    font-family: 'Orbitron', sans-serif !important;
    font-weight: 700 !important;
    color: #F8FAFC !important;
    text-shadow: 0 0 8px rgba(248, 250, 252, 0.35);
}

/* Custom styled tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 10px;
    background-color: #0F1D30;
    padding: 6px;
    border-radius: 8px;
    border: 1px solid #1E3A8A;
}

.stTabs [data-baseweb="tab"] {
    background-color: #050B14;
    border: 1px solid #1E293B;
    border-radius: 6px;
    color: #38BDF8;
    padding: 10px 20px;
    font-family: 'Orbitron', sans-serif;
    font-size: 13px;
    font-weight: 500;
    transition: all 0.3s ease;
}

.stTabs [data-baseweb="tab"]:hover {
    color: #F8FAFC;
    border-color: #38BDF8;
    box-shadow: 0 0 8px rgba(56, 189, 248, 0.3);
}

.stTabs [aria-selected="true"] {
    background-color: #38BDF8 !important;
    color: #081120 !important;
    font-weight: 700 !important;
    box-shadow: 0 0 15px rgba(56, 189, 248, 0.6) !important;
    border: 1px solid #38BDF8 !important;
}

/* Glassmorphic Cyber Cards in Navy/Blue Theme */
.kpi-card {
    background: linear-gradient(135deg, #0F1D30 0%, #050B14 100%);
    border: 1.5px solid #1E3A8A;
    border-left: 5px solid #38BDF8;
    border-radius: 10px;
    padding: 18px 24px;
    margin-bottom: 15px;
    box-shadow: 0 8px 16px rgba(0, 0, 0, 0.5);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.kpi-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 12px 20px rgba(56, 189, 248, 0.25);
}
.kpi-card.green {
    border-left-color: #10B981;
}
.kpi-card.green:hover {
    box-shadow: 0 12px 20px rgba(16, 185, 129, 0.2);
}
.kpi-card.amber {
    border-left-color: #F59E0B;
}
.kpi-card.amber:hover {
    box-shadow: 0 12px 20px rgba(245, 158, 11, 0.2);
}
.kpi-card.red {
    border-left-color: #EF4444;
}
.kpi-card.red:hover {
    box-shadow: 0 12px 20px rgba(239, 68, 68, 0.2);
}

.kpi-title {
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 1.8px;
    color: #38BDF8;
    font-weight: 600;
}
.kpi-value {
    font-family: 'Rajdhani', sans-serif;
    font-size: 32px;
    font-weight: bold;
    color: #FAFAFA;
    margin-top: 4px;
    text-shadow: 0 0 5px rgba(250, 250, 250, 0.2);
}
.kpi-subtitle {
    font-size: 11px;
    color: #CBD5E1;
    margin-top: 4px;
}

/* Control Terminal styling */
.terminal-block {
    background-color: #040914;
    border: 1px solid #1E3A8A;
    border-radius: 6px;
    padding: 12px;
    font-family: 'Share Tech Mono', monospace;
    font-size: 13px;
    color: #CBD5E1;
    box-shadow: inset 0 2px 8px rgba(0,0,0,0.9);
}

/* Info text styling with spacious margins */
.info-text {
    font-size: 14px !important;
    color: #38BDF8 !important;
    margin-bottom: 30px !important;
    margin-top: 5px !important;
    line-height: 1.6 !important;
}
</style>
""", unsafe_allow_html=True)

# ----------------- SIDEBAR CONFIGURATION -----------------
st.sidebar.markdown("<h2 style='text-align: center; margin-bottom: 0px;'>🛰️ CRATER-X</h2>", unsafe_allow_html=True)
st.sidebar.markdown("<p style='text-align: center; color: #64748B; font-size: 12px; margin-top: 0px;'>Lunar South Pole Control Room</p>", unsafe_allow_html=True)
st.sidebar.markdown("---")

st.sidebar.subheader("🎯 Target Selection")
crater_choice = st.sidebar.selectbox(
    "Select Lunar Crater/Zone",
    ["Shackleton Crater", "Shoemaker Crater", "Haworth Crater", "Crater-X (Custom Zone)"]
)

# Set crater parameters
if crater_choice == "Shackleton Crater":
    c_name = "Shackleton"
    c_peak = False
    c_radius = 42.0
    c_depth = 900.0
    c_rim = 120.0
elif crater_choice == "Shoemaker Crater":
    c_name = "Shoemaker"
    c_peak = False
    c_radius = 65.0
    c_depth = 1100.0
    c_rim = 180.0
elif crater_choice == "Haworth Crater":
    c_name = "Haworth"
    c_peak = True
    c_radius = 55.0
    c_depth = 1200.0
    c_rim = 200.0
else:
    c_name = "Crater-X"
    c_peak = True
    c_radius = 48.0
    c_depth = 1000.0
    c_rim = 150.0

st.sidebar.subheader("📡 Telemetry Settings")
glitch_rate = st.sidebar.slider(
    "Raw Telemetry Glitch Rate (%)",
    min_value=0,
    max_value=30,
    value=12,
    step=1,
    help="Simulates data packet drops, cosmic ray spikes, and line scan failures in Chandrayaan-2 feeds."
) / 100.0

st.sidebar.subheader("☀️ Solar Geometry")
sun_azimuth = st.sidebar.slider(
    "Solar Azimuth (deg)",
    min_value=0,
    max_value=360,
    value=145,
    step=5,
    help="Compass direction of solar illumination."
)
sun_elevation = st.sidebar.slider(
    "Solar Elevation (deg)",
    min_value=0.2,
    max_value=8.0,
    value=1.5,
    step=0.1,
    help="Lunar polar regions have extremely grazing sunlit angles, rarely exceeding 2-3 degrees."
)

st.sidebar.subheader("🚜 Rover Configuration")
start_battery = st.sidebar.slider("Starting Battery (%)", 50, 100, 100, step=5)
max_slope = st.sidebar.slider("Max Passable Slope (deg)", 12, 25, 20, step=1)

# Statically defined battery discharge and solar charge rates
shadow_drain = 0.8
solar_recharge = 1.2

# ----------------- SIMULATION INITIALIZATION -----------------

# Initialize simulator
sim = LunarCraterSimulator(size=120, resolution=12.0)
raw_data = sim.generate_crater(
    name=c_name, 
    center_peak=c_peak, 
    rim_radius=c_radius, 
    depth=c_depth, 
    height_rim=c_rim
)

# Apply solar shadow raytracing
shadow_map = sim.calculate_shadows(raw_data["dem"], sun_azimuth, sun_elevation)
raw_data["shadow_map"] = shadow_map

# Inject glitches into polarimetric datasets (CPR, DOP)
glitched_data = raw_data.copy()
glitched_data["cpr"] = sim.inject_telemetry_glitches(raw_data["cpr"], glitch_rate=glitch_rate)
glitched_data["dop"] = sim.inject_telemetry_glitches(raw_data["dop"], glitch_rate=glitch_rate)

# Run the self-healing polarimetric glitch eraser
eraser = SelfHealingGlitchEraser()
healed_data = eraser.heal_telemetry_packet(glitched_data)
# Add non-glitched elements back
healed_data["dem"] = raw_data["dem"]
healed_data["slope"] = raw_data["slope"]
healed_data["shadow_map"] = shadow_map
healed_data["rock_density"] = raw_data["rock_density"]

# Reset session state values if target crater changed (MUST be done before any widgets are rendered)
if "current_crater" not in st.session_state or st.session_state["current_crater"] != c_name:
    st.session_state["current_crater"] = c_name
    
    # Default end point is where we have ice
    ice_y, ice_x = np.where((healed_data["cpr"] > 1.2) & (healed_data["dop"] < 0.1))
    def_end_y = int(ice_y[0]) if len(ice_y) > 0 else healed_data["dem"].shape[0] // 2
    def_end_x = int(ice_x[0]) if len(ice_x) > 0 else healed_data["dem"].shape[1] // 2
    
    st.session_state["endx_slider"] = def_end_x
    st.session_state["endx_number"] = def_end_x
    st.session_state["endy_slider"] = def_end_y
    st.session_state["endy_number"] = def_end_y
    
    st.session_state["landx_slider"] = healed_data["dem"].shape[1] // 2
    st.session_state["landx_number"] = healed_data["dem"].shape[1] // 2
    st.session_state["landy_slider"] = healed_data["dem"].shape[0] // 3
    st.session_state["landy_number"] = healed_data["dem"].shape[0] // 3
    
    st.session_state["drillx_slider"] = healed_data["dem"].shape[1] // 2
    st.session_state["drillx_number"] = healed_data["dem"].shape[1] // 2
    st.session_state["drilly_slider"] = healed_data["dem"].shape[0] // 3
    st.session_state["drilly_number"] = healed_data["dem"].shape[0] // 3
    
    st.session_state["startx_slider"] = 15
    st.session_state["startx_number"] = 15
    st.session_state["starty_slider"] = 15
    st.session_state["starty_number"] = 15

# Run the 3D Subsurface Ice Estimator
estimator = SubsurfaceIceEstimator(max_depth=5.0, dz=0.1)
ice_3d, temp_3d = estimator.estimate_3d_ice_grid(healed_data["cpr"], healed_data["dop"], resolution=12.0)
water_mass = estimator.calculate_total_water_mass(ice_3d, resolution=12.0)

# Calculate statistics for KPI cards
surface_ice_pixels = np.sum((healed_data["cpr"] > 1.0) & (healed_data["dop"] < 0.13))
surface_ice_area = surface_ice_pixels * (12.0 * 12.0) # square meters
avg_ice_wt = np.mean(ice_3d[ice_3d > 0]) if np.sum(ice_3d > 0) > 0 else 0.0
total_glitched_cpr = np.sum(healed_data["cpr_mask"])
pct_healed = (total_glitched_cpr / glitched_data["cpr"].size) * 100.0

# Header Title Block
st.markdown(f"<h1 style='margin-bottom: 2px;'>Crater-X</h1>", unsafe_allow_html=True)
st.markdown(f"<p style='color: #64748B; margin-bottom: 35px; font-size: 14px;'>Autonomous Ingestion, Subsurface Ice Estimation & Traverse Planning</p>", unsafe_allow_html=True)

# ----------------- TABS SETUP -----------------
tab1, tab2, tab3, tab4 = st.tabs([
    "📡 Telemetry & Glitch Healing",
    "🛡️ Landing Safety & Hazards",
    "🧊 3D Subsurface Ice-Core Volumetric Estimator",
    "🚜 Solar-Constrained Route Finder"
])

# ----------------- TAB 1: TELEMETRY & GLITCH HEALING -----------------
with tab1:
    st.subheader("🛰️ Chandrayaan-2 Radar Telemetry Pipeline")
    st.markdown("<p class='info-text'>Cleans and heals raw polarimetric radar telemetry to remove point anomalies, line dropouts, and cosmic noise spikes on the fly.</p>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Raw Stream Status</div>
            <div class="kpi-value" style="color: #EF4444;">CORRUPTED</div>
            <div class="kpi-subtitle">Glitch Rate: {glitch_rate*100:.1f}% telemetry dropouts</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="kpi-card green">
            <div class="kpi-title">Self-Healing Engine</div>
            <div class="kpi-value" style="color: #10B981;">ACTIVE</div>
            <div class="kpi-subtitle">Healed Pixels: {total_glitched_cpr} ({pct_healed:.1f}%)</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        # Determine signal health rating
        if glitch_rate <= 0.05:
            health_color, health_lbl = "#10B981", "EXCELLENT (A+)"
        elif glitch_rate <= 0.15:
            health_color, health_lbl = "#F59E0B", "NOMINAL (B)"
        else:
            health_color, health_lbl = "#EF4444", "DEGRADED (C-)"
            
        st.markdown(f"""
        <div class="kpi-card" style="border-left-color: {health_color};">
            <div class="kpi-title">Clean Signal Quality</div>
            <div class="kpi-value" style="color: {health_color};">{health_lbl}</div>
            <div class="kpi-subtitle">Telemetry reconstructed using Fast-Marching Inpainting</div>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("### 🔍 Live Dual-Channel Radar Comparison (Raw vs. Restored)")
    
    view_layer = st.radio("Select Polarimetric Layer", ["Circular Polarization Ratio (CPR)", "Degree of Polarization (DOP)"], horizontal=True)
    
    col_raw, col_healed = st.columns(2)
    
    if view_layer == "Circular Polarization Ratio (CPR)":
        raw_mat = glitched_data["cpr"]
        healed_mat = healed_data["cpr"]
        cmap = "Plasma"
        lbl = "CPR (Ratio)"
        c_range = [0, 2.0]
    else:
        raw_mat = glitched_data["dop"]
        healed_mat = healed_data["dop"]
        cmap = "RdBu"
        lbl = "DOP (Fraction)"
        c_range = [0, 1.0]

    # Plotting helper
    def plot_matrix(matrix, title, cmap, c_min, c_max):
        fig = go.Figure(data=go.Heatmap(
            z=matrix,
            colorscale=cmap,
            zmin=c_min,
            zmax=c_max,
            colorbar=dict(title=lbl, thickness=12)
        ))
        fig.update_layout(
            title=dict(text=title, font=dict(family="Orbitron", size=14, color="#06B6D4")),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            width=600,
            height=450,
            margin=dict(l=10, r=10, t=40, b=10),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        return fig

    with col_raw:
        st.plotly_chart(plot_matrix(raw_mat, "Corrupted Telemetry Feed (Raw)", cmap, c_range[0], c_range[1]), use_container_width=True)
    with col_healed:
        st.plotly_chart(plot_matrix(healed_mat, "Reconstructed & Healed Grid (Output)", cmap, c_range[0], c_range[1]), use_container_width=True)
 
    st.markdown("---")
    st.markdown("##### 🔬 Polarimetric Radar Decision Rules for Surface Ice Detection")
    st.latex(r"\text{Confirmed Surface Water Ice} \iff \text{Circular Polarization Ratio (CPR)} > 1.0 \quad \text{and} \quad \text{Degree of Polarization (DOP)} < 0.13")

    st.subheader("💻 Telemetry Glitch Eraser Log")
    log_text = f"""
    [CRATER-X TELEMETRY INGESTION PROCESSOR]
    Initializing Chandrayaan-2 DFSAR L-Band and P-Band arrays...
    Injesting metadata for target region: {c_name}
    WARNING: Detected {total_glitched_cpr} invalid or dropped pixels (NaN/Spikes/Lines) out of {glitched_data["cpr"].size} telemetry records.
    Booting Self-Healing Polarimetric Pipeline...
    Creating Glitch Mask array (Shape: {glitched_data["cpr"].shape}).
    Executing cv2.INPAINT_TELEA Fast Marching Interpolation.
    Applying SciPy 3x3 Median Spatial filter on repaired pixels to remove jitter...
    Success! Telemetry healed. Average interpolation delta: {np.nanmean(np.abs(healed_mat - np.nan_to_num(raw_mat))):.4f}.
    Data fed to Core Resource locator & Pathfinding engines.
    """
    st.markdown(f"<pre class='terminal-block'>{log_text}</pre>", unsafe_allow_html=True)


# ----------------- TAB 2: LANDING SAFETY & HAZARDS -----------------
with tab2:
    st.subheader("🛡️ Landing Safety Analyzer & Landing Ellipse Selection")
    st.markdown("<p class='info-text'>Analyzes slopes, rock hazards, and solar shadows to identify level, safe landing zones near key water-ice deposits.</p>", unsafe_allow_html=True)
    
    # Calculate Safety Index: 0 (Deadly) to 100 (Extremely Safe)
    slope_factor = np.clip(1.0 - (healed_data["slope"] / 15.0), 0.0, 1.0)
    rock_factor = np.clip(1.0 - healed_data["rock_density"], 0.0, 1.0)
    sun_factor = 0.3 + 0.7 * healed_data["shadow_map"] # partial penalty for landing in shadows
    
    safety_score = slope_factor * rock_factor * sun_factor * 100.0
    
    # Highlight regions matching strict landing parameters (Safe Ellipses)
    safe_landing_mask = (healed_data["slope"] < 12.0) & (healed_data["rock_density"] < 0.2) & (healed_data["shadow_map"] > 0.5)
    
    col_a, col_b = st.columns([2, 1])
    
    with col_a:
        st.markdown("### Landing Zone Safety Map")
        map_view = st.selectbox("Select Safety Heatmap Layer", ["Combined Safety Score", "Slope Map", "Rock Hazard Map", "Solar Shadow Mask"])
        
        if map_view == "Combined Safety Score":
            z_data = safety_score
            colorscale = "RdYlGn"
            c_label = "Safety Score (%)"
            c_lims = [0, 100]
        elif map_view == "Slope Map":
            z_data = healed_data["slope"]
            colorscale = "Viridis"
            c_label = "Slope (Degrees)"
            c_lims = [0, 30]
        elif map_view == "Rock Hazard Map":
            z_data = healed_data["rock_density"]
            colorscale = "Reds"
            c_label = "Rock Density Fraction"
            c_lims = [0, 1.0]
        else:
            z_data = healed_data["shadow_map"]
            colorscale = "Greys"
            c_label = "Sunlight (1=Sun, 0=Shadow)"
            c_lims = [0, 1.0]
            
        fig_safe = go.Figure(data=go.Heatmap(
            z=z_data,
            colorscale=colorscale,
            zmin=c_lims[0],
            zmax=c_lims[1],
            colorbar=dict(title=c_label, thickness=12)
        ))
        
        # Overlay landing sites if selected
        fig_safe.update_layout(
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            width=700,
            height=500,
            margin=dict(l=10, r=10, t=10, b=10),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig_safe, use_container_width=True)
        
    with col_b:
        st.markdown("### 📍 Drill down on Target Coordinates")
        st.markdown("Analyze localized safety metrics at specific pixel coordinates on the map:")
        
        coord_x = synced_slider_input("Select Pixel Coordinate X", 0, healed_data["dem"].shape[1]-1, healed_data["dem"].shape[1]//2, "landx")
        coord_y = synced_slider_input("Select Pixel Coordinate Y", 0, healed_data["dem"].shape[0]-1, healed_data["dem"].shape[0]//3, "landy")
        
        sel_slope = healed_data["slope"][coord_y, coord_x]
        sel_rocks = healed_data["rock_density"][coord_y, coord_x]
        sel_shadow = "SUNLIT" if healed_data["shadow_map"][coord_y, coord_x] > 0.5 else "SHADOW (PSR)"
        sel_safety = safety_score[coord_y, coord_x]
        sel_elev = healed_data["dem"][coord_y, coord_x]
        
        # Proximity to ice
        # Find indices of ice
        ice_indices = np.argwhere((healed_data["cpr"] > 1.0) & (healed_data["dop"] < 0.13))
        if len(ice_indices) > 0:
            distances_to_ice = np.sqrt((ice_indices[:, 0] - coord_y)**2 + (ice_indices[:, 1] - coord_x)**2) * 12.0
            min_dist_ice = np.min(distances_to_ice)
        else:
            min_dist_ice = float('inf')
            
        st.markdown(f"**Elevation:** `{sel_elev:.1f} m` (relative)")
        st.markdown(f"**Local Slope:** `{sel_slope:.1f}°` {'⚠️ Steep' if sel_slope > 15 else '✅ Safe'}")
        st.markdown(f"**Rock Hazard Fraction:** `{sel_rocks:.2f}` {'⚠️ Heavy' if sel_rocks > 0.3 else '✅ Safe'}")
        st.markdown(f"**Thermal Environment:** `{sel_shadow}`")
        st.markdown(f"**Distance to nearest Ice Deposit:** `{min_dist_ice:.1f} m`")
        
        if sel_safety >= 75.0:
            safety_badge = "🟢 HIGHLY RECOMMENDED"
        elif sel_safety >= 50.0:
            safety_badge = "🟡 MARGINAL - PROCEED WITH CAUTION"
        else:
            safety_badge = "🔴 DANGEROUS ZONE - DO NOT LAND"
            
        st.markdown(f"**Landing Rating:** **{safety_badge}** (Score: `{sel_safety:.1f}/100`)")
        
        # Print landing site feasibility statistics
        safe_area = np.sum(safe_landing_mask) * (12.0*12.0)
        st.markdown(f"---")
        st.markdown(f"**Total Safe Landing Area in Grid:** `{safe_area/10000:.2f} hectares` ({np.sum(safe_landing_mask)} safe pixel spots)")


# ----------------- TAB 3: 3D SUBSURFACE ICE-CORE ESTIMATOR -----------------
with tab3:
    st.subheader("🧊 3D Subsurface 'Ice-Core' Volumetric Estimator (up to 5m depth)")
    st.markdown("<p class='info-text'>Models subsurface temperature gradients and ice sublimation stability to estimate 3D water-ice yield down to 5 meters.</p>", unsafe_allow_html=True)
    
    # LaTeX Scientific Criteria for Subsurface Ice Confirmation
    st.markdown("##### 🔬 Subsurface Ice Stability & Confirmation Criteria")
    st.latex(r"\text{Subsurface Ice Presence Confirmed if:} \quad \text{CPR}_{surface} > 1.0 \quad \text{and} \quad \text{DOP}_{surface} < 0.13 \quad \text{and} \quad T(z) < 110\text{ K}")
    st.markdown("<div style='margin-bottom: 25px;'></div>", unsafe_allow_html=True)
    
    col_v1, col_v2, col_v3 = st.columns(3)
    with col_v1:
        st.markdown(f"""
        <div class="kpi-card green">
            <div class="kpi-title">Estimated Resource Yield</div>
            <div class="kpi-value">{water_mass:.1f} MT</div>
            <div class="kpi-subtitle">Total estimated water mass in simulated grid</div>
        </div>
        """, unsafe_allow_html=True)
    with col_v2:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Average Ice Purity</div>
            <div class="kpi-value">{avg_ice_wt:.2f} wt.%</div>
            <div class="kpi-subtitle">Weighted average in ice-bearing cells</div>
        </div>
        """, unsafe_allow_html=True)
    with col_v3:
        st.markdown(f"""
        <div class="kpi-card amber">
            <div class="kpi-title">Ice Surface footprint</div>
            <div class="kpi-value">{surface_ice_area:.0f} m²</div>
            <div class="kpi-subtitle">Detected at surface: {surface_ice_pixels} pixels</div>
        </div>
        """, unsafe_allow_html=True)
        
    col_3d, col_drill = st.columns([1, 1])
    
    with col_3d:
        st.markdown("### 🌍 3D Subsurface Volumetric Scatter")
        st.markdown("This interactive 3D model extracts cells containing ice (>2 wt.% concentration) to map the subsurface reservoir shapes and volumes.")
        
        # Subsample indices to keep plotting fast and lightweight
        # Find points where ice_3d > 2.0
        y_indices, x_indices, z_indices = np.where(ice_3d > 2.0)
        
        if len(y_indices) > 0:
            # Subsample up to 2000 points for performance
            max_points = 2000
            if len(y_indices) > max_points:
                step = len(y_indices) // max_points
                y_pts = y_indices[::step]
                x_pts = x_indices[::step]
                z_pts = z_indices[::step]
            else:
                x_pts, y_pts, z_pts = x_indices, y_indices, z_indices
                
            # Convert indices to physical dimensions
            val_pts = ice_3d[y_pts, x_pts, z_pts]
            x_m = x_pts * 12.0
            y_m = y_pts * 12.0
            depth_m = estimator.depths[z_pts]
            
            # Plotly 3D scatter
            fig_3d = go.Figure(data=[go.Scatter3d(
                x=x_m,
                y=y_m,
                z=depth_m,
                mode='markers',
                marker=dict(
                    size=3.5,
                    color=val_pts,
                    colorscale='Blues',
                    colorbar=dict(title="Ice Wt.%", thickness=15),
                    opacity=0.75
                )
            )])
            
            fig_3d.update_layout(
                scene=dict(
                    xaxis_title='X (meters)',
                    yaxis_title='Y (meters)',
                    zaxis_title='Subsurface Depth (m)',
                    zaxis=dict(autorange="reversed"), # depth goes down!
                    camera=dict(
                        eye=dict(x=1.5, y=1.5, z=1.2)
                    )
                ),
                width=600,
                height=500,
                margin=dict(l=10, r=10, t=10, b=10),
                paper_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig_3d, use_container_width=True)
        else:
            st.info("No substantial subsurface ice detected in this crater configuration. Increase the solar shadow or select a darker crater.")
            
    with col_drill:
        st.markdown("### 🧬 Virtual Ice-Core Drilling Simulator")
        st.markdown("Place a virtual drill core anywhere on the landing site to query soil chemistry, temperatures, and water mass percentage layer-by-layer:")
        
        drill_x = synced_slider_input("Drill Site X coordinate", 0, healed_data["dem"].shape[1]-1, healed_data["dem"].shape[1]//2, "drillx")
        drill_y = synced_slider_input("Drill Site Y coordinate", 0, healed_data["dem"].shape[0]-1, healed_data["dem"].shape[0]//3, "drilly")
        
        # Get core info
        core_info = estimator.get_drill_core(drill_y, drill_x, ice_3d, temp_3d)
        
        # Plots
        fig_core, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharey=True)
        fig_core.patch.set_facecolor('#070913')
        
        # Set text colors for cyber-theme compatibility
        for ax in [ax1, ax2]:
            ax.set_facecolor('#0B0F19')
            ax.tick_params(colors='#94A3B8')
            ax.xaxis.label.set_color('#94A3B8')
            ax.yaxis.label.set_color('#94A3B8')
            ax.title.set_color('#06B6D4')
            ax.spines['bottom'].set_color('#1E293B')
            ax.spines['top'].set_color('#1E293B')
            ax.spines['left'].set_color('#1E293B')
            ax.spines['right'].set_color('#1E293B')
            
        # Temperature chart
        ax1.plot(core_info["temperature"], core_info["depths"], color='#EF4444', linewidth=2.5, label="Temp (K)")
        ax1.axvline(110.0, color='#38BDF8', linestyle='--', label="Sublimation (110K)")
        ax1.set_xlabel("Temperature (K)")
        ax1.set_ylabel("Depth (meters)")
        ax1.set_title("Thermal Gradient Profile", fontsize=10)
        ax1.invert_yaxis()
        ax1.legend(loc='lower left', prop={'size': 7})
        ax1.grid(True, color='#1E293B', linestyle=':')
        
        # Concentration chart
        ax2.fill_betweenx(core_info["depths"], 0, core_info["ice_concentration"], color='#06B6D4', alpha=0.4)
        ax2.plot(core_info["ice_concentration"], core_info["depths"], color='#06B6D4', linewidth=2.5, label="Ice wt.%")
        ax2.set_xlabel("Ice Concentration (wt.%)")
        ax2.set_title("Volumetric Ice Grade", fontsize=10)
        ax2.grid(True, color='#1E293B', linestyle=':')
        
        plt.tight_layout()
        st.pyplot(fig_core)
        
        max_ice_val = np.max(core_info["ice_concentration"])
        sub_temp = core_info["temperature"][0]
        st.markdown(f"**Drill Site Status:** Surface Temp: `{sub_temp:.1f} K` | Max Subsurface Ice Grade: `{max_ice_val:.1f} wt.%` at depth: `{core_info['depths'][np.argmax(core_info['ice_concentration'])]:.1f} m`")


# ----------------- TAB 4: SOLAR-CONSTRAINED ROUTE FINDER -----------------
with tab4:
    st.subheader("🚜 Solar-Constrained & Battery-Safe Route Finder")
    st.markdown("<p class='info-text'>Plans battery-safe routes that maximize solar illumination on crater rims and schedule quick resource extraction runs into dark PSRs.</p>", unsafe_allow_html=True)
    
    col_path_a, col_path_b = st.columns([1.5, 1])
    
    # Let user pick start and end coordinates
    with col_path_b:
        st.markdown("### 🧭 Set Mission Route Points")
        
        # Calculate default end points for target location
        ice_y, ice_x = np.where((healed_data["cpr"] > 1.2) & (healed_data["dop"] < 0.1))
        def_end_y = int(ice_y[0]) if len(ice_y) > 0 else healed_data["dem"].shape[0] // 2
        def_end_x = int(ice_x[0]) if len(ice_x) > 0 else healed_data["dem"].shape[1] // 2
        


        st.markdown("Select Start Location (typically a sunlit rim landing site):")
        start_x = synced_slider_input("Start Coordinate X", 0, healed_data["dem"].shape[1]-1, 15, "startx")
        start_y = synced_slider_input("Start Coordinate Y", 0, healed_data["dem"].shape[0]-1, 15, "starty")
        
        st.markdown("Select Target Resource Location (typically deep inside dark crater):")
        
        end_x = synced_slider_input("Target Coordinate X", 0, healed_data["dem"].shape[1]-1, def_end_x, "endx")
        end_y = synced_slider_input("Target Coordinate Y", 0, healed_data["dem"].shape[0]-1, def_end_y, "endy")
        
        shadow_penalty = st.slider("Shadow Avoidance Strictness", 1.0, 10.0, 5.0, step=0.5, help="Higher penalty forces the rover to take winding, longer paths that follow sunlit rims. Lower values allow shorter paths cutting directly through the shadow.")
        
        # Calculate Path
        planner = LunarPathPlanner(resolution=12.0, max_slope_passable=max_slope)
        path, success = planner.plan_traverse(
            healed_data["dem"], 
            healed_data["slope"], 
            healed_data["shadow_map"], 
            (start_y, start_x), 
            (end_y, end_x), 
            downsample_factor=2, 
            shadow_penalty_factor=shadow_penalty
        )
        
        if success:
            # Simulate Battery Level
            bat_sim = planner.simulate_battery(
                path, 
                healed_data["shadow_map"], 
                start_charge=start_battery, 
                drain_rate=shadow_drain, 
                recharge_rate=solar_recharge
            )
            
            st.success("✅ Path Found and Safety Validated!")
            
            # Print Route telemetry KPIs
            tot_dist = (len(path) - 1) * 12.0
            shd_exposure = bat_sim["shadow_steps"] * 12.0
            
            st.markdown(f"**Total Distance:** `{tot_dist:.1f} meters`")
            st.markdown(f"**Shadow Exposure:** `{shd_exposure:.1f} meters` ({bat_sim['shadow_steps']} steps)")
            st.markdown(f"**Final Battery Charge:** `{bat_sim['final_charge']:.1f}%` (Min: `{bat_sim['min_charge']:.1f}%`)")
            
            if bat_sim["min_charge"] <= 0.0:
                mission_status = "<span style='color: #EF4444; font-weight:bold;'>❌ CRITICAL FAILURE: ROVER FROZE IN SHADOWS</span>"
            elif bat_sim["min_charge"] <= 15.0:
                mission_status = "<span style='color: #F59E0B; font-weight:bold;'>⚠️ MARGINAL: BATTERY CRITICALLY DEPLETED</span>"
            else:
                mission_status = "<span style='color: #10B981; font-weight:bold;'>🟢 NOMINAL: SAFE TRAVERSE SUCCESSFUL</span>"
                
            st.markdown(f"**Mission Status:** {mission_status}", unsafe_allow_html=True)
            
            if len(bat_sim["events"]) > 0:
                for e in bat_sim["events"]:
                    st.warning(e["msg"])
        else:
            st.error("❌ PATH BLOCKED: Slopes exceed safety parameters or terrain is disconnected.")
            path = []

    with col_path_a:
        st.markdown("### 🗺️ Autonomous Traverse Overlay")
        st.markdown("This map shows the elevation contours and shades of the crater, with the dynamic path planning overlaid.")
        
        # Plot Path on elevation map
        fig_path = go.Figure()
        
        # Add elevation contour/heatmap
        fig_path.add_trace(go.Heatmap(
            z=healed_data["dem"],
            colorscale="Cividis",
            colorbar=dict(title="Elevation (m)", thickness=10),
            name="DEM"
        ))
        
        # Add shadows as a translucent overlay (dark regions)
        # We represent shadow as binary black, 0 opacity for sun, 0.45 opacity for shadow
        shadow_overlay = np.zeros_like(healed_data["shadow_map"])
        shadow_overlay[healed_data["shadow_map"] < 0.5] = 1.0
        
        # Draw Start Node
        fig_path.add_trace(go.Scatter(
            x=[start_x],
            y=[start_y],
            mode='markers',
            marker=dict(size=14, color='#10B981', symbol='circle'),
            name='Landing Site (Start)'
        ))
        
        # Draw End Node
        fig_path.add_trace(go.Scatter(
            x=[end_x],
            y=[end_y],
            mode='markers',
            marker=dict(size=14, color='#06B6D4', symbol='star'),
            name='Ice Target (End)'
        ))
        
        # Draw Path
        if len(path) > 0:
            py, px_val = zip(*path)
            fig_path.add_trace(go.Scatter(
                x=px_val,
                y=py,
                mode='lines+markers',
                line=dict(color='#06B6D4', width=3),
                marker=dict(size=4, color='#38BDF8'),
                name='Rover Path'
            ))
            
        fig_path.update_layout(
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, autorange="reversed"), # flip Y to match image matrix
            width=700,
            height=500,
            margin=dict(l=10, r=10, t=10, b=10),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        st.plotly_chart(fig_path, use_container_width=True)
        
    st.markdown("### 🔋 Battery Charge Profile vs Distance")
    
    if len(path) > 0 and success:
        steps = np.arange(len(path)) * 12.0 # distance in meters
        
        fig_bat = go.Figure()
        
        # Add trace for battery
        fig_bat.add_trace(go.Scatter(
            x=steps,
            y=bat_sim["battery_profile"],
            mode='lines',
            line=dict(color='#06B6D4', width=3.5),
            name="Battery Level (%)",
            fill='tozeroy',
            fillcolor='rgba(6, 182, 212, 0.08)'
        ))
        
        # Draw horizontal lines for warning thresholds
        fig_bat.add_hline(y=15.0, line_dash="dash", line_color="#F59E0B", annotation_text="Critical (15%)")
        
        # Highlight shadowed intervals in red along the distance line
        # Find continuous shadow segments
        shadow_mask = np.array([healed_data["shadow_map"][p[0], p[1]] < 0.5 for p in path])
        
        fig_bat.update_layout(
            xaxis_title="Accumulated Traverse Distance (meters)",
            yaxis_title="Rover Charge (%)",
            yaxis=dict(range=[0, 105]),
            height=280,
            margin=dict(l=40, r=40, t=20, b=40),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='#0B0F19'
        )
        fig_bat.update_xaxes(showgrid=True, gridcolor='#1E293B')
        fig_bat.update_yaxes(showgrid=True, gridcolor='#1E293B')
        st.plotly_chart(fig_bat, use_container_width=True)
