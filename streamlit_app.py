import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tempfile
import os
from pathlib import Path
import warnings
from scipy import signal
from scipy.fft import fft, fftfreq, fftshift
from scipy.interpolate import interp1d
warnings.filterwarnings('ignore')
# Configure the page at the VERY TOP (must be first Streamlit command)
st.set_page_config(
    page_title="My App",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://docs.streamlit.io',
        'Report a bug': "https://github.com/streamlit/streamlit/issues",
        'About': "# This is a header. This is an *extremely* cool app!"
    }
)

# Set page config
st.set_page_config(
    page_title="GPR Data Processor",
    page_icon="📡",
    layout="wide"
)

# Title
st.title("📡 Advanced GPR Data Processor")
st.markdown("Process GPR data with coordinate import, aspect control, and advanced windowing")

# Custom CSS
st.markdown("""
<style>
    .stProgress > div > div > div > div {
        background-color: #4CAF50;
    }
    .st-bb {
        background-color: #f0f2f6;
    }
    .st-at {
        background-color: #ffffff;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 15px;
        border-left: 4px solid #4CAF50;
    }
    .window-box {
        background-color: #e8f4fd;
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
        border-left: 4px solid #2196F3;
    }
    .coordinate-box {
        background-color: #e8f5e9;
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
        border-left: 4px solid #4CAF50;
    }
    .aspect-box {
        background-color: #fff3e0;
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
        border-left: 4px solid #FF9800;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'original_array' not in st.session_state:
    st.session_state.original_array = None
if 'processed_array' not in st.session_state:
    st.session_state.processed_array = None
if 'coordinates' not in st.session_state:
    st.session_state.coordinates = None
if 'interpolated_coords' not in st.session_state:
    st.session_state.interpolated_coords = None

# Sidebar
with st.sidebar:
    st.header("📂 File Upload")
    
    dzt_file = st.file_uploader("Upload DZT file", type=['dzt', 'DZT', '.dzt'])
    dzg_file = st.file_uploader("Upload DZG file (GPS data)", type=['dzg', 'DZG'], 
                                help="Optional: For GPS-based distance normalization")
    
    st.markdown("---")
    st.header("🗺️ Coordinate Import (Optional)")
    
    # Coordinate CSV upload
    coord_csv = st.file_uploader("Upload CSV with coordinates", type=['csv'], 
                                help="CSV with columns: Easting, Northing, Elevation")
    
    if coord_csv:
        st.markdown('<div class="coordinate-box">', unsafe_allow_html=True)
        st.subheader("Coordinate Settings")
        
        # Preview CSV
        try:
            coords_df = pd.read_csv(coord_csv)
            st.info(f"CSV loaded: {len(coords_df)} rows, {len(coords_df.columns)} columns")
            if st.checkbox("Show CSV Preview"):
                st.dataframe(coords_df.head())
        except:
            st.warning("Could not preview CSV")
        
        # Coordinate column mapping
        col1, col2 = st.columns(2)
        with col1:
            easting_col = st.text_input("Easting Column", "Easting")
            northing_col = st.text_input("Northing Column", "Northing")
        with col2:
            elevation_col = st.text_input("Elevation Column", "Elevation")
            trace_col = st.text_input("Trace Column (optional)", "", 
                                     help="If CSV has trace numbers")
        
        # Coordinate interpolation method
        interp_method = st.selectbox("Interpolation Method", 
                                    ["Linear", "Cubic", "Nearest", "Previous", "Next"],
                                    help="Interpolate coordinates between points")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    st.header("📏 Axis Scaling")
    
    # Depth scaling (Y-axis)
    st.subheader("Depth Scaling (Y-axis)")
    depth_unit = st.selectbox("Depth Unit", ["samples", "meters", "nanoseconds", "feet"])
    
    if depth_unit != "samples":
        max_depth = st.number_input(f"Max Depth ({depth_unit})", 0.1, 1000.0, 12.0, 0.1,
                                   help=f"Maximum depth in {depth_unit}")
    
    # Distance scaling (X-axis)
    st.subheader("Distance Scaling (X-axis)")
    use_coords_for_distance = coord_csv is not None and st.checkbox("Use Coordinates for Distance", False,
                                                                    help="Use imported coordinates for X-axis")
    
    if not use_coords_for_distance:
        distance_unit = st.selectbox("Distance Unit", ["traces", "meters", "feet", "kilometers"])
        
        if distance_unit != "traces":
            total_distance = st.number_input(f"Total Distance ({distance_unit})", 0.1, 10000.0, 250.0, 0.1,
                                            help=f"Total survey distance in {distance_unit}")
    else:
        st.info("Using coordinate-based distance calculation")
        distance_unit = "meters"
    
    st.markdown("---")
    st.header("📐 Plot Aspect Ratio")
    
    st.markdown('<div class="aspect-box">', unsafe_allow_html=True)
    # Aspect ratio control
    aspect_mode = st.selectbox("Aspect Ratio Mode", 
                              ["Auto", "Equal", "Manual", "Realistic"],
                              help="Control Y:X scale of the plot")
    
    if aspect_mode == "Manual":
        aspect_ratio = st.selectbox("Aspect Ratio (Y:X)", 
                                   ["1:1", "1:2", "1:4", "1:5", "1:10", "2:1", "4:1", "5:1", "10:1"])
        aspect_ratio_float = float(aspect_ratio.split(":")[0]) / float(aspect_ratio.split(":")[1])
    elif aspect_mode == "Realistic":
        realistic_ratio = st.selectbox("Realistic Ratio", 
                                      ["1:5 (Shallow)", "1:10 (Standard)", "1:20 (Deep)", "1:50 (Very Deep)"])
        aspect_ratio_float = 1 / float(realistic_ratio.split(":")[1].split()[0])
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    st.header("🔍 Plot Windowing")
    
    use_custom_window = st.checkbox("Use Custom Plot Window", False,
                                   help="Define custom depth and distance ranges")
    
    if use_custom_window:
        st.markdown('<div class="window-box">', unsafe_allow_html=True)
        
        # Depth window
        st.subheader("Depth Window (Y-axis)")
        if depth_unit != "samples":
            depth_min = st.number_input(f"Min Depth ({depth_unit})", 0.0, max_depth, 0.0, 0.1)
            depth_max = st.number_input(f"Max Depth ({depth_unit})", 0.0, max_depth, max_depth, 0.1)
        else:
            depth_min = st.number_input("Min Depth (samples)", 0, 5000, 0)
            depth_max = st.number_input("Max Depth (samples)", 0, 5000, 255)
        
        # Distance window
        st.subheader("Distance Window (X-axis)")
        if not use_coords_for_distance:
            if distance_unit != "traces":
                distance_min = st.number_input(f"Min Distance ({distance_unit})", 0.0, total_distance, 0.0, 0.1)
                distance_max = st.number_input(f"Max Distance ({distance_unit})", 0.0, total_distance, total_distance, 0.1)
            else:
                distance_min = st.number_input("Min Distance (traces)", 0, 10000, 0)
                distance_max = st.number_input("Max Distance (traces)", 0, 10000, 800)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Multiple windows option
    multiple_windows = st.checkbox("Enable Multiple Windows", False,
                                  help="Plot multiple windows in same view")
    
    if multiple_windows and use_custom_window:
        num_windows = st.number_input("Number of Additional Windows", 1, 5, 1)
        
        windows = []
        for i in range(num_windows):
            st.markdown(f"**Window {i+2}**")
            col1, col2 = st.columns(2)
            with col1:
                d_min = st.number_input(f"Depth Min {i+2} ({depth_unit})", 0.0, max_depth, 2.0 + i*2, 0.1)
                d_max = st.number_input(f"Depth Max {i+2} ({depth_unit})", 0.0, max_depth, 5.0 + i*2, 0.1)
            with col2:
                if not use_coords_for_distance:
                    dist_min = st.number_input(f"Dist Min {i+2} ({distance_unit})", 0.0, total_distance, 50.0 + i*50, 0.1)
                    dist_max = st.number_input(f"Dist Max {i+2} ({distance_unit})", 0.0, total_distance, 150.0 + i*50, 0.1)
                else:
                    dist_min = st.number_input(f"Dist Min {i+2} (m)", 0.0, 10000.0, 50.0 + i*50, 0.1)
                    dist_max = st.number_input(f"Dist Max {i+2} (m)", 0.0, 10000.0, 150.0 + i*50, 0.1)
            
            windows.append({
                'depth_min': d_min,
                'depth_max': d_max,
                'distance_min': dist_min,
                'distance_max': dist_max,
                'color': f'C{i+1}'
            })
    
    st.markdown("---")
    st.header("🎛️ Processing Parameters")
    
    time_zero = st.number_input("Time Zero (samples)", 0, 2000, 2, 
                               help="Adjust start time of each trace")
    
    stacking = st.selectbox("Stacking", ["none", "auto", "manual"], 
                           help="Reduce noise by averaging traces")
    
    if stacking == "manual":
        stack_value = st.number_input("Stack Value", 1, 50, 3)
    
    st.markdown("---")
    st.header("📈 Time Gain Control")
    
    gain_type = st.selectbox(
        "Gain Type",
        ["Constant", "Linear", "Exponential", "AGC (Automatic Gain Control)", "Spherical"],
        help="Apply gain to amplify weak deep signals"
    )
    
    if gain_type == "Constant":
        const_gain = st.slider("Gain (%)", 0, 500, 100)
    
    elif gain_type == "Linear":
        min_gain = st.slider("Gain at Top (%)", 0, 200, 50)
        max_gain = st.slider("Gain at Bottom (%)", 0, 1000, 500)
    
    elif gain_type == "Exponential":
        base_gain = st.slider("Base Gain (%)", 0, 300, 100)
        exp_factor = st.slider("Exponential Factor", 0.1, 5.0, 1.5, 0.1)
    
    elif gain_type == "AGC (Automatic Gain Control)":
        window_size = st.slider("AGC Window (samples)", 10, 500, 100)
        target_amplitude = st.slider("Target Amplitude", 0.1, 1.0, 0.3, 0.05)
    
    elif gain_type == "Spherical":
        power_gain = st.slider("Power Gain", 1.0, 3.0, 2.0, 0.1)
        attenuation = st.slider("Attenuation Factor", 0.01, 0.1, 0.05, 0.01)
    
    st.markdown("---")
    st.header("⚙️ Advanced Processing")
    
    bgr = st.checkbox("Apply Background Removal", False)
    if bgr:
        bgr_type = st.selectbox("BGR Type", ["Full-width", "Boxcar"])
        if bgr_type == "Boxcar":
            bgr_window = st.slider("Boxcar Window", 10, 500, 100)
    
    freq_filter = st.checkbox("Apply Frequency Filter", False)
    if freq_filter:
        col1, col2 = st.columns(2)
        with col1:
            freq_min = st.number_input("Min Freq (MHz)", 10, 500, 60)
        with col2:
            freq_max = st.number_input("Max Freq (MHz)", 10, 1000, 130)
    
    process_btn = st.button("🚀 Process Data", type="primary", use_container_width=True)

# Helper functions
def apply_gain(array, gain_type, **kwargs):
    """Apply time-varying gain to radar data"""
    n_samples, n_traces = array.shape
    
    if gain_type == "Constant":
        gain = 1 + kwargs.get('const_gain', 1.0) / 100
        return array * gain
    
    elif gain_type == "Linear":
        min_g = 1 + kwargs.get('min_gain', 0.5) / 100
        max_g = 1 + kwargs.get('max_gain', 5.0) / 100
        gain_vector = np.linspace(min_g, max_g, n_samples)
        return array * gain_vector[:, np.newaxis]
    
    elif gain_type == "Exponential":
        base_g = 1 + kwargs.get('base_gain', 1.0) / 100
        exp_f = kwargs.get('exp_factor', 1.5)
        t = np.linspace(0, 1, n_samples)
        gain_vector = base_g * np.exp(exp_f * t)
        return array * gain_vector[:, np.newaxis]
    
    elif gain_type == "AGC (Automatic Gain Control)":
        window = kwargs.get('window_size', 100)
        target = kwargs.get('target_amplitude', 0.3)
        
        result = np.zeros_like(array)
        half_window = window // 2
        
        for i in range(n_traces):
            trace = array[:, i]
            agc_trace = np.zeros(n_samples)
            
            for j in range(n_samples):
                start = max(0, j - half_window)
                end = min(n_samples, j + half_window + 1)
                
                window_data = trace[start:end]
                rms = np.sqrt(np.mean(window_data**2))
                
                if rms > 0:
                    agc_trace[j] = trace[j] * (target / rms)
                else:
                    agc_trace[j] = trace[j]
            
            result[:, i] = agc_trace
        
        return result
    
    elif gain_type == "Spherical":
        power = kwargs.get('power_gain', 2.0)
        attenuation = kwargs.get('attenuation', 0.05)
        
        t = np.arange(n_samples) / n_samples
        gain_vector = (1 + attenuation * t) ** power
        gain_vector = gain_vector[:, np.newaxis]
        
        return array * gain_vector
    
    return array

def calculate_fft(trace, sampling_rate=1000):
    """Calculate FFT of a trace"""
    n = len(trace)
    yf = fft(trace)
    xf = fftfreq(n, 1/sampling_rate)[:n//2]
    
    magnitude = 2.0/n * np.abs(yf[:n//2])
    return xf, magnitude

def process_coordinates(coords_df, n_traces, easting_col='Easting', northing_col='Northing', 
                       elevation_col='Elevation', trace_col=None, method='linear'):
    """Process and interpolate coordinates to match GPR traces"""
    try:
        # Extract data
        easting = coords_df[easting_col].values
        northing = coords_df[northing_col].values
        elevation = coords_df[elevation_col].values
        
        # Determine x positions for coordinate points
        if trace_col and trace_col in coords_df.columns:
            coord_trace_indices = coords_df[trace_col].values
        else:
            # Use cumulative distance along profile
            dx = np.diff(easting)
            dy = np.diff(northing)
            distances = np.sqrt(dx**2 + dy**2)
            cumulative_dist = np.concatenate(([0], np.cumsum(distances)))
            coord_trace_indices = np.linspace(0, n_traces-1, len(cumulative_dist))
        
        # Target trace indices
        target_trace_indices = np.arange(n_traces)
        
        # Map method names
        method_map = {
            'linear': 'linear',
            'cubic': 'cubic',
            'nearest': 'nearest',
            'previous': 'previous',
            'next': 'next'
        }
        interp_kind = method_map.get(method.lower(), 'linear')
        
        # Create interpolation functions
        f_easting = interp1d(coord_trace_indices, easting, kind=interp_kind, fill_value='extrapolate')
        f_northing = interp1d(coord_trace_indices, northing, kind=interp_kind, fill_value='extrapolate')
        f_elevation = interp1d(coord_trace_indices, elevation, kind=interp_kind, fill_value='extrapolate')
        
        # Interpolate
        easting_interp = f_easting(target_trace_indices)
        northing_interp = f_northing(target_trace_indices)
        elevation_interp = f_elevation(target_trace_indices)
        
        # Calculate distance along profile
        dx_interp = np.diff(easting_interp)
        dy_interp = np.diff(northing_interp)
        dist_interp = np.sqrt(dx_interp**2 + dy_interp**2)
        cumulative_distance = np.concatenate(([0], np.cumsum(dist_interp)))
        
        return {
            'easting': easting_interp,
            'northing': northing_interp,
            'elevation': elevation_interp,
            'distance': cumulative_distance,
            'trace_indices': target_trace_indices,
            'original_points': len(easting),
            'interpolated_points': n_traces
        }
        
    except Exception as e:
        st.error(f"Error processing coordinates: {str(e)}")
        return None

def scale_axes(array_shape, depth_unit, max_depth, distance_unit, total_distance, coordinates=None):
    """Create scaled axis arrays based on user input"""
    n_samples, n_traces = array_shape
    
    # Scale Y-axis
    if depth_unit == "samples":
        y_axis = np.arange(n_samples)
        y_label = "Sample Number"
    elif depth_unit == "meters":
        y_axis = np.linspace(0, max_depth, n_samples)
        y_label = "Depth (m)"
    elif depth_unit == "nanoseconds":
        y_axis = np.linspace(0, max_depth, n_samples)
        y_label = "Two-way Time (ns)"
    elif depth_unit == "feet":
        y_axis = np.linspace(0, max_depth, n_samples)
        y_label = "Depth (ft)"
    
    # Scale X-axis
    if coordinates is not None:
        x_axis = coordinates['distance']
        x_label = "Distance along profile (m)"
        distance_unit = "meters"
        total_distance = x_axis[-1]
    elif distance_unit == "traces":
        x_axis = np.arange(n_traces)
        x_label = "Trace Number"
    elif distance_unit == "meters":
        x_axis = np.linspace(0, total_distance, n_traces)
        x_label = "Distance (m)"
    elif distance_unit == "feet":
        x_axis = np.linspace(0, total_distance, n_traces)
        x_label = "Distance (ft)"
    elif distance_unit == "kilometers":
        x_axis = np.linspace(0, total_distance, n_traces)
        x_label = "Distance (km)"
    
    return x_axis, y_axis, x_label, y_label, distance_unit, total_distance

def get_aspect_ratio(mode, manual_ratio=None):
    """Calculate aspect ratio based on mode"""
    if mode == "Auto":
        return "auto"
    elif mode == "Equal":
        return "equal"
    elif mode == "Manual" and manual_ratio is not None:
        return manual_ratio
    elif mode == "Realistic" and manual_ratio is not None:
        return manual_ratio
    else:
        return "auto"

def get_window_indices(x_axis, y_axis, depth_min, depth_max, distance_min, distance_max):
    """Convert user-specified window coordinates to array indices"""
    # Find depth indices
    depth_idx_min = np.argmin(np.abs(y_axis - depth_min))
    depth_idx_max = np.argmin(np.abs(y_axis - depth_max))
    
    if depth_idx_min > depth_idx_max:
        depth_idx_min, depth_idx_max = depth_idx_max, depth_idx_min
    
    # Find distance indices
    dist_idx_min = np.argmin(np.abs(x_axis - distance_min))
    dist_idx_max = np.argmin(np.abs(x_axis - distance_max))
    
    if dist_idx_min > dist_idx_max:
        dist_idx_min, dist_idx_max = dist_idx_max, dist_idx_min
    
    return {
        'depth_min_idx': depth_idx_min,
        'depth_max_idx': depth_idx_max,
        'dist_min_idx': dist_idx_min,
        'dist_max_idx': dist_idx_max,
        'depth_min_val': y_axis[depth_idx_min],
        'depth_max_val': y_axis[depth_idx_max],
        'dist_min_val': x_axis[dist_idx_min],
        'dist_max_val': x_axis[dist_idx_max]
    }

# Main content
if dzt_file and process_btn:
    with st.spinner("Processing radar data..."):
        try:
            # Import readgssi
            try:
                from readgssi import readgssi
            except ImportError:
                st.error("⚠️ readgssi not installed! Please run: pip install readgssi")
                st.stop()
            
            progress_bar = st.progress(0)
            
            with tempfile.TemporaryDirectory() as tmpdir:
                progress_bar.progress(10)
                
                # Save DZT
                dzt_path = os.path.join(tmpdir, "input.dzt")
                with open(dzt_path, "wb") as f:
                    f.write(dzt_file.getbuffer())
                
                # Save DZG if provided
                dzg_path = None
                if dzg_file:
                    dzg_path = os.path.join(tmpdir, "input.dzg")
                    with open(dzg_path, "wb") as f:
                        f.write(dzg_file.getbuffer())
                
                progress_bar.progress(30)
                
                # Process coordinates if provided
                coordinates_data = None
                if coord_csv:
                    try:
                        coords_df = pd.read_csv(coord_csv)
                        st.session_state.coordinates = coords_df
                    except Exception as e:
                        st.warning(f"Could not read CSV coordinates: {str(e)}")
                        coord_csv = None
                
                progress_bar.progress(40)
                
                # Build parameters for readgssi
                params = {
                    'infile': dzt_path,
                    'zero': [time_zero],
                    'verbose': False
                }
                
                if stacking == "auto":
                    params['stack'] = 'auto'
                elif stacking == "manual":
                    params['stack'] = stack_value
                
                if bgr:
                    if bgr_type == "Full-width":
                        params['bgr'] = 0
                    else:
                        params['bgr'] = bgr_window
                
                if freq_filter:
                    params['freqmin'] = freq_min
                    params['freqmax'] = freq_max
                
                progress_bar.progress(50)
                
                # Read data
                header, arrays, gps = readgssi.readgssi(**params)
                
                progress_bar.progress(70)
                
                if arrays and len(arrays) > 0:
                    original_array = arrays[0]
                    processed_array = original_array.copy()
                    
                    # Apply gain
                    if gain_type == "Constant":
                        processed_array = apply_gain(processed_array, "Constant", const_gain=const_gain)
                    elif gain_type == "Linear":
                        processed_array = apply_gain(processed_array, "Linear", min_gain=min_gain, max_gain=max_gain)
                    elif gain_type == "Exponential":
                        processed_array = apply_gain(processed_array, "Exponential", base_gain=base_gain, exp_factor=exp_factor)
                    elif gain_type == "AGC (Automatic Gain Control)":
                        processed_array = apply_gain(processed_array, "AGC (Automatic Gain Control)", window_size=window_size, target_amplitude=target_amplitude)
                    elif gain_type == "Spherical":
                        processed_array = apply_gain(processed_array, "Spherical", power_gain=power_gain, attenuation=attenuation)
                    
                    progress_bar.progress(80)
                    
                    # Process coordinates if provided
                    if coord_csv and st.session_state.coordinates is not None:
                        coordinates_data = process_coordinates(
                            st.session_state.coordinates,
                            processed_array.shape[1],
                            easting_col=easting_col,
                            northing_col=northing_col,
                            elevation_col=elevation_col,
                            trace_col=trace_col if trace_col else None,
                            method=interp_method.lower()
                        )
                        st.session_state.interpolated_coords = coordinates_data
                    
                    progress_bar.progress(90)
                    
                    # Store in session state
                    st.session_state.header = header
                    st.session_state.original_array = original_array
                    st.session_state.processed_array = processed_array
                    st.session_state.gps = gps
                    st.session_state.data_loaded = True
                    
                    # Store settings
                    st.session_state.depth_unit = depth_unit
                    st.session_state.max_depth = max_depth if depth_unit != "samples" else None
                    st.session_state.use_coords_for_distance = 'use_coords_for_distance' in locals() and use_coords_for_distance
                    st.session_state.coordinates_data = coordinates_data
                    
                    if not st.session_state.use_coords_for_distance:
                        st.session_state.distance_unit = distance_unit
                        st.session_state.total_distance = total_distance if distance_unit != "traces" else None
                    else:
                        st.session_state.distance_unit = "meters"
                        st.session_state.total_distance = coordinates_data['distance'][-1] if coordinates_data else None
                    
                    # Store aspect ratio
                    st.session_state.aspect_mode = aspect_mode
                    if aspect_mode == "Manual" and 'aspect_ratio_float' in locals():
                        st.session_state.aspect_ratio = aspect_ratio_float
                    elif aspect_mode == "Realistic" and 'aspect_ratio_float' in locals():
                        st.session_state.aspect_ratio = aspect_ratio_float
                    else:
                        st.session_state.aspect_ratio = None
                    
                    # Store window parameters
                    st.session_state.use_custom_window = use_custom_window
                    if use_custom_window:
                        st.session_state.depth_min = depth_min if 'depth_min' in locals() else 0
                        st.session_state.depth_max = depth_max if 'depth_max' in locals() else max_depth
                        if not st.session_state.use_coords_for_distance:
                            st.session_state.distance_min = distance_min if 'distance_min' in locals() else 0
                            st.session_state.distance_max = distance_max if 'distance_max' in locals() else total_distance
                        else:
                            st.session_state.distance_min = 0
                            st.session_state.distance_max = coordinates_data['distance'][-1] if coordinates_data else total_distance
                    
                    st.session_state.multiple_windows = multiple_windows
                    if multiple_windows and use_custom_window:
                        st.session_state.additional_windows = windows if 'windows' in locals() else []
                    
                    progress_bar.progress(100)
                    st.success("✅ Data processed successfully!")
                    
                else:
                    st.error("No radar data found in file")
                    
        except Exception as e:
            st.error(f"Error processing data: {str(e)}")

# Display results if data is loaded
if st.session_state.data_loaded:
    # Create tabs
    tab_names = ["📊 Header Info", "📈 Full View", "🔍 Custom Window", "🗺️ Coordinate View", "📉 FFT Analysis", "🎛️ Gain Analysis", "💾 Export"]
    tabs = st.tabs(tab_names)
    
    with tabs[0]:  # Header Info
        st.subheader("File Information & Settings")
        
        # Display coordinate info
        if st.session_state.interpolated_coords is not None:
            st.markdown("### Coordinate Information")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Original Points", st.session_state.interpolated_coords['original_points'])
                st.metric("Total Distance", f"{st.session_state.interpolated_coords['distance'][-1]:.1f} m")
            with col2:
                st.metric("Interpolated Points", st.session_state.interpolated_coords['interpolated_points'])
                st.metric("Easting Range", f"{st.session_state.interpolated_coords['easting'].min():.1f} - {st.session_state.interpolated_coords['easting'].max():.1f}")
        
        # Display settings
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### Axis Scaling")
            settings_data = {
                "Y-axis (Depth)": f"{st.session_state.depth_unit}",
                "Max Depth": f"{st.session_state.max_depth if st.session_state.max_depth else 'Auto'}",
                "X-axis (Distance)": f"{st.session_state.distance_unit}",
                "Total Distance": f"{st.session_state.total_distance if st.session_state.total_distance else 'Auto'}"
            }
            for key, value in settings_data.items():
                st.markdown(f"**{key}:** {value}")
            
            st.markdown(f"**Aspect Mode:** {st.session_state.aspect_mode}")
            if st.session_state.aspect_ratio:
                st.markdown(f"**Aspect Ratio:** {st.session_state.aspect_ratio:.3f}")
        
        with col2:
            if st.session_state.header:
                st.markdown("### File Header")
                info_data = {
                    "System": st.session_state.header.get('system', 'Unknown'),
                    "Antenna Frequency": f"{st.session_state.header.get('ant_freq', 'N/A')} MHz",
                    "Samples per Trace": st.session_state.header.get('spt', 'N/A'),
                    "Number of Traces": st.session_state.header.get('ntraces', 'N/A')
                }
                for key, value in info_data.items():
                    st.markdown(f"**{key}:** {value}")
    
    with tabs[1]:  # Full View
        st.subheader("Full Radar Profile")
        
        # Get aspect ratio
        aspect_value = get_aspect_ratio(
            st.session_state.aspect_mode,
            st.session_state.aspect_ratio
        )
        
        # Create scaled axes
        x_axis_full, y_axis_full, x_label_full, y_label_full, _, _ = scale_axes(
            st.session_state.processed_array.shape,
            st.session_state.depth_unit,
            st.session_state.max_depth,
            st.session_state.distance_unit,
            st.session_state.total_distance,
            coordinates=st.session_state.interpolated_coords if st.session_state.use_coords_for_distance else None
        )
        
        # Display options
        col1, col2, col3 = st.columns(3)
        with col1:
            show_colorbar = st.checkbox("Show Colorbar", True, key="full_cbar")
            interpolation = st.selectbox("Interpolation", ["none", "bilinear", "bicubic", "gaussian"], key="full_interp")
        with col2:
            colormap = st.selectbox("Colormap", ["seismic", "RdBu", "gray", "viridis", "jet"], key="full_cmap")
            display_aspect = st.selectbox("Display Aspect", ["auto", "equal", 0.1, 0.2, 0.5, 1.0, 2.0, 5.0], index=0)
        with col3:
            vmin = st.number_input("Color Min", -1.0, 0.0, -0.5, 0.01, key="full_vmin")
            vmax = st.number_input("Color Max", 0.0, 1.0, 0.5, 0.01, key="full_vmax")
            normalize_colors = st.checkbox("Auto-normalize Colors", True, key="full_norm")
        
        # Create figure
        fig_full, (ax1_full, ax2_full) = plt.subplots(1, 2, figsize=(18, 8))
        
        # Plot original
        if normalize_colors:
            vmax_plot = np.percentile(np.abs(st.session_state.original_array), 99)
            vmin_plot = -vmax_plot
        else:
            vmin_plot, vmax_plot = vmin, vmax
        
        im1 = ax1_full.imshow(st.session_state.original_array, 
                             extent=[x_axis_full[0], x_axis_full[-1], y_axis_full[-1], y_axis_full[0]],
                             aspect=display_aspect, cmap=colormap, 
                             vmin=vmin_plot, vmax=vmax_plot,
                             interpolation=interpolation)
        ax1_full.set_xlabel(x_label_full)
        ax1_full.set_ylabel(y_label_full)
        ax1_full.set_title("Original Data - Full View")
        ax1_full.grid(True, alpha=0.3)
        if show_colorbar:
            plt.colorbar(im1, ax=ax1_full, label='Amplitude')
        
        # Plot processed
        im2 = ax2_full.imshow(st.session_state.processed_array,
                             extent=[x_axis_full[0], x_axis_full[-1], y_axis_full[-1], y_axis_full[0]],
                             aspect=display_aspect, cmap=colormap,
                             vmin=vmin_plot, vmax=vmax_plot,
                             interpolation=interpolation)
        ax2_full.set_xlabel(x_label_full)
        ax2_full.set_ylabel(y_label_full)
        ax2_full.set_title(f"Processed ({gain_type} Gain) - Full View")
        ax2_full.grid(True, alpha=0.3)
        if show_colorbar:
            plt.colorbar(im2, ax=ax2_full, label='Amplitude')
        
        # Add window overlay
        if st.session_state.use_custom_window:
            window_info = get_window_indices(
                x_axis_full, y_axis_full,
                st.session_state.depth_min, st.session_state.depth_max,
                st.session_state.distance_min, st.session_state.distance_max
            )
            rect = plt.Rectangle((window_info['dist_min_val'], window_info['depth_min_val']),
                               window_info['dist_max_val'] - window_info['dist_min_val'],
                               window_info['depth_max_val'] - window_info['depth_min_val'],
                               linewidth=2, edgecolor='yellow', facecolor='none', alpha=0.8)
            ax1_full.add_patch(rect.copy())
            ax2_full.add_patch(rect.copy())
        
        plt.tight_layout()
        st.pyplot(fig_full)
        
        # Display aspect info
        st.info(f"**Aspect Ratio:** {aspect_value} | **Dimensions:** {st.session_state.processed_array.shape[1]} × {st.session_state.processed_array.shape[0]}")
    
    with tabs[2]:  # Custom Window
        st.subheader("Custom Window Analysis")
        
        if not st.session_state.use_custom_window:
            st.warning("Enable 'Use Custom Plot Window' in sidebar")
        else:
            # Create scaled axes
            x_axis, y_axis, x_label, y_label, _, _ = scale_axes(
                st.session_state.processed_array.shape,
                st.session_state.depth_unit,
                st.session_state.max_depth,
                st.session_state.distance_unit,
                st.session_state.total_distance,
                coordinates=st.session_state.interpolated_coords if st.session_state.use_coords_for_distance else None
            )
            
            # Get window indices
            window_info = get_window_indices(
                x_axis, y_axis,
                st.session_state.depth_min, st.session_state.depth_max,
                st.session_state.distance_min, st.session_state.distance_max
            )
            
            # Extract window data
            window_data = st.session_state.processed_array[
                window_info['depth_min_idx']:window_info['depth_max_idx'],
                window_info['dist_min_idx']:window_info['dist_max_idx']
            ]
            
            window_data_original = st.session_state.original_array[
                window_info['depth_min_idx']:window_info['depth_max_idx'],
                window_info['dist_min_idx']:window_info['dist_max_idx']
            ]
            
            x_axis_window = x_axis[window_info['dist_min_idx']:window_info['dist_max_idx']]
            y_axis_window = y_axis[window_info['depth_min_idx']:window_info['depth_max_idx']]
            
            # Display window stats
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Depth Range", f"{window_info['depth_min_val']:.1f} - {window_info['depth_max_val']:.1f} {st.session_state.depth_unit}")
            with col2:
                st.metric("Distance Range", f"{window_info['dist_min_val']:.1f} - {window_info['dist_max_val']:.1f} {st.session_state.distance_unit}")
            with col3:
                st.metric("Window Size", f"{window_data.shape[0]} × {window_data.shape[1]}")
            with col4:
                st.metric("Data Points", f"{window_data.size:,}")
            
            # Plot windowed data
            fig_window, (ax1_window, ax2_window) = plt.subplots(1, 2, figsize=(16, 6))
            
            aspect_window = get_aspect_ratio(st.session_state.aspect_mode, st.session_state.aspect_ratio)
            
            im1_window = ax1_window.imshow(window_data_original,
                                          extent=[x_axis_window[0], x_axis_window[-1], 
                                                  y_axis_window[-1], y_axis_window[0]],
                                          aspect=aspect_window, cmap='seismic')
            ax1_window.set_xlabel(x_label)
            ax1_window.set_ylabel(y_label)
            ax1_window.set_title(f"Original - Custom Window")
            ax1_window.grid(True, alpha=0.3)
            plt.colorbar(im1_window, ax=ax1_window, label='Amplitude')
            
            im2_window = ax2_window.imshow(window_data,
                                          extent=[x_axis_window[0], x_axis_window[-1], 
                                                  y_axis_window[-1], y_axis_window[0]],
                                          aspect=aspect_window, cmap='seismic')
            ax2_window.set_xlabel(x_label)
            ax2_window.set_ylabel(y_label)
            ax2_window.set_title(f"Processed - Custom Window")
            ax2_window.grid(True, alpha=0.3)
            plt.colorbar(im2_window, ax=ax2_window, label='Amplitude')
            
            plt.tight_layout()
            st.pyplot(fig_window)
            
            # Multiple windows
            if st.session_state.multiple_windows and hasattr(st.session_state, 'additional_windows'):
                st.subheader("Multiple Windows View")
                num_windows_total = 1 + len(st.session_state.additional_windows)
                cols = min(2, num_windows_total)
                rows = (num_windows_total + cols - 1) // cols
                
                fig_multi, axes = plt.subplots(rows, cols, figsize=(cols*8, rows*6))
                if rows * cols == 1:
                    axes = np.array([[axes]])
                elif rows == 1:
                    axes = axes.reshape(1, -1)
                elif cols == 1:
                    axes = axes.reshape(-1, 1)
                
                # Plot main window
                ax = axes[0, 0]
                im = ax.imshow(window_data,
                             extent=[x_axis_window[0], x_axis_window[-1], 
                                     y_axis_window[-1], y_axis_window[0]],
                             aspect=aspect_window, cmap='seismic')
                ax.set_xlabel(x_label)
                ax.set_ylabel(y_label)
                ax.set_title(f"Window 1")
                ax.grid(True, alpha=0.3)
                plt.colorbar(im, ax=ax, label='Amplitude')
                
                # Plot additional windows
                window_idx = 1
                for i in range(rows):
                    for j in range(cols):
                        if window_idx >= num_windows_total:
                            if i == 0 and j == 0:
                                continue
                            axes[i, j].axis('off')
                            continue
                        
                        if window_idx == 0:
                            continue
                        
                        ax = axes[i, j]
                        win = st.session_state.additional_windows[window_idx-1]
                        
                        win_info = get_window_indices(
                            x_axis, y_axis,
                            win['depth_min'], win['depth_max'],
                            win['distance_min'], win['distance_max']
                        )
                        
                        win_data = st.session_state.processed_array[
                            win_info['depth_min_idx']:win_info['depth_max_idx'],
                            win_info['dist_min_idx']:win_info['dist_max_idx']
                        ]
                        
                        x_axis_win = x_axis[win_info['dist_min_idx']:win_info['dist_max_idx']]
                        y_axis_win = y_axis[win_info['depth_min_idx']:win_info['depth_max_idx']]
                        
                        im = ax.imshow(win_data,
                                     extent=[x_axis_win[0], x_axis_win[-1], 
                                             y_axis_win[-1], y_axis_win[0]],
                                     aspect=aspect_window, cmap='seismic')
                        ax.set_xlabel(x_label)
                        ax.set_ylabel(y_label)
                        ax.set_title(f"Window {window_idx+1}")
                        ax.grid(True, alpha=0.3)
                        plt.colorbar(im, ax=ax, label='Amplitude')
                        
                        window_idx += 1
                
                plt.tight_layout()
                st.pyplot(fig_multi)
    
    with tabs[3]:  # Coordinate View
        st.subheader("Coordinate-Based Visualization")
        
        if st.session_state.interpolated_coords is None:
            st.warning("No coordinates imported")
        else:
            # Display coordinate stats
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Profile Length", f"{st.session_state.interpolated_coords['distance'][-1]:.1f} m")
            with col2:
                st.metric("Elevation Change", f"{st.session_state.interpolated_coords['elevation'].max() - st.session_state.interpolated_coords['elevation'].min():.1f} m")
            with col3:
                st.metric("Easting Range", f"{np.ptp(st.session_state.interpolated_coords['easting']):.1f} m")

            with col4:
                st.metric("Northing Range", f"{np.ptp(st.session_state.interpolated_coords['northing']):.1f} m")
            
            # Create coordinate visualizations
            fig_coords, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
            
            # Plan view
            ax1.plot(st.session_state.interpolated_coords['easting'], 
                    st.session_state.interpolated_coords['northing'], 
                    'b-', linewidth=1, alpha=0.7)
            scatter1 = ax1.scatter(st.session_state.interpolated_coords['easting'], 
                                 st.session_state.interpolated_coords['northing'], 
                                 c=st.session_state.interpolated_coords['distance'], 
                                 cmap='viridis', s=20, alpha=0.8)
            ax1.set_xlabel('Easting (m)')
            ax1.set_ylabel('Northing (m)')
            ax1.set_title('Plan View - Survey Line')
            ax1.grid(True, alpha=0.3)
            ax1.axis('equal')
            plt.colorbar(scatter1, ax=ax1, label='Distance (m)')
            
            # Elevation profile
            ax2.plot(st.session_state.interpolated_coords['distance'], 
                    st.session_state.interpolated_coords['elevation'], 
                    'g-', linewidth=2, alpha=0.8)
            ax2.fill_between(st.session_state.interpolated_coords['distance'],
                            st.session_state.interpolated_coords['elevation'].min(),
                            st.session_state.interpolated_coords['elevation'],
                            alpha=0.3, color='green')
            ax2.set_xlabel('Distance (m)')
            ax2.set_ylabel('Elevation (m)')
            ax2.set_title('Elevation Profile')
            ax2.grid(True, alpha=0.3)
            
            # 3D view
            from mpl_toolkits.mplot3d import Axes3D
            ax3 = fig_coords.add_subplot(2, 2, 3, projection='3d')
            ax3.plot(st.session_state.interpolated_coords['easting'],
                    st.session_state.interpolated_coords['northing'],
                    st.session_state.interpolated_coords['elevation'],
                    'b-', linewidth=1, alpha=0.7)
            scatter3 = ax3.scatter(st.session_state.interpolated_coords['easting'],
                                 st.session_state.interpolated_coords['northing'],
                                 st.session_state.interpolated_coords['elevation'],
                                 c=st.session_state.interpolated_coords['distance'],
                                 cmap='viridis', s=20, alpha=0.8)
            ax3.set_xlabel('Easting (m)')
            ax3.set_ylabel('Northing (m)')
            ax3.set_zlabel('Elevation (m)')
            ax3.set_title('3D Survey Line')
            plt.colorbar(scatter3, ax=ax3, label='Distance (m)')
            
            # GPR with coordinate-based X-axis
            aspect_value_coords = get_aspect_ratio(st.session_state.aspect_mode, st.session_state.aspect_ratio)
            
            if st.session_state.depth_unit != "samples":
                depth_axis = np.linspace(0, st.session_state.max_depth, 
                                        st.session_state.processed_array.shape[0])
            else:
                depth_axis = np.arange(st.session_state.processed_array.shape[0])
            
            im4 = ax4.imshow(st.session_state.processed_array,
                          extent=[st.session_state.interpolated_coords['distance'][0],
                                 st.session_state.interpolated_coords['distance'][-1],
                                 depth_axis[-1], depth_axis[0]],
                          aspect=aspect_value_coords, cmap='seismic', alpha=0.9)
            ax4.set_xlabel('Distance (m)')
            ax4.set_ylabel(f'Depth ({st.session_state.depth_unit})')
            ax4.set_title(f'GPR Data with Coordinate Scaling')
            ax4.grid(True, alpha=0.2)
            plt.colorbar(im4, ax=ax4, label='Amplitude')
            
            plt.tight_layout()
            st.pyplot(fig_coords)
    
    with tabs[4]:  # FFT Analysis
        st.subheader("Frequency vs Amplitude Analysis (FFT)")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            trace_for_fft = st.slider("Select Trace for FFT", 
                                     0, st.session_state.processed_array.shape[1]-1, 
                                     st.session_state.processed_array.shape[1]//2,
                                     key="fft_trace")
        with col2:
            sampling_rate = st.number_input("Sampling Rate (MHz)", 100, 5000, 1000, 100,
                                           help="Antenna sampling rate in MHz",
                                           key="fft_sampling")
        with col3:
            fft_mode = st.selectbox("FFT Mode", ["Single Trace", "Average of All Traces", "Trace Range"],
                                   key="fft_mode")
        
        if fft_mode == "Trace Range":
            trace_start = st.number_input("Start Trace", 0, st.session_state.processed_array.shape[1]-1, 0,
                                         key="fft_start")
            trace_end = st.number_input("End Trace", 0, st.session_state.processed_array.shape[1]-1, 
                                       st.session_state.processed_array.shape[1]-1,
                                       key="fft_end")
        
        # Calculate FFT
        if fft_mode == "Single Trace":
            trace_data = st.session_state.processed_array[:, trace_for_fft]
            freq, amplitude = calculate_fft(trace_data, sampling_rate)
            title = f"FFT - Trace {trace_for_fft}"
        elif fft_mode == "Average of All Traces":
            avg_trace = np.mean(st.session_state.processed_array, axis=1)
            freq, amplitude = calculate_fft(avg_trace, sampling_rate)
            title = "FFT - Average of All Traces"
        elif fft_mode == "Trace Range":
            avg_trace = np.mean(st.session_state.processed_array[:, trace_start:trace_end+1], axis=1)
            freq, amplitude = calculate_fft(avg_trace, sampling_rate)
            title = f"FFT - Traces {trace_start} to {trace_end}"
        else:
            freq, amplitude = [], []
            title = ""
        
        if len(freq) > 0:
            fig_fft, (ax_fft1, ax_fft2) = plt.subplots(1, 2, figsize=(16, 6))
            
            ax_fft1.plot(freq, amplitude, 'b-', linewidth=2, alpha=0.8)
            ax_fft1.fill_between(freq, 0, amplitude, alpha=0.3, color='blue')
            ax_fft1.set_xlabel("Frequency (MHz)")
            ax_fft1.set_ylabel("Amplitude")
            ax_fft1.set_title(f"{title} - Linear Scale")
            ax_fft1.grid(True, alpha=0.3)
            ax_fft1.set_xlim([0, sampling_rate/2])
            
            ax_fft2.semilogy(freq, amplitude, 'r-', linewidth=2, alpha=0.8)
            ax_fft2.fill_between(freq, 0.001, amplitude, alpha=0.3, color='red')
            ax_fft2.set_xlabel("Frequency (MHz)")
            ax_fft2.set_ylabel("Amplitude (log)")
            ax_fft2.set_title(f"{title} - Log Scale")
            ax_fft2.grid(True, alpha=0.3)
            ax_fft2.set_xlim([0, sampling_rate/2])
            
            plt.tight_layout()
            st.pyplot(fig_fft)
            
            # FFT statistics
            peak_idx = np.argmax(amplitude)
            peak_freq = freq[peak_idx]
            peak_amp = amplitude[peak_idx]
            
            max_amp = np.max(amplitude)
            half_power = max_amp / np.sqrt(2)
            mask = amplitude >= half_power
            
            if np.any(mask):
                low_freq = freq[mask][0]
                high_freq = freq[mask][-1]
                bandwidth = high_freq - low_freq
            else:
                low_freq = high_freq = bandwidth = 0
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Peak Frequency", f"{peak_freq:.1f} MHz")
            with col2:
                st.metric("Peak Amplitude", f"{peak_amp:.3e}")
            with col3:
                st.metric("Bandwidth (-3dB)", f"{bandwidth:.1f} MHz")
            with col4:
                st.metric("Center Freq", f"{(low_freq + high_freq)/2:.1f} MHz")
    
    with tabs[5]:  # Gain Analysis
        st.subheader("Gain Analysis")
        
        n_samples = st.session_state.original_array.shape[0]
        with np.errstate(divide='ignore', invalid='ignore'):
            gain_profile = np.zeros(n_samples)
            for i in range(n_samples):
                orig_slice = st.session_state.original_array[i, :]
                proc_slice = st.session_state.processed_array[i, :]
                mask = np.abs(orig_slice) > 1e-10
                if np.any(mask):
                    gains = np.abs(proc_slice[mask]) / np.abs(orig_slice[mask])
                    gain_profile[i] = np.median(gains)
                else:
                    gain_profile[i] = 1.0
        
        y_axis_analysis, _, _, y_label_analysis = scale_axes(
            (n_samples, 1),
            st.session_state.depth_unit,
            st.session_state.max_depth,
            "traces",
            None
        )
        
        fig_gain, ax_gain = plt.subplots(figsize=(10, 6))
        ax_gain.plot(gain_profile, y_axis_analysis, 'b-', linewidth=2, label='Gain Factor')
        ax_gain.fill_betweenx(y_axis_analysis, 1, gain_profile, alpha=0.3, color='blue')
        ax_gain.set_xlabel("Gain Factor (multiplier)")
        ax_gain.set_ylabel(y_label_analysis)
        ax_gain.set_title("Gain Applied vs Depth")
        ax_gain.grid(True, alpha=0.3)
        ax_gain.legend()
        ax_gain.invert_yaxis()
        
        st.pyplot(fig_gain)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Min Gain", f"{gain_profile.min():.2f}x")
        with col2:
            st.metric("Max Gain", f"{gain_profile.max():.2f}x")
        with col3:
            st.metric("Mean Gain", f"{gain_profile.mean():.2f}x")
    
    with tabs[6]:  # Export
        st.subheader("Export Processed Data")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("💾 Save Full Image", use_container_width=True):
                x_axis_export, y_axis_export, x_label_export, y_label_export, _, _ = scale_axes(
                    st.session_state.processed_array.shape,
                    st.session_state.depth_unit,
                    st.session_state.max_depth,
                    st.session_state.distance_unit,
                    st.session_state.total_distance,
                    coordinates=st.session_state.interpolated_coords if st.session_state.use_coords_for_distance else None
                )
                
                fig, ax = plt.subplots(figsize=(12, 8))
                im = ax.imshow(st.session_state.processed_array,
                             extent=[x_axis_export[0], x_axis_export[-1], 
                                    y_axis_export[-1], y_axis_export[0]],
                             aspect='auto', cmap='seismic')
                ax.set_xlabel(x_label_export)
                ax.set_ylabel(y_label_export)
                ax.set_title(f"GPR Data - {gain_type} Gain")
                plt.colorbar(im, ax=ax, label='Amplitude')
                plt.tight_layout()
                plt.savefig("gpr_data_full.png", dpi=300, bbox_inches='tight')
                st.success("Saved as 'gpr_data_full.png'")
        
        with col2:
            if st.session_state.use_custom_window:
                if st.button("💾 Save Window Image", use_container_width=True):
                    x_axis, y_axis, x_label, y_label, _, _ = scale_axes(
                        st.session_state.processed_array.shape,
                        st.session_state.depth_unit,
                        st.session_state.max_depth,
                        st.session_state.distance_unit,
                        st.session_state.total_distance,
                        coordinates=st.session_state.interpolated_coords if st.session_state.use_coords_for_distance else None
                    )
                    
                    window_info = get_window_indices(
                        x_axis, y_axis,
                        st.session_state.depth_min, st.session_state.depth_max,
                        st.session_state.distance_min, st.session_state.distance_max
                    )
                    
                    window_data = st.session_state.processed_array[
                        window_info['depth_min_idx']:window_info['depth_max_idx'],
                        window_info['dist_min_idx']:window_info['dist_max_idx']
                    ]
                    
                    x_axis_window = x_axis[window_info['dist_min_idx']:window_info['dist_max_idx']]
                    y_axis_window = y_axis[window_info['depth_min_idx']:window_info['depth_max_idx']]
                    
                    fig, ax = plt.subplots(figsize=(10, 6))
                    im = ax.imshow(window_data,
                                 extent=[x_axis_window[0], x_axis_window[-1], 
                                         y_axis_window[-1], y_axis_window[0]],
                                 aspect='auto', cmap='seismic')
                    ax.set_xlabel(x_label)
                    ax.set_ylabel(y_label)
                    ax.set_title(f"GPR Data - Custom Window")
                    plt.colorbar(im, ax=ax, label='Amplitude')
                    plt.tight_layout()
                    plt.savefig("gpr_data_windowed.png", dpi=300, bbox_inches='tight')
                    st.success("Saved as 'gpr_data_windowed.png'")
        
        with col3:
            # Export CSV
            x_axis_csv, _, _, _, _, _ = scale_axes(
                st.session_state.processed_array.shape,
                st.session_state.depth_unit,
                st.session_state.max_depth,
                st.session_state.distance_unit,
                st.session_state.total_distance,
                coordinates=st.session_state.interpolated_coords if st.session_state.use_coords_for_distance else None
            )
            
            csv_data = pd.DataFrame(st.session_state.processed_array, 
                                  columns=[f"{xi:.2f}" for xi in x_axis_csv])
            csv_string = csv_data.to_csv(index=False)
            
            st.download_button(
                label="📥 Download Full CSV",
                data=csv_string,
                file_name="gpr_data_full.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with col4:
            if st.session_state.interpolated_coords is not None:
                coord_df = pd.DataFrame({
                    'Trace_Index': st.session_state.interpolated_coords['trace_indices'],
                    'Distance_m': st.session_state.interpolated_coords['distance'],
                    'Easting_m': st.session_state.interpolated_coords['easting'],
                    'Northing_m': st.session_state.interpolated_coords['northing'],
                    'Elevation_m': st.session_state.interpolated_coords['elevation']
                })
                coord_csv = coord_df.to_csv(index=False)
                
                st.download_button(
                    label="📥 Download Coordinates",
                    data=coord_csv,
                    file_name="interpolated_coordinates.csv",
                    mime="text/csv",
                    use_container_width=True
                )

# Initial state message
elif not dzt_file:
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.info("""
        👈 **Upload a DZT file to begin processing**
        
        **Features included:**
        1. **Coordinate Import:** CSV with Easting, Northing, Elevation
        2. **Aspect Ratio Control:** Y:X scaling (1:1 to 1:50)
        3. **Custom Windowing:** Zoom to specific depth/distance ranges
        4. **Multiple Windows:** Compare different areas
        5. **FFT Analysis:** Frequency vs amplitude
        6. **Gain Control:** Time-varying gain for deep signals
        
        **Coordinate CSV Example:**
        ```
        Easting, Northing, Elevation
        100.0, 200.0, 50.0
        101.0, 201.0, 49.8
        ...
        ```
        
        **All features integrated in one powerful app!**
        """)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666;'>"
    "📡 <b>Advanced GPR Data Processor v6.0</b> | Integrated Features | "
    "Built with Streamlit & readgssi"
    "</div>",
    unsafe_allow_html=True
)

