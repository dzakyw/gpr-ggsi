import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tempfile
import os
from pathlib import Path
import warnings
from scipy import signal, interpolate
from scipy.fft import fft, fftfreq
import plotly.graph_objects as go
from plotly.subplots import make_subplots
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="GPR Data Processor",
    page_icon="📡",
    layout="wide"
)

# Title
st.title("📡 GPR Data Processor with Window Selection & Coordinate Integration")
st.markdown("Process GPR data with custom windows, aspect ratios, and coordinate mapping")

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
    .window-control {
        background-color: #e8f4fd;
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 15px;
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
if 'coordinates_loaded' not in st.session_state:
    st.session_state.coordinates_loaded = False

# Sidebar
with st.sidebar:
    st.header("📂 File Upload")
    
    dzt_file = st.file_uploader("Upload DZT file", type=['dzt', 'DZT', '.dzt'])
    dzg_file = st.file_uploader("Upload DZG file (GPS data)", type=['dzg', 'DZG'])
    
    # Coordinate CSV upload
    st.markdown("---")
    st.header("📍 Coordinate Import (Optional)")
    coord_file = st.file_uploader("Upload Coordinate CSV", type=['csv', 'txt'],
                                  help="CSV with columns: Easting, Northing, Elevation (Optional)")
    
    if coord_file:
        try:
            coords_df = pd.read_csv(coord_file)
            required_cols = ['Easting', 'Northing']
            if all(col in coords_df.columns for col in required_cols):
                st.session_state.coordinates_df = coords_df
                st.session_state.coordinates_loaded = True
                st.success(f"✅ Loaded {len(coords_df)} coordinate points")
                
                # Show preview
                with st.expander("Preview Coordinates"):
                    st.dataframe(coords_df.head())
                    
                # Check for trace numbers
                if 'TraceNumber' in coords_df.columns:
                    st.info("Found TraceNumber column")
                else:
                    st.info("TraceNumber not found - will interpolate")
            else:
                st.error("CSV must contain 'Easting' and 'Northing' columns")
                st.session_state.coordinates_loaded = False
        except Exception as e:
            st.error(f"Error reading CSV: {str(e)}")
    
    st.markdown("---")
    st.header("📏 Axis Scaling")
    
    # Depth scaling (Y-axis)
    st.subheader("Depth Scaling (Y-axis)")
    depth_unit = st.selectbox("Depth Unit", ["samples", "meters", "nanoseconds", "feet"])
    
    if depth_unit != "samples":
        max_depth = st.number_input(f"Max Depth ({depth_unit})", 0.1, 1000.0, 12.0, 0.1,
                                   help=f"Set maximum depth in {depth_unit}")
        velocity = None
        if depth_unit == "nanoseconds":
            velocity = st.number_input("Wave Velocity (m/ns)", 0.01, 0.3, 0.1, 0.01)
    
    # Distance scaling (X-axis)
    st.subheader("Distance Scaling (X-axis)")
    distance_unit = st.selectbox("Distance Unit", ["traces", "meters", "feet", "kilometers"])
    
    if distance_unit != "traces":
        total_distance = st.number_input(f"Total Distance ({distance_unit})", 0.1, 10000.0, 250.0, 0.1,
                                        help=f"Set total survey distance in {distance_unit}")
    
    st.markdown("---")
    st.header("🖼️ Plot Windows")
    
    window_mode = st.selectbox("Window Mode", ["Full Profile", "Custom Windows"])
    
    if window_mode == "Custom Windows":
        st.subheader("Depth Window (Y-axis)")
        depth_min = st.number_input("Min Depth", 0.0, 1000.0, 0.0, 0.1)
        depth_max = st.number_input("Max Depth", 0.0, 1000.0, 5.0, 0.1)
        
        st.subheader("Distance Window (X-axis)")
        distance_min = st.number_input("Min Distance", 0.0, 10000.0, 0.0, 0.1)
        distance_max = st.number_input("Max Distance", 0.0, 10000.0, 100.0, 0.1)
    
    st.markdown("---")
    st.header("📐 Aspect Ratio")
    
    aspect_ratio = st.selectbox("Y:X Aspect Ratio", 
                               ["Auto", "1:1", "1:2", "1:3", "1:4", "1:5", "1:10", "Custom"])
    
    if aspect_ratio == "Custom":
        custom_aspect = st.number_input("Custom Aspect Ratio (Y:X)", 0.1, 10.0, 1.0, 0.1)
    
    st.markdown("---")
    st.header("🎛️ Processing Parameters")
    
    time_zero = st.number_input("Time Zero (samples)", 0, 2000, 2)
    
    stacking = st.selectbox("Stacking", ["none", "auto", "manual"])
    
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

def scale_axes(array_shape, depth_unit, max_depth, distance_unit, total_distance):
    """Create scaled axis arrays based on user input"""
    n_samples, n_traces = array_shape
    
    # Scale Y-axis (depth/time)
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
    
    # Scale X-axis (distance)
    if distance_unit == "traces":
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
    
    return x_axis, y_axis, x_label, y_label

def interpolate_coordinates(coords_df, n_traces):
    """Interpolate coordinates for all traces"""
    n_coords = len(coords_df)
    
    # Create trace indices for known coordinates
    if 'TraceNumber' in coords_df.columns:
        trace_indices = coords_df['TraceNumber'].values
    else:
        # Distribute coordinates evenly along the line
        trace_indices = np.linspace(0, n_traces-1, n_coords, dtype=int)
    
    # Interpolate each coordinate component
    easting_interp = interpolate.interp1d(trace_indices, coords_df['Easting'].values, 
                                         kind='linear', fill_value='extrapolate')
    northing_interp = interpolate.interp1d(trace_indices, coords_df['Northing'].values, 
                                          kind='linear', fill_value='extrapolate')
    
    # Create arrays for all traces
    all_traces = np.arange(n_traces)
    easting_all = easting_interp(all_traces)
    northing_all = northing_interp(all_traces)
    
    # Calculate cumulative distance
    dx = np.diff(easting_all)
    dy = np.diff(northing_all)
    distances = np.sqrt(dx**2 + dy**2)
    cumulative_distance = np.zeros(n_traces)
    cumulative_distance[1:] = np.cumsum(distances)
    
    # Elevation if available
    if 'Elevation' in coords_df.columns:
        elev_interp = interpolate.interp1d(trace_indices, coords_df['Elevation'].values, 
                                          kind='linear', fill_value='extrapolation')
        elevation_all = elev_interp(all_traces)
    else:
        elevation_all = np.zeros(n_traces)
    
    result = pd.DataFrame({
        'Trace': all_traces,
        'Easting': easting_all,
        'Northing': northing_all,
        'Elevation': elevation_all,
        'Distance': cumulative_distance
    })
    
    return result

def get_window_indices(x_axis, y_axis, x_min, x_max, y_min, y_max):
    """Get array indices for window selection"""
    x_mask = (x_axis >= x_min) & (x_axis <= x_max)
    y_mask = (y_axis >= y_min) & (y_axis <= y_max)
    
    x_indices = np.where(x_mask)[0]
    y_indices = np.where(y_mask)[0]
    
    if len(x_indices) == 0:
        x_indices = np.arange(len(x_axis))
    if len(y_indices) == 0:
        y_indices = np.arange(len(y_axis))
    
    return (x_indices[0], x_indices[-1]), (y_indices[0], y_indices[-1])

# Main content
if dzt_file and process_btn:
    with st.spinner("Processing radar data..."):
        try:
            # Try to import readgssi
            try:
                from readgssi import readgssi
            except ImportError:
                st.error("⚠️ readgssi not installed! Please run:")
                st.code("pip install readgssi")
                st.stop()
            
            # Create progress bar
            progress_bar = st.progress(0)
            
            # Save files to temp location
            with tempfile.TemporaryDirectory() as tmpdir:
                progress_bar.progress(10)
                
                # Save DZT
                dzt_path = os.path.join(tmpdir, "input.dzt")
                with open(dzt_path, "wb") as f:
                    f.write(dzt_file.getbuffer())
                
                progress_bar.progress(30)
                
                # Build parameters for readgssi
                params = {
                    'infile': dzt_path,
                    'zero': [time_zero],
                    'verbose': False
                }
                
                # Add stacking
                if stacking == "auto":
                    params['stack'] = 'auto'
                elif stacking == "manual":
                    params['stack'] = stack_value
                
                # Add BGR
                if bgr:
                    if bgr_type == "Full-width":
                        params['bgr'] = 0
                    else:
                        params['bgr'] = bgr_window
                
                # Add frequency filter
                if freq_filter:
                    params['freqmin'] = freq_min
                    params['freqmax'] = freq_max
                
                progress_bar.progress(50)
                
                # Read data
                header, arrays, gps = readgssi.readgssi(**params)
                
                progress_bar.progress(70)
                
                # Store original array
                if arrays and len(arrays) > 0:
                    original_array = arrays[0]
                    
                    # Apply time-varying gain
                    processed_array = original_array.copy()
                    
                    # Apply selected gain
                    if gain_type == "Constant":
                        processed_array = apply_gain(processed_array, "Constant", 
                                                    const_gain=const_gain)
                    elif gain_type == "Linear":
                        processed_array = apply_gain(processed_array, "Linear",
                                                    min_gain=min_gain, max_gain=max_gain)
                    elif gain_type == "Exponential":
                        processed_array = apply_gain(processed_array, "Exponential",
                                                    base_gain=base_gain, exp_factor=exp_factor)
                    elif gain_type == "AGC (Automatic Gain Control)":
                        processed_array = apply_gain(processed_array, "AGC (Automatic Gain Control)",
                                                    window_size=window_size, target_amplitude=target_amplitude)
                    elif gain_type == "Spherical":
                        processed_array = apply_gain(processed_array, "Spherical",
                                                    power_gain=power_gain, attenuation=attenuation)
                    
                    # Interpolate coordinates if provided
                    if st.session_state.coordinates_loaded:
                        coords_df = st.session_state.coordinates_df
                        n_traces = processed_array.shape[1]
                        interpolated_coords = interpolate_coordinates(coords_df, n_traces)
                        st.session_state.interpolated_coords = interpolated_coords
                        st.success(f"✅ Interpolated coordinates for {n_traces} traces")
                    
                    progress_bar.progress(90)
                    
                    # Store in session state
                    st.session_state.header = header
                    st.session_state.original_array = original_array
                    st.session_state.processed_array = processed_array
                    st.session_state.gps = gps
                    st.session_state.data_loaded = True
                    
                    # Store axis scaling parameters
                    st.session_state.depth_unit = depth_unit
                    st.session_state.max_depth = max_depth if depth_unit != "samples" else None
                    st.session_state.distance_unit = distance_unit
                    st.session_state.total_distance = total_distance if distance_unit != "traces" else None
                    
                    # Store window parameters
                    st.session_state.window_mode = window_mode
                    if window_mode == "Custom Windows":
                        st.session_state.depth_min = depth_min
                        st.session_state.depth_max = depth_max
                        st.session_state.distance_min = distance_min
                        st.session_state.distance_max = distance_max
                    
                    # Store aspect ratio
                    st.session_state.aspect_ratio = aspect_ratio
                    if aspect_ratio == "Custom":
                        st.session_state.custom_aspect = custom_aspect
                    
                    progress_bar.progress(100)
                    st.success("✅ Data processed successfully!")
                    
                else:
                    st.error("No radar data found in file")
                    
        except Exception as e:
            st.error(f"Error processing data: {str(e)}")
            st.code(str(e))

# Display results if data is loaded
if st.session_state.data_loaded:
    # Create tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["📊 Header Info", "📈 Radar Profile", "🖼️ Window Views", 
                                                   "📍 Coordinates", "📉 FFT Analysis", "💾 Export"])
    
    with tab1:
        st.subheader("File Information & Settings")
        
        # Display settings
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Processing Settings")
            settings = {
                "Gain Type": gain_type,
                "Time Zero": f"{time_zero} samples",
                "Stacking": stacking,
                "Window Mode": window_mode,
                "Aspect Ratio": aspect_ratio
            }
            
            for key, value in settings.items():
                st.markdown(f"**{key}:** {value}")
        
        with col2:
            if st.session_state.header:
                st.markdown("### File Header")
                info = {
                    "System": st.session_state.header.get('system', 'Unknown'),
                    "Antenna Frequency": f"{st.session_state.header.get('ant_freq', 'N/A')} MHz",
                    "Samples": st.session_state.header.get('spt', 'N/A'),
                    "Traces": st.session_state.header.get('ntraces', 'N/A'),
                    "Sampling Depth": f"{st.session_state.header.get('depth', 'N/A'):.2f} m"
                }
                
                for key, value in info.items():
                    st.markdown(f"**{key}:** {value}")
    
    with tab2:
        st.subheader("Full Radar Profile")
        
        # Create scaled axes
        x_axis, y_axis, x_label, y_label = scale_axes(
            st.session_state.processed_array.shape,
            st.session_state.depth_unit,
            st.session_state.max_depth,
            st.session_state.distance_unit,
            st.session_state.total_distance
        )
        
        # Determine aspect ratio
        if aspect_ratio == "Auto":
            plot_aspect = 'auto'
        elif aspect_ratio == "Custom":
            plot_aspect = custom_aspect
        else:
            plot_aspect = float(aspect_ratio.split(':')[1])  # Convert "1:4" to 4
        
        # Plotting options
        col1, col2 = st.columns(2)
        with col1:
            colormap = st.selectbox("Colormap", ["seismic", "RdBu", "gray", "viridis", "jet", "hot"])
            show_grid = st.checkbox("Show Grid", True)
        
        with col2:
            interpolation = st.selectbox("Interpolation", ["none", "bilinear", "bicubic", "gaussian"])
            auto_normalize = st.checkbox("Auto-normalize Colors", True)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Prepare data for plotting
        plot_data = st.session_state.processed_array
        
        # Apply color normalization
        if auto_normalize:
            vmax = np.percentile(np.abs(plot_data), 99)
            vmin = -vmax
        else:
            vmin, vmax = -0.5, 0.5
        
        # Create the plot
        im = ax.imshow(plot_data, 
                      extent=[x_axis[0], x_axis[-1], y_axis[-1], y_axis[0]],
                      aspect=plot_aspect, cmap=colormap, 
                      vmin=vmin, vmax=vmax,
                      interpolation=interpolation)
        
        ax.set_xlabel(x_label, fontsize=12)
        ax.set_ylabel(y_label, fontsize=12)
        ax.set_title(f"Full Radar Profile - {gain_type} Gain", fontsize=14, pad=20)
        
        if show_grid:
            ax.grid(True, alpha=0.3, linestyle='--')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, pad=0.01)
        cbar.set_label('Amplitude', rotation=270, labelpad=15)
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Add window selection overlay if in custom mode
        if window_mode == "Custom Windows":
            # Draw rectangle for selected window
            rect = plt.Rectangle((distance_min, depth_min), 
                                distance_max - distance_min,
                                depth_max - depth_min,
                                linewidth=2, edgecolor='yellow', 
                                facecolor='none', linestyle='--')
            ax.add_patch(rect)
            st.pyplot(fig)  # Re-plot with rectangle
    
    with tab3:
        st.subheader("Windowed Views")
        
        if window_mode == "Custom Windows":
            # Create scaled axes
            x_axis, y_axis, x_label, y_label = scale_axes(
                st.session_state.processed_array.shape,
                st.session_state.depth_unit,
                st.session_state.max_depth,
                st.session_state.distance_unit,
                st.session_state.total_distance
            )
            
            # Get window indices
            (x_start, x_end), (y_start, y_end) = get_window_indices(
                x_axis, y_axis,
                distance_min, distance_max,
                depth_min, depth_max
            )
            
            # Extract windowed data
            windowed_data = st.session_state.processed_array[y_start:y_end, x_start:x_end]
            window_x_axis = x_axis[x_start:x_end]
            window_y_axis = y_axis[y_start:y_end]
            
            # Create multiple window views
            st.markdown("### Multiple Window Views")
            
            # Determine number of rows based on window size
            n_rows = 1
            if (depth_max - depth_min) > (distance_max - distance_min) / 3:
                n_rows = 2
            
            fig_windows, axes = plt.subplots(n_rows, 2, figsize=(16, 6*n_rows))
            axes = axes.flatten() if n_rows > 1 else [axes]
            
            # Different aspect ratios for comparison
            aspect_ratios = [1, 2, 4, 5, 10, 'auto']
            
            for idx, ax in enumerate(axes[:6]):
                if idx < len(aspect_ratios):
                    aspect_val = aspect_ratios[idx]
                    
                    # Prepare data for plotting
                    vmax = np.percentile(np.abs(windowed_data), 99)
                    vmin = -vmax
                    
                    im = ax.imshow(windowed_data,
                                 extent=[window_x_axis[0], window_x_axis[-1], 
                                        window_y_axis[-1], window_y_axis[0]],
                                 aspect=aspect_val, cmap='seismic',
                                 vmin=vmin, vmax=vmax,
                                 interpolation='bilinear')
                    
                    ax.set_xlabel(x_label)
                    ax.set_ylabel(y_label)
                    ax.set_title(f"Aspect Ratio: {aspect_val}:1" if aspect_val != 'auto' else "Aspect: Auto")
                    ax.grid(True, alpha=0.3)
                
                if idx == 0:  # Add colorbar to first subplot
                    plt.colorbar(im, ax=ax, label='Amplitude')
            
            plt.tight_layout()
            st.pyplot(fig_windows)
            
            # Window statistics
            st.subheader("Window Statistics")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Window Size", f"{windowed_data.shape[1]} × {windowed_data.shape[0]}")
                st.metric("X Range", f"{distance_min:.1f} - {distance_max:.1f} {distance_unit}")
            
            with col2:
                st.metric("Y Range", f"{depth_min:.1f} - {depth_max:.1f} {depth_unit}")
                st.metric("Data Min/Max", f"{windowed_data.min():.2e} / {windowed_data.max():.2e}")
            
            with col3:
                st.metric("Mean Amplitude", f"{windowed_data.mean():.2e}")
                st.metric("Std Deviation", f"{windowed_data.std():.2e}")
            
            # Interactive window selection
            st.subheader("Interactive Window Selection")
            
            col1, col2 = st.columns(2)
            
            with col1:
                new_depth_min = st.slider("New Depth Min", float(y_axis[0]), float(y_axis[-1]), 
                                         float(depth_min), 0.1)
                new_depth_max = st.slider("New Depth Max", float(y_axis[0]), float(y_axis[-1]), 
                                         float(depth_max), 0.1)
            
            with col2:
                new_distance_min = st.slider("New Distance Min", float(x_axis[0]), float(x_axis[-1]), 
                                            float(distance_min), 0.1)
                new_distance_max = st.slider("New Distance Max", float(x_axis[0]), float(x_axis[-1]), 
                                            float(distance_max), 0.1)
            
            if st.button("Apply New Window"):
                st.session_state.depth_min = new_depth_min
                st.session_state.depth_max = new_depth_max
                st.session_state.distance_min = new_distance_min
                st.session_state.distance_max = new_distance_max
                st.rerun()
        
        else:
            st.info("Switch to 'Custom Windows' mode in the sidebar to use window selection")
    
    with tab4:
        st.subheader("Coordinate Integration")
        
        if st.session_state.coordinates_loaded and hasattr(st.session_state, 'interpolated_coords'):
            coords_df = st.session_state.interpolated_coords
            
            # Display coordinate statistics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Traces", len(coords_df))
                st.metric("Total Distance", f"{coords_df['Distance'].iloc[-1]:.2f} m")
            
            with col2:
                st.metric("Easting Range", f"{coords_df['Easting'].min():.2f} - {coords_df['Easting'].max():.2f}")
                st.metric("Northing Range", f"{coords_df['Northing'].min():.2f} - {coords_df['Northing'].max():.2f}")
            
            with col3:
                if 'Elevation' in coords_df.columns:
                    st.metric("Elevation Range", f"{coords_df['Elevation'].min():.2f} - {coords_df['Elevation'].max():.2f}")
            
            # Plot coordinate visualization
            fig_coords, axes = plt.subplots(2, 2, figsize=(16, 12))
            
            # Plot 1: Easting vs Northing (plan view)
            axes[0, 0].plot(coords_df['Easting'], coords_df['Northing'], 'b-', linewidth=1, alpha=0.7)
            axes[0, 0].scatter(coords_df['Easting'], coords_df['Northing'], c=coords_df['Trace'], 
                              cmap='viridis', s=20, alpha=0.6)
            axes[0, 0].set_xlabel('Easting')
            axes[0, 0].set_ylabel('Northing')
            axes[0, 0].set_title('Survey Line (Plan View)')
            axes[0, 0].grid(True, alpha=0.3)
            axes[0, 0].axis('equal')
            
            # Plot 2: Distance vs Elevation (profile view)
            if 'Elevation' in coords_df.columns:
                axes[0, 1].plot(coords_df['Distance'], coords_df['Elevation'], 'r-', linewidth=2)
                axes[0, 1].set_xlabel('Distance (m)')
                axes[0, 1].set_ylabel('Elevation')
                axes[0, 1].set_title('Elevation Profile')
                axes[0, 1].grid(True, alpha=0.3)
                axes[0, 1].fill_between(coords_df['Distance'], coords_df['Elevation'].min(), 
                                       coords_df['Elevation'], alpha=0.3, color='red')
            
            # Plot 3: Trace distribution
            axes[1, 0].scatter(coords_df['Trace'], coords_df['Distance'], c='green', s=10, alpha=0.6)
            axes[1, 0].set_xlabel('Trace Number')
            axes[1, 0].set_ylabel('Cumulative Distance (m)')
            axes[1, 0].set_title('Trace vs Distance Mapping')
            axes[1, 0].grid(True, alpha=0.3)
            
            # Plot 4: 3D view
            ax3d = fig_coords.add_subplot(2, 2, 4, projection='3d')
            scatter = ax3d.scatter(coords_df['Easting'], coords_df['Northing'], 
                                  coords_df['Elevation'] if 'Elevation' in coords_df.columns else np.zeros(len(coords_df)),
                                  c=coords_df['Trace'], cmap='plasma', s=20, alpha=0.8)
            ax3d.set_xlabel('Easting')
            ax3d.set_ylabel('Northing')
            ax3d.set_zlabel('Elevation')
            ax3d.set_title('3D Survey Line')
            plt.colorbar(scatter, ax=ax3d, label='Trace Number')
            
            plt.tight_layout()
            st.pyplot(fig_coords)
            
            # Display coordinate table
            with st.expander("View Interpolated Coordinates"):
                st.dataframe(coords_df.head(50))
            
            # Option to plot radar data using coordinates
            st.subheader("Plot Radar Data with Coordinates")
            
            if st.checkbox("Show Radar Profile with Coordinate Overlay"):
                fig_combined, ax_combined = plt.subplots(figsize=(14, 6))
                
                # Plot radar data (using distance from coordinates)
                if hasattr(st.session_state, 'processed_array'):
                    radar_data = st.session_state.processed_array
                    
                    # Use coordinate-based distance for X-axis
                    x_coords = coords_df['Distance'].values
                    y_coords = scale_axes(
                        radar_data.shape,
                        depth_unit,
                        max_depth,
                        "meters",  # Force meters for depth
                        None
                    )[1]
                    
                    # Ensure dimensions match
                    if len(x_coords) == radar_data.shape[1]:
                        im = ax_combined.imshow(radar_data,
                                              extent=[x_coords[0], x_coords[-1], 
                                                     y_coords[-1], y_coords[0]],
                                              aspect='auto', cmap='seismic',
                                              alpha=0.8)
                        
                        ax_combined.set_xlabel('Distance from Start (m)')
                        ax_combined.set_ylabel('Depth (m)')
                        ax_combined.set_title('Radar Profile with Coordinate-based Distance')
                        ax_combined.grid(True, alpha=0.3)
                        
                        plt.colorbar(im, ax=ax_combined, label='Amplitude')
                        
                        # Overlay elevation if available
                        if 'Elevation' in coords_df.columns:
                            ax_elev = ax_combined.twinx()
                            ax_elev.plot(x_coords, coords_df['Elevation'], 'g-', linewidth=2, alpha=0.7)
                            ax_elev.set_ylabel('Elevation', color='green')
                            ax_elev.tick_params(axis='y', labelcolor='green')
                        
                        st.pyplot(fig_combined)
            
            # Export coordinates
            st.subheader("Export Coordinates")
            csv_coords = coords_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Interpolated Coordinates",
                data=csv_coords,
                file_name="interpolated_coordinates.csv",
                mime="text/csv"
            )
        
        else:
            st.info("No coordinates loaded. Upload a CSV file with Easting and Northing columns in the sidebar.")
    
    with tab5:
        st.subheader("FFT Analysis")
        
        # FFT options
        col1, col2 = st.columns(2)
        
        with col1:
            trace_for_fft = st.slider("Select Trace for FFT", 
                                     0, st.session_state.processed_array.shape[1]-1, 
                                     st.session_state.processed_array.shape[1]//2)
            sampling_rate = st.number_input("Sampling Rate (MHz)", 100, 5000, 1000, 100)
        
        with col2:
            fft_mode = st.selectbox("FFT Mode", ["Single Trace", "Average of All Traces"])
            fft_window = st.selectbox("FFT Window", ["Rectangular", "Hanning", "Hamming", "Blackman"])
        
        # Calculate FFT
        if fft_mode == "Single Trace":
            trace_data = st.session_state.processed_array[:, trace_for_fft]
        else:
            trace_data = np.mean(st.session_state.processed_array, axis=1)
        
        # Apply window function
        n = len(trace_data)
        if fft_window == "Hanning":
            window = np.hanning(n)
        elif fft_window == "Hamming":
            window = np.hamming(n)
        elif fft_window == "Blackman":
            window = np.blackman(n)
        else:
            window = np.ones(n)
        
        windowed_data = trace_data * window
        freq, amplitude = calculate_fft(windowed_data, sampling_rate)
        
        # Plot FFT
        fig_fft, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        ax1.plot(freq, amplitude, 'b-', linewidth=2, alpha=0.8)
        ax1.fill_between(freq, 0, amplitude, alpha=0.3, color='blue')
        ax1.set_xlabel("Frequency (MHz)")
        ax1.set_ylabel("Amplitude")
        ax1.set_title(f"FFT - {fft_mode}")
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim([0, sampling_rate/2])
        
        ax2.semilogy(freq, amplitude, 'r-', linewidth=2, alpha=0.8)
        ax2.fill_between(freq, 0.001, amplitude, alpha=0.3, color='red')
        ax2.set_xlabel("Frequency (MHz)")
        ax2.set_ylabel("Amplitude (log)")
        ax2.set_title(f"FFT - Log Scale")
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim([0, sampling_rate/2])
        
        plt.tight_layout()
        st.pyplot(fig_fft)
    
    with tab6:
        st.subheader("Export Data")
        
        # Export options
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("💾 Save Full Profile", use_container_width=True):
                fig, ax = plt.subplots(figsize=(12, 8))
                im = ax.imshow(st.session_state.processed_array, aspect='auto', cmap='seismic')
                ax.set_title("GPR Data")
                plt.colorbar(im, ax=ax)
                plt.savefig("gpr_full_profile.png", dpi=300, bbox_inches='tight')
                st.success("Saved as 'gpr_full_profile.png'")
        
        with col2:
            if window_mode == "Custom Windows":
                if st.button("💾 Save Windowed View", use_container_width=True):
                    fig, ax = plt.subplots(figsize=(12, 8))
                    
                    # Extract windowed data
                    x_axis, y_axis, _, _ = scale_axes(
                        st.session_state.processed_array.shape,
                        depth_unit, max_depth,
                        distance_unit, total_distance
                    )
                    
                    (x_start, x_end), (y_start, y_end) = get_window_indices(
                        x_axis, y_axis,
                        distance_min, distance_max,
                        depth_min, depth_max
                    )
                    
                    windowed_data = st.session_state.processed_array[y_start:y_end, x_start:x_end]
                    
                    im = ax.imshow(windowed_data, aspect='auto', cmap='seismic')
                    ax.set_title(f"Window: {depth_min}-{depth_max}m, {distance_min}-{distance_max}{distance_unit}")
                    plt.colorbar(im, ax=ax)
                    plt.savefig("gpr_windowed.png", dpi=300, bbox_inches='tight')
                    st.success("Saved as 'gpr_windowed.png'")
        
        with col3:
            # Export data as CSV
            csv_data = pd.DataFrame(st.session_state.processed_array)
            csv_string = csv_data.to_csv(index=False)
            
            st.download_button(
                label="📥 Download Data CSV",
                data=csv_string,
                file_name="gpr_data.csv",
                mime="text/csv",
                use_container_width=True
            )

# Initial state message
elif not dzt_file:
    st.info("""
    👈 **Upload a DZT file to begin processing**
    
    **New Features:**
    1. **Window Selection:** View specific depth and distance ranges
    2. **Aspect Ratio Control:** Change Y:X ratio for better visualization
    3. **Coordinate Integration:** Import CSV with Easting, Northing, Elevation
    4. **Automatic Coordinate Interpolation:** Map sparse coordinates to all traces
    
    **Example Usage:**
    - Set Depth: 0-12 meters, select window 5-9 meters
    - Set Distance: 0-250 meters, select window 50-150 meters
    - Upload coordinates CSV with 40 points, auto-interpolate to 1000 traces
    """)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666;'>"
    "📡 <b>GPR Data Processor v4.0</b> | Window Selection & Coordinate Mapping | "
    "Built with Streamlit & readgssi"
    "</div>",
    unsafe_allow_html=True
)
