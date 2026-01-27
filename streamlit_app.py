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
from pathlib import Path

# Path to the Streamlit config directory
config_dir = Path.home() / ".streamlit"
config_dir.mkdir(exist_ok=True)

# Path to the config file
config_file = config_dir / "config.toml"

# Create config file if it doesn't exist
if not config_file.exists():
    config_file.write_text("""[server]
runOnSave = true
""")
    st.info(f"Created config file at {config_file}")
else:
    # Check if runOnSave is already set
    config_content = config_file.read_text()
    if "runOnSave = true" not in config_content:
        # Append the setting if file exists but doesn't have it
        with open(config_file, 'a') as f:
            f.write("\n[server]\nrunOnSave = true\n")
        st.info("Added runOnSave setting to config file")

# Now run your Streamlit app normally
st.title("My App")
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="GPR Data Processor",
    page_icon="📡",
    layout="wide"
)

# Title
st.title("📡 GPR Data Processor with Custom Depth Windows")
st.markdown("Process GPR data with custom depth windows and zoom capabilities")

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
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        border: 2px dashed #1e88e5;
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
if 'depth_windows' not in st.session_state:
    st.session_state.depth_windows = []

# Sidebar
with st.sidebar:
    st.header("📂 File Upload")
    
    dzt_file = st.file_uploader("Upload DZT file", type=['dzt', 'DZT', '.dzt'])
    dzg_file = st.file_uploader("Upload DZG file (GPS data)", type=['dzg', 'DZG'], 
                                help="Optional: Required for GPS-based distance normalization")
    
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
            velocity = st.number_input("Wave Velocity (m/ns)", 0.01, 0.3, 0.1, 0.01,
                                      help="Wave velocity for time-depth conversion")
    
    # Distance scaling (X-axis)
    st.subheader("Distance Scaling (X-axis)")
    distance_unit = st.selectbox("Distance Unit", ["traces", "meters", "feet", "kilometers"])
    
    if distance_unit != "traces":
        total_distance = st.number_input(f"Total Distance ({distance_unit})", 0.1, 10000.0, 250.0, 0.1,
                                        help=f"Set total survey distance in {distance_unit}")
    
    st.markdown("---")
    st.header("📊 Depth Windows")
    
    st.markdown("""
    **Define custom depth windows:**
    - Example: For 0-12m total depth
    - Window 1: 0-5m
    - Window 2: 5-9m
    - Window 3: 9-12m
    """)
    
    num_windows = st.number_input("Number of Windows", 1, 10, 1, 
                                 help="How many depth windows to create")
    
    # Initialize windows in session state
    if 'windows' not in st.session_state:
        st.session_state.windows = [{"name": f"Window {i+1}", "min": 0.0, "max": max_depth if depth_unit != "samples" else 100} 
                                   for i in range(num_windows)]
    
    # Dynamic window inputs
    for i in range(num_windows):
        st.markdown(f"**Window {i+1}**")
        col1, col2 = st.columns(2)
        with col1:
            min_val = st.number_input(f"Min {depth_unit}", 0.0, max_depth if depth_unit != "samples" else 1000.0, 
                                     float(i * (max_depth/num_windows)) if depth_unit != "samples" else 0.0, 0.1,
                                     key=f"window_{i}_min")
        with col2:
            max_val = st.number_input(f"Max {depth_unit}", 0.0, max_depth if depth_unit != "samples" else 1000.0, 
                                     float((i+1) * (max_depth/num_windows)) if depth_unit != "samples" else 100.0, 0.1,
                                     key=f"window_{i}_max")
        
        # Update session state
        if min_val < max_val:
            st.session_state.windows[i] = {"name": f"Window {i+1}", "min": min_val, "max": max_val}
    
    st.markdown("---")
    st.header("🎛️ Processing Parameters")
    
    time_zero = st.number_input("Time Zero (samples)", 0, 2000, 2, 
                               help="Adjust the start time of each trace")
    
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
        
        # Create spherical spreading correction
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
    
    # Take magnitude
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

def get_depth_window_indices(y_axis, window_min, window_max):
    """Get array indices for a specific depth window"""
    # Find indices within the window
    mask = (y_axis >= window_min) & (y_axis <= window_max)
    indices = np.where(mask)[0]
    
    if len(indices) > 0:
        return indices[0], indices[-1]
    else:
        return 0, len(y_axis) - 1

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
                
                # Save DZG if provided
                dzg_path = None
                if dzg_file:
                    dzg_path = os.path.join(tmpdir, "input.dzg")
                    with open(dzg_path, "wb") as f:
                        f.write(dzg_file.getbuffer())
                
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
                    
                    progress_bar.progress(90)
                    
                    # Store in session state
                    st.session_state.header = header
                    st.session_state.original_array = original_array
                    st.session_state.processed_array = processed_array
                    st.session_state.gps = gps
                    st.session_state.data_loaded = True
                    
                    # Store axis scaling parameters
                    st.session_state.depth_unit = depth_unit
                    st.session_state.max_depth = max_depth if depth_unit != "samples" else original_array.shape[0]
                    st.session_state.distance_unit = distance_unit
                    st.session_state.total_distance = total_distance if distance_unit != "traces" else original_array.shape[1]
                    
                    progress_bar.progress(100)
                    st.success("✅ Data processed successfully!")
                    
                else:
                    st.error("No radar data found in file")
                    
        except Exception as e:
            st.error(f"Error processing data: {str(e)}")
            st.code(str(e))

# Display results if data is loaded
if st.session_state.data_loaded:
    # Create scaled axes
    x_axis, y_axis, x_label, y_label = scale_axes(
        st.session_state.processed_array.shape,
        st.session_state.depth_unit,
        st.session_state.max_depth if hasattr(st.session_state, 'max_depth') else None,
        st.session_state.distance_unit,
        st.session_state.total_distance if hasattr(st.session_state, 'total_distance') else None
    )
    
    # Create tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Header Info", "📈 Full Profile", "🔍 Depth Windows", "📉 FFT Analysis", "💾 Export"])
    
    with tab1:
        st.subheader("File Information & Scaling Settings")
        
        # Display scaling settings
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Axis Scaling Settings")
            settings_data = {
                "Y-axis (Depth)": f"{st.session_state.depth_unit}",
                "Max Y-value": f"{st.session_state.max_depth if st.session_state.max_depth else 'Auto'}",
                "X-axis (Distance)": f"{st.session_state.distance_unit}",
                "Total X-distance": f"{st.session_state.total_distance if st.session_state.total_distance else 'Auto'}"
            }
            
            for key, value in settings_data.items():
                st.markdown(f"**{key}:** {value}")
        
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
        
        # Array shape info
        st.markdown("### Data Dimensions")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"**Original Shape:** {st.session_state.original_array.shape}")
            st.markdown(f"**Samples (Y):** {st.session_state.original_array.shape[0]}")
        
        with col2:
            st.markdown(f"**Traces (X):** {st.session_state.original_array.shape[1]}")
            st.markdown(f"**Data Type:** {st.session_state.original_array.dtype}")
        
        with col3:
            st.markdown(f"**Min Amplitude:** {st.session_state.original_array.min():.2e}")
            st.markdown(f"**Max Amplitude:** {st.session_state.original_array.max():.2e}")
        
        # Show complete header in expander
        if st.session_state.header:
            with st.expander("Show Complete Header"):
                st.json(st.session_state.header)
    
    with tab2:
        st.subheader("Full Radar Profile")
        
        # Display options
        col1, col2 = st.columns(2)
        
        with col1:
            show_full_profile = st.checkbox("Show Full Profile", True)
            colormap = st.selectbox("Colormap", ["seismic", "RdBu", "gray", "viridis", "jet", "coolwarm"])
        
        with col2:
            show_colorbar = st.checkbox("Show Colorbar", True)
            normalize_colors = st.checkbox("Auto-normalize Colors", True)
        
        if show_full_profile:
            # Create figure for full profile
            fig_full, ax_full = plt.subplots(figsize=(14, 8))
            
            if normalize_colors:
                vmax = np.percentile(np.abs(st.session_state.processed_array), 99)
                vmin = -vmax
            else:
                vmin, vmax = -0.5, 0.5
            
            im_full = ax_full.imshow(st.session_state.processed_array, 
                                    extent=[x_axis[0], x_axis[-1], y_axis[-1], y_axis[0]],
                                    aspect='auto', cmap=colormap, 
                                    vmin=vmin, vmax=vmax)
            
            ax_full.set_xlabel(x_label)
            ax_full.set_ylabel(y_label)
            ax_full.set_title(f"Full Profile: 0 to {y_axis[-1]:.2f} {st.session_state.depth_unit}")
            ax_full.grid(True, alpha=0.2, linestyle='--')
            
            # Mark depth windows on full profile
            for i, window in enumerate(st.session_state.windows):
                ax_full.axhline(y=window['min'], color='yellow', linestyle='--', alpha=0.7, linewidth=1)
                ax_full.axhline(y=window['max'], color='yellow', linestyle='--', alpha=0.7, linewidth=1)
                ax_full.fill_between([x_axis[0], x_axis[-1]], window['min'], window['max'], 
                                    alpha=0.1, color='yellow')
                ax_full.text(x_axis[0], (window['min'] + window['max'])/2, 
                            f"W{i+1}", color='yellow', fontweight='bold',
                            backgroundcolor='black', alpha=0.7)
            
            if show_colorbar:
                plt.colorbar(im_full, ax=ax_full, label='Amplitude')
            
            plt.tight_layout()
            st.pyplot(fig_full)
        
        # Depth statistics
        st.subheader("Depth Statistics")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"**Total Depth Range:**")
            st.metric("Min", f"{y_axis[0]:.2f} {st.session_state.depth_unit}")
            st.metric("Max", f"{y_axis[-1]:.2f} {st.session_state.depth_unit}")
        
        with col2:
            st.markdown(f"**Data Statistics:**")
            st.metric("Mean Amplitude", f"{st.session_state.processed_array.mean():.2e}")
            st.metric("Std Deviation", f"{st.session_state.processed_array.std():.2e}")
        
        with col3:
            st.markdown(f"**Defined Windows:**")
            for i, window in enumerate(st.session_state.windows):
                window_size = window['max'] - window['min']
                st.markdown(f"**W{i+1}:** {window['min']:.2f} - {window['max']:.2f} {st.session_state.depth_unit} ({window_size:.2f})")
    
    with tab3:
        st.subheader("Custom Depth Windows Analysis")
        
        # Window selection
        st.markdown("### Select Windows to Display")
        
        # Create checkboxes for each window
        selected_windows = []
        cols = st.columns(min(4, len(st.session_state.windows)))
        
        for i, window in enumerate(st.session_state.windows):
            with cols[i % 4]:
                if st.checkbox(f"Window {i+1}", True, key=f"show_window_{i}"):
                    selected_windows.append(i)
        
        # Display options for windows
        if selected_windows:
            col1, col2 = st.columns(2)
            
            with col1:
                window_colormap = st.selectbox("Window Colormap", 
                                              ["seismic", "RdBu", "gray", "viridis", "jet", "coolwarm"],
                                              key="window_cmap")
                window_interp = st.selectbox("Interpolation", 
                                            ["none", "bilinear", "bicubic", "gaussian"],
                                            key="window_interp")
            
            with col2:
                show_window_colorbar = st.checkbox("Show Colorbars", True, key="window_cbar")
                normalize_windows = st.checkbox("Normalize Each Window", False,
                                               help="Normalize color scale for each window independently")
            
            # Determine layout
            n_windows = len(selected_windows)
            if n_windows <= 2:
                n_rows, n_cols = 1, n_windows
                fig_size = (6 * n_cols, 8)
            elif n_windows <= 4:
                n_rows, n_cols = 2, 2
                fig_size = (12, 16)
            else:
                n_rows, n_cols = (n_windows + 1) // 2, 2
                fig_size = (12, 4 * n_rows)
            
            # Create figure for windows
            fig_windows, axes = plt.subplots(n_rows, n_cols, figsize=fig_size)
            
            if n_windows == 1:
                axes = np.array([axes])
            if n_rows == 1 and n_cols > 1:
                axes = axes.flatten()
            elif n_rows > 1 and n_cols > 1:
                axes = axes.flatten()
            
            # Plot each selected window
            for idx, window_idx in enumerate(selected_windows):
                if idx < len(axes):
                    window = st.session_state.windows[window_idx]
                    
                    # Get indices for this window
                    start_idx, end_idx = get_depth_window_indices(y_axis, window['min'], window['max'])
                    
                    # Extract window data
                    window_data = st.session_state.processed_array[start_idx:end_idx+1, :]
                    window_y_axis = y_axis[start_idx:end_idx+1]
                    
                    # Determine color scale
                    if normalize_windows:
                        vmax_window = np.percentile(np.abs(window_data), 99)
                        vmin_window = -vmax_window
                    else:
                        vmax_window = np.percentile(np.abs(st.session_state.processed_array), 99)
                        vmin_window = -vmax_window
                    
                    # Plot window
                    im = axes[idx].imshow(window_data,
                                         extent=[x_axis[0], x_axis[-1], window_y_axis[-1], window_y_axis[0]],
                                         aspect='auto', cmap=window_colormap,
                                         vmin=vmin_window, vmax=vmax_window,
                                         interpolation=window_interp)
                    
                    axes[idx].set_xlabel(x_label)
                    axes[idx].set_ylabel(y_label)
                    axes[idx].set_title(f"Window {window_idx+1}: {window['min']:.2f} - {window['max']:.2f} {st.session_state.depth_unit}")
                    axes[idx].grid(True, alpha=0.2, linestyle='--')
                    
                    if show_window_colorbar:
                        plt.colorbar(im, ax=axes[idx], label='Amplitude')
            
            # Hide unused subplots
            for idx in range(len(selected_windows), len(axes)):
                axes[idx].axis('off')
            
            plt.tight_layout()
            st.pyplot(fig_windows)
            
            # Window statistics
            st.subheader("Window Statistics")
            
            stats_data = []
            for window_idx in selected_windows:
                window = st.session_state.windows[window_idx]
                start_idx, end_idx = get_depth_window_indices(y_axis, window['min'], window['max'])
                window_data = st.session_state.processed_array[start_idx:end_idx+1, :]
                
                stats_data.append({
                    "Window": f"W{window_idx+1}",
                    "Depth Range": f"{window['min']:.2f} - {window['max']:.2f} {st.session_state.depth_unit}",
                    "Size": f"{window['max'] - window['min']:.2f} {st.session_state.depth_unit}",
                    "Min Amp": f"{window_data.min():.2e}",
                    "Max Amp": f"{window_data.max():.2e}",
                    "Mean Amp": f"{window_data.mean():.2e}",
                    "Std Dev": f"{window_data.std():.2e}"
                })
            
            stats_df = pd.DataFrame(stats_data)
            st.dataframe(stats_df, use_container_width=True)
            
            # Trace comparison across windows
            st.subheader("Trace Comparison Across Windows")
            
            trace_idx = st.slider("Select Trace for Comparison", 
                                 0, st.session_state.processed_array.shape[1]-1, 
                                 st.session_state.processed_array.shape[1]//2,
                                 key="window_trace")
            
            fig_trace_comparison, ax_trace = plt.subplots(figsize=(12, 8))
            
            colors = plt.cm.tab10(np.linspace(0, 1, len(selected_windows)))
            
            for window_idx, color in zip(selected_windows, colors):
                window = st.session_state.windows[window_idx]
                start_idx, end_idx = get_depth_window_indices(y_axis, window['min'], window['max'])
                
                window_y = y_axis[start_idx:end_idx+1]
                window_trace = st.session_state.processed_array[start_idx:end_idx+1, trace_idx]
                
                ax_trace.plot(window_trace, window_y, 
                             color=color, linewidth=2, alpha=0.8,
                             label=f"W{window_idx+1}: {window['min']:.2f}-{window['max']:.2f}")
                ax_trace.fill_betweenx(window_y, 0, window_trace, 
                                      alpha=0.2, color=color)
            
            ax_trace.set_xlabel("Amplitude")
            ax_trace.set_ylabel(f"Depth ({st.session_state.depth_unit})")
            ax_trace.set_title(f"Trace {trace_idx} Comparison Across Windows")
            ax_trace.grid(True, alpha=0.3)
            ax_trace.legend(loc='upper right')
            ax_trace.invert_yaxis()
            
            st.pyplot(fig_trace_comparison)
            
            # Export windows data
            st.subheader("Export Window Data")
            
            export_cols = st.columns(min(4, len(selected_windows)))
            
            for idx, window_idx in enumerate(selected_windows):
                with export_cols[idx % 4]:
                    window = st.session_state.windows[window_idx]
                    start_idx, end_idx = get_depth_window_indices(y_axis, window['min'], window['max'])
                    window_data = st.session_state.processed_array[start_idx:end_idx+1, :]
                    
                    # Create DataFrame
                    window_df = pd.DataFrame(window_data)
                    
                    # Download button
                    csv_string = window_df.to_csv(index=False)
                    
                    st.download_button(
                        label=f"📥 W{window_idx+1}",
                        data=csv_string,
                        file_name=f"window_{window_idx+1}_data.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
        
        else:
            st.info("No windows selected. Please select at least one window to display.")
    
    with tab4:
        st.subheader("Frequency vs Amplitude Analysis (FFT)")
        
        # FFT analysis options
        col1, col2, col3 = st.columns(3)
        
        with col1:
            fft_mode = st.selectbox("FFT Mode", 
                                   ["Single Trace", "Average of All Traces", "Trace Range", "Depth Window"],
                                   key="fft_mode")
            
            if fft_mode == "Depth Window":
                window_for_fft = st.selectbox("Select Window", 
                                             [f"Window {i+1}" for i in range(len(st.session_state.windows))],
                                             key="fft_window")
                window_idx = int(window_for_fft.split(" ")[1]) - 1
                window = st.session_state.windows[window_idx]
                start_idx, end_idx = get_depth_window_indices(y_axis, window['min'], window['max'])
        
        with col2:
            if fft_mode == "Single Trace":
                trace_for_fft = st.slider("Select Trace for FFT", 
                                         0, st.session_state.processed_array.shape[1]-1, 
                                         st.session_state.processed_array.shape[1]//2,
                                         key="fft_trace")
            elif fft_mode == "Trace Range":
                trace_start = st.number_input("Start Trace", 0, st.session_state.processed_array.shape[1]-1, 0,
                                             key="fft_start")
                trace_end = st.number_input("End Trace", 0, st.session_state.processed_array.shape[1]-1, 
                                           st.session_state.processed_array.shape[1]-1,
                                           key="fft_end")
        
        with col3:
            sampling_rate = st.number_input("Sampling Rate (MHz)", 100, 5000, 1000, 100,
                                           help="Antenna sampling rate in MHz",
                                           key="fft_sampling")
        
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
        
        elif fft_mode == "Depth Window":
            window_data = st.session_state.processed_array[start_idx:end_idx+1, :]
            avg_trace = np.mean(window_data, axis=1)
            freq, amplitude = calculate_fft(avg_trace, sampling_rate)
            title = f"FFT - Window: {window['min']:.2f}-{window['max']:.2f} {st.session_state.depth_unit}"
        
        # Plot FFT
        fig_fft, (ax_fft1, ax_fft2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Linear scale
        ax_fft1.plot(freq, amplitude, 'b-', linewidth=2, alpha=0.8)
        ax_fft1.fill_between(freq, 0, amplitude, alpha=0.3, color='blue')
        ax_fft1.set_xlabel("Frequency (MHz)")
        ax_fft1.set_ylabel("Amplitude")
        ax_fft1.set_title(f"{title} - Linear Scale")
        ax_fft1.grid(True, alpha=0.3)
        ax_fft1.set_xlim([0, sampling_rate/2])
        
        # Log scale
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
        st.subheader("FFT Statistics")
        
        # Find peak frequencies
        peak_idx = np.argmax(amplitude)
        peak_freq = freq[peak_idx]
        peak_amp = amplitude[peak_idx]
        
        # Calculate bandwidth at -3dB
        max_amp = np.max(amplitude)
        half_power = max_amp / np.sqrt(2)
        
        # Find frequencies where amplitude is above half power
        mask = amplitude >= half_power
        if np.any(mask):
            low_freq = freq[mask][0]
            high_freq = freq[mask][-1]
            bandwidth = high_freq - low_freq
        else:
            low_freq = high_freq = bandwidth = 0
        
        # Display statistics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Peak Frequency", f"{peak_freq:.1f} MHz")
        with col2:
            st.metric("Peak Amplitude", f"{peak_amp:.3e}")
        with col3:
            st.metric("Bandwidth (-3dB)", f"{bandwidth:.1f} MHz")
        with col4:
            st.metric("Center Freq", f"{(low_freq + high_freq)/2:.1f} MHz")
    
    with tab5:
        st.subheader("Export Processed Data")
        
        # Export options in columns
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("💾 Save Full Profile", use_container_width=True):
                fig, ax = plt.subplots(figsize=(12, 8))
                
                im = ax.imshow(st.session_state.processed_array,
                             extent=[x_axis[0], x_axis[-1], y_axis[-1], y_axis[0]],
                             aspect='auto', cmap='seismic')
                ax.set_xlabel(x_label)
                ax.set_ylabel(y_label)
                ax.set_title(f"GPR Full Profile - {gain_type} Gain")
                plt.colorbar(im, ax=ax, label='Amplitude')
                plt.tight_layout()
                plt.savefig("gpr_full_profile.png", dpi=300, bbox_inches='tight')
                st.success("Saved as 'gpr_full_profile.png'")
        
        with col2:
            # Export all windows
            if len(selected_windows) > 0:
                if st.button("💾 Save All Windows", use_container_width=True):
                    n_windows = len(selected_windows)
                    fig_w, axes_w = plt.subplots(n_windows, 1, figsize=(12, 4*n_windows))
                    
                    if n_windows == 1:
                        axes_w = [axes_w]
                    
                    for idx, window_idx in enumerate(selected_windows):
                        window = st.session_state.windows[window_idx]
                        start_idx, end_idx = get_depth_window_indices(y_axis, window['min'], window['max'])
                        window_data = st.session_state.processed_array[start_idx:end_idx+1, :]
                        window_y_axis = y_axis[start_idx:end_idx+1]
                        
                        im = axes_w[idx].imshow(window_data,
                                              extent=[x_axis[0], x_axis[-1], window_y_axis[-1], window_y_axis[0]],
                                              aspect='auto', cmap='seismic')
                        axes_w[idx].set_xlabel(x_label)
                        axes_w[idx].set_ylabel(y_label)
                        axes_w[idx].set_title(f"Window {window_idx+1}: {window['min']:.2f} - {window['max']:.2f} {st.session_state.depth_unit}")
                        plt.colorbar(im, ax=axes_w[idx], label='Amplitude')
                    
                    plt.tight_layout()
                    plt.savefig("gpr_windows.png", dpi=300, bbox_inches='tight')
                    st.success("Saved as 'gpr_windows.png'")
        
        with col3:
            # Export as CSV with scaled axes
            csv_data = pd.DataFrame(st.session_state.processed_array, 
                                  columns=[f"{xi:.2f}" for xi in x_axis])
            csv_string = csv_data.to_csv(index=False)
            
            st.download_button(
                label="📥 Full Data CSV",
                data=csv_string,
                file_name="gpr_full_data.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with col4:
            # Export settings
            settings = {
                "Gain Type": gain_type,
                "Time Zero": time_zero,
                "Depth Unit": st.session_state.depth_unit,
                "Max Depth": st.session_state.max_depth if hasattr(st.session_state, 'max_depth') else "Auto",
                "Distance Unit": st.session_state.distance_unit,
                "Total Distance": st.session_state.total_distance if hasattr(st.session_state, 'total_distance') else "Auto",
                "Windows": str(st.session_state.windows),
                "Date Processed": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            settings_df = pd.DataFrame(list(settings.items()), columns=["Parameter", "Value"])
            settings_csv = settings_df.to_csv(index=False)
            
            st.download_button(
                label="📥 Settings CSV",
                data=settings_csv,
                file_name="processing_settings.csv",
                mime="text/csv",
                use_container_width=True
            )

# Initial state message
elif not dzt_file:
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.info("""
        👈 **Upload a DZT file to begin processing**
        
        **New Features:**
        1. **Custom Depth Windows:** Zoom into specific depth ranges
        2. **Multiple Window Display:** View multiple windows side-by-side
        3. **Window Statistics:** Compare statistics across windows
        4. **Window FFT:** Perform FFT on specific depth windows
        
        **Example for 0-12m data:**
        - Window 1: 0-5m (shallow features)
        - Window 2: 5-9m (mid-depth features)
        - Window 3: 9-12m (deep features)
        
        **No GPS file needed!** Manual scaling available.
        """)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666;'>"
    "📡 <b>GPR Data Processor v4.0</b> | Custom Depth Windows | "
    "Built with Streamlit & readgssi"
    "</div>",
    unsafe_allow_html=True
)

