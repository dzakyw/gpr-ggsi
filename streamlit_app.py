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
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="GPR Data Processor",
    page_icon="📡",
    layout="wide"
)

# Title
st.title("📡 GPR Data Processor with Custom Windowing")
st.markdown("Process GPR data with customizable depth/distance windows and zoom capabilities")

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
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'original_array' not in st.session_state:
    st.session_state.original_array = None
if 'processed_array' not in st.session_state:
    st.session_state.processed_array = None

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
    st.header("🔍 Plot Windowing")
    
    use_custom_window = st.checkbox("Use Custom Plot Window", False,
                                   help="Define custom depth and distance ranges for plotting")
    
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
        if distance_unit != "traces":
            distance_min = st.number_input(f"Min Distance ({distance_unit})", 0.0, total_distance, 0.0, 0.1)
            distance_max = st.number_input(f"Max Distance ({distance_unit})", 0.0, total_distance, total_distance, 0.1)
        else:
            distance_min = st.number_input("Min Distance (traces)", 0, 10000, 0)
            distance_max = st.number_input("Max Distance (traces)", 0, 10000, 800)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Multiple windows option
    multiple_windows = st.checkbox("Enable Multiple Windows", False,
                                  help="Plot multiple windows in the same view")
    
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
                dist_min = st.number_input(f"Dist Min {i+2} ({distance_unit})", 0.0, total_distance, 50.0 + i*50, 0.1)
                dist_max = st.number_input(f"Dist Max {i+2} ({distance_unit})", 0.0, total_distance, 150.0 + i*50, 0.1)
            
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

def get_window_indices(x_axis, y_axis, depth_min, depth_max, distance_min, distance_max):
    """Convert user-specified window coordinates to array indices"""
    # Find depth indices
    depth_idx_min = np.argmin(np.abs(y_axis - depth_min))
    depth_idx_max = np.argmin(np.abs(y_axis - depth_max))
    
    # Ensure correct ordering
    if depth_idx_min > depth_idx_max:
        depth_idx_min, depth_idx_max = depth_idx_max, depth_idx_min
    
    # Find distance indices
    dist_idx_min = np.argmin(np.abs(x_axis - distance_min))
    dist_idx_max = np.argmin(np.abs(x_axis - distance_max))
    
    # Ensure correct ordering
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
                    st.session_state.max_depth = max_depth if depth_unit != "samples" else None
                    st.session_state.distance_unit = distance_unit
                    st.session_state.total_distance = total_distance if distance_unit != "traces" else None
                    
                    # Store window parameters
                    st.session_state.use_custom_window = use_custom_window
                    if use_custom_window:
                        st.session_state.depth_min = depth_min if 'depth_min' in locals() else 0
                        st.session_state.depth_max = depth_max if 'depth_max' in locals() else max_depth
                        st.session_state.distance_min = distance_min if 'distance_min' in locals() else 0
                        st.session_state.distance_max = distance_max if 'distance_max' in locals() else total_distance
                    
                    st.session_state.multiple_windows = multiple_windows
                    if multiple_windows and use_custom_window:
                        st.session_state.additional_windows = windows if 'windows' in locals() else []
                    
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
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["📊 Header Info", "📈 Full View", "🔍 Custom Window", "📉 FFT Analysis", "🎛️ Gain Analysis", "💾 Export"])
    
    with tab1:
        st.subheader("File Information & Settings")
        
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
            
            if st.session_state.use_custom_window:
                st.markdown("### Custom Window Settings")
                window_data = {
                    "Depth Window": f"{st.session_state.depth_min:.1f} - {st.session_state.depth_max:.1f} {st.session_state.depth_unit}",
                    "Distance Window": f"{st.session_state.distance_min:.1f} - {st.session_state.distance_max:.1f} {st.session_state.distance_unit}"
                }
                
                for key, value in window_data.items():
                    st.markdown(f"**{key}:** {value}")
        
        with col2:
            if st.session_state.header:
                st.markdown("### File Header")
                info_data = {
                    "System": st.session_state.header.get('system', 'Unknown'),
                    "Antenna Frequency": f"{st.session_state.header.get('ant_freq', 'N/A')} MHz",
                    "Samples per Trace": st.session_state.header.get('spt', 'N/A'),
                    "Number of Traces": st.session_state.header.get('ntraces', 'N/A'),
                    "Sampling Depth": f"{st.session_state.header.get('depth', 'N/A'):.2f} m"
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
    
    with tab2:
        st.subheader("Full Radar Profile")
        
        # Create scaled axes for full view
        x_axis_full, y_axis_full, x_label_full, y_label_full = scale_axes(
            st.session_state.processed_array.shape,
            st.session_state.depth_unit,
            st.session_state.max_depth if hasattr(st.session_state, 'max_depth') else None,
            st.session_state.distance_unit,
            st.session_state.total_distance if hasattr(st.session_state, 'total_distance') else None
        )
        
        # Display options
        col1, col2, col3 = st.columns(3)
        
        with col1:
            show_colorbar = st.checkbox("Show Colorbar", True, key="full_cbar")
            interpolation = st.selectbox("Interpolation", ["none", "bilinear", "bicubic", "gaussian"], key="full_interp")
        
        with col2:
            colormap = st.selectbox("Colormap", ["seismic", "RdBu", "gray", "viridis", "jet"], key="full_cmap")
            aspect_ratio = st.selectbox("Aspect Ratio", ["auto", "equal", 0.5, 1.0, 2.0], key="full_aspect")
        
        with col3:
            vmin = st.number_input("Color Min", -1.0, 0.0, -0.5, 0.01, key="full_vmin")
            vmax = st.number_input("Color Max", 0.0, 1.0, 0.5, 0.01, key="full_vmax")
            normalize_colors = st.checkbox("Auto-normalize Colors", True, key="full_norm")
        
        # Create figure
        fig_full, (ax1_full, ax2_full) = plt.subplots(1, 2, figsize=(18, 8))
        
        # Plot original full view
        if normalize_colors:
            vmax_plot = np.percentile(np.abs(st.session_state.original_array), 99)
            vmin_plot = -vmax_plot
        else:
            vmin_plot, vmax_plot = vmin, vmax
        
        im1 = ax1_full.imshow(st.session_state.original_array, 
                             extent=[x_axis_full[0], x_axis_full[-1], y_axis_full[-1], y_axis_full[0]],
                             aspect=aspect_ratio, cmap=colormap, 
                             vmin=vmin_plot, vmax=vmax_plot,
                             interpolation=interpolation)
        
        ax1_full.set_xlabel(x_label_full)
        ax1_full.set_ylabel(y_label_full)
        ax1_full.set_title("Original Data - Full View")
        ax1_full.grid(True, alpha=0.3, linestyle='--')
        
        if show_colorbar:
            plt.colorbar(im1, ax=ax1_full, label='Amplitude')
        
        # Plot processed full view
        im2 = ax2_full.imshow(st.session_state.processed_array,
                             extent=[x_axis_full[0], x_axis_full[-1], y_axis_full[-1], y_axis_full[0]],
                             aspect=aspect_ratio, cmap=colormap,
                             vmin=vmin_plot, vmax=vmax_plot,
                             interpolation=interpolation)
        
        ax2_full.set_xlabel(x_label_full)
        ax2_full.set_ylabel(y_label_full)
        ax2_full.set_title(f"Processed ({gain_type} Gain) - Full View")
        ax2_full.grid(True, alpha=0.3, linestyle='--')
        
        if show_colorbar:
            plt.colorbar(im2, ax=ax2_full, label='Amplitude')
        
        plt.tight_layout()
        st.pyplot(fig_full)
        
        # Add window overlay if custom window is enabled
        if st.session_state.use_custom_window:
            # Get window indices
            window_info = get_window_indices(
                x_axis_full, y_axis_full,
                st.session_state.depth_min, st.session_state.depth_max,
                st.session_state.distance_min, st.session_state.distance_max
            )
            
            # Add rectangle to show selected window
            rect = plt.Rectangle((window_info['dist_min_val'], window_info['depth_min_val']),
                               window_info['dist_max_val'] - window_info['dist_min_val'],
                               window_info['depth_max_val'] - window_info['depth_min_val'],
                               linewidth=2, edgecolor='yellow', facecolor='none', alpha=0.8)
            
            ax1_full.add_patch(rect.copy())
            ax2_full.add_patch(rect.copy())
            
            # Add text label
            ax1_full.text(window_info['dist_min_val'], window_info['depth_min_val'] - 0.1,
                         f"Selected Window", color='yellow', fontsize=10,
                         bbox=dict(boxstyle="round,pad=0.3", facecolor="black", alpha=0.7))
            
            st.pyplot(fig_full)
    
    with tab3:
        st.subheader("Custom Window Analysis")
        
        if not st.session_state.use_custom_window:
            st.warning("⚠️ Enable 'Use Custom Plot Window' in the sidebar to use this feature.")
        else:
            # Create scaled axes
            x_axis, y_axis, x_label, y_label = scale_axes(
                st.session_state.processed_array.shape,
                st.session_state.depth_unit,
                st.session_state.max_depth if hasattr(st.session_state, 'max_depth') else None,
                st.session_state.distance_unit,
                st.session_state.total_distance if hasattr(st.session_state, 'total_distance') else None
            )
            
            # Get window indices
            window_info = get_window_indices(
                x_axis, y_axis,
                st.session_state.depth_min, st.session_state.depth_max,
                st.session_state.distance_min, st.session_state.distance_max
            )
            
            # Extract windowed data
            window_data = st.session_state.processed_array[
                window_info['depth_min_idx']:window_info['depth_max_idx'],
                window_info['dist_min_idx']:window_info['dist_max_idx']
            ]
            
            window_data_original = st.session_state.original_array[
                window_info['depth_min_idx']:window_info['depth_max_idx'],
                window_info['dist_min_idx']:window_info['dist_max_idx']
            ]
            
            # Create windowed axes
            x_axis_window = x_axis[window_info['dist_min_idx']:window_info['dist_max_idx']]
            y_axis_window = y_axis[window_info['depth_min_idx']:window_info['depth_max_idx']]
            
            # Display window statistics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Window Depth Range", 
                         f"{window_info['depth_min_val']:.1f} - {window_info['depth_max_val']:.1f} {st.session_state.depth_unit}")
            
            with col2:
                st.metric("Window Distance Range", 
                         f"{window_info['dist_min_val']:.1f} - {window_info['dist_max_val']:.1f} {st.session_state.distance_unit}")
            
            with col3:
                st.metric("Window Size (samples×traces)", 
                         f"{window_data.shape[0]} × {window_data.shape[1]}")
            
            with col4:
                st.metric("Data Points", 
                         f"{window_data.size:,}")
            
            # Plot windowed data
            fig_window, (ax1_window, ax2_window) = plt.subplots(1, 2, figsize=(16, 6))
            
            # Windowed original
            im1_window = ax1_window.imshow(window_data_original,
                                          extent=[x_axis_window[0], x_axis_window[-1], 
                                                  y_axis_window[-1], y_axis_window[0]],
                                          aspect='auto', cmap='seismic')
            
            ax1_window.set_xlabel(x_label)
            ax1_window.set_ylabel(y_label)
            ax1_window.set_title(f"Original Data - Custom Window\n"
                               f"Depth: {window_info['depth_min_val']:.1f}-{window_info['depth_max_val']:.1f} {st.session_state.depth_unit}\n"
                               f"Distance: {window_info['dist_min_val']:.1f}-{window_info['dist_max_val']:.1f} {st.session_state.distance_unit}")
            ax1_window.grid(True, alpha=0.3)
            plt.colorbar(im1_window, ax=ax1_window, label='Amplitude')
            
            # Windowed processed
            im2_window = ax2_window.imshow(window_data,
                                          extent=[x_axis_window[0], x_axis_window[-1], 
                                                  y_axis_window[-1], y_axis_window[0]],
                                          aspect='auto', cmap='seismic')
            
            ax2_window.set_xlabel(x_label)
            ax2_window.set_ylabel(y_label)
            ax2_window.set_title(f"Processed Data - Custom Window\n"
                               f"Depth: {window_info['depth_min_val']:.1f}-{window_info['depth_max_val']:.1f} {st.session_state.depth_unit}\n"
                               f"Distance: {window_info['dist_min_val']:.1f}-{window_info['dist_max_val']:.1f} {st.session_state.distance_unit}")
            ax2_window.grid(True, alpha=0.3)
            plt.colorbar(im2_window, ax=ax2_window, label='Amplitude')
            
            plt.tight_layout()
            st.pyplot(fig_window)
            
            # Multiple windows view
            if st.session_state.multiple_windows and hasattr(st.session_state, 'additional_windows'):
                st.subheader("Multiple Windows View")
                
                # Create figure with subplots
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
                             aspect='auto', cmap='seismic')
                
                ax.set_xlabel(x_label)
                ax.set_ylabel(y_label)
                ax.set_title(f"Window 1\n{window_info['depth_min_val']:.1f}-{window_info['depth_max_val']:.1f} {st.session_state.depth_unit}")
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
                        
                        if window_idx == 0:  # Skip first (already plotted)
                            continue
                        
                        ax = axes[i, j]
                        win = st.session_state.additional_windows[window_idx-1]
                        
                        # Get indices for this window
                        win_info = get_window_indices(
                            x_axis, y_axis,
                            win['depth_min'], win['depth_max'],
                            win['distance_min'], win['distance_max']
                        )
                        
                        # Extract window data
                        win_data = st.session_state.processed_array[
                            win_info['depth_min_idx']:win_info['depth_max_idx'],
                            win_info['dist_min_idx']:win_info['dist_max_idx']
                        ]
                        
                        # Create windowed axes
                        x_axis_win = x_axis[win_info['dist_min_idx']:win_info['dist_max_idx']]
                        y_axis_win = y_axis[win_info['depth_min_idx']:win_info['depth_max_idx']]
                        
                        # Plot
                        im = ax.imshow(win_data,
                                     extent=[x_axis_win[0], x_axis_win[-1], 
                                             y_axis_win[-1], y_axis_win[0]],
                                     aspect='auto', cmap='seismic')
                        
                        ax.set_xlabel(x_label)
                        ax.set_ylabel(y_label)
                        ax.set_title(f"Window {window_idx+1}\n{win_info['depth_min_val']:.1f}-{win_info['depth_max_val']:.1f} {st.session_state.depth_unit}")
                        ax.grid(True, alpha=0.3)
                        plt.colorbar(im, ax=ax, label='Amplitude')
                        
                        window_idx += 1
                
                plt.tight_layout()
                st.pyplot(fig_multi)
            
            # Windowed trace analysis
            st.subheader("Windowed Trace Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Select trace within window
                trace_in_window = st.slider(
                    "Select Trace in Window", 
                    0, window_data.shape[1]-1,
                    window_data.shape[1]//2,
                    key="window_trace"
                )
                
                # Get the actual trace index
                actual_trace_idx = window_info['dist_min_idx'] + trace_in_window
                
                # Plot trace
                fig_trace, ax_trace = plt.subplots(figsize=(10, 6))
                
                ax_trace.plot(y_axis_window, window_data[:, trace_in_window], 
                             'b-', linewidth=1.5, alpha=0.8)
                ax_trace.fill_between(y_axis_window, 0, window_data[:, trace_in_window], 
                                     alpha=0.3, color='blue')
                ax_trace.set_xlabel(y_label)
                ax_trace.set_ylabel("Amplitude")
                ax_trace.set_title(f"Trace {actual_trace_idx} in Window\n"
                                 f"Distance: {x_axis_window[trace_in_window]:.1f} {st.session_state.distance_unit}")
                ax_trace.grid(True, alpha=0.3)
                ax_trace.invert_xaxis()
                
                st.pyplot(fig_trace)
            
            with col2:
                # Select depth slice within window
                depth_slice_in_window = st.slider(
                    "Select Depth Slice in Window", 
                    0, window_data.shape[0]-1,
                    window_data.shape[0]//2,
                    key="window_depth"
                )
                
                # Get actual depth value
                actual_depth = y_axis_window[depth_slice_in_window]
                
                # Plot depth slice
                fig_slice, ax_slice = plt.subplots(figsize=(10, 6))
                
                ax_slice.plot(x_axis_window, window_data[depth_slice_in_window, :], 
                             'r-', linewidth=1.5, alpha=0.8)
                ax_slice.fill_between(x_axis_window, 0, window_data[depth_slice_in_window, :], 
                                     alpha=0.3, color='red')
                ax_slice.set_xlabel(x_label)
                ax_slice.set_ylabel("Amplitude")
                ax_slice.set_title(f"Depth Slice at {actual_depth:.2f} {st.session_state.depth_unit}")
                ax_slice.grid(True, alpha=0.3)
                
                st.pyplot(fig_slice)
            
            # Window statistics
            st.subheader("Window Statistics")
            
            stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
            
            with stat_col1:
                st.metric("Mean Amplitude", f"{window_data.mean():.3e}")
                st.metric("Std Deviation", f"{window_data.std():.3e}")
            
            with stat_col2:
                st.metric("Min Amplitude", f"{window_data.min():.3e}")
                st.metric("Max Amplitude", f"{window_data.max():.3e}")
            
            with stat_col3:
                st.metric("Depth Resolution", 
                         f"{(y_axis_window[1] - y_axis_window[0]):.3f} {st.session_state.depth_unit}/sample")
                st.metric("Distance Resolution", 
                         f"{(x_axis_window[1] - x_axis_window[0]):.3f} {st.session_state.distance_unit}/trace")
            
            with stat_col4:
                st.metric("Window Area", 
                         f"{(window_info['depth_max_val'] - window_info['depth_min_val']) * (window_info['dist_max_val'] - window_info['dist_min_val']):.1f} {st.session_state.depth_unit}×{st.session_state.distance_unit}")
                st.metric("Data Density", 
                         f"{window_data.size / ((window_info['depth_max_val'] - window_info['depth_min_val']) * (window_info['dist_max_val'] - window_info['dist_min_val'])):.1f} points/unit²")
    
    with tab4:
        st.subheader("Frequency vs Amplitude Analysis (FFT)")
        
        # FFT analysis options
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
            fft_mode = st.selectbox("FFT Mode", ["Single Trace", "Average of All Traces", "Trace Range", "Windowed Traces"],
                                   key="fft_mode")
        
        if fft_mode == "Trace Range":
            trace_start = st.number_input("Start Trace", 0, st.session_state.processed_array.shape[1]-1, 0,
                                         key="fft_start")
            trace_end = st.number_input("End Trace", 0, st.session_state.processed_array.shape[1]-1, 
                                       st.session_state.processed_array.shape[1]-1,
                                       key="fft_end")
        
        if fft_mode == "Windowed Traces" and st.session_state.use_custom_window:
            # Use windowed traces for FFT
            st.info(f"Using traces from custom window: {window_info['dist_min_idx']} to {window_info['dist_max_idx']}")
        
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
        
        elif fft_mode == "Windowed Traces" and st.session_state.use_custom_window:
            # Use windowed traces
            windowed_traces = st.session_state.processed_array[
                :, window_info['dist_min_idx']:window_info['dist_max_idx']
            ]
            avg_trace = np.mean(windowed_traces, axis=1)
            freq, amplitude = calculate_fft(avg_trace, sampling_rate)
            title = f"FFT - Windowed Traces ({window_info['dist_min_idx']} to {window_info['dist_max_idx']})"
        
        else:
            st.warning("Please select a valid FFT mode")
            freq, amplitude = [], []
            title = ""
        
        if len(freq) > 0:
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
        st.subheader("Gain Analysis")
        
        # Calculate gain profile
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
        
        # Create scaled depth axis
        y_axis_analysis, _, _, y_label_analysis = scale_axes(
            (n_samples, 1),
            st.session_state.depth_unit,
            st.session_state.max_depth if hasattr(st.session_state, 'max_depth') else None,
            "traces",
            None
        )
        
        # Plot gain profile
        fig_gain, ax_gain = plt.subplots(figsize=(10, 6))
        
        ax_gain.plot(gain_profile, y_axis_analysis, 'b-', linewidth=2, label='Gain Factor')
        ax_gain.fill_betweenx(y_axis_analysis, 1, gain_profile, alpha=0.3, color='blue')
        
        ax_gain.set_xlabel("Gain Factor (multiplier)")
        ax_gain.set_ylabel(y_label_analysis)
        ax_gain.set_title("Gain Applied vs Depth")
        ax_gain.grid(True, alpha=0.3)
        ax_gain.legend()
        ax_gain.invert_yaxis()  # Depth increases downward
        
        st.pyplot(fig_gain)
        
        # Show statistics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Min Gain", f"{gain_profile.min():.2f}x")
        with col2:
            st.metric("Max Gain", f"{gain_profile.max():.2f}x")
        with col3:
            st.metric("Mean Gain", f"{gain_profile.mean():.2f}x")
    
    with tab6:
        st.subheader("Export Processed Data")
        
        # Export options in columns
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("💾 Save Full Radar Image", use_container_width=True):
                fig, ax = plt.subplots(figsize=(12, 8))
                
                x_axis_export, y_axis_export, x_label_export, y_label_export = scale_axes(
                    st.session_state.processed_array.shape,
                    st.session_state.depth_unit,
                    st.session_state.max_depth if hasattr(st.session_state, 'max_depth') else None,
                    st.session_state.distance_unit,
                    st.session_state.total_distance if hasattr(st.session_state, 'total_distance') else None
                )
                
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
                if st.button("💾 Save Windowed Image", use_container_width=True):
                    fig, ax = plt.subplots(figsize=(10, 6))
                    
                    # Get window data
                    x_axis, y_axis, x_label, y_label = scale_axes(
                        st.session_state.processed_array.shape,
                        st.session_state.depth_unit,
                        st.session_state.max_depth if hasattr(st.session_state, 'max_depth') else None,
                        st.session_state.distance_unit,
                        st.session_state.total_distance if hasattr(st.session_state, 'total_distance') else None
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
                    
                    im = ax.imshow(window_data,
                                 extent=[x_axis_window[0], x_axis_window[-1], 
                                         y_axis_window[-1], y_axis_window[0]],
                                 aspect='auto', cmap='seismic')
                    ax.set_xlabel(x_label)
                    ax.set_ylabel(y_label)
                    ax.set_title(f"GPR Data - Custom Window\n"
                               f"Depth: {window_info['depth_min_val']:.1f}-{window_info['depth_max_val']:.1f} {st.session_state.depth_unit}\n"
                               f"Distance: {window_info['dist_min_val']:.1f}-{window_info['dist_max_val']:.1f} {st.session_state.distance_unit}")
                    plt.colorbar(im, ax=ax, label='Amplitude')
                    plt.tight_layout()
                    plt.savefig("gpr_data_windowed.png", dpi=300, bbox_inches='tight')
                    st.success("Saved as 'gpr_data_windowed.png'")
        
        with col3:
            # Export as CSV with scaled axes
            x_axis_csv = scale_axes(
                st.session_state.processed_array.shape,
                st.session_state.depth_unit,
                st.session_state.max_depth if hasattr(st.session_state, 'max_depth') else None,
                st.session_state.distance_unit,
                st.session_state.total_distance if hasattr(st.session_state, 'total_distance') else None
            )[0]
            
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
            if st.session_state.use_custom_window:
                # Export windowed data
                x_axis, y_axis, x_label, y_label = scale_axes(
                    st.session_state.processed_array.shape,
                    st.session_state.depth_unit,
                    st.session_state.max_depth if hasattr(st.session_state, 'max_depth') else None,
                    st.session_state.distance_unit,
                    st.session_state.total_distance if hasattr(st.session_state, 'total_distance') else None
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
                
                window_csv = pd.DataFrame(window_data, 
                                        columns=[f"{xi:.2f}" for xi in x_axis_window])
                window_csv_string = window_csv.to_csv(index=False)
                
                st.download_button(
                    label="📥 Download Window CSV",
                    data=window_csv_string,
                    file_name="gpr_data_window.csv",
                    mime="text/csv",
                    use_container_width=True
                )

# Initial state message
elif not dzt_file:
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.info("""
        👈 **Upload a DZT file to begin processing**
        
        **New Windowing Features:**
        1. **Custom Depth Windows:** Plot specific depth ranges (e.g., 0-5 meters)
        2. **Custom Distance Windows:** Plot specific distance ranges
        3. **Multiple Windows:** Compare different areas simultaneously
        4. **Window Statistics:** Get detailed info for selected regions
        
        **Example Usage:**
        - Full data: 0-12m depth, 0-250m distance
        - Window 1: 0-5m depth, 50-150m distance
        - Window 2: 5-9m depth, 100-200m distance
        
        **No GPS file needed!** Manual scaling and windowing available.
        """)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666;'>"
    "📡 <b>GPR Data Processor v4.0</b> | Custom Windowing & Multi-View Analysis | "
    "Built with Streamlit & readgssi"
    "</div>",
    unsafe_allow_html=True
)
