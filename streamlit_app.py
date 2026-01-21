import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tempfile
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="GPR Data Processor",
    page_icon="📡",
    layout="wide"
)

# Title
st.title("📡 GSSI Radar Data Processor with Time Gain Control")
st.markdown("Process GPR data with automatic gain normalization for deep signals")

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
    dzg_file = st.file_uploader("Upload DZG file (GPS data)", type=['dzg', 'DZG'], help="Optional: Required for distance normalization")
    
    st.markdown("---")
    st.header("🎛️ Basic Parameters")
    
    time_zero = st.number_input("Time Zero (samples)", 0, 2000, 2, 
                               help="Adjust the start time of each trace")
    
    stacking = st.selectbox("Stacking", ["none", "auto", "manual"], 
                           help="Reduce noise by averaging traces")
    
    if stacking == "manual":
        stack_value = st.number_input("Stack Value", 1, 50, 3)
    
    st.markdown("---")
    st.header("📈 Time Gain Control")
    
    # Time-varying gain parameters
    st.markdown("**Amplify deep signals that are weak:**")
    
    gain_type = st.selectbox(
        "Gain Type",
        ["Constant", "Linear", "Exponential", "AGC (Automatic Gain Control)"],
        help="Apply gain that increases with time to amplify deep signals"
    )
    
    if gain_type == "Constant":
        const_gain = st.slider("Gain (%)", 0, 500, 100, 
                              help="Apply constant gain to entire profile")
    
    elif gain_type == "Linear":
        min_gain = st.slider("Gain at Top (%)", 0, 200, 50, 
                            help="Gain at time zero (shallow signals)")
        max_gain = st.slider("Gain at Bottom (%)", 0, 1000, 500, 
                            help="Gain at max time (deep signals)")
    
    elif gain_type == "Exponential":
        base_gain = st.slider("Base Gain (%)", 0, 300, 100, 
                             help="Starting gain at time zero")
        exp_factor = st.slider("Exponential Factor", 0.1, 5.0, 1.5, 0.1,
                              help="How quickly gain increases with time")
    
    elif gain_type == "AGC (Automatic Gain Control)":
        window_size = st.slider("AGC Window (samples)", 10, 500, 100, 
                               help="Window size for automatic gain calculation")
        target_amplitude = st.slider("Target Amplitude", 0.1, 1.0, 0.3, 0.05,
                                    help="Target amplitude level for normalization")
    
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
    
    normalize_distance = st.checkbox("Distance Normalization", False,
                                    help="Requires DZG file with GPS data")
    
    process_btn = st.button("🚀 Process Data", type="primary", use_container_width=True)

# Main content
if dzt_file and process_btn:
    with st.spinner("Processing radar data..."):
        try:
            # Try to import readgssi
            try:
                from readgssi import readgssi
                import readgssi.arrayops as arrayops
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
                    
                    # Function to apply different gain types
                    def apply_gain(array, gain_type, **kwargs):
                        """Apply time-varying gain to radar data"""
                        n_samples, n_traces = array.shape
                        
                        if gain_type == "Constant":
                            gain = 1 + kwargs.get('const_gain', 1.0) / 100
                            return array * gain
                        
                        elif gain_type == "Linear":
                            min_g = 1 + kwargs.get('min_gain', 0.5) / 100
                            max_g = 1 + kwargs.get('max_gain', 5.0) / 100
                            # Create linear gain vector
                            gain_vector = np.linspace(min_g, max_g, n_samples)
                            return array * gain_vector[:, np.newaxis]
                        
                        elif gain_type == "Exponential":
                            base_g = 1 + kwargs.get('base_gain', 1.0) / 100
                            exp_f = kwargs.get('exp_factor', 1.5)
                            # Create exponential gain vector
                            t = np.linspace(0, 1, n_samples)
                            gain_vector = base_g * np.exp(exp_f * t)
                            return array * gain_vector[:, np.newaxis]
                        
                        elif gain_type == "AGC (Automatic Gain Control)":
                            window = kwargs.get('window_size', 100)
                            target = kwargs.get('target_amplitude', 0.3)
                            
                            # Apply AGC
                            result = np.zeros_like(array)
                            half_window = window // 2
                            
                            for i in range(n_traces):
                                trace = array[:, i]
                                agc_trace = np.zeros(n_samples)
                                
                                for j in range(n_samples):
                                    # Window boundaries
                                    start = max(0, j - half_window)
                                    end = min(n_samples, j + half_window + 1)
                                    
                                    # Calculate RMS in window
                                    window_data = trace[start:end]
                                    rms = np.sqrt(np.mean(window_data**2))
                                    
                                    # Avoid division by zero
                                    if rms > 0:
                                        agc_trace[j] = trace[j] * (target / rms)
                                    else:
                                        agc_trace[j] = trace[j]
                                
                                result[:, i] = agc_trace
                            
                            return result
                        
                        return array
                    
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
                    
                    progress_bar.progress(90)
                    
                    # Store in session state
                    st.session_state.header = header
                    st.session_state.original_array = original_array
                    st.session_state.processed_array = processed_array
                    st.session_state.gps = gps
                    st.session_state.data_loaded = True
                    
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
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Header Info", "📈 Compare Views", "🎛️ Gain Analysis", "💾 Export"])
    
    with tab1:
        st.subheader("File Information")
        
        if st.session_state.header:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("System", st.session_state.header.get('system', 'Unknown'))
                st.metric("Antenna Frequency", f"{st.session_state.header.get('ant_freq', 'N/A')} MHz")
                st.metric("Samples per Trace", st.session_state.header.get('spt', 'N/A'))
            
            with col2:
                st.metric("Traces", st.session_state.header.get('ntraces', 'N/A'))
                st.metric("Bits per Sample", st.session_state.header.get('bps', 'N/A'))
                st.metric("Sampling Depth", f"{st.session_state.header.get('depth', 'N/A'):.2f} m")
            
            with col3:
                st.metric("Dielectric Constant", st.session_state.header.get('epsr', 'N/A'))
                st.metric("Traces per Second", st.session_state.header.get('tps', 'N/A'))
                st.metric("GPS Enabled", "Yes" if st.session_state.gps is not None else "No")
            
            # Array statistics
            st.subheader("Data Statistics")
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Original Min Amplitude", f"{st.session_state.original_array.min():.2e}")
                st.metric("Original Max Amplitude", f"{st.session_state.original_array.max():.2e}")
                st.metric("Original Mean", f"{st.session_state.original_array.mean():.2e}")
            
            with col2:
                st.metric("Processed Min Amplitude", f"{st.session_state.processed_array.min():.2e}")
                st.metric("Processed Max Amplitude", f"{st.session_state.processed_array.max():.2e}")
                st.metric("Processed Mean", f"{st.session_state.processed_array.mean():.2e}")
    
    with tab2:
        st.subheader("Original vs Processed Data")
        
        # Display side by side
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Original Data (No Gain)**")
            
            # Create figure for original data
            fig1, ax1 = plt.subplots(figsize=(10, 6))
            
            # Normalize for display
            vmax = np.percentile(np.abs(st.session_state.original_array), 99)
            im1 = ax1.imshow(st.session_state.original_array, aspect='auto', 
                            cmap='seismic', vmin=-vmax, vmax=vmax)
            
            ax1.set_xlabel("Trace Number")
            ax1.set_ylabel("Sample (Time)")
            ax1.set_title("Original Radar Data")
            plt.colorbar(im1, ax=ax1, label='Amplitude')
            
            st.pyplot(fig1)
            
            # Show amplitude histogram
            fig1h, ax1h = plt.subplots(figsize=(8, 3))
            ax1h.hist(st.session_state.original_array.flatten(), bins=100, alpha=0.7)
            ax1h.set_xlabel("Amplitude")
            ax1h.set_ylabel("Frequency")
            ax1h.set_title("Original Data Amplitude Distribution")
            st.pyplot(fig1h)
        
        with col2:
            st.markdown(f"**Processed Data ({gain_type} Gain)**")
            
            # Create figure for processed data
            fig2, ax2 = plt.subplots(figsize=(10, 6))
            
            # Normalize for display
            vmax = np.percentile(np.abs(st.session_state.processed_array), 99)
            im2 = ax2.imshow(st.session_state.processed_array, aspect='auto', 
                            cmap='seismic', vmin=-vmax, vmax=vmax)
            
            ax2.set_xlabel("Trace Number")
            ax2.set_ylabel("Sample (Time)")
            ax2.set_title(f"Processed with {gain_type} Gain")
            plt.colorbar(im2, ax=ax2, label='Amplitude')
            
            st.pyplot(fig2)
            
            # Show amplitude histogram
            fig2h, ax2h = plt.subplots(figsize=(8, 3))
            ax2h.hist(st.session_state.processed_array.flatten(), bins=100, alpha=0.7, color='green')
            ax2h.set_xlabel("Amplitude")
            ax2h.set_ylabel("Frequency")
            ax2h.set_title("Processed Data Amplitude Distribution")
            st.pyplot(fig2h)
    
    with tab3:
        st.subheader("Gain Analysis")
        
        # Calculate gain applied at different depths
        n_samples = st.session_state.original_array.shape[0]
        
        # Calculate average gain at each depth
        with np.errstate(divide='ignore', invalid='ignore'):
            gain_profile = np.zeros(n_samples)
            for i in range(n_samples):
                orig_slice = st.session_state.original_array[i, :]
                proc_slice = st.session_state.processed_array[i, :]
                
                # Avoid division by zero
                mask = np.abs(orig_slice) > 1e-10
                if np.any(mask):
                    gains = np.abs(proc_slice[mask]) / np.abs(orig_slice[mask])
                    gain_profile[i] = np.median(gains)
                else:
                    gain_profile[i] = 1.0
        
        # Plot gain profile
        fig_gain, ax_gain = plt.subplots(figsize=(10, 6))
        
        depth = np.arange(n_samples)
        ax_gain.plot(gain_profile, depth, 'b-', linewidth=2, label='Gain Factor')
        ax_gain.fill_betweenx(depth, 1, gain_profile, alpha=0.3, color='blue')
        
        ax_gain.set_xlabel("Gain Factor (multiplier)")
        ax_gain.set_ylabel("Depth (samples)")
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
        
        # Show trace comparison
        st.subheader("Single Trace Comparison")
        
        trace_idx = st.slider("Select Trace Number", 0, st.session_state.original_array.shape[1]-1, 
                             st.session_state.original_array.shape[1]//2)
        
        fig_trace, ax_trace = plt.subplots(figsize=(10, 6))
        
        depth = np.arange(n_samples)
        ax_trace.plot(depth, st.session_state.original_array[:, trace_idx], 'b-', 
                     alpha=0.7, label='Original', linewidth=1)
        ax_trace.plot(depth, st.session_state.processed_array[:, trace_idx], 'r-', 
                     alpha=0.9, label='Processed', linewidth=1.5)
        
        ax_trace.set_xlabel("Depth (samples)")
        ax_trace.set_ylabel("Amplitude")
        ax_trace.set_title(f"Trace {trace_idx} Comparison")
        ax_trace.grid(True, alpha=0.3)
        ax_trace.legend()
        ax_trace.invert_xaxis()
        
        st.pyplot(fig_trace)
    
    with tab4:
        st.subheader("Export Processed Data")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("💾 Save Processed Image", use_container_width=True):
                fig, ax = plt.subplots(figsize=(12, 8))
                vmax = np.percentile(np.abs(st.session_state.processed_array), 99)
                im = ax.imshow(st.session_state.processed_array, aspect='auto', 
                              cmap='seismic', vmin=-vmax, vmax=vmax)
                ax.set_title(f"GPR Data - {gain_type} Gain Applied")
                ax.set_xlabel("Trace Number")
                ax.set_ylabel("Depth (samples)")
                plt.colorbar(im, ax=ax, label='Amplitude')
                plt.tight_layout()
                plt.savefig("processed_gpr_data.png", dpi=300, bbox_inches='tight')
                st.success("Saved as 'processed_gpr_data.png'")
        
        with col2:
            # Export as CSV
            csv_data = pd.DataFrame(st.session_state.processed_array)
            csv_string = csv_data.to_csv(index=False)
            
            st.download_button(
                label="📥 Download as CSV",
                data=csv_string,
                file_name="processed_gpr_data.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with col3:
            # Export as NumPy binary
            np_bytes = st.session_state.processed_array.tobytes()
            
            st.download_button(
                label="📥 Download as Binary",
                data=np_bytes,
                file_name="processed_gpr_data.npy",
                mime="application/octet-stream",
                use_container_width=True
            )
        
        # Export settings summary
        st.subheader("Processing Settings Summary")
        
        settings = {
            "Gain Type": gain_type,
            "Time Zero": time_zero,
            "Stacking": stacking,
            "BGR Applied": bgr,
            "Frequency Filter": freq_filter,
            "Date Processed": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        if gain_type == "Constant":
            settings["Constant Gain"] = f"{const_gain}%"
        elif gain_type == "Linear":
            settings["Min Gain"] = f"{min_gain}%"
            settings["Max Gain"] = f"{max_gain}%"
        elif gain_type == "Exponential":
            settings["Base Gain"] = f"{base_gain}%"
            settings["Exponential Factor"] = exp_factor
        
        settings_df = pd.DataFrame(list(settings.items()), columns=["Parameter", "Value"])
        st.table(settings_df)

# Initial state message
elif not dzt_file:
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.info("""
        👈 **Upload a DZT file to begin processing**
        
        **How it works:**
        1. Upload your GPR data (.DZT file)
        2. Select gain type to amplify deep signals
        3. Adjust parameters in the sidebar
        4. Click "Process Data"
        5. View and export results
        
        **Time Gain Control Options:**
        - **Constant**: Same gain for all depths
        - **Linear**: Gain increases linearly with depth
        - **Exponential**: Gain increases exponentially (best for deep weak signals)
        - **AGC**: Automatic gain control per trace
        """)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666;'>"
    "📡 <b>GPR Data Processor v2.0</b> | Time Gain Control for Radar Data"
    "</div>",
    unsafe_allow_html=True
)
