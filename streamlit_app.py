# app.py - Main Streamlit Application

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tempfile
import os
import sys
from pathlib import Path
import traceback

# Set page configuration
st.set_page_config(
    page_title="GSSI Radar Data Processor",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #3B82F6;
        margin-top: 1rem;
        margin-bottom: 1rem;
    }
    .info-box {
        background-color: #F0F8FF;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #3B82F6;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #FEF3C7;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #F59E0B;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Title and description
st.markdown('<h1 class="main-header">📡 GSSI Radar Data Processor</h1>', unsafe_allow_html=True)
st.markdown("""
This application processes GSSI radar data (.DZT files) using the `readgssi` library.
Upload your radar data files and configure processing parameters in the sidebar.
""")

# Initialize session state for storing data
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'header' not in st.session_state:
    st.session_state.header = None
if 'radar_array' not in st.session_state:
    st.session_state.radar_array = None
if 'gps_data' not in st.session_state:
    st.session_state.gps_data = None
if 'processing_params' not in st.session_state:
    st.session_state.processing_params = {}

# Sidebar for file upload and parameters
with st.sidebar:
    st.markdown('<h2 class="sub-header">📂 File Upload</h2>', unsafe_allow_html=True)
    
    # File uploaders
    dzt_file = st.file_uploader("Upload DZT file", type=['dzt', 'DZT'])
    dzg_file = st.file_uploader("Upload DZG file (optional - GPS data)", type=['dzg', 'DZG'])
    
    st.markdown("---")
    st.markdown('<h2 class="sub-header">⚙️ Processing Parameters</h2>', unsafe_allow_html=True)
    
    # Time zero parameter
    time_zero = st.number_input(
        "Time Zero (samples)",
        min_value=0,
        max_value=5000,
        value=2,
        help="Time zero offset in samples. Default is 2."
    )
    
    # Stacking options
    stacking_method = st.selectbox(
        "Stacking Method",
        ["None", "Auto", "Manual"],
        help="Stack traces to reduce noise and condense display"
    )
    
    stack_value = 1
    if stacking_method == "Manual":
        stack_value = st.number_input(
            "Stack Value",
            min_value=1,
            max_value=50,
            value=3,
            help="Number of traces to stack together"
        )
    
    # Gain parameter
    gain = st.slider(
        "Gain",
        min_value=0,
        max_value=100,
        value=30,
        help="Signal amplification factor"
    )
    
    # Background Removal (BGR)
    st.markdown("### Background Removal")
    bgr_method = st.selectbox(
        "BGR Method",
        ["None", "Full-width", "Boxcar"],
        help="Remove horizontal noise from data"
    )
    
    bgr_window = 100
    if bgr_method == "Boxcar":
        bgr_window = st.slider(
            "Boxcar Window Size",
            min_value=10,
            max_value=500,
            value=100,
            help="Moving window size for boxcar filter"
        )
    
    # Frequency filtering
    st.markdown("### Frequency Filtering")
    freq_filter = st.checkbox("Apply Frequency Filter", value=False)
    
    freq_min = 60
    freq_max = 130
    if freq_filter:
        col1, col2 = st.columns(2)
        with col1:
            freq_min = st.number_input("Minimum Frequency (MHz)", value=60)
        with col2:
            freq_max = st.number_input("Maximum Frequency (MHz)", value=130)
    
    # Additional options
    st.markdown("### Additional Options")
    normalize = st.checkbox("Distance Normalization", value=False, 
                           help="Requires GPS data in DZG file")
    reverse = st.checkbox("Reverse Travel Direction", value=False)
    pause_correction = st.checkbox("GPS Pause Correction", value=False,
                                  help="Correct for pauses during GPS recording")
    
    # Process button
    process_btn = st.button("🚀 Process Data", type="primary", use_container_width=True)
    
    # Clear button
    if st.button("🧹 Clear Data", use_container_width=True):
        st.session_state.data_loaded = False
        st.session_state.header = None
        st.session_state.radar_array = None
        st.session_state.gps_data = None
        st.rerun()

# Main content area
tab1, tab2, tab3, tab4 = st.tabs(["📊 Data Overview", "📈 Visualization", "⚙️ Processing", "📋 Documentation"])

with tab1:
    if dzt_file and process_btn:
        try:
            # Save uploaded files to temporary directory
            with tempfile.TemporaryDirectory() as tmpdir:
                # Save DZT file
                dzt_path = os.path.join(tmpdir, dzt_file.name)
                with open(dzt_path, 'wb') as f:
                    f.write(dzt_file.getbuffer())
                
                # Save DZG file if provided
                dzg_path = None
                if dzg_file:
                    dzg_path = os.path.join(tmpdir, dzg_file.name)
                    with open(dzg_path, 'wb') as f:
                        f.write(dzg_file.getbuffer())
                
                # Import readgssi and process data
                try:
                    from readgssi import readgssi
                    
                    with st.spinner("Reading radar data..."):
                        # Prepare parameters for readgssi
                        params = {
                            'infile': dzt_path,
                            'zero': [int(time_zero)],
                            'verbose': False
                        }
                        
                        # Add stacking parameter
                        if stacking_method == "Auto":
                            params['stack'] = 'auto'
                        elif stacking_method == "Manual":
                            params['stack'] = stack_value
                        
                        # Add BGR parameter
                        if bgr_method == "Full-width":
                            params['bgr'] = 0
                        elif bgr_method == "Boxcar":
                            params['bgr'] = bgr_window
                        
                        # Add frequency filter
                        if freq_filter:
                            params['freqmin'] = freq_min
                            params['freqmax'] = freq_max
                        
                        # Add normalization
                        if normalize:
                            params['normalize'] = True
                        
                        # Add reverse
                        if reverse:
                            params['reverse'] = True
                        
                        # Add pause correction
                        if pause_correction:
                            params['pausecorrect'] = True
                        
                        # Read data
                        header, arrays, gps = readgssi.readgssi(**params)
                        
                        # Store in session state
                        st.session_state.header = header
                        st.session_state.radar_array = arrays[0] if arrays else None
                        st.session_state.gps_data = gps
                        st.session_state.data_loaded = True
                        st.session_state.processing_params = params
                        
                        st.success("✅ Data loaded successfully!")
                        
                except ImportError as e:
                    st.error(f"Error importing readgssi: {e}")
                    st.info("Please ensure readgssi is installed. Check the Documentation tab for installation instructions.")
                    
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
            st.code(traceback.format_exc())
    
    # Display header information if data is loaded
    if st.session_state.data_loaded and st.session_state.header:
        st.markdown('<h2 class="sub-header">📋 File Header Information</h2>', unsafe_allow_html=True)
        
        # Create columns for better organization
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("System", st.session_state.header.get('system', 'N/A'))
            st.metric("Antenna Frequency", f"{st.session_state.header.get('ant_freq', 'N/A')} MHz")
            st.metric("Samples per Trace", st.session_state.header.get('spt', 'N/A'))
            st.metric("Traces per Second", st.session_state.header.get('tps', 'N/A'))
        
        with col2:
            st.metric("Number of Channels", st.session_state.header.get('nchannels', 'N/A'))
            st.metric("Traces", st.session_state.header.get('ntraces', 'N/A'))
            st.metric("Sampling Depth", f"{st.session_state.header.get('depth', 'N/A'):.2f} m")
            st.metric("Dielectric Constant", st.session_state.header.get('epsr', 'N/A'))
        
        # Display additional information in expander
        with st.expander("Show Complete Header"):
            st.json(st.session_state.header)
        
        # Display GPS information if available
        if st.session_state.gps_data is not None and not isinstance(st.session_state.gps_data, bool):
            st.markdown("### 📍 GPS Data Information")
            st.dataframe(st.session_state.gps_data.head())
            st.metric("GPS Points", len(st.session_state.gps_data))

with tab2:
    if st.session_state.data_loaded and st.session_state.radar_array is not None:
        st.markdown('<h2 class="sub-header">📈 Radar Data Visualization</h2>', unsafe_allow_html=True)
        
        # Visualization options
        col1, col2, col3 = st.columns(3)
        with col1:
            show_colorbar = st.checkbox("Show Colorbar", value=True)
        with col2:
            show_title = st.checkbox("Show Title", value=True)
        with col3:
            aspect_ratio = st.selectbox("Aspect Ratio", ["auto", "equal", "1:2", "1:3"])
        
        # Plot the radar data
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Apply gain if specified
        array_to_plot = st.session_state.radar_array * (1 + gain/100)
        
        # Create the plot
        im = ax.imshow(array_to_plot, aspect=aspect_ratio, 
                      cmap='seismic', interpolation='bilinear')
        
        if show_title:
            title = f"Radar Profile - Gain: {gain}%"
            if stacking_method != "None":
                title += f" | Stacking: {stacking_method}"
            ax.set_title(title, fontsize=14, pad=20)
        
        ax.set_xlabel("Trace Number", fontsize=12)
        ax.set_ylabel("Sample Depth", fontsize=12)
        
        if show_colorbar:
            cbar = fig.colorbar(im, ax=ax)
            cbar.set_label('Amplitude', rotation=270, labelpad=15)
        
        st.pyplot(fig)
        
        # Additional visualization options
        with st.expander("Advanced Visualization Options"):
            st.markdown("### 3D Surface Plot")
            if st.button("Generate 3D View (Experimental)"):
                try:
                    import plotly.graph_objects as go
                    
                    # Downsample for performance
                    downsampled_array = array_to_plot[::10, ::10]
                    
                    fig_3d = go.Figure(data=[go.Surface(z=downsampled_array, colorscale='Viridis')])
                    fig_3d.update_layout(
                        title='3D Radar Profile',
                        scene=dict(
                            xaxis_title='Trace',
                            yaxis_title='Depth',
                            zaxis_title='Amplitude'
                        ),
                        height=600
                    )
                    st.plotly_chart(fig_3d, use_container_width=True)
                except Exception as e:
                    st.warning(f"3D plot not available: {e}")
        
        # Export options
        st.markdown("### 💾 Export Visualization")
        export_col1, export_col2 = st.columns(2)
        with export_col1:
            if st.button("Save as PNG"):
                # Save figure
                fig.savefig("radar_profile.png", dpi=300, bbox_inches='tight')
                st.success("Saved as radar_profile.png")
        
        with export_col2:
            # Provide download link for data
            if st.button("Export Data as CSV"):
                # Create a flattened version for export
                df_export = pd.DataFrame(array_to_plot)
                csv = df_export.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name="radar_data.csv",
                    mime="text/csv"
                )
    
    elif not st.session_state.data_loaded:
        st.info("👈 Please upload a DZT file and click 'Process Data' to visualize radar data.")

with tab3:
    st.markdown('<h2 class="sub-header">⚙️ Advanced Processing</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    ### Available Processing Functions
    
    The following processing methods are available through `readgssi`:
    
    1. **Stacking** - Reduce noise by averaging neighboring traces
    2. **Background Removal (BGR)** - Remove horizontal noise patterns
    3. **Frequency Filtering** - Apply bandpass filters based on antenna frequency
    4. **Distance Normalization** - Convert time-based data to distance-based using GPS
    5. **Pause Correction** - Correct GPS data for recording pauses
    
    ### Processing Pipeline
    """)
    
    # Create a processing pipeline visualization
    steps = ["Raw Data", "Time Zero Correction", "Stacking", "BGR", "Frequency Filter", "Normalization"]
    current_step = 0
    
    if st.session_state.data_loaded:
        # Determine which steps are active based on parameters
        active_steps = [True]  # Raw data is always active
        
        # Time zero correction
        active_steps.append(time_zero != 2)
        
        # Stacking
        active_steps.append(stacking_method != "None")
        
        # BGR
        active_steps.append(bgr_method != "None")
        
        # Frequency filter
        active_steps.append(freq_filter)
        
        # Normalization
        active_steps.append(normalize)
        
        # Create visual pipeline
        cols = st.columns(len(steps))
        for idx, (col, step, active) in enumerate(zip(cols, steps, active_steps)):
            with col:
                if active:
                    st.markdown(f"**✅ {step}**")
                    st.progress(1.0 if idx == len(steps)-1 else 0.7)
                else:
                    st.markdown(f"⚪ {step}")
                    st.progress(0.0)
    
    # Display current processing parameters
    if st.session_state.processing_params:
        st.markdown("### Current Processing Parameters")
        st.json(st.session_state.processing_params)
    
    # Quick processing presets
    st.markdown("### 🚀 Quick Processing Presets")
    preset_col1, preset_col2, preset_col3 = st.columns(3)
    
    with preset_col1:
        if st.button("Basic Processing", use_container_width=True):
            st.session_state.processing_params.update({
                'stack': 'auto',
                'gain': 30,
                'bgr': 0
            })
            st.info("Basic processing preset applied")
    
    with preset_col2:
        if st.button("High Resolution", use_container_width=True):
            st.session_state.processing_params.update({
                'stack': 1,
                'gain': 50,
                'bgr': 50,
                'freqmin': 70,
                'freqmax': 130
            })
            st.info("High resolution preset applied")
    
    with preset_col3:
        if st.button("Noise Reduction", use_container_width=True):
            st.session_state.processing_params.update({
                'stack': 'auto',
                'bgr': 100,
                'freqmin': 60,
                'freqmax': 100
            })
            st.info("Noise reduction preset applied")

with tab4:
    st.markdown('<h2 class="sub-header">📖 Documentation & Help</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    ### Installation Requirements
    
    This application requires the following Python packages:
    
    ```bash
    streamlit==1.28.0
    obspy==1.4.0
    numpy==1.24.0
    scipy==1.10.0
    matplotlib==3.7.0
    pandas==2.0.0
    h5py==3.8.0
    geopy==2.3.0
    pytz==2023.3
    readgssi==0.0.12
    ```
    
    ### Installation Methods
    
    #### 1. Using pip (Recommended)
    ```bash
    pip install -r requirements.txt
    ```
    
    #### 2. Using conda
    ```bash
    conda create -n radar python=3.9
    conda activate radar
    conda install -c conda-forge obspy numpy scipy matplotlib pandas h5py
    pip install geopy pytz readgssi streamlit
    ```
    
    ### File Formats
    
    - **DZT**: GSSI radar data file (required)
    - **DZG**: GPS data file (optional, required for distance normalization)
    
    ### Processing Parameters Explained
    
    1. **Time Zero**: Adjusts the starting time of each trace
    2. **Stacking**: Averages multiple traces to reduce noise
    3. **Gain**: Amplifies weak signals
    4. **BGR**: Removes horizontal noise patterns
    5. **Frequency Filter**: Filters data based on antenna frequency
    6. **Distance Normalization**: Converts time-based data to distance using GPS
    
    ### Troubleshooting
    
    #### Common Issues:
    
    1. **readgssi not found**: Ensure readgssi is installed via pip
    2. **Memory errors**: Large files may require more memory
    3. **GPS data issues**: Ensure DZG file matches DZT file
    
    #### Getting Help:
    
    - Check the [readgssi GitHub repository](https://github.com/iannesbitt/readgssi)
    - Review the [Obspy documentation](https://docs.obspy.org)
    - Create an issue on the GitHub repo for bugs
    """)
    
    # Quick test section
    st.markdown("### 🧪 Quick Installation Test")
    if st.button("Test Installation"):
        try:
            import readgssi
            st.success("✅ readgssi is properly installed!")
            
            # Test basic functionality
            version = readgssi.__version__ if hasattr(readgssi, '__version__') else "Unknown"
            st.info(f"readgssi version: {version}")
            
        except ImportError:
            st.error("❌ readgssi is not installed. Please install using the commands above.")
    
    # Version information
    st.markdown("### 📊 Version Information")
    version_info = {}
    try:
        import streamlit as st_lib
        version_info['Streamlit'] = st_lib.__version__
    except:
        version_info['Streamlit'] = 'N/A'
    
    for lib in ['numpy', 'pandas', 'matplotlib', 'obspy']:
        try:
            module = __import__(lib)
            version_info[lib.capitalize()] = getattr(module, '__version__', 'N/A')
        except:
            version_info[lib.capitalize()] = 'N/A'
    
    st.table(pd.DataFrame(list(version_info.items()), columns=['Library', 'Version']))

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; font-size: 0.9rem;'>
    <p>GSSI Radar Data Processor | Built with Streamlit and readgssi</p>
    <p>For support and documentation, visit the 
    <a href='https://github.com/iannesbitt/readgssi' target='_blank'>readgssi GitHub repository</a></p>
</div>
""", unsafe_allow_html=True)
