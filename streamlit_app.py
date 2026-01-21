import streamlit as st
import numpy as np
from readgssi import readgssi
import tempfile
import os
import traceback
from contextlib import redirect_stdout
import io
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Configure the Streamlit page
st.set_page_config(page_title="Advanced GPR Data Viewer", layout="wide")
st.title("📡 Advanced GSSI Radar Data Processor")
st.markdown("Upload a GSSI `.DZT` file to process and visualize ground-penetrating radar data with multiple filters.")

# Initialize session state
if 'processing_params' not in st.session_state:
    st.session_state.processing_params = {}
if 'current_file' not in st.session_state:
    st.session_state.current_file = None
if 'arrs' not in st.session_state:
    st.session_state.arrs = None
if 'hdr' not in st.session_state:
    st.session_state.hdr = None

# File uploader
uploaded_file = st.file_uploader("Choose a DZT file", type=['dzt', 'DZT'], 
                                 help="Upload GSSI radar data file for processing")

if uploaded_file is not None:
    # Check if file changed
    file_changed = st.session_state.current_file != uploaded_file.name
    
    if file_changed:
        st.session_state.current_file = uploaded_file.name
        st.session_state.processing_params = {}
        
        # Save and read new file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.DZT') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name
        
        try:
            with st.spinner('Loading radar data...'):
                st.session_state.hdr, st.session_state.arrs, st.session_state.gps = readgssi.readgssi(infile=tmp_path, verbose=False)
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    
    # Get data from session state
    hdr = st.session_state.hdr
    arrs = st.session_state.arrs
    gps = st.session_state.gps
    
    if hdr is not None and arrs is not None:
        # Use tabs for organization
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Header Info", "⚙️ Processing", "📈 Visualization", "📍 GPS Data"])
        
        with tab1:
            st.header("File Information")
            
            col_info, col_raw = st.columns([1, 1])
            
            with col_info:
                st.subheader("Key Parameters")
                if hdr:
                    info_col1, info_col2 = st.columns(2)
                    
                    with info_col1:
                        st.metric("System", hdr.get('system', 'N/A'))
                        st.metric("Antenna Frequency", f"{hdr.get('antfreq', ['N/A'])[0]} MHz")
                        st.metric("Number of Traces", hdr.get('numsamp', 'N/A'))
                        
                    with info_col2:
                        antennas = hdr.get('antennas', ['N/A'])
                        st.metric("Antenna Type", antennas[0] if antennas else 'N/A')
                        st.metric("Samples per Trace", hdr.get('spp', 'N/A'))
                        created = hdr.get('created', 'N/A')
                        if created != 'N/A':
                            st.metric("Survey Date", created.strftime('%Y-%m-%d'))
                        else:
                            st.metric("Survey Date", 'N/A')
            
            with col_raw:
                with st.expander("📋 Raw Header Output", expanded=False):
                    # Create a temporary file for verbose reading
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.DZT') as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        tmp_path = tmp_file.name
                    
                    try:
                        f = io.StringIO()
                        with redirect_stdout(f):
                            readgssi.readgssi(infile=tmp_path, frmt=None, verbose=True)
                        st.text(f.getvalue())
                    finally:
                        if os.path.exists(tmp_path):
                            os.unlink(tmp_path)
        
        with tab2:
            st.header("Data Processing Filters")
            
            with st.form("processing_form"):
                st.subheader("Noise Reduction Filters")
                
                col_stack, col_hfilter, col_vfilter = st.columns(3)
                
                with col_stack:
                    st.markdown("#### Stacking")
                    stacking_type = st.selectbox(
                        "Stacking Method",
                        ["None", "Auto (2.5:1 ratio)", "Manual"],
                        help="Average neighboring traces to reduce noise and condense X-axis",
                        key="stacking_type"
                    )
                    
                    if stacking_type == "Manual":
                        stack_value = st.slider(
                            "Stacking Factor", 
                            min_value=2, max_value=20, value=3, step=1,
                            help="Higher values = more averaging, shorter X-axis",
                            key="stack_value"
                        )
                        st.session_state.processing_params['stack'] = stack_value
                    elif stacking_type == "Auto (2.5:1 ratio)":
                        st.session_state.processing_params['stack'] = 'auto'
                    else:
                        st.session_state.processing_params.pop('stack', None)
                
                with col_hfilter:
                    st.markdown("#### Horizontal Filter (BGR)")
                    bgr_type = st.selectbox(
                        "BGR Type",
                        ["None", "Full-width Average", "Moving Window"],
                        help="Remove horizontal banding noise",
                        key="bgr_type"
                    )
                    
                    if bgr_type == "Moving Window":
                        window_size = st.slider(
                            "Window Size (traces)", 
                            min_value=10, max_value=500, value=100, step=10,
                            help="Larger windows remove more noise but may affect horizontal layers",
                            key="window_size"
                        )
                        st.session_state.processing_params['bgr'] = window_size
                    elif bgr_type == "Full-width Average":
                        st.session_state.processing_params['bgr'] = 0
                    else:
                        st.session_state.processing_params.pop('bgr', None)
                
                with col_vfilter:
                    st.markdown("#### Frequency Filter")
                    use_freq_filter = st.checkbox(
                        "Apply Bandpass Filter",
                        value=False,
                        help="Filter frequencies outside antenna range",
                        key="use_freq_filter"
                    )
                    
                    if use_freq_filter:
                        ant_freq = hdr.get('antfreq', [100])[0]
                        default_min = max(10, int(ant_freq * 0.6))
                        default_max = int(ant_freq * 1.3)
                        
                        freq_min = st.slider(
                            "Minimum Frequency (MHz)",
                            min_value=10, max_value=500, value=default_min, step=10,
                            key="freq_min"
                        )
                        freq_max = st.slider(
                            "Maximum Frequency (MHz)",
                            min_value=freq_min + 10, max_value=1000, value=default_max, step=10,
                            key="freq_max"
                        )
                        
                        st.session_state.processing_params['freqmin'] = freq_min
                        st.session_state.processing_params['freqmax'] = freq_max
                    else:
                        st.session_state.processing_params.pop('freqmin', None)
                        st.session_state.processing_params.pop('freqmax', None)
                
                st.markdown("---")
                st.subheader("Signal Enhancement")
                
                col_gain, col_misc = st.columns(2)
                
                with col_gain:
                    gain_value = st.slider(
                        "Amplitude Gain",
                        min_value=0.1, max_value=100.0, value=30.0, step=0.5,
                        help="Increase signal strength (careful with high values)",
                        key="gain_value"
                    )
                    st.session_state.processing_params['gain'] = gain_value
                
                with col_misc:
                    reverse_line = st.checkbox(
                        "Reverse Survey Direction",
                        value=False,
                        help="Flip array for comparison with opposite direction lines",
                        key="reverse_line"
                    )
                    if reverse_line:
                        st.session_state.processing_params['reverse'] = True
                    else:
                        st.session_state.processing_params.pop('reverse', None)
                    
                    # X-axis units (simpler approach)
                    x_units = st.selectbox(
                        "X-axis Units",
                        ["traces", "seconds", "distance"],
                        index=0,
                        key="x_units"
                    )
                    if x_units == "distance":
                        st.warning("⚠️ Distance units require proper GPS data (DZG file)")
                        st.session_state.processing_params['x'] = 'm'
                    elif x_units == "seconds":
                        st.session_state.processing_params['x'] = 's'
                    else:
                        st.session_state.processing_params.pop('x', None)
                
                process_submitted = st.form_submit_button("🚀 Apply Processing", type="primary")
        
        with tab3:
            st.header("Visualization Settings")
            
            with st.container():
                vis_col1, vis_col2 = st.columns(2)
                
                with vis_col1:
                    colormap = st.selectbox(
                        "Color Map",
                        ["gray", "seismic", "RdBu", "viridis", "plasma", "hot", "coolwarm"],
                        index=0,
                        help="Choose color scheme for radargram",
                        key="colormap"
                    )
                    
                    z_units = st.selectbox(
                        "Depth/Time Units",
                        ["samples", "nanoseconds", "depth"],
                        index=0,
                        key="z_units"
                    )
                    if z_units == "nanoseconds":
                        st.session_state.processing_params['z'] = 'ns'
                    elif z_units == "depth":
                        st.session_state.processing_params['z'] = 'm'
                    else:
                        st.session_state.processing_params.pop('z', None)
                
                with vis_col2:
                    fig_width = st.slider("Figure Width", 8, 20, 12, 1, key="fig_width")
                    fig_height = st.slider("Figure Height", 4, 12, 8, 1, key="fig_height")
                    dpi_value = st.slider("Image DPI", 72, 600, 150, 50, key="dpi_value")
                    
                    st.session_state.processing_params['figsize'] = (fig_width, fig_height)
                    st.session_state.processing_params['dpi'] = dpi_value
                    st.session_state.processing_params['colormap'] = colormap
            
            # Generate plot button
            if st.button("📊 Generate Radargram", type="primary", use_container_width=True, key="generate_plot"):
                with st.spinner('Processing and generating radargram...'):
                    plot_temp = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
                    plot_path = plot_temp.name
                    plot_temp.close()
                    
                    try:
                        # Apply processing to the data
                        from readgssi.arrayops import process_array
                        
                        # Get the first array channel
                        data = arrs[0].copy()
                        
                        # Apply processing filters if specified
                        processing_kwargs = {}
                        if 'bgr' in st.session_state.processing_params:
                            processing_kwargs['bgr'] = st.session_state.processing_params['bgr']
                        if 'freqmin' in st.session_state.processing_params and 'freqmax' in st.session_state.processing_params:
                            processing_kwargs['freqmin'] = st.session_state.processing_params['freqmin']
                            processing_kwargs['freqmax'] = st.session_state.processing_params['freqmax']
                        if 'stack' in st.session_state.processing_params:
                            processing_kwargs['stack'] = st.session_state.processing_params['stack']
                        if 'reverse' in st.session_state.processing_params:
                            from readgssi.arrayops import flip
                            data = flip(data)
                        
                        # Apply gain
                        if 'gain' in st.session_state.processing_params:
                            gain = st.session_state.processing_params['gain']
                            data = data * gain
                        
                        # Create the plot using matplotlib directly
                        fig, ax = plt.subplots(figsize=st.session_state.processing_params.get('figsize', (12, 8)))
                        
                        # Plot the data
                        im = ax.imshow(data, 
                                     aspect='auto',
                                     cmap=st.session_state.processing_params.get('colormap', 'gray'),
                                     interpolation='nearest')
                        
                        # Add labels
                        ax.set_xlabel('Trace Number')
                        
                        # Set Y-axis label based on units
                        if 'z' in st.session_state.processing_params:
                            if st.session_state.processing_params['z'] == 'ns':
                                ax.set_ylabel('Time (nanoseconds)')
                            elif st.session_state.processing_params['z'] == 'm':
                                ax.set_ylabel('Depth (meters)')
                        else:
                            ax.set_ylabel('Sample Number')
                        
                        # Add title
                        ax.set_title(f"Radargram: {uploaded_file.name}")
                        
                        # Add colorbar
                        cbar = plt.colorbar(im, ax=ax)
                        cbar.set_label('Amplitude')
                        
                        plt.tight_layout()
                        
                        # Save the figure
                        fig.savefig(plot_path, 
                                  dpi=st.session_state.processing_params.get('dpi', 150), 
                                  bbox_inches='tight')
                        plt.close(fig)
                        
                        # Display results
                        st.image(plot_path, use_container_width=True)
                        
                        # Show applied parameters
                        with st.expander("🔧 Applied Processing Parameters", expanded=True):
                            param_display = st.session_state.processing_params.copy()
                            
                            # Group parameters
                            st.markdown("**Noise Reduction:**")
                            noise_params = {k: v for k, v in param_display.items() 
                                          if k in ['stack', 'bgr', 'freqmin', 'freqmax']}
                            if noise_params:
                                st.json(noise_params)
                            else:
                                st.caption("No noise reduction filters applied")
                            
                            st.markdown("**Signal Enhancement:**")
                            signal_params = {k: v for k, v in param_display.items() 
                                           if k in ['gain', 'reverse']}
                            if signal_params:
                                st.json(signal_params)
                            
                            st.markdown("**Visualization:**")
                            vis_params = {k: v for k, v in param_display.items() 
                                        if k in ['colormap', 'figsize', 'dpi', 'z', 'x']}
                            if vis_params:
                                st.json(vis_params)
                        
                        # Download button
                        with open(plot_path, "rb") as file:
                            st.download_button(
                                label="💾 Download Processed Radargram",
                                data=file,
                                file_name=f"{uploaded_file.name.replace('.DZT', '')}_processed.png",
                                mime="image/png",
                                use_container_width=True
                            )
                        
                        st.success("✅ Radargram processed successfully!")
                        
                    except Exception as plot_error:
                        st.error(f"Plot generation error: {str(plot_error)}")
                        
                        with st.expander("🔍 Debug Information"):
                            st.code(traceback.format_exc())
                            st.markdown("**Try these solutions:**")
                            st.markdown("""
                            1. Reduce the gain value
                            2. Remove frequency filters
                            3. Use simpler stacking values
                            4. Ensure file is valid GSSI DZT format
                            """)
                        
                        # Fallback basic plot
                        if st.button("Generate Basic Plot Instead", key="fallback_plot"):
                            try:
                                fig, ax = plt.subplots(figsize=(12, 8))
                                ax.imshow(arrs[0], aspect='auto', cmap='gray')
                                ax.set_title(f"Basic Plot: {uploaded_file.name}")
                                ax.set_xlabel('Trace')
                                ax.set_ylabel('Sample')
                                plt.tight_layout()
                                fig.savefig(plot_path, dpi=150)
                                st.image(plot_path, use_container_width=True)
                            except:
                                st.error("Could not generate fallback plot")
                    
                    finally:
                        if os.path.exists(plot_path):
                            os.unlink(plot_path)
        
        with tab4:
            st.header("GPS Information")
            if gps is not False and gps is not None and hasattr(gps, 'shape') and gps.shape[0] > 0:
                st.success(f"✅ GPS data found ({len(gps)} points)")
                
                # GPS statistics
                gps_col1, gps_col2, gps_col3 = st.columns(3)
                with gps_col1:
                    st.metric("Latitude Range", 
                             f"{gps['latitude'].min():.6f} to {gps['latitude'].max():.6f}")
                with gps_col2:
                    st.metric("Longitude Range", 
                             f"{gps['longitude'].min():.6f} to {gps['longitude'].max():.6f}")
                with gps_col3:
                    if 'elevation' in gps.columns:
                        st.metric("Elevation Range", 
                                 f"{gps['elevation'].min():.2f} to {gps['elevation'].max():.2f} m")
                    else:
                        st.metric("Elevation", "Not available")
                
                # Interactive GPS table
                with st.expander("📋 GPS Data Table", expanded=True):
                    st.dataframe(
                        gps,
                        use_container_width=True
                    )
            else:
                st.warning("No GPS data found in file")
                st.info("""
                **To enable distance normalization:**
                1. Ensure GPS data was recorded with the survey
                2. GPS data should be in DZG format alongside DZT file
                3. For manual GPS alignment, use gpx2dzg software
                """)

else:
    # Welcome screen
    st.info("👆 Please upload a GSSI DZT file to begin processing.")
    
    with st.expander("📚 About This Application", expanded=True):
        st.markdown("""
        ### **Advanced GPR Data Processing Features**
        
        #### **Noise Reduction Filters:**
        - **Stacking**: Average neighboring traces (auto or manual)
        - **BGR Filters**: Remove horizontal banding noise
        - **Frequency Bandpass**: Filter outside antenna frequency range
        
        #### **Signal Enhancement:**
        - **Gain Control**: Amplify signal strength
        - **Line Reversal**: Flip survey direction
        
        #### **Visualization Options:**
        - Multiple color maps
        - Adjustable figure size and resolution
        - Flexible axis units
        
        ### **Recommended Processing Workflows:**
        1. **Quick Quality Check**: Auto-stacking + moderate gain
        2. **Noise Reduction**: BGR + frequency filter + stacking
        3. **Detailed Analysis**: Manual stacking + custom filters
        """)
