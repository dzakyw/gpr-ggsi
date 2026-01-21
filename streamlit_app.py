import streamlit as st
import numpy as np
from readgssi import readgssi
import tempfile
import os
import traceback
from contextlib import redirect_stdout
import io

# Configure the Streamlit page
st.set_page_config(page_title="Advanced GPR Data Viewer", layout="wide")
st.title("📡 Advanced GSSI Radar Data Processor")
st.markdown("Upload a GSSI `.DZT` file to process and visualize ground-penetrating radar data with multiple filters.")

# Initialize session state for parameters
if 'processing_params' not in st.session_state:
    st.session_state.processing_params = {}
if 'current_file' not in st.session_state:
    st.session_state.current_file = None

# File uploader with caching
uploaded_file = st.file_uploader("Choose a DZT file", type=['dzt', 'DZT'], 
                                 help="Upload GSSI radar data file for processing")

if uploaded_file is not None:
    # Check if file changed
    file_changed = st.session_state.current_file != uploaded_file.name
    
    if file_changed:
        st.session_state.current_file = uploaded_file.name
        st.session_state.processing_params = {}  # Reset params on new file
    
    # Save uploaded file to a temporary location
    with tempfile.NamedTemporaryFile(delete=False, suffix='.DZT') as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name
    
    try:
        # Read data once and cache
        with st.spinner('Loading radar data...'):
            hdr, arrs, gps = readgssi.readgssi(infile=tmp_path, verbose=False)
        
        # Use tabs for better organization
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Header Info", "⚙️ Processing", "📈 Visualization", "📍 GPS Data"])
        
        with tab1:
            st.header("File Information")
            
            # Create columns for header display
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
                        st.metric("Survey Date", hdr.get('created', 'N/A').strftime('%Y-%m-%d') 
                                  if hdr.get('created') else 'N/A')
            
            with col_raw:
                with st.expander("📋 Raw Header Output", expanded=False):
                    f = io.StringIO()
                    with redirect_stdout(f):
                        readgssi.readgssi(infile=tmp_path, frmt=None, verbose=True)
                    st.text(f.getvalue())
        
        with tab2:
            st.header("Data Processing Filters")
            
            # Create processing form
            with st.form("processing_form"):
                st.subheader("Noise Reduction Filters")
                
                # Three columns for filter organization
                col_stack, col_hfilter, col_vfilter = st.columns(3)
                
                with col_stack:
                    st.markdown("#### Stacking")
                    stacking_type = st.selectbox(
                        "Stacking Method",
                        ["None", "Auto (2.5:1 ratio)", "Manual"],
                        help="Average neighboring traces to reduce noise and condense X-axis"
                    )
                    
                    if stacking_type == "Manual":
                        stack_value = st.slider(
                            "Stacking Factor", 
                            min_value=2, max_value=20, value=3, step=1,
                            help="Higher values = more averaging, shorter X-axis"
                        )
                        st.session_state.processing_params['stack'] = stack_value
                    elif stacking_type == "Auto (2.5:1 ratio)":
                        st.session_state.processing_params['stack'] = 'auto'
                    else:
                        if 'stack' in st.session_state.processing_params:
                            del st.session_state.processing_params['stack']
                
                with col_hfilter:
                    st.markdown("#### Horizontal Filter (BGR)")
                    bgr_type = st.selectbox(
                        "BGR Type",
                        ["None", "Full-width Average", "Moving Window"],
                        help="Remove horizontal banding noise"
                    )
                    
                    if bgr_type == "Moving Window":
                        window_size = st.slider(
                            "Window Size (traces)", 
                            min_value=10, max_value=500, value=100, step=10,
                            help="Larger windows remove more noise but may affect horizontal layers"
                        )
                        st.session_state.processing_params['bgr'] = window_size
                    elif bgr_type == "Full-width Average":
                        st.session_state.processing_params['bgr'] = 0
                    else:
                        if 'bgr' in st.session_state.processing_params:
                            del st.session_state.processing_params['bgr']
                
                with col_vfilter:
                    st.markdown("#### Frequency Filter")
                    use_freq_filter = st.checkbox(
                        "Apply Bandpass Filter",
                        value=False,
                        help="Filter frequencies outside antenna range"
                    )
                    
                    if use_freq_filter:
                        ant_freq = hdr.get('antfreq', [100])[0]
                        default_min = max(10, int(ant_freq * 0.6))
                        default_max = int(ant_freq * 1.3)
                        
                        freq_min = st.slider(
                            "Minimum Frequency (MHz)",
                            min_value=10, max_value=500, value=default_min, step=10
                        )
                        freq_max = st.slider(
                            "Maximum Frequency (MHz)",
                            min_value=freq_min + 10, max_value=1000, value=default_max, step=10
                        )
                        
                        st.session_state.processing_params['freqmin'] = freq_min
                        st.session_state.processing_params['freqmax'] = freq_max
                    else:
                        if 'freqmin' in st.session_state.processing_params:
                            del st.session_state.processing_params['freqmin']
                        if 'freqmax' in st.session_state.processing_params:
                            del st.session_state.processing_params['freqmax']
                
                st.markdown("---")
                st.subheader("Signal Enhancement")
                
                col_gain, col_misc = st.columns(2)
                
                with col_gain:
                    gain_value = st.slider(
                        "Amplitude Gain",
                        min_value=0.1, max_value=100.0, value=30.0, step=0.5,
                        help="Increase signal strength (careful with high values)"
                    )
                    st.session_state.processing_params['gain'] = gain_value
                
                with col_misc:
                    reverse_line = st.checkbox(
                        "Reverse Survey Direction",
                        value=False,
                        help="Flip array for comparison with opposite direction lines"
                    )
                    if reverse_line:
                        st.session_state.processing_params['reverse'] = True
                    else:
                        if 'reverse' in st.session_state.processing_params:
                            del st.session_state.processing_params['reverse']
                    
                    # Distance normalization warning
                    if st.checkbox("Enable Distance Units (requires GPS)", value=False):
                        st.warning("⚠️ Requires DZG GPS file for accurate distance normalization")
                        x_units = st.selectbox(
                            "Distance Units",
                            ["meters", "kilometers", "centimeters"]
                        )
                        st.session_state.processing_params['x'] = x_units[0]
                        st.session_state.processing_params['normalize'] = True
                    else:
                        if 'x' in st.session_state.processing_params:
                            del st.session_state.processing_params['x']
                        if 'normalize' in st.session_state.processing_params:
                            del st.session_state.processing_params['normalize']
                
                # Submit button for processing
                process_submitted = st.form_submit_button("🚀 Apply Processing", type="primary")
        
        with tab3:
            st.header("Visualization Settings")
            
            # Visualization controls
            with st.container():
                vis_col1, vis_col2 = st.columns(2)
                
                with vis_col1:
                    colormap = st.selectbox(
                        "Color Map",
                        ["gray", "seismic", "RdBu", "viridis", "plasma", "hot", "coolwarm"],
                        index=0,
                        help="Choose color scheme for radargram"
                    )
                    st.session_state.processing_params['colormap'] = colormap
                    
                    z_units = st.selectbox(
                        "Depth/Time Units",
                        ["samples", "nanoseconds", "depth"],
                        index=0
                    )
                    if z_units == "nanoseconds":
                        st.session_state.processing_params['z'] = 'ns'
                    elif z_units == "depth":
                        st.session_state.processing_params['z'] = 'm'
                    else:
                        if 'z' in st.session_state.processing_params:
                            del st.session_state.processing_params['z']
                
                with vis_col2:
                    fig_width = st.slider("Figure Width", 8, 20, 12, 1)
                    fig_height = st.slider("Figure Height", 4, 12, 8, 1)
                    dpi_value = st.slider("Image DPI", 72, 600, 150, 50)
                    
                    st.session_state.processing_params['figsize'] = (fig_width, fig_height)
                    st.session_state.processing_params['dpi'] = dpi_value
            
            # Generate plot button
            if st.button("📊 Generate Radargram", type="primary", use_container_width=True):
                with st.spinner('Processing and generating radargram...'):
                    plot_temp = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
                    plot_path = plot_temp.name
                    plot_temp.close()
                    
                    try:
                        # Prepare processing parameters
                        params = {
                            'infile': tmp_path,
                            'outfile': plot_path.replace('.png', ''),
                            'frmt': None,
                            'plot': True,
                            'noshow': True,
                            'verbose': False,
                            **st.session_state.processing_params
                        }
                        
                        # Apply processing and generate plot
                        readgssi.readgssi(**params)
                        
                        # Display results
                        st.image(plot_path, use_column_width=True)
                        
                        # Show applied parameters
                        with st.expander("🔧 Applied Processing Parameters", expanded=True):
                            param_display = {k: v for k, v in params.items() 
                                           if k not in ['infile', 'outfile', 'noshow', 'verbose']}
                            
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
                                        if k in ['colormap', 'figsize', 'dpi', 'z', 'x', 'normalize']}
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
                        if st.button("Generate Basic Plot Instead"):
                            import matplotlib
                            matplotlib.use('Agg')
                            import matplotlib.pyplot as plt
                            
                            fig, ax = plt.subplots(figsize=(12, 8))
                            ax.imshow(arrs[0], aspect='auto', cmap='gray')
                            ax.set_title(f"Basic Plot: {uploaded_file.name}")
                            ax.set_xlabel('Trace')
                            ax.set_ylabel('Sample')
                            plt.tight_layout()
                            fig.savefig(plot_path, dpi=150)
                            st.image(plot_path, use_container_width=True)
                    
                    finally:
                        if os.path.exists(plot_path):
                            os.unlink(plot_path)
        
        with tab4:
            st.header("GPS Information")
            if gps is not False and hasattr(gps, 'shape') and gps.shape[0] > 0:
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
                    st.metric("Elevation Range", 
                             f"{gps['elevation'].min():.2f} to {gps['elevation'].max():.2f} m")
                
                # Interactive GPS table
                with st.expander("📋 GPS Data Table", expanded=True):
                    st.dataframe(
                        gps,
                        use_container_width=True,
                        column_config={
                            "latitude": st.column_config.NumberColumn(format="%.6f"),
                            "longitude": st.column_config.NumberColumn(format="%.6f"),
                            "elevation": st.column_config.NumberColumn(format="%.2f")
                        }
                    )
                
                # GPS visualization option
                if st.checkbox("Show GPS Track Visualization"):
                    try:
                        import pandas as pd
                        import plotly.express as px
                        
                        fig = px.line_mapbox(
                            gps,
                            lat="latitude",
                            lon="longitude",
                            color="elevation",
                            color_continuous_scale=px.colors.cyclical.IceFire,
                            zoom=10,
                            height=400
                        )
                        fig.update_layout(mapbox_style="open-street-map")
                        st.plotly_chart(fig, use_container_width=True)
                    except:
                        st.warning("Map visualization requires plotly. Install with: pip install plotly")
            else:
                st.warning("No GPS data found in file")
                st.info("""
                **To enable distance normalization:**
                1. Ensure GPS data was recorded with the survey
                2. GPS data should be in DZG format alongside DZT file
                3. For manual GPS alignment, use gpx2dzg software
                """)
    
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.info("Please ensure you've uploaded a valid GSSI DZT file.")
        
        with st.expander("Troubleshooting Tips"):
            st.markdown("""
            1. **File Format**: Ensure it's a valid GSSI DZT file
            2. **readgssi Version**: Check installation with `pip show readgssi`
            3. **Dependencies**: Ensure all requirements are installed
            4. **File Size**: Very large files may need more memory
            """)
    
    finally:
        # Clean up the temporary uploaded file
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

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
        - **Distance Normalization**: Convert to distance units (requires GPS)
        
        #### **Visualization Options:**
        - Multiple color maps
        - Adjustable figure size and resolution
        - Flexible axis units
        
        #### **Data Inspection:**
        - Detailed header information
        - GPS data visualization (if available)
        - Interactive parameter controls
        
        ### **Recommended Processing Workflows:**
        1. **Quick Quality Check**: Auto-stacking + moderate gain
        2. **Noise Reduction**: BGR + frequency filter + stacking
        3. **Detailed Analysis**: Manual stacking + custom filters
        
        ### **File Requirements:**
        - Valid GSSI `.DZT` radar data file
        - Optional: `.DZG` GPS file (for distance normalization)
        - File naming: `filename.DZT` (with optional `filename.DZG`)
        """)
    
    # Add quick tips
    st.markdown("---")
    col_tip1, col_tip2, col_tip3 = st.columns(3)
    with col_tip1:
        st.markdown("**💡 Tip 1**")
        st.caption("Start with auto-stacking for long survey lines")
    with col_tip2:
        st.markdown("**💡 Tip 2**")
        st.caption("Use BGR=100 for moving window horizontal filtering")
    with col_tip3:
        st.markdown("**💡 Tip 3**")
        st.caption("Combine filters for best results (BGR + frequency)")
