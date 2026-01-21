import streamlit as st
import numpy as np
from readgssi import readgssi
import tempfile
import os
import traceback

# Configure the Streamlit page
st.set_page_config(page_title="GPR Data Viewer", layout="wide")
st.title("📡 GSSI Radar Data Viewer with readgssi")
st.markdown("Upload a GSSI `.DZT` file to read its header and plot the radargram.")

# File uploader
uploaded_file = st.file_uploader("Choose a DZT file", type=['dzt', 'DZT'])

if uploaded_file is not None:
    # Save uploaded file to a temporary location
    with tempfile.NamedTemporaryFile(delete=False, suffix='.DZT') as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name

    try:
        # Use two columns for layout
        col1, col2 = st.columns([1, 2])

        with col1:
            st.header("File Header Information")
            # Read file header (verbose output)
            with st.spinner('Reading file header...'):
                # Capture the verbose output from readgssi
                import io
                from contextlib import redirect_stdout
                f = io.StringIO()
                with redirect_stdout(f):
                    # Read with verbose=True to get header info
                    readgssi.readgssi(infile=tmp_path, frmt=None, verbose=True)
                header_output = f.getvalue()
            
            # Display header in an expandable section
            with st.expander("View Raw Header Output", expanded=True):
                st.text(header_output)

            # Now read the data into Python objects for processing
            with st.spinner('Loading radar data...'):
                hdr, arrs, gps = readgssi.readgssi(infile=tmp_path, verbose=False)
            
            # Display key header parameters in a user-friendly format
            st.subheader("Key Parameters")
            if hdr:
                st.markdown(f"""
                **System:** {hdr.get('system', 'N/A')}  
                **Antenna:** {hdr.get('antennas', ['N/A'])[0]}  
                **Frequency:** {hdr.get('antfreq', ['N/A'])[0]} MHz  
                **Samples per Trace:** {hdr.get('spp', 'N/A')}  
                **Number of Traces:** {hdr.get('numsamp', 'N/A')}  
                **Survey Date:** {hdr.get('created', 'N/A')}  
                """)

        with col2:
            st.header("Data Processing & Radargram")
            
            # Create an expandable section for processing controls
            with st.expander("🛠️ Processing Filters", expanded=True):
                st.subheader("Noise Reduction & Enhancement")
                
                # Filter controls organized in columns
                col_f1, col_f2 = st.columns(2)
                
                with col_f1:
                    # Horizontal Filter (BGR)
                    bgr_type = st.selectbox(
                        "Horizontal Filter (BGR)",
                        ["None", "Full-width average", "Moving window"],
                        help="Removes horizontal banding noise. 'Moving window' requires a window size."
                    )
                    bgr_value = 0
                    if bgr_type == "Moving window":
                        bgr_value = st.slider("Window size (traces)", 10, 500, 100, 10)
                    elif bgr_type == "Full-width average":
                        bgr_value = 0  # readgssi uses 0 for full-width
                    
                    # Stacking
                    stack_method = st.selectbox(
                        "Stacking",
                        ["None", "Auto (2.5:1 ratio)", "Manual"],
                        help="Adds neighboring traces to reduce noise and condense the X-axis"
                    )
                    stack_value = None
                    if stack_method == "Manual":
                        stack_value = st.slider("Stacking factor", 2, 20, 3, 1)
                    elif stack_method == "Auto (2.5:1 ratio)":
                        stack_value = 'auto'
                
                with col_f2:
                    # Frequency Filter
                    use_freq_filter = st.checkbox("Apply Frequency Bandpass Filter", value=False)
                    freq_min = 0
                    freq_max = 0
                    if use_freq_filter:
                        # Get antenna frequency for sensible defaults
                        ant_freq = hdr.get('antfreq', [100])[0]
                        default_min = max(10, int(ant_freq * 0.6))
                        default_max = int(ant_freq * 1.3)
                        
                        freq_min = st.slider("Minimum frequency (MHz)", 10, 500, default_min, 10)
                        freq_max = st.slider("Maximum frequency (MHz)", freq_min+10, 1000, default_max, 10)
                    
                    # Line Reversal
                    reverse_line = st.checkbox(
                        "Reverse survey direction",
                        value=False,
                        help="Flip the array to show comparison with opposite direction lines"
                    )
                    
                    # Gain is always available
                    gain = st.slider("Amplitude Gain", 0.1, 100.0, 30.0, 0.1)
            
            # Display settings section
            with st.expander("📐 Display Settings", expanded=True):
                col_d1, col_d2 = st.columns(2)
                with col_d1:
                    colormap = st.selectbox("Colormap", ["gray", "seismic", "RdBu", "viridis", "plasma"], index=0)
                    x_units = st.selectbox("X-axis units", ["traces", "seconds", "meters", "kilometers", "centimeters"], index=0)
                with col_d2:
                    z_units = st.selectbox("Z-axis units", ["samples", "nanoseconds", "depth"], index=0)
                    dpi = st.slider("Image resolution (DPI)", 72, 600, 150, 50)
            
            # Plot button
            plot_button = st.button("🚀 Generate Processed Radargram", type="primary")
        
            if plot_button and len(arrs) > 0:
                with st.spinner('Processing and plotting radar data...'):
                    # Create a temporary file for the plot
                    plot_temp = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
                    plot_path = plot_temp.name
                    plot_temp.close()
                    
                    try:
                        # Prepare processing parameters based on user selections
                        processing_params = {
                            'infile': tmp_path,
                            'outfile': plot_path.replace('.png', ''),  # Without extension
                            'frmt': None,
                            'plot': True,
                            'figsize': (12, 8),
                            'gain': gain,
                            'colormap': colormap,
                            'dpi': dpi,
                            'noshow': True,
                            'verbose': False
                        }
                        
                        # Apply stacking if selected
                        if stack_value is not None:
                            processing_params['stack'] = stack_value
                        
                        # Apply BGR filter if selected
                        if bgr_type != "None":
                            processing_params['bgr'] = bgr_value
                        
                        # Apply frequency filter if selected
                        if use_freq_filter and freq_max > freq_min:
                            processing_params['freqmin'] = freq_min
                            processing_params['freqmax'] = freq_max
                        
                        # Apply line reversal if selected
                        if reverse_line:
                            processing_params['reverse'] = True
                        
                        # Apply X-axis units (with caution about distance units)
                        if x_units != "traces":
                            if x_units in ["meters", "kilometers", "centimeters"]:
                                st.warning(f"Using '{x_units}' for X-axis requires proper GPS data. Ensure your file has associated DZG data for accurate distance normalization.")
                            processing_params['x'] = x_units[0]  # 'm', 'k', or 'c'
                        
                        # Apply Z-axis units
                        if z_units == "nanoseconds":
                            processing_params['z'] = 'ns'
                        elif z_units == "depth":
                            processing_params['z'] = 'm'
                        
                        # Generate the processed plot
                        from readgssi import readgssi
                        import matplotlib
                        matplotlib.use('Agg')
                        
                        # Call readgssi with all processing parameters
                        readgssi.readgssi(**processing_params)
                        
                        # Display the result
                        st.image(plot_path, use_column_width=True)
                        
                        # Show applied parameters
                        st.success("✅ Radargram processed successfully!")
                        with st.expander("📋 Applied Processing Parameters"):
                            param_display = {k: v for k, v in processing_params.items() if k not in ['infile', 'outfile', 'noshow', 'verbose']}
                            st.json(param_display)
                        
                        # Download button
                        with open(plot_path, "rb") as file:
                            st.download_button(
                                label="Download Processed Radargram",
                                data=file,
                                file_name=f"{uploaded_file.name.replace('.DZT', '')}_processed.png",
                                mime="image/png"
                            )
                        
                    except Exception as e:
                        st.error(f"Processing error: {str(e)}")
                        st.info("Try simplifying the filter combination or reducing parameter values.")
                    finally:
                        if os.path.exists(plot_path):
                            os.unlink(plot_path)

        # GPS data section (if available)
        if gps is not False and hasattr(gps, 'shape') and gps.shape[0] > 0:
            st.subheader("📡 GPS Data")
            st.dataframe(gps.head())
            st.caption(f"Total GPS points: {len(gps)}")

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.info("Please ensure you've uploaded a valid GSSI DZT file.")
    finally:
        # Clean up the temporary uploaded file
        os.unlink(tmp_path)
else:
    st.info("👆 Please upload a GSSI DZT file to begin.")
    st.markdown("""
    ### What this app does:
    1. **Reads GSSI DZT files** using the `readgssi` library
    2. **Displays header information** like antenna type, frequency, and survey details
    3. **Generates radargram plots** with two methods:
       - **Simple Matplotlib**: Reliable basic visualization
       - **readgssi radargram**: Advanced plotting (may have compatibility issues)
    
    ### Troubleshooting:
    - If you get the "cannot access local variable 'figx'" error, use the **Simple Matplotlib** method
    - Ensure your `readgssi` installation is complete
    - Check that uploaded files are valid GSSI DZT files
    """)

