import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tempfile
import os
from pathlib import Path
from PIL import Image

# Set page config
st.set_page_config(
    page_title="GPR Data Processor",
    page_icon="📡",
    layout="wide"
)

# Title
st.title("📡 GSSI Radar Data Processor")
st.markdown("Process and visualize GPR data from GSSI (.DZT) files")

# Sidebar for file upload
with st.sidebar:
    st.header("📂 File Upload")
    
    # Upload DZT file
    dzt_file = st.file_uploader("Upload DZT file", type=['dzt', 'DZT'])
    
    # Upload DZG file (optional)
    dzg_file = st.file_uploader("Upload DZG file (optional, GPS data)", type=['dzg', 'DZG'])
    
    st.markdown("---")
    
    # Processing parameters
    st.header("⚙️ Processing Parameters")
    
    # Time zero
    time_zero = st.slider("Time Zero", 0, 500, 2)
    
    # Stacking
    stacking = st.selectbox("Stacking", ["None", "Auto", "Manual"])
    
    if stacking == "Manual":
        stack_value = st.number_input("Stack Value", 1, 50, 3)
    
    # Gain
    gain = st.slider("Gain (%)", 0, 100, 30)
    
    # BGR
    bgr = st.selectbox("Background Removal", ["None", "Full-width", "Boxcar"])
    
    if bgr == "Boxcar":
        bgr_window = st.slider("Boxcar Window", 10, 500, 100)
    
    # Frequency filter
    freq_filter = st.checkbox("Apply Frequency Filter", False)
    
    if freq_filter:
        col1, col2 = st.columns(2)
        with col1:
            freq_min = st.number_input("Min Freq (MHz)", 10, 500, 60)
        with col2:
            freq_max = st.number_input("Max Freq (MHz)", 10, 500, 130)
    
    # Process button
    process_btn = st.button("🚀 Process Data", type="primary")

# Main content
if dzt_file and process_btn:
    with st.spinner("Processing data..."):
        try:
            # Import readgssi
            try:
                from readgssi import readgssi
            except ImportError:
                st.error("readgssi not installed! Run: pip install readgssi")
                st.stop()
            
            # Save files to temp location
            with tempfile.TemporaryDirectory() as tmpdir:
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
                
                # Build parameters for readgssi
                params = {
                    'infile': dzt_path,
                    'zero': [time_zero],
                    'verbose': False
                }
                
                # Add stacking
                if stacking == "Auto":
                    params['stack'] = 'auto'
                elif stacking == "Manual":
                    params['stack'] = stack_value
                
                # Add BGR
                if bgr == "Full-width":
                    params['bgr'] = 0
                elif bgr == "Boxcar":
                    params['bgr'] = bgr_window
                
                # Add frequency filter
                if freq_filter:
                    params['freqmin'] = freq_min
                    params['freqmax'] = freq_max
                
                # Read data
                header, arrays, gps = readgssi.readgssi(**params)
                
                # Store in session state
                st.session_state.header = header
                st.session_state.array = arrays[0] if arrays else None
                st.session_state.gps = gps
                
                st.success("Data processed successfully!")
                
        except Exception as e:
            st.error(f"Error: {str(e)}")
            st.code(str(e))

# Display results if data is loaded
if 'header' in st.session_state and st.session_state.header:
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["📊 Header Info", "📈 Visualization", "📁 Export"])
    
    with tab1:
        st.subheader("File Header Information")
        
        # Display key header info
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("System", st.session_state.header.get('system', 'N/A'))
            st.metric("Antenna Frequency", f"{st.session_state.header.get('ant_freq', 'N/A')} MHz")
            st.metric("Samples per Trace", st.session_state.header.get('spt', 'N/A'))
            st.metric("Traces", st.session_state.header.get('ntraces', 'N/A'))
        
        with col2:
            st.metric("Bits per Sample", st.session_state.header.get('bps', 'N/A'))
            st.metric("Sampling Depth", f"{st.session_state.header.get('depth', 'N/A'):.2f} m")
            st.metric("Dielectric Constant", st.session_state.header.get('epsr', 'N/A'))
            st.metric("Traces per Second", st.session_state.header.get('tps', 'N/A'))
        
        # Show complete header in expander
        with st.expander("Show Complete Header"):
            st.json(st.session_state.header)
    
    with tab2:
        if st.session_state.array is not None:
            st.subheader("Radar Data Visualization")
            
            # Apply gain
            array_to_plot = st.session_state.array * (1 + gain/100)
            
            # Create figure
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Plot
            im = ax.imshow(array_to_plot, aspect='auto', cmap='seismic', 
                          interpolation='bilinear')
            
            ax.set_xlabel("Trace Number")
            ax.set_ylabel("Depth")
            ax.set_title(f"Radar Profile - Gain: {gain}%")
            
            # Add colorbar
            plt.colorbar(im, ax=ax, label='Amplitude')
            
            st.pyplot(fig)
            
            # Optional: Show GPS data
            if st.session_state.gps is not None and isinstance(st.session_state.gps, pd.DataFrame):
                with st.expander("GPS Data"):
                    st.dataframe(st.session_state.gps.head())
    
    with tab3:
        st.subheader("Export Data")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Save as PNG"):
                # Create and save figure
                fig, ax = plt.subplots(figsize=(12, 6))
                ax.imshow(array_to_plot, aspect='auto', cmap='seismic')
                ax.set_title("Radar Profile")
                plt.tight_layout()
                plt.savefig("radar_profile.png", dpi=150)
                st.success("Saved as radar_profile.png")
        
        with col2:
            if st.button("Export to CSV"):
                # Convert array to DataFrame and save
                df = pd.DataFrame(st.session_state.array)
                df.to_csv("radar_data.csv", index=False)
                st.success("Saved as radar_data.csv")

# Instructions if no file uploaded
elif not dzt_file:
    st.info("👈 Please upload a DZT file using the sidebar")

# Footer
st.markdown("---")
st.markdown("**GSSI Radar Data Processor** | Built with Streamlit & readgssi")
