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
st.set_page_config(page_title="GPR Data Processor", layout="wide")
st.title("📡 GSSI Radar Data Processor with Matplotlib")
st.markdown("Upload a GSSI `.DZT` file, apply filters, and visualize with matplotlib")

# Initialize session state
if 'data' not in st.session_state:
    st.session_state.data = None
if 'hdr' not in st.session_state:
    st.session_state.hdr = None
if 'current_file' not in st.session_state:
    st.session_state.current_file = None
if 'original_data' not in st.session_state:
    st.session_state.original_data = None

# File uploader
uploaded_file = st.file_uploader("Choose a DZT file", type=['dzt', 'DZT'])

if uploaded_file is not None:
    # Check if file changed
    if st.session_state.current_file != uploaded_file.name:
        st.session_state.current_file = uploaded_file.name
        
        # Save and read file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.DZT') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name
        
        try:
            with st.spinner('Loading radar data...'):
                # Read data without any processing
                hdr, arrs, gps = readgssi.readgssi(infile=tmp_path, verbose=False)
                if len(arrs) > 0:
                    st.session_state.original_data = arrs[0].copy()
                    st.session_state.data = arrs[0].copy()
                    st.session_state.hdr = hdr
                    st.session_state.gps = gps
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    
    # Main interface
    if st.session_state.data is not None and st.session_state.hdr is not None:
        # Create tabs
        tab1, tab2, tab3 = st.tabs(["📊 Data Info", "⚙️ Processing", "📈 Visualization"])
        
        with tab1:
            st.header("File Information")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Key Parameters")
                hdr = st.session_state.hdr
                st.write(f"**System:** {hdr.get('system', 'N/A')}")
                st.write(f"**Antenna:** {hdr.get('antennas', ['N/A'])[0]}")
                st.write(f"**Frequency:** {hdr.get('antfreq', ['N/A'])[0]} MHz")
                st.write(f"**Samples per Trace:** {hdr.get('spp', 'N/A')}")
                st.write(f"**Number of Traces:** {hdr.get('numsamp', 'N/A')}")
                if hdr.get('created'):
                    st.write(f"**Survey Date:** {hdr.get('created').strftime('%Y-%m-%d %H:%M:%S')}")
            
            with col2:
                st.subheader("Data Statistics")
                data = st.session_state.data
                st.write(f"**Data Shape:** {data.shape}")
                st.write(f"**Min Value:** {np.nanmin(data):.2f}")
                st.write(f"**Max Value:** {np.nanmax(data):.2f}")
                st.write(f"**Mean Value:** {np.nanmean(data):.2f}")
                st.write(f"**Standard Deviation:** {np.nanstd(data):.2f}")
                
                # Data preview
                with st.expander("Raw Data Preview (first 5x5)"):
                    st.dataframe(data[:5, :5].round(2))
        
        with tab2:
            st.header("Data Processing Filters")
            
            # Reset to original button
            if st.button("Reset to Original Data", type="secondary"):
                st.session_state.data = st.session_state.original_data.copy()
                st.success("Data reset to original!")
            
            # Create filter controls
            with st.form("filter_form"):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown("#### Stacking")
                    apply_stacking = st.checkbox("Apply Stacking", value=False)
                    if apply_stacking:
                        stack_factor = st.slider("Stack Factor", 2, 10, 3, 1,
                                               help="Number of traces to average together")
                    
                    st.markdown("#### Gain")
                    apply_gain = st.checkbox("Apply Gain", value=True)
                    if apply_gain:
                        gain_value = st.slider("Gain Value", 0.1, 10.0, 1.0, 0.1,
                                             help="Multiply data by this value")
                
                with col2:
                    st.markdown("#### BGR Filter")
                    apply_bgr = st.checkbox("Apply BGR", value=False)
                    if apply_bgr:
                        bgr_type = st.selectbox("BGR Type", ["Full-width", "Moving Window"])
                        if bgr_type == "Moving Window":
                            window_size = st.slider("Window Size", 10, 200, 50, 10,
                                                  help="Size of moving window in traces")
                    
                    st.markdown("#### Reverse")
                    apply_reverse = st.checkbox("Reverse Data", value=False,
                                              help="Flip the data horizontally")
                
                with col3:
                    st.markdown("#### Frequency Filter")
                    apply_freq = st.checkbox("Apply Frequency Filter", value=False)
                    if apply_freq:
                        ant_freq = st.session_state.hdr.get('antfreq', [100])[0]
                        freq_min = st.slider("Min Freq (MHz)", 1, ant_freq-1, 
                                           max(1, int(ant_freq * 0.6)), 1)
                        freq_max = st.slider("Max Freq (MHz)", freq_min+1, 500,
                                           min(500, int(ant_freq * 1.4)), 1)
                    
                    st.markdown("#### Normalization")
                    apply_norm = st.checkbox("Normalize Data", value=True,
                                           help="Normalize to 0-1 range")
                
                # Apply filters button
                apply_filters = st.form_submit_button("Apply Filters", type="primary")
            
            # Apply filters when button is clicked
            if apply_filters:
                with st.spinner("Applying filters..."):
                    # Start with original data
                    processed_data = st.session_state.original_data.copy()
                    
                    # Apply BGR filter (background removal)
                    if apply_bgr:
                        try:
                            from readgssi.arrayops import bgr
                            if bgr_type == "Full-width":
                                processed_data = bgr(processed_data, window=0)
                            else:
                                processed_data = bgr(processed_data, window=window_size)
                            st.success(f"Applied {bgr_type} BGR filter")
                        except Exception as e:
                            st.warning(f"BGR filter not available: {str(e)}")
                    
                    # Apply frequency filter
                    if apply_freq:
                        try:
                            from readgssi.arrayops import freqfilter
                            processed_data = freqfilter(processed_data, header=hdr,
                                                      freqmin=freq_min, freqmax=freq_max)
                            st.success(f"Applied frequency filter ({freq_min}-{freq_max} MHz)")
                        except Exception as e:
                            st.warning(f"Frequency filter not available: {str(e)}")
                    
                    # Apply stacking
                    if apply_stacking:
                        try:
                            from readgssi.arrayops import stack
                            processed_data = stack(processed_data, factor=stack_factor)
                            st.success(f"Applied stacking (factor={stack_factor})")
                        except Exception as e:
                            st.warning(f"Stacking not available: {str(e)}")
                            # Simple manual stacking
                            if stack_factor > 1:
                                new_shape = (processed_data.shape[0], 
                                           processed_data.shape[1] // stack_factor, 
                                           stack_factor)
                                processed_data = processed_data[:, :new_shape[1]*stack_factor]
                                processed_data = processed_data.reshape(new_shape).mean(axis=2)
                                st.success(f"Applied manual stacking (factor={stack_factor})")
                    
                    # Apply gain
                    if apply_gain:
                        processed_data = processed_data * gain_value
                        st.success(f"Applied gain ({gain_value}x)")
                    
                    # Apply reverse
                    if apply_reverse:
                        processed_data = np.fliplr(processed_data)
                        st.success("Reversed data")
                    
                    # Apply normalization
                    if apply_norm:
                        data_min = np.nanmin(processed_data)
                        data_max = np.nanmax(processed_data)
                        if data_max > data_min:
                            processed_data = (processed_data - data_min) / (data_max - data_min)
                            st.success("Normalized data")
                    
                    # Update session state
                    st.session_state.data = processed_data
                    
                    # Show statistics
                    st.info(f"""
                    **New Statistics:**
                    - Shape: {processed_data.shape}
                    - Min: {np.nanmin(processed_data):.4f}
                    - Max: {np.nanmax(processed_data):.4f}
                    - Mean: {np.nanmean(processed_data):.4f}
                    """)
        
        with tab3:
            st.header("Matplotlib Visualization")
            
            # Visualization settings
            with st.expander("Plot Settings", expanded=True):
                col1, col2 = st.columns(2)
                
                with col1:
                    colormap = st.selectbox("Colormap", 
                                          ["gray", "seismic", "RdBu", "viridis", 
                                           "plasma", "hot", "coolwarm", "jet"],
                                          index=0)
                    
                    aspect_ratio = st.selectbox("Aspect Ratio",
                                              ["auto", "equal", "0.5", "1", "2"],
                                              index=0)
                    
                    # Parse aspect ratio
                    if aspect_ratio == "auto":
                        aspect = "auto"
                    elif aspect_ratio == "equal":
                        aspect = "equal"
                    else:
                        try:
                            aspect = float(aspect_ratio)
                        except:
                            aspect = "auto"
                
                with col2:
                    interpolation = st.selectbox("Interpolation",
                                               ["nearest", "bilinear", "bicubic", 
                                                "spline16", "spline36", "hanning"],
                                               index=0)
                    
                    dpi = st.slider("DPI", 72, 600, 150, 50)
                    
                    show_colorbar = st.checkbox("Show Colorbar", value=True)
                    show_grid = st.checkbox("Show Grid", value=False)
            
            # Plot button
            if st.button("Generate Plot with Matplotlib", type="primary", use_container_width=True):
                with st.spinner("Creating plot..."):
                    # Create temporary file for plot
                    plot_temp = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
                    plot_path = plot_temp.name
                    plot_temp.close()
                    
                    try:
                        # Get current data
                        plot_data = st.session_state.data
                        
                        # Create figure
                        fig, ax = plt.subplots(figsize=(12, 8))
                        
                        # Plot the radar data
                        im = ax.imshow(plot_data,
                                     aspect=aspect,
                                     cmap=colormap,
                                     interpolation=interpolation,
                                     origin='upper')
                        
                        # Add labels
                        ax.set_xlabel("Trace Number", fontsize=12)
                        
                        # Calculate and set Y-axis label
                        hdr = st.session_state.hdr
                        if 'spp' in hdr and 'antfreq' in hdr:
                            spp = hdr['spp']
                            freq = hdr['antfreq']
                            if isinstance(freq, list):
                                freq = freq[0]
                            
                            if freq > 0:
                                # Calculate time per sample (nanoseconds)
                                time_per_sample = (1 / (freq * 1e6)) * 1e9  # ns
                                total_time = spp * time_per_sample
                                
                                # Create time axis
                                y_ticks = np.linspace(0, spp, 6)
                                y_labels = [f"{t * time_per_sample:.0f}" for t in y_ticks]
                                
                                ax.set_yticks(y_ticks)
                                ax.set_yticklabels(y_labels)
                                ax.set_ylabel(f"Two-way Time (ns) - {freq} MHz", fontsize=12)
                            else:
                                ax.set_ylabel("Sample Number", fontsize=12)
                        else:
                            ax.set_ylabel("Sample Number", fontsize=12)
                        
                        # Add title
                        title = f"Radargram: {uploaded_file.name}"
                        if apply_gain:
                            title += f" (Gain: {gain_value}x)"
                        ax.set_title(title, fontsize=14, fontweight='bold')
                        
                        # Add colorbar
                        if show_colorbar:
                            cbar = plt.colorbar(im, ax=ax, pad=0.01)
                            cbar.set_label('Amplitude', fontsize=11)
                        
                        # Add grid
                        if show_grid:
                            ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
                        
                        # Adjust layout
                        plt.tight_layout()
                        
                        # Save figure
                        fig.savefig(plot_path, dpi=dpi, bbox_inches='tight')
                        plt.close(fig)
                        
                        # Display plot
                        st.image(plot_path, use_container_width=True)
                        
                        # Download button
                        with open(plot_path, "rb") as f:
                            st.download_button(
                                label="Download Plot",
                                data=f,
                                file_name=f"{uploaded_file.name.replace('.DZT', '')}_matplotlib.png",
                                mime="image/png",
                                use_container_width=True
                            )
                        
                        st.success("Plot generated successfully!")
                        
                        # Show data info
                        with st.expander("Plot Information"):
                            st.write(f"**Data Shape:** {plot_data.shape}")
                            st.write(f"**Color Range:** {np.nanmin(plot_data):.4f} to {np.nanmax(plot_data):.4f}")
                            st.write(f"**Colormap:** {colormap}")
                            st.write(f"**Interpolation:** {interpolation}")
                        
                    except Exception as e:
                        st.error(f"Plot generation error: {str(e)}")
                        with st.expander("Error Details"):
                            st.code(traceback.format_exc())
                    finally:
                        if os.path.exists(plot_path):
                            os.unlink(plot_path)
            
            # Quick plot without settings
            if st.button("Quick Plot (Default Settings)", type="secondary"):
                with st.spinner("Creating quick plot..."):
                    plot_temp = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
                    plot_path = plot_temp.name
                    plot_temp.close()
                    
                    try:
                        fig, ax = plt.subplots(figsize=(10, 6))
                        im = ax.imshow(st.session_state.data, 
                                     aspect='auto', 
                                     cmap='gray',
                                     interpolation='nearest')
                        ax.set_xlabel("Trace")
                        ax.set_ylabel("Sample")
                        ax.set_title(f"Quick Plot: {uploaded_file.name}")
                        plt.colorbar(im, ax=ax)
                        plt.tight_layout()
                        fig.savefig(plot_path, dpi=150)
                        st.image(plot_path, use_container_width=True)
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
                    finally:
                        if os.path.exists(plot_path):
                            os.unlink(plot_path)
        
        # Show GPS data if available
        if hasattr(st.session_state, 'gps') and st.session_state.gps is not False:
            with st.expander("GPS Data", expanded=False):
                st.write(f"GPS points: {len(st.session_state.gps)}")
                st.dataframe(st.session_state.gps.head())
    
    else:
        st.error("Failed to load data. Please check the file format.")
else:
    # Welcome screen
    st.info("👆 Please upload a GSSI DZT file to begin.")
    
    st.markdown("""
    ### How to use this app:
    
    1. **Upload** a GSSI `.DZT` file
    2. **View** file information and statistics in the first tab
    3. **Apply filters** in the second tab (BGR, frequency, stacking, gain, etc.)
    4. **Visualize** the processed data with matplotlib in the third tab
    
    ### Available Filters:
    
    - **Stacking**: Average neighboring traces to reduce noise
    - **BGR**: Remove horizontal banding noise (background removal)
    - **Frequency Filter**: Bandpass filter around antenna frequency
    - **Gain**: Amplify signal strength
    - **Reverse**: Flip data horizontally
    - **Normalization**: Scale data to 0-1 range
    
    ### Features:
    
    - Full control over matplotlib plotting parameters
    - Real-time filter application
    - Data statistics display
    - High-resolution export
    - Multiple colormap options
    """)
