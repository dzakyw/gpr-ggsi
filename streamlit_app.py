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
            st.header("Radargram Plot")
            # Create a form for plot parameters
            with st.form("plot_form"):
                st.subheader("Plot Settings")
                col_a, col_b = st.columns(2)
                with col_a:
                    gain = st.slider("Gain", 0.1, 5.0, 1.0, 0.1)
                    colormap = st.selectbox("Colormap", ["gray", "seismic", "RdBu", "viridis", "plasma"], index=0)
                    plot_method = st.selectbox("Plot Method", ["Simple Matplotlib", "readgssi radargram"], index=0)
                
                with col_b:
                    if plot_method == "Simple Matplotlib":
                        normalize = st.checkbox("Normalize data", value=True)
                        aspect_ratio = st.selectbox("Aspect Ratio", ["auto", "equal"], index=0)
                    else:
                        x_units = st.selectbox("X-axis units", ["seconds", "distance", "traces"], index=0)
                        z_units = st.selectbox("Z-axis units", ["nanoseconds", "depth", "samples"], index=0)
                
                plot_button = st.form_submit_button("Generate Radargram")

            if plot_button and len(arrs) > 0:
                with st.spinner('Generating radargram...'):
                    # Create a temporary file for the plot image
                    plot_temp = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
                    plot_path = plot_temp.name
                    plot_temp.close()
                    
                    try:
                        if plot_method == "readgssi radargram":
                            # Try the readgssi radargram function with safer parameters
                            from readgssi.plot import radargram
                            import matplotlib
                            matplotlib.use('Agg')
                            import matplotlib.pyplot as plt
                            
                            # Try with a simpler parameter set
                            radargram(
                                ar=arrs[0],
                                ant=0,
                                header=hdr,
                                freq=hdr.get('antfreq', [100])[0],
                                figsize=(10, 6),
                                gain=gain,
                                stack=1,
                                x=x_units,
                                z=z_units,
                                title=f"Radargram: {uploaded_file.name}",
                                colormap=colormap,
                                colorbar=True,
                                absval=False,
                                noshow=True,
                                outfile=plot_path.replace('.png', ''),
                                fmt='png',
                                dpi=150,
                                verbose=False
                            )
                        else:
                            # Use simple matplotlib plotting as fallback
                            import matplotlib
                            matplotlib.use('Agg')
                            import matplotlib.pyplot as plt
                            
                            # Get the data array
                            data = arrs[0].astype(np.float32)
                            
                            # Apply gain
                            if gain != 1.0:
                                data = data * gain
                            
                            # Normalize if requested
                            if normalize:
                                data_min = np.nanmin(data)
                                data_max = np.nanmax(data)
                                if data_max > data_min:
                                    data = (data - data_min) / (data_max - data_min)
                            
                            fig, ax = plt.subplots(figsize=(10, 6))
                            
                            # Plot the radargram
                            im = ax.imshow(data, 
                                          aspect=aspect_ratio, 
                                          cmap=colormap,
                                          interpolation='nearest')
                            
                            # Add colorbar
                            cbar = plt.colorbar(im, ax=ax)
                            cbar.set_label('Amplitude')
                            
                            # Set labels based on header information
                            ax.set_xlabel('Trace Number')
                            
                            # Calculate depth/time axis if possible
                            if 'spp' in hdr and 'antfreq' in hdr:
                                samples = hdr['spp']
                                freq = hdr['antfreq'][0] if isinstance(hdr['antfreq'], list) else hdr['antfreq']
                                if freq > 0:
                                    # Convert samples to time (nanoseconds)
                                    time_ns = np.arange(samples) * (1 / (freq * 1e6)) * 1e9
                                    ax.set_ylabel(f'Time (ns) - {freq} MHz Antenna')
                            else:
                                ax.set_ylabel('Sample Number')
                            
                            ax.set_title(f"Radargram: {uploaded_file.name}")
                            plt.tight_layout()
                            
                            # Save the figure
                            fig.savefig(plot_path, dpi=150, bbox_inches='tight')
                            plt.close(fig)
                        
                        # Display the plot in Streamlit
                        st.image(plot_path, use_column_width=True)
                        st.success("Radargram generated successfully!")
                        
                        # Offer download of the plot
                        with open(plot_path, "rb") as file:
                            btn = st.download_button(
                                label="Download Radargram (PNG)",
                                data=file,
                                file_name=f"{uploaded_file.name.replace('.DZT', '')}_radargram.png",
                                mime="image/png"
                            )
                        
                    except Exception as plot_error:
                        st.error(f"Plot generation error: {str(plot_error)}")
                        
                        # Show simplified traceback for debugging
                        with st.expander("Technical Details (for debugging)"):
                            st.code(traceback.format_exc())
                        
                        # Offer a basic plot as fallback
                        if st.button("Generate Basic Plot Instead"):
                            import matplotlib
                            matplotlib.use('Agg')
                            import matplotlib.pyplot as plt
                            
                            fig, ax = plt.subplots(figsize=(10, 6))
                            ax.imshow(arrs[0], aspect='auto', cmap=colormap)
                            ax.set_title(f"Basic Plot: {uploaded_file.name}")
                            ax.set_xlabel('Trace')
                            ax.set_ylabel('Sample')
                            plt.tight_layout()
                            fig.savefig(plot_path, dpi=150)
                            st.image(plot_path, use_column_width=True)
                    
                    finally:
                        # Clean up temporary plot file
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
