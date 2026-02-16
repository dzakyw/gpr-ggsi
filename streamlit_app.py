# -*- coding: utf-8 -*-
"""
GPR Data Processor with Advanced Deconvolution
Streamlit application for processing GPR data (DZT files) with deconvolution,
coordinate import, muting, and advanced visualization.

Author: Original by asus, rewritten for clarity
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import tempfile
import os
import json
import warnings
from scipy import signal
from scipy.fft import fft, fftfreq
from scipy.interpolate import interp1d
from scipy.linalg import toeplitz, solve_toeplitz
from scipy.optimize import minimize
from scipy.stats import norm
from pathlib import Path
from readgssi import readgssi

warnings.filterwarnings('ignore')

# ------------------------------------------------------------------------------
# Page configuration
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="GPR Data Processor",
    page_icon="📡",
    layout="wide"
)

st.title("📡 GPR Data Processor with Deconvolution")
st.markdown("Process GPR data with advanced deconvolution, coordinate import, and trace muting")

# ------------------------------------------------------------------------------
# Custom CSS styling
# ------------------------------------------------------------------------------
st.markdown("""
<style>
    .stProgress > div > div > div > div { background-color: #4CAF50; }
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
    .coordinate-box {
        background-color: #e8f5e9;
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
        border-left: 4px solid #4CAF50;
    }
    .near-surface-box {
        background-color: #fff3e0;
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
        border-left: 4px solid #FF9800;
    }
    .mute-box {
        background-color: #fff0f0;
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
        border-left: 4px solid #FF5252;
    }
    .deconv-box {
        background-color: #f3e5f5;
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
        border-left: 4px solid #9C27B0;
    }
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# Initialize session state
# ------------------------------------------------------------------------------
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'original_array' not in st.session_state:
    st.session_state.original_array = None
if 'processed_array' not in st.session_state:
    st.session_state.processed_array = None
if 'deconvolved_array' not in st.session_state:
    st.session_state.deconvolved_array = None
if 'coordinates_df' not in st.session_state:
    st.session_state.coordinates_df = None
if 'interpolated_coords' not in st.session_state:
    st.session_state.interpolated_coords = None
if 'pole_data' not in st.session_state:
    st.session_state.pole_data = None
if 'header' not in st.session_state:
    st.session_state.header = None
if 'gps' not in st.session_state:
    st.session_state.gps = None
if 'estimated_wavelet' not in st.session_state:
    st.session_state.estimated_wavelet = None
if 'mute_mask' not in st.session_state:
    st.session_state.mute_mask = None
if 'mute_zones' not in st.session_state:
    st.session_state.mute_zones = None

# ------------------------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------------------------

def calculate_fft(trace, sampling_rate=1000):
    """Compute FFT magnitude spectrum."""
    n = len(trace)
    yf = fft(trace)
    xf = fftfreq(n, 1/sampling_rate)[:n//2]
    magnitude = 2.0/n * np.abs(yf[:n//2])
    return xf, magnitude


def estimate_wavelet(trace, wavelet_length=51):
    """Estimate wavelet from trace using autocorrelation."""
    autocorr = np.correlate(trace, trace, mode='full')
    autocorr = autocorr[len(autocorr)//2:]
    autocorr = autocorr / autocorr[0]
    wavelet = autocorr[:wavelet_length]
    return wavelet


def wiener_deconvolution(trace, wavelet, noise_level=0.01, regularization=0.1):
    """Wiener deconvolution using Toeplitz system."""
    n = len(trace)
    m = len(wavelet)
    col = np.zeros(n)
    col[:m] = wavelet
    row = np.zeros(n)
    row[0] = wavelet[0]
    H = toeplitz(col, row)
    R = regularization * np.eye(n)
    try:
        HTy = H.T @ trace
        # solve (H^T H + λI) x = H^T y
        result = np.linalg.lstsq(H.T @ H + R, HTy, rcond=None)[0]
    except:
        result = trace  # fallback
    return result[:n]


def predictive_deconvolution(trace, prediction_distance=10, filter_length=50,
                             prewhitening=0.1, iterations=3):
    """Predictive deconvolution."""
    n = len(trace)
    result = trace.copy()
    for _ in range(iterations):
        autocorr = np.correlate(result, result, mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        autocorr[0] *= (1 + prewhitening)
        try:
            from scipy.linalg import solve_toeplitz
            r = autocorr[:filter_length]
            b = autocorr[prediction_distance:prediction_distance+filter_length]
            prediction_filter = solve_toeplitz((r, r), b)
        except:
            R = toeplitz(autocorr[:filter_length])
            b = autocorr[prediction_distance:prediction_distance+filter_length]
            prediction_filter = np.linalg.lstsq(R, b, rcond=None)[0]
        predicted = np.convolve(result, prediction_filter, mode='same')
        result = result - predicted
    return result


def spiking_deconvolution(trace, desired_spike=0.8, spike_length=21,
                          noise_level=0.01, iterations=5):
    """Spiking deconvolution (Wiener shaping filter)."""
    n = len(trace)
    desired = np.zeros(n)
    desired[spike_length//2] = desired_spike
    autocorr = np.correlate(trace, trace, mode='full')
    autocorr = autocorr[len(autocorr)//2:]
    autocorr[0] *= (1 + noise_level)
    crosscorr = np.correlate(trace, desired, mode='full')
    crosscorr = crosscorr[len(crosscorr)//2:]
    flen = min(100, n//2)
    R = toeplitz(autocorr[:flen])
    R += noise_level * np.eye(flen)
    P = crosscorr[:flen]
    try:
        inv_filter = np.linalg.solve(R, P)
    except:
        inv_filter = np.linalg.lstsq(R, P, rcond=None)[0]
    deconv = np.convolve(trace, inv_filter, mode='same')
    return deconv


def minimum_entropy_deconvolution(trace, filter_length=80, iterations=10,
                                  convergence=0.001, noise_estimate=0.01):
    """Minimum Entropy Deconvolution (MED)."""
    n = len(trace)
    h = np.zeros(filter_length)
    h[filter_length//2] = 1.0
    h_prev = h.copy()
    for _ in range(iterations):
        y = np.convolve(trace, h, mode='same')
        X = np.zeros((n, filter_length))
        for i in range(filter_length):
            X[:, i] = np.roll(trace, i - filter_length//2)[:n]
        y3 = y**3
        gradient = X.T @ y3 / (np.sum(y**4) + 1e-10)
        h = gradient / (np.linalg.norm(gradient) + 1e-10)
        if np.linalg.norm(h - h_prev) < convergence:
            break
        h_prev = h.copy()
    result = np.convolve(trace, h, mode='same')
    return result


def homomorphic_deconvolution(trace, window_type='hanning', cutoff=0.1,
                              prewhitening=0.01, iterations=3):
    """Homomorphic deconvolution using cepstral liftering."""
    n = len(trace)
    # ensure positivity
    trace_min = np.min(trace)
    if trace_min <= 0:
        trace = trace - trace_min + 0.001 * np.std(trace)
    result = trace.copy()
    for _ in range(iterations):
        spectrum = np.fft.fft(result)
        log_spec = np.log(np.abs(spectrum) + prewhitening)
        cepstrum = np.fft.ifft(log_spec).real
        # liftering window
        if window_type == 'hanning':
            window = np.hanning(n)
        elif window_type == 'hamming':
            window = np.hamming(n)
        elif window_type == 'blackman':
            window = np.blackman(n)
        else:  # bartlett
            window = np.bartlett(n)
        cutoff_idx = int(cutoff * n)
        window[:cutoff_idx] = 1
        window[-cutoff_idx:] = 1
        window[cutoff_idx:-cutoff_idx] = 0
        filtered_cep = cepstrum * window
        filtered_log_spec = np.fft.fft(filtered_cep)
        wavelet_spec = np.exp(filtered_log_spec)
        deconv_spec = spectrum / (wavelet_spec + prewhitening)
        result = np.fft.ifft(deconv_spec).real
    return result


def bayesian_deconvolution(trace, prior='Laplace', iterations=1000,
                           burnin=500, noise_std=0.01):
    """Simplified Bayesian deconvolution (Wiener for Gaussian, L1 for Laplace)."""
    if prior == 'Gaussian':
        wavelet = estimate_wavelet(trace)
        return wiener_deconvolution(trace, wavelet, noise_std, 0.1)
    else:
        # fallback to spiking deconvolution for Laplace/Jeffreys
        return spiking_deconvolution(trace, noise_level=noise_std)


def apply_deconvolution_to_array(array, method='Wiener Filter', **kwargs):
    """Apply chosen deconvolution to entire 2D array."""
    n_samples, n_traces = array.shape
    deconvolved = np.zeros_like(array)

    start = kwargs.get('deconv_window_start', 0)
    end = kwargs.get('deconv_window_end', n_samples)
    start = max(0, min(start, n_samples-1))
    end = max(0, min(end, n_samples-1))

    # wavelet estimation
    trace_for_wavelet = kwargs.get('trace_for_wavelet', 0)
    use_average = kwargs.get('use_average_wavelet', True)
    wavelet_trace_range = kwargs.get('wavelet_trace_range', 10)
    wavelet_length = kwargs.get('wavelet_length', 51)

    if use_average and wavelet_trace_range > 1:
        s = max(0, trace_for_wavelet - wavelet_trace_range//2)
        e = min(n_traces, trace_for_wavelet + wavelet_trace_range//2)
        avg_trace = np.mean(array[:, s:e], axis=1)
        wavelet = estimate_wavelet(avg_trace, wavelet_length)
    else:
        tidx = min(max(0, trace_for_wavelet), n_traces-1)
        wavelet = estimate_wavelet(array[:, tidx], wavelet_length)

    st.session_state.estimated_wavelet = wavelet

    for i in range(n_traces):
        trace = array[:, i].copy()
        if method == "Wiener Filter":
            deconv_trace = wiener_deconvolution(
                trace, wavelet,
                noise_level=kwargs.get('noise_level', 0.01),
                regularization=kwargs.get('regularization', 0.1)
            )
        elif method == "Predictive Deconvolution":
            deconv_trace = predictive_deconvolution(
                trace,
                prediction_distance=kwargs.get('prediction_distance', 10),
                filter_length=kwargs.get('filter_length', 50),
                prewhitening=kwargs.get('prewhitening', 0.1)/100,
                iterations=kwargs.get('iterations', 3)
            )
        elif method == "Spiking Deconvolution":
            deconv_trace = spiking_deconvolution(
                trace,
                desired_spike=kwargs.get('spike_strength', 0.8),
                spike_length=kwargs.get('spike_length', 21),
                noise_level=kwargs.get('spike_noise', 0.01),
                iterations=kwargs.get('spike_iterations', 5)
            )
        elif method == "Minimum Entropy Deconvolution":
            deconv_trace = minimum_entropy_deconvolution(
                trace,
                filter_length=kwargs.get('med_filter_length', 80),
                iterations=kwargs.get('med_iterations', 10),
                convergence=kwargs.get('med_convergence', 0.001),
                noise_estimate=kwargs.get('med_noise', 0.01)
            )
        elif method == "Homomorphic Deconvolution":
            deconv_trace = homomorphic_deconvolution(
                trace,
                window_type=kwargs.get('homo_window', 'hanning'),
                cutoff=kwargs.get('homo_cutoff', 0.1),
                prewhitening=kwargs.get('homo_prewhiten', 0.01),
                iterations=kwargs.get('homo_iterations', 3)
            )
        elif method == "Bayesian Deconvolution":
            deconv_trace = bayesian_deconvolution(
                trace,
                prior=kwargs.get('bayesian_prior', 'Laplace'),
                iterations=kwargs.get('bayesian_iterations', 1000),
                burnin=kwargs.get('bayesian_burnin', 500),
                noise_std=kwargs.get('bayesian_noise', 0.01)
            )
        else:
            deconv_trace = trace

        # apply only within window with edge blending
        if start > 0 or end < n_samples:
            deconvolved[start:end, i] = deconv_trace[start:end]
            # blend start edge
            if start > 0:
                blend_len = min(50, start)
                alpha = np.linspace(0, 1, blend_len)
                deconvolved[start-blend_len:start, i] = (
                    (1-alpha) * trace[start-blend_len:start] +
                    alpha * deconv_trace[start-blend_len:start]
                )
            # blend end edge
            if end < n_samples:
                blend_len = min(50, n_samples - end)
                alpha = np.linspace(1, 0, blend_len)
                deconvolved[end:end+blend_len, i] = (
                    alpha * deconv_trace[end:end+blend_len] +
                    (1-alpha) * trace[end:end+blend_len]
                )
        else:
            deconvolved[:, i] = deconv_trace
    return deconvolved


def apply_gain(array, gain_type, **kwargs):
    """Apply time-varying gain."""
    n_samples, n_traces = array.shape
    if gain_type == "Constant":
        gain = 1 + kwargs.get('const_gain', 100) / 100
        return array * gain
    elif gain_type == "Linear":
        min_g = 1 + kwargs.get('min_gain', 50) / 100
        max_g = 1 + kwargs.get('max_gain', 500) / 100
        gain_vec = np.linspace(min_g, max_g, n_samples)
        return array * gain_vec[:, np.newaxis]
    elif gain_type == "Exponential":
        base = 1 + kwargs.get('base_gain', 100) / 100
        expf = kwargs.get('exp_factor', 1.5)
        t = np.linspace(0, 1, n_samples)
        gain_vec = base * np.exp(expf * t)
        return array * gain_vec[:, np.newaxis]
    elif gain_type == "AGC (Automatic Gain Control)":
        window = kwargs.get('window_size', 100)
        target = kwargs.get('target_amplitude', 0.3)
        result = np.zeros_like(array)
        half = window // 2
        for i in range(n_traces):
            tr = array[:, i]
            agc = np.zeros(n_samples)
            for j in range(n_samples):
                s = max(0, j - half)
                e = min(n_samples, j + half + 1)
                rms = np.sqrt(np.mean(tr[s:e]**2))
                if rms > 0:
                    agc[j] = tr[j] * (target / rms)
                else:
                    agc[j] = tr[j]
            result[:, i] = agc
        return result
    elif gain_type == "Spherical":
        power = kwargs.get('power_gain', 2.0)
        atten = kwargs.get('attenuation', 0.05)
        t = np.arange(n_samples) / n_samples
        gain_vec = (1 + atten * t) ** power
        return array * gain_vec[:, np.newaxis]
    return array


def apply_near_surface_correction(array, correction_type, correction_depth,
                                   max_depth, **kwargs):
    """Reduce amplitudes in shallow region."""
    n_samples, n_traces = array.shape
    if max_depth is not None:
        corr_samples = int((correction_depth / max_depth) * n_samples)
    else:
        corr_samples = int(0.1 * n_samples)
    corr_samples = max(1, min(corr_samples, n_samples))
    result = array.copy()

    if correction_type == "Linear Reduction":
        surface_red = kwargs.get('surface_reduction', 80) / 100.0
        depth_factor = kwargs.get('depth_factor', 1.0)
        red_vec = np.ones(n_samples)
        ratios = np.linspace(0, 1, corr_samples)
        red_vec[:corr_samples] = 1 - surface_red * (1 - ratios**depth_factor)
        result = result * red_vec[:, np.newaxis]

    elif correction_type == "Exponential Reduction":
        expf = kwargs.get('exp_factor', 2.0)
        max_red = kwargs.get('max_reduction', 90) / 100.0
        red_vec = np.ones(n_samples)
        ratios = np.linspace(0, 1, corr_samples)
        red_vec[:corr_samples] = 1 - max_red * np.exp(-expf * ratios)
        result = result * red_vec[:, np.newaxis]

    elif correction_type == "Gaussian Filter":
        sigma = kwargs.get('filter_sigma', 1.0)
        window = kwargs.get('filter_window', 21)
        from scipy.ndimage import gaussian_filter1d
        near_surface = array[:corr_samples, :]
        filt = gaussian_filter1d(near_surface, sigma=sigma, axis=0, mode='reflect')
        alpha = np.linspace(1.0, 0.0, corr_samples)[:, np.newaxis]
        result[:corr_samples, :] = near_surface * alpha + filt * (1 - alpha)

    elif correction_type == "Windowed Normalization":
        win_size = kwargs.get('window_size', 50)
        target = kwargs.get('target_amplitude', 0.3)
        half = win_size // 2
        for i in range(n_traces):
            tr = result[:corr_samples, i]
            norm_tr = np.zeros_like(tr)
            for j in range(len(tr)):
                s = max(0, j - half)
                e = min(len(tr), j + half + 1)
                rms = np.sqrt(np.mean(tr[s:e]**2))
                depth_fac = 1.0 - (j / len(tr))
                if rms > 0:
                    norm_tr[j] = tr[j] * (target / rms) * depth_fac
                else:
                    norm_tr[j] = tr[j]
            result[:corr_samples, i] = norm_tr
    return result


def reverse_array(array):
    """Reverse traces (flip line direction)."""
    return array[:, ::-1]


def apply_trace_mute(array, mute_params, x_axis=None):
    """Mute a single zone."""
    n_samples, n_traces = array.shape
    muted = array.copy()
    mask = np.zeros_like(array, dtype=bool)

    method = mute_params['method']
    start = mute_params['start']
    end = mute_params['end']
    strength = mute_params.get('strength', 100) / 100.0
    apply_taper = mute_params.get('apply_taper', False)

    # determine indices
    if method == "By Distance":
        if x_axis is None:
            return array, mask
        start_idx = np.argmin(np.abs(x_axis - start))
        end_idx = np.argmin(np.abs(x_axis - end))
    else:  # By Trace Index
        start_idx = int(start)
        end_idx = int(end)
    start_idx = max(0, min(start_idx, n_traces-1))
    end_idx = max(0, min(end_idx, n_traces-1))
    if start_idx > end_idx:
        start_idx, end_idx = end_idx, start_idx

    if apply_taper:
        taper_len = int((end_idx - start_idx) * mute_params.get('taper_length', 0.1))
        taper_len = max(1, min(taper_len, (end_idx - start_idx)//2))
        taper_left = np.linspace(1, 0, taper_len)
        taper_right = np.linspace(0, 1, taper_len)

        # middle section full mute
        mid_start = start_idx + taper_len
        mid_end = end_idx - taper_len
        if mid_start < mid_end:
            muted[:, mid_start:mid_end] *= (1 - strength)
            mask[:, mid_start:mid_end] = True

        # tapered edges
        for k in range(taper_len):
            fac_left = 1 - strength * taper_left[k]
            muted[:, start_idx + k] *= fac_left
            mask[:, start_idx + k] = taper_left[k] > 0.5

            fac_right = 1 - strength * taper_right[k]
            muted[:, end_idx - taper_len + k] *= fac_right
            mask[:, end_idx - taper_len + k] = taper_right[k] > 0.5
    else:
        muted[:, start_idx:end_idx] *= (1 - strength)
        mask[:, start_idx:end_idx] = True

    return muted, mask


def apply_multiple_mute_zones(array, mute_zones, x_axis=None):
    """Apply multiple mute zones."""
    muted = array.copy()
    combined_mask = np.zeros_like(array, dtype=bool)
    for zone in mute_zones:
        zone_muted, zone_mask = apply_trace_mute(muted, zone, x_axis)
        # combine: for overlapping, we take the one with stronger muting (min amplitude)
        muted = np.minimum(muted, zone_muted)
        combined_mask = combined_mask | zone_mask
    return muted, combined_mask


def process_coordinates(coords_df, n_traces, trace_col=None, method='linear'):
    """Interpolate coordinates to match number of traces."""
    # column guessing
    east_col = next((c for c in coords_df.columns if 'east' in c.lower()), None)
    north_col = next((c for c in coords_df.columns if 'north' in c.lower()), None)
    elev_col = next((c for c in coords_df.columns if 'elev' in c.lower()), None)
    if east_col is None or north_col is None or elev_col is None:
        st.error("CSV must contain Easting, Northing, Elevation columns.")
        return None

    east = coords_df[east_col].values
    north = coords_df[north_col].values
    elev = coords_df[elev_col].values

    if trace_col and trace_col in coords_df.columns:
        trace_indices = coords_df[trace_col].values
    else:
        # compute cumulative distance along coordinate points
        dx = np.diff(east)
        dy = np.diff(north)
        dist = np.sqrt(dx**2 + dy**2)
        cumdist = np.concatenate(([0], np.cumsum(dist)))
        trace_indices = np.linspace(0, n_traces-1, len(cumdist))

    target = np.arange(n_traces)
    kind_map = {'linear': 'linear', 'cubic': 'cubic', 'nearest': 'nearest',
                'previous': 'previous', 'next': 'next'}
    kind = kind_map.get(method.lower(), 'linear')

    try:
        f_east = interp1d(trace_indices, east, kind=kind, fill_value='extrapolate')
        f_north = interp1d(trace_indices, north, kind=kind, fill_value='extrapolate')
        f_elev = interp1d(trace_indices, elev, kind=kind, fill_value='extrapolate')
        east_interp = f_east(target)
        north_interp = f_north(target)
        elev_interp = f_elev(target)

        # recompute cumulative distance along interpolated path
        dx = np.diff(east_interp)
        dy = np.diff(north_interp)
        dist_step = np.sqrt(dx**2 + dy**2)
        cumdist = np.concatenate(([0], np.cumsum(dist_step)))

        return {
            'easting': east_interp,
            'northing': north_interp,
            'elevation': elev_interp,
            'distance': cumdist,
            'trace_indices': target,
            'original_points': len(east),
            'interpolated_points': n_traces
        }
    except Exception as e:
        st.error(f"Coordinate interpolation failed: {e}")
        return None


def scale_axes(array_shape, depth_unit, max_depth, distance_unit, total_distance, coordinates=None):
    """Create x and y axis arrays with labels."""
    n_samples, n_traces = array_shape
    if depth_unit == "samples":
        y_axis = np.arange(n_samples)
        y_label = "Sample Number"
    elif depth_unit == "meters":
        y_axis = np.linspace(0, max_depth, n_samples)
        y_label = f"Depth ({depth_unit})"
    elif depth_unit == "nanoseconds":
        y_axis = np.linspace(0, max_depth, n_samples)
        y_label = "Two-way Time (ns)"
    elif depth_unit == "feet":
        y_axis = np.linspace(0, max_depth, n_samples)
        y_label = f"Depth ({depth_unit})"
    else:
        y_axis = np.arange(n_samples)
        y_label = "Sample Number"

    if coordinates is not None:
        x_axis = coordinates['distance']
        x_label = "Distance along profile (m)"
        distance_unit = "meters"
        total_distance = x_axis[-1]
    elif distance_unit == "traces":
        x_axis = np.arange(n_traces)
        x_label = "Trace Number"
    elif distance_unit == "meters":
        x_axis = np.linspace(0, total_distance, n_traces)
        x_label = f"Distance ({distance_unit})"
    elif distance_unit == "feet":
        x_axis = np.linspace(0, total_distance, n_traces)
        x_label = f"Distance ({distance_unit})"
    elif distance_unit == "kilometers":
        x_axis = np.linspace(0, total_distance, n_traces)
        x_label = f"Distance ({distance_unit})"
    else:
        x_axis = np.arange(n_traces)
        x_label = "Trace Number"

    return x_axis, y_axis, x_label, y_label


def get_aspect_ratio(mode, manual_ratio=None, data_shape=None):
    """Return aspect ratio string or float."""
    if mode == "Auto":
        return "auto"
    elif mode == "Equal":
        return "equal"
    elif mode in ("Manual", "Realistic") and manual_ratio is not None:
        return manual_ratio
    elif data_shape is not None:
        return data_shape[0] / data_shape[1] * 0.5
    else:
        return "auto"


def get_window_indices(x_axis, y_axis, depth_min, depth_max, distance_min, distance_max):
    """Convert user-specified window limits to array indices."""
    depth_min_idx = np.argmin(np.abs(y_axis - depth_min))
    depth_max_idx = np.argmin(np.abs(y_axis - depth_max))
    if depth_min_idx > depth_max_idx:
        depth_min_idx, depth_max_idx = depth_max_idx, depth_min_idx

    dist_min_idx = np.argmin(np.abs(x_axis - distance_min))
    dist_max_idx = np.argmin(np.abs(x_axis - distance_max))
    if dist_min_idx > dist_max_idx:
        dist_min_idx, dist_max_idx = dist_max_idx, dist_min_idx

    return {
        'depth_min_idx': depth_min_idx,
        'depth_max_idx': depth_max_idx,
        'dist_min_idx': dist_min_idx,
        'dist_max_idx': dist_max_idx,
        'depth_min_val': y_axis[depth_min_idx],
        'depth_max_val': y_axis[depth_max_idx],
        'dist_min_val': x_axis[dist_min_idx],
        'dist_max_val': x_axis[dist_max_idx]
    }


def process_pole_csv(pole_file, interp_coords, max_dist_threshold=10.0):
    """Process electric pole CSV and project onto GPR line."""
    try:
        pole_df = pd.read_csv(pole_file)
        # guess columns
        east_col = next((c for c in pole_df.columns if 'east' in c.lower()), None)
        north_col = next((c for c in pole_df.columns if 'north' in c.lower()), None)
        name_col = next((c for c in pole_df.columns if 'name' in c.lower()), None)
        if east_col is None or north_col is None or name_col is None:
            st.error("Pole CSV must contain Easting, Northing, Name columns.")
            return None

        easting = pole_df[east_col].values
        northing = pole_df[north_col].values
        names = pole_df[name_col].values.astype(str)

        gpr_east = interp_coords['easting']
        gpr_north = interp_coords['northing']
        gpr_dist = interp_coords['distance']

        proj_dists = []
        min_dists = []
        for e, n in zip(easting, northing):
            d = np.sqrt((gpr_east - e)**2 + (gpr_north - n)**2)
            min_idx = np.argmin(d)
            proj_dists.append(gpr_dist[min_idx])
            min_dists.append(d[min_idx])

        # filter within threshold
        filtered = [i for i, d in enumerate(min_dists) if d <= max_dist_threshold]
        if filtered:
            return {
                'easting': easting[filtered],
                'northing': northing[filtered],
                'names': names[filtered],
                'projected_distances': np.array(proj_dists)[filtered],
                'min_distances': np.array(min_dists)[filtered]
            }
        else:
            st.warning(f"No poles within {max_dist_threshold}m of GPR line.")
            return None
    except Exception as e:
        st.error(f"Error processing pole CSV: {e}")
        return None


# ------------------------------------------------------------------------------
# Sidebar UI
# ------------------------------------------------------------------------------
with st.sidebar:
    st.header("📂 File Upload")
    dzt_file = st.file_uploader("Upload DZT file", type=['dzt', 'DZT', '.dzt'])
    dzg_file = st.file_uploader("Upload DZG file (GPS data)", type=['dzg', 'DZG'],
                                 help="Optional: GPS data for distance normalization")

    st.markdown("---")
    st.header("🗺️ Coordinate Import (Optional)")
    coord_csv = st.file_uploader("Upload CSV with coordinates", type=['csv'],
                                  help="CSV with columns: Easting, Northing, Elevation")
    if coord_csv:
        st.markdown('<div class="coordinate-box">', unsafe_allow_html=True)
        st.subheader("Coordinate Settings")
        col1, col2 = st.columns(2)
        with col1:
            easting_col = st.text_input("Easting Column", "Easting")
            northing_col = st.text_input("Northing Column", "Northing")
        with col2:
            elevation_col = st.text_input("Elevation Column", "Elevation")
            trace_col = st.text_input("Trace Column (optional)", "",
                                      help="If CSV has trace numbers")
        interp_method = st.selectbox("Interpolation Method",
                                     ["Linear", "Cubic", "Nearest", "Previous", "Next"])
        coord_units = st.selectbox("Coordinate Units", ["Meters", "Feet", "Kilometers", "Miles"])
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.header("📏 Axis Scaling")
    st.subheader("Depth Scaling (Y-axis)")
    depth_unit = st.selectbox("Depth Unit", ["samples", "meters", "nanoseconds", "feet"])
    if depth_unit != "samples":
        max_depth = st.number_input(f"Max Depth ({depth_unit})", 0.1, 1000.0, 12.0, 0.1)
        velocity = None
        if depth_unit == "nanoseconds":
            velocity = st.number_input("Wave Velocity (m/ns)", 0.01, 0.3, 0.1, 0.01)

    st.subheader("Distance Scaling (X-axis)")
    use_coords_for_distance = coord_csv is not None and st.checkbox("Use Coordinates for Distance", False,
                                                                     help="Use imported coordinates for X-axis")
    if not use_coords_for_distance:
        distance_unit = st.selectbox("Distance Unit", ["traces", "meters", "feet", "kilometers"])
        if distance_unit != "traces":
            total_distance = st.number_input(f"Total Distance ({distance_unit})", 0.1, 10000.0, 250.0, 0.1)
    else:
        distance_unit = "meters"
        total_distance = None
        st.info("Using coordinate-based distance calculation")

    st.markdown("---")
    st.header("📐 Plot Aspect Ratio")
    aspect_mode = st.selectbox("Aspect Ratio Mode",
                               ["Auto", "Equal", "Manual", "Realistic"],
                               help="Control Y:X scale")
    if aspect_mode == "Manual":
        aspect_ratio = st.selectbox("Aspect Ratio (Y:X)",
                                   ["1:1", "1:2", "1:4", "1:5", "1:10", "2:1", "4:1", "5:1", "10:1"])
        aspect_ratio_float = float(aspect_ratio.split(":")[0]) / float(aspect_ratio.split(":")[1])
    elif aspect_mode == "Realistic":
        realistic_ratio = st.selectbox("Realistic Ratio",
                                      ["1:5 (Shallow)", "1:10 (Standard)", "1:20 (Deep)", "1:50 (Very Deep)"])
        aspect_ratio_float = 1 / float(realistic_ratio.split(":")[1].split()[0])
    else:
        aspect_ratio_float = None

    st.markdown("---")
    st.header("🔍 Plot Windowing")
    use_custom_window = st.checkbox("Use Custom Plot Window", False,
                                    help="Define custom depth/distance ranges")
    if use_custom_window:
        st.markdown('<div class="window-box">', unsafe_allow_html=True)
        st.subheader("Depth Window")
        if depth_unit != "samples":
            depth_min = st.number_input(f"Min Depth ({depth_unit})", 0.0, max_depth, 0.0, 0.1)
            depth_max = st.number_input(f"Max Depth ({depth_unit})", 0.0, max_depth, max_depth, 0.1)
        else:
            depth_min = st.number_input("Min Depth (samples)", 0, 5000, 0)
            depth_max = st.number_input("Max Depth (samples)", 0, 5000, 255)
        st.subheader("Distance Window")
        if not use_coords_for_distance:
            if distance_unit != "traces":
                distance_min = st.number_input(f"Min Distance ({distance_unit})", 0.0, total_distance, 0.0, 0.1)
                distance_max = st.number_input(f"Max Distance ({distance_unit})", 0.0, total_distance, total_distance, 0.1)
            else:
                distance_min = st.number_input("Min Distance (traces)", 0, 10000, 0)
                distance_max = st.number_input("Max Distance (traces)", 0, 10000, 800)
        st.markdown('</div>', unsafe_allow_html=True)

    multiple_windows = st.checkbox("Enable Multiple Windows", False,
                                   help="Plot multiple windows in same view")
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
    st.header("🔄 Line Adjustment & Muting")
    reverse_line = st.checkbox("Reverse Line Direction (A→B to B→A)", False,
                               help="Reverse trace order")
    mute_traces = st.checkbox("Mute Traces", False,
                              help="Set specific trace ranges to zero")
    if mute_traces:
        st.markdown('<div class="mute-box">', unsafe_allow_html=True)
        mute_method = st.selectbox("Mute Method",
                                  ["By Distance", "By Trace Index", "Multiple Zones"])
        if mute_method == "By Distance":
            col1, col2 = st.columns(2)
            with col1:
                mute_start_dist = st.number_input("Mute Start Distance", 0.0, 10000.0, 2.0, 0.1)
            with col2:
                mute_end_dist = st.number_input("Mute End Distance", 0.0, 10000.0, 6.0, 0.1)
            apply_taper = st.checkbox("Apply Taper", True)
            if apply_taper:
                taper_length = st.slider("Taper Length (% of zone)", 1, 50, 10, 1)
        elif mute_method == "By Trace Index":
            col1, col2 = st.columns(2)
            with col1:
                mute_start_idx = st.number_input("Mute Start Trace", 0, 10000, 100)
            with col2:
                mute_end_idx = st.number_input("Mute End Trace", 0, 10000, 200)
            apply_taper = st.checkbox("Apply Taper", True)
            if apply_taper:
                taper_samples = st.slider("Taper Samples", 1, 100, 10, 1)
        elif mute_method == "Multiple Zones":
            num_zones = st.number_input("Number of Mute Zones", 1, 5, 1)
            mute_zones_input = []
            for i in range(num_zones):
                st.markdown(f"**Zone {i+1}**")
                col1, col2 = st.columns(2)
                with col1:
                    zone_method = st.selectbox(f"Zone {i+1} Method", ["By Distance", "By Trace Index"])
                    if zone_method == "By Distance":
                        zone_start = st.number_input(f"Zone {i+1} Start", 0.0, 10000.0, 10.0 + i*10, 0.1)
                        zone_end = st.number_input(f"Zone {i+1} End", 0.0, 10000.0, 15.0 + i*10, 0.1)
                    else:
                        zone_start = st.number_input(f"Zone {i+1} Start Trace", 0, 10000, 150 + i*50)
                        zone_end = st.number_input(f"Zone {i+1} End Trace", 0, 10000, 200 + i*50)
                with col2:
                    zone_taper = st.checkbox(f"Taper Zone {i+1}", True)
                    zone_label = st.text_input(f"Zone {i+1} Label", f"Zone {i+1}")
                mute_zones_input.append({
                    'method': zone_method,
                    'start': zone_start,
                    'end': zone_end,
                    'taper': zone_taper,
                    'label': zone_label
                })
        mute_strength = st.slider("Muting Strength (%)", 0, 100, 100, 5)
        show_mute_preview = st.checkbox("Show Mute Zone Preview", True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.header("🎛️ Processing Parameters")
    time_zero = st.number_input("Time Zero (samples)", 0, 2000, 2)
    stacking = st.selectbox("Stacking", ["none", "auto", "manual"])
    if stacking == "manual":
        stack_value = st.number_input("Stack Value", 1, 50, 3)

    st.markdown("---")
    st.header("🌍 Near-Surface Amplitude Correction")
    apply_near_surface_correction = st.checkbox("Apply Near-Surface Amplitude Correction", False)
    if apply_near_surface_correction:
        st.markdown('<div class="near-surface-box">', unsafe_allow_html=True)
        correction_type = st.selectbox("Correction Type",
                                      ["Linear Reduction", "Exponential Reduction",
                                       "Gaussian Filter", "Windowed Normalization"])
        correction_depth = st.number_input("Correction Depth (m)", 0.1, 10.0, 2.5, 0.1)
        if correction_type == "Linear Reduction":
            surface_reduction = st.slider("Surface Reduction (%)", 0, 95, 80, 5)
            depth_factor = st.slider("Reduction Depth Factor", 0.1, 2.0, 1.0, 0.1)
        elif correction_type == "Exponential Reduction":
            exp_factor = st.slider("Exponential Factor", 0.5, 5.0, 2.0, 0.1)
            max_reduction = st.slider("Maximum Reduction (%)", 0, 95, 90, 5)
        elif correction_type == "Gaussian Filter":
            filter_sigma = st.slider("Filter Sigma", 0.1, 5.0, 1.0, 0.1)
            filter_window = st.slider("Filter Window (samples)", 5, 100, 21, 2)
        elif correction_type == "Windowed Normalization":
            window_size = st.slider("Normalization Window (samples)", 10, 200, 50, 5)
            target_amplitude = st.slider("Target Amplitude", 0.1, 1.0, 0.3, 0.05)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.header("🔬 Advanced Deconvolution")
    apply_deconvolution = st.checkbox("Apply Deconvolution", False)
    if apply_deconvolution:
        st.markdown('<div class="deconv-box">', unsafe_allow_html=True)
        deconv_method = st.selectbox("Deconvolution Method",
                                    ["Wiener Filter", "Predictive Deconvolution",
                                     "Spiking Deconvolution", "Minimum Entropy Deconvolution",
                                     "Homomorphic Deconvolution", "Bayesian Deconvolution"])
        # Method-specific parameters
        if deconv_method == "Wiener Filter":
            col1, col2 = st.columns(2)
            with col1:
                wiener_window = st.slider("Wiener Window (samples)", 5, 101, 21, 2)
                noise_level = st.slider("Noise Level", 0.001, 0.1, 0.01, 0.001)
            with col2:
                wavelet_length = st.slider("Wavelet Length (samples)", 5, 101, 51, 2)
                regularization = st.slider("Regularization", 0.0, 1.0, 0.1, 0.01)
        elif deconv_method == "Predictive Deconvolution":
            col1, col2 = st.columns(2)
            with col1:
                prediction_distance = st.slider("Prediction Distance (samples)", 1, 100, 10, 1)
                filter_length = st.slider("Filter Length (samples)", 10, 200, 50, 5)
            with col2:
                prewhitening = st.slider("Pre-whitening (%)", 0.0, 10.0, 0.1, 0.1)
                iterations = st.slider("Iterations", 1, 10, 3, 1)
        elif deconv_method == "Spiking Deconvolution":
            col1, col2 = st.columns(2)
            with col1:
                spike_strength = st.slider("Spike Strength", 0.1, 2.0, 0.8, 0.1)
                spike_length = st.slider("Spike Length (samples)", 5, 101, 21, 2)
            with col2:
                spike_noise = st.slider("Spike Noise", 0.001, 0.1, 0.01, 0.001)
                spike_iterations = st.slider("Iterations", 1, 20, 5, 1)
        elif deconv_method == "Minimum Entropy Deconvolution":
            col1, col2 = st.columns(2)
            with col1:
                med_filter_length = st.slider("Filter Length (samples)", 10, 200, 80, 5)
                med_iterations = st.slider("Iterations", 1, 50, 10, 1)
            with col2:
                med_convergence = st.slider("Convergence Threshold", 0.0001, 0.1, 0.001, 0.0001)
                med_noise = st.slider("Noise Estimate", 0.001, 0.1, 0.01, 0.001)
        elif deconv_method == "Homomorphic Deconvolution":
            col1, col2 = st.columns(2)
            with col1:
                homo_window = st.selectbox("Smoothing Window", ["hanning", "hamming", "blackman", "bartlett"])
                homo_cutoff = st.slider("Cepstral Cutoff", 0.01, 0.5, 0.1, 0.01)
            with col2:
                homo_prewhiten = st.slider("Pre-whitening", 0.0, 0.1, 0.01, 0.001)
                homo_iterations = st.slider("Iterations", 1, 10, 3, 1)
        elif deconv_method == "Bayesian Deconvolution":
            col1, col2 = st.columns(2)
            with col1:
                bayesian_prior = st.selectbox("Prior Distribution", ["Laplace", "Gaussian", "Jeffreys"])
                bayesian_iterations = st.slider("MCMC Iterations", 100, 5000, 1000, 100)
            with col2:
                bayesian_burnin = st.slider("Burn-in Samples", 100, 2000, 500, 100)
                bayesian_noise = st.slider("Noise Estimate", 0.001, 0.1, 0.01, 0.001)

        st.subheader("Common Parameters")
        col1, col2 = st.columns(2)
        with col1:
            deconv_window_start = st.number_input("Deconvolution Start (samples)", 0, 5000, 0)
            deconv_window_end = st.number_input("Deconvolution End (samples)", 0, 5000, 1000)
        with col2:
            trace_for_wavelet = st.number_input("Trace for Wavelet Estimation", 0, 10000, 0)
            use_average_wavelet = st.checkbox("Use Average Wavelet", True)
        if use_average_wavelet:
            wavelet_trace_range = st.slider("Wavelet Trace Range", 0, 100, 10, 1)

        output_type = st.selectbox("Output Type",
                                  ["Deconvolved Only", "Deconvolved + Original",
                                   "Difference (Deconvolved - Original)"])
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.header("📈 Time Gain Control")
    gain_type = st.selectbox("Gain Type",
                            ["Constant", "Linear", "Exponential",
                             "AGC (Automatic Gain Control)", "Spherical"])
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

# ------------------------------------------------------------------------------
# Main processing
# ------------------------------------------------------------------------------
if dzt_file and process_btn:
    with st.spinner("Processing radar data..."):
        try:
            from readgssi import readgssi
        except ImportError:
            st.error("⚠️ readgssi not installed! Run: pip install readgssi")
            st.stop()

        progress_bar = st.progress(0)
        with tempfile.TemporaryDirectory() as tmpdir:
            # save DZT
            dzt_path = os.path.join(tmpdir, "input.dzt")
            with open(dzt_path, "wb") as f:
                f.write(dzt_file.getbuffer())
            progress_bar.progress(10)

            # save DZG if provided
            dzg_path = None
            if dzg_file:
                dzg_path = os.path.join(tmpdir, "input.dzg")
                with open(dzg_path, "wb") as f:
                    f.write(dzg_file.getbuffer())
            progress_bar.progress(20)

            # load coordinates if provided
            if coord_csv:
                try:
                    st.session_state.coordinates_df = pd.read_csv(coord_csv)
                    st.info(f"Loaded {len(st.session_state.coordinates_df)} coordinate points")
                except Exception as e:
                    st.warning(f"Could not read CSV coordinates: {e}")
                    st.session_state.coordinates_df = None
            progress_bar.progress(30)

            # readgssi parameters
            params = {
                'infile': dzt_path,
                'zero': [time_zero],
                'verbose': False
            }
            if stacking == "auto":
                params['stack'] = 'auto'
            elif stacking == "manual":
                params['stack'] = stack_value
            if bgr:
                if bgr_type == "Full-width":
                    params['bgr'] = 0
                else:
                    params['bgr'] = bgr_window
            if freq_filter:
                params['freqmin'] = freq_min
                params['freqmax'] = freq_max

            header, arrays, gps = readgssi.readgssi(**params)
            progress_bar.progress(50)

            if not arrays or len(arrays) == 0:
                st.error("No radar data found")
                st.stop()

            original_array = arrays[0]

            # line reversal
            if reverse_line:
                original_array = reverse_array(original_array)
                st.session_state.line_reversed = True
            else:
                st.session_state.line_reversed = False

            # near-surface correction
            if apply_near_surface_correction:
                corr_kwargs = {}
                if correction_type == "Linear Reduction":
                    corr_kwargs = {'surface_reduction': surface_reduction, 'depth_factor': depth_factor}
                elif correction_type == "Exponential Reduction":
                    corr_kwargs = {'exp_factor': exp_factor, 'max_reduction': max_reduction}
                elif correction_type == "Gaussian Filter":
                    corr_kwargs = {'filter_sigma': filter_sigma, 'filter_window': filter_window}
                elif correction_type == "Windowed Normalization":
                    corr_kwargs = {'window_size': window_size, 'target_amplitude': target_amplitude}
                original_array = apply_near_surface_correction(
                    original_array, correction_type, correction_depth,
                    max_depth if depth_unit != "samples" else None,
                    **corr_kwargs
                )

            # deconvolution
            if apply_deconvolution:
                deconv_params = {
                    'deconv_window_start': deconv_window_start,
                    'deconv_window_end': deconv_window_end,
                    'trace_for_wavelet': trace_for_wavelet,
                    'use_average_wavelet': use_average_wavelet,
                }
                # add method-specific params
                if deconv_method == "Wiener Filter":
                    deconv_params.update({
                        'noise_level': noise_level,
                        'wavelet_length': wavelet_length,
                        'regularization': regularization
                    })
                elif deconv_method == "Predictive Deconvolution":
                    deconv_params.update({
                        'prediction_distance': prediction_distance,
                        'filter_length': filter_length,
                        'prewhitening': prewhitening,
                        'iterations': iterations
                    })
                elif deconv_method == "Spiking Deconvolution":
                    deconv_params.update({
                        'spike_strength': spike_strength,
                        'spike_length': spike_length,
                        'spike_noise': spike_noise,
                        'spike_iterations': spike_iterations
                    })
                elif deconv_method == "Minimum Entropy Deconvolution":
                    deconv_params.update({
                        'med_filter_length': med_filter_length,
                        'med_iterations': med_iterations,
                        'med_convergence': med_convergence,
                        'med_noise': med_noise
                    })
                elif deconv_method == "Homomorphic Deconvolution":
                    deconv_params.update({
                        'homo_window': homo_window,
                        'homo_cutoff': homo_cutoff,
                        'homo_prewhiten': homo_prewhiten,
                        'homo_iterations': homo_iterations
                    })
                elif deconv_method == "Bayesian Deconvolution":
                    deconv_params.update({
                        'bayesian_prior': bayesian_prior,
                        'bayesian_iterations': bayesian_iterations,
                        'bayesian_burnin': bayesian_burnin,
                        'bayesian_noise': bayesian_noise
                    })
                if use_average_wavelet:
                    deconv_params['wavelet_trace_range'] = wavelet_trace_range

                deconvolved_array = apply_deconvolution_to_array(original_array, deconv_method, **deconv_params)
                st.session_state.deconvolved_array = deconvolved_array
                st.session_state.deconv_params = deconv_params

                if output_type == "Deconvolved Only":
                    processed_array = deconvolved_array.copy()
                elif output_type == "Deconvolved + Original":
                    processed_array = 0.7 * deconvolved_array + 0.3 * original_array
                else:  # Difference
                    processed_array = deconvolved_array - original_array
            else:
                processed_array = original_array.copy()

            # gain
            gain_kwargs = {}
            if gain_type == "Constant":
                gain_kwargs['const_gain'] = const_gain
            elif gain_type == "Linear":
                gain_kwargs['min_gain'] = min_gain
                gain_kwargs['max_gain'] = max_gain
            elif gain_type == "Exponential":
                gain_kwargs['base_gain'] = base_gain
                gain_kwargs['exp_factor'] = exp_factor
            elif gain_type == "AGC (Automatic Gain Control)":
                gain_kwargs['window_size'] = window_size
                gain_kwargs['target_amplitude'] = target_amplitude
            elif gain_type == "Spherical":
                gain_kwargs['power_gain'] = power_gain
                gain_kwargs['attenuation'] = attenuation
            processed_array = apply_gain(processed_array, gain_type, **gain_kwargs)

            # coordinate interpolation
            if st.session_state.coordinates_df is not None:
                interp_coords = process_coordinates(
                    st.session_state.coordinates_df,
                    processed_array.shape[1],
                    trace_col=trace_col if trace_col else None,
                    method=interp_method.lower()
                )
                st.session_state.interpolated_coords = interp_coords
                if interp_coords:
                    st.success(f"Interpolated {interp_coords['original_points']} points to {interp_coords['interpolated_points']} traces")

            # build x_axis for muting if needed
            x_axis_mute = None
            if mute_traces:
                if st.session_state.interpolated_coords and use_coords_for_distance:
                    x_axis_mute = st.session_state.interpolated_coords['distance']
                elif distance_unit != "traces" and total_distance is not None:
                    x_axis_mute = np.linspace(0, total_distance, processed_array.shape[1])

            # apply muting after gain and deconv (but before storing processed)
            if mute_traces:
                mute_strength_val = mute_strength if 'mute_strength' in locals() else 100
                if mute_method == "Multiple Zones":
                    mute_zones_list = []
                    for z in mute_zones_input:
                        zone_params = {
                            'method': z['method'],
                            'start': z['start'],
                            'end': z['end'],
                            'apply_taper': z['taper'],
                            'strength': mute_strength_val,
                            'taper_length': 0.1 if z['taper'] else 0,
                            'taper_samples': 10 if z['taper'] else 0
                        }
                        mute_zones_list.append(zone_params)
                    processed_array, mute_mask = apply_multiple_mute_zones(processed_array, mute_zones_list, x_axis_mute)
                    st.session_state.mute_zones = mute_zones_list
                elif mute_method == "By Distance":
                    zone_params = {
                        'method': 'By Distance',
                        'start': mute_start_dist,
                        'end': mute_end_dist,
                        'apply_taper': apply_taper if 'apply_taper' in locals() else False,
                        'strength': mute_strength_val,
                        'taper_length': taper_length/100 if 'taper_length' in locals() else 0.1
                    }
                    processed_array, mute_mask = apply_trace_mute(processed_array, zone_params, x_axis_mute)
                    st.session_state.mute_zones = [zone_params]
                else:  # By Trace Index
                    zone_params = {
                        'method': 'By Trace Index',
                        'start': mute_start_idx,
                        'end': mute_end_idx,
                        'apply_taper': apply_taper if 'apply_taper' in locals() else False,
                        'strength': mute_strength_val,
                        'taper_samples': taper_samples if 'taper_samples' in locals() else 10
                    }
                    processed_array, mute_mask = apply_trace_mute(processed_array, zone_params, x_axis_mute)
                    st.session_state.mute_zones = [zone_params]
                st.session_state.mute_mask = mute_mask
                st.session_state.mute_applied = True
            else:
                st.session_state.mute_applied = False
                st.session_state.mute_mask = None
                st.session_state.mute_zones = None

            # store everything
            st.session_state.original_array = original_array
            st.session_state.processed_array = processed_array
            st.session_state.header = header
            st.session_state.gps = gps
            st.session_state.data_loaded = True
            st.session_state.depth_unit = depth_unit
            st.session_state.max_depth = max_depth if depth_unit != "samples" else None
            st.session_state.distance_unit = distance_unit
            st.session_state.total_distance = total_distance if distance_unit != "traces" else None
            st.session_state.use_coords_for_distance = use_coords_for_distance
            st.session_state.aspect_mode = aspect_mode
            st.session_state.aspect_ratio = aspect_ratio_float
            st.session_state.use_custom_window = use_custom_window
            if use_custom_window:
                st.session_state.depth_min = depth_min
                st.session_state.depth_max = depth_max
                if not use_coords_for_distance:
                    st.session_state.distance_min = distance_min
                    st.session_state.distance_max = distance_max
            st.session_state.multiple_windows = multiple_windows
            if multiple_windows and use_custom_window:
                st.session_state.additional_windows = windows

            progress_bar.progress(100)
            st.success("✅ Data processed successfully!")

        except Exception as e:
            st.error(f"Error processing data: {str(e)}")
            st.code(str(e))

# ------------------------------------------------------------------------------
# Display results
# ------------------------------------------------------------------------------
if st.session_state.data_loaded:
    tab_names = ["📊 Header Info", "📈 Full View", "🔍 Custom Window",
                 "🗺️ Coordinate View", "📉 FFT Analysis", "🎛️ Gain Analysis",
                 "🔬 Deconvolution Analysis", "💾 Export"]
    tabs = st.tabs(tab_names)

    # ---- Header Info ----
    with tabs[0]:
        st.subheader("File Information & Settings")
        if st.session_state.interpolated_coords is not None:
            st.markdown("### Coordinate Information")
            c = st.session_state.interpolated_coords
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Original Points", c['original_points'])
                st.metric("Total Distance", f"{c['distance'][-1]:.1f} m")
            with col2:
                st.metric("Interpolated Points", c['interpolated_points'])
                st.metric("Avg Point Spacing", f"{c['distance'][-1]/c['original_points']:.1f} m")
            with col3:
                st.metric("Easting Range", f"{c['easting'].min():.1f} - {c['easting'].max():.1f}")
                st.metric("Elevation Range", f"{c['elevation'].min():.1f} - {c['elevation'].max():.1f}")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### Axis Scaling Settings")
            st.markdown(f"**Y-axis:** {st.session_state.depth_unit}")
            st.markdown(f"**Max Y:** {st.session_state.max_depth if st.session_state.max_depth else 'Auto'}")
            st.markdown(f"**X-axis:** {st.session_state.distance_unit}")
            st.markdown(f"**Total X:** {st.session_state.total_distance if st.session_state.total_distance else 'Auto'}")
            st.markdown(f"**Aspect Mode:** {st.session_state.aspect_mode}")
            if st.session_state.aspect_ratio:
                st.markdown(f"**Aspect Ratio:** {st.session_state.aspect_ratio:.3f}")
            if hasattr(st.session_state, 'line_reversed') and st.session_state.line_reversed:
                st.markdown("**Line Direction:** Reversed (B→A)")
            if hasattr(st.session_state, 'near_surface_correction') and st.session_state.near_surface_correction:
                st.markdown("### Near-Surface Correction")
                st.markdown(f"**Type:** {st.session_state.correction_type}")
                st.markdown(f"**Depth:** {st.session_state.correction_depth} m")
            if hasattr(st.session_state, 'mute_applied') and st.session_state.mute_applied:
                st.markdown("### Trace Muting")
                st.markdown(f"**Muting Applied:** ✓")
                if st.session_state.mute_zones:
                    for i, z in enumerate(st.session_state.mute_zones):
                        label = z.get('label', f'Zone {i+1}')
                        if z['method'] == 'By Distance':
                            st.markdown(f"**{label}:** {z['start']:.1f} - {z['end']:.1f} {st.session_state.distance_unit}")
                        else:
                            st.markdown(f"**{label}:** Traces {z['start']} - {z['end']}")
            if hasattr(st.session_state, 'deconvolution_applied') and st.session_state.deconvolution_applied:
                st.markdown("### Deconvolution")
                st.markdown(f"**Method:** {st.session_state.deconv_method}")
                if hasattr(st.session_state, 'deconv_params'):
                    p = st.session_state.deconv_params
                    st.markdown(f"**Window:** {p.get('deconv_window_start',0)} - {p.get('deconv_window_end',1000)} samples")
        with col2:
            if st.session_state.header:
                st.markdown("### File Header")
                h = st.session_state.header
                st.markdown(f"**System:** {h.get('system','Unknown')}")
                st.markdown(f"**Antenna Freq:** {h.get('ant_freq','N/A')} MHz")
                st.markdown(f"**Samples/Trace:** {h.get('spt','N/A')}")
                st.markdown(f"**Number of Traces:** {h.get('ntraces','N/A')}")

    # ---- Full View ----
    with tabs[1]:
        st.subheader("Full Radar Profile")
        x_full, y_full, xlab, ylab, _, _ = scale_axes(
            st.session_state.processed_array.shape,
            st.session_state.depth_unit,
            st.session_state.max_depth,
            st.session_state.distance_unit,
            st.session_state.total_distance,
            st.session_state.interpolated_coords if st.session_state.use_coords_for_distance else None
        )
        aspect_val = get_aspect_ratio(st.session_state.aspect_mode,
                                      st.session_state.aspect_ratio,
                                      st.session_state.processed_array.shape)

        col1, col2, col3 = st.columns(3)
        with col1:
            show_cbar = st.checkbox("Show Colorbar", True, key="full_cbar")
            interp = st.selectbox("Interpolation", ["none", "bilinear", "bicubic", "gaussian"], key="full_interp")
        with col2:
            cmap = st.selectbox("Colormap", ["seismic", "RdBu", "gray", "viridis", "jet", "coolwarm"], key="full_cmap")
            disp_aspect = st.selectbox("Display Aspect", ["auto", "equal", 0.1, 0.2, 0.5, 1.0, 2.0, 5.0],
                                       index=0, key="full_disp_aspect")
        with col3:
            vmin = st.number_input("Color Min", -1.0, 0.0, -0.5, 0.01, key="full_vmin")
            vmax = st.number_input("Color Max", 0.0, 1.0, 0.5, 0.01, key="full_vmax")
            norm_colors = st.checkbox("Auto-normalize Colors", True, key="full_norm")

        if norm_colors:
            vmax_plot = np.percentile(np.abs(st.session_state.original_array), 99)
            vmin_plot = -vmax_plot
        else:
            vmin_plot, vmax_plot = vmin, vmax

        if hasattr(st.session_state, 'deconvolution_applied') and st.session_state.deconvolution_applied:
            fig_full, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 8))
        else:
            fig_full, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

        # Original
        im1 = ax1.imshow(st.session_state.original_array,
                         extent=[x_full[0], x_full[-1], y_full[-1], y_full[0]],
                         aspect=disp_aspect, cmap=cmap, vmin=vmin_plot, vmax=vmax_plot,
                         interpolation=interp)
        ax1.set_xlabel(xlab)
        ax1.set_ylabel(ylab)
        ax1.set_title("Original Data")
        ax1.grid(True, alpha=0.3)
        if show_cbar:
            plt.colorbar(im1, ax=ax1, label='Amplitude')

        # Processed
        im2 = ax2.imshow(st.session_state.processed_array,
                         extent=[x_full[0], x_full[-1], y_full[-1], y_full[0]],
                         aspect=disp_aspect, cmap=cmap, vmin=vmin_plot, vmax=vmax_plot,
                         interpolation=interp)
        title2 = f"Processed ({gain_type} Gain"
        if hasattr(st.session_state, 'deconvolution_applied') and st.session_state.deconvolution_applied:
            title2 += f" + {st.session_state.deconv_method}"
        title2 += ")"
        ax2.set_title(title2)
        ax2.set_xlabel(xlab)
        ax2.set_ylabel(ylab)
        ax2.grid(True, alpha=0.3)
        if show_cbar:
            plt.colorbar(im2, ax=ax2, label='Amplitude')

        # Deconvolved if exists
        if hasattr(st.session_state, 'deconvolution_applied') and st.session_state.deconvolution_applied:
            if st.session_state.deconvolved_array is not None:
                im3 = ax3.imshow(st.session_state.deconvolved_array,
                                 extent=[x_full[0], x_full[-1], y_full[-1], y_full[0]],
                                 aspect=disp_aspect, cmap=cmap, vmin=vmin_plot, vmax=vmax_plot,
                                 interpolation=interp)
                ax3.set_xlabel(xlab)
                ax3.set_ylabel(ylab)
                ax3.set_title(f"Deconvolved Only ({st.session_state.deconv_method})")
                ax3.grid(True, alpha=0.3)
                if show_cbar:
                    plt.colorbar(im3, ax=ax3, label='Amplitude')

        # Overlay mute mask if any
        if st.session_state.mute_mask is not None:
            mute_cmap = ListedColormap([(1, 0, 0, 0.3)])
            ax1.imshow(st.session_state.mute_mask,
                       extent=[x_full[0], x_full[-1], y_full[-1], y_full[0]],
                       aspect=disp_aspect, cmap=mute_cmap, alpha=0.3, interpolation='nearest')
            ax2.imshow(st.session_state.mute_mask,
                       extent=[x_full[0], x_full[-1], y_full[-1], y_full[0]],
                       aspect=disp_aspect, cmap=mute_cmap, alpha=0.3, interpolation='nearest')
            if hasattr(st.session_state, 'deconvolution_applied') and st.session_state.deconvolution_applied:
                ax3.imshow(st.session_state.mute_mask,
                           extent=[x_full[0], x_full[-1], y_full[-1], y_full[0]],
                           aspect=disp_aspect, cmap=mute_cmap, alpha=0.3, interpolation='nearest')
            # add legend
            mute_patch = Patch(facecolor='red', alpha=0.3, label='Mute Zone')
            ax1.legend(handles=[mute_patch], loc='upper right')
            ax2.legend(handles=[mute_patch], loc='upper right')
            if hasattr(st.session_state, 'deconvolution_applied') and st.session_state.deconvolution_applied:
                ax3.legend(handles=[mute_patch], loc='upper right')

        plt.tight_layout()
        st.pyplot(fig_full)

    # ---- Custom Window ----
    with tabs[2]:
        st.subheader("Custom Window Analysis")
        if not st.session_state.use_custom_window:
            st.warning("Enable 'Use Custom Plot Window' in sidebar to use this feature.")
        else:
            x_full, y_full, xlab, ylab, _, _ = scale_axes(
                st.session_state.processed_array.shape,
                st.session_state.depth_unit,
                st.session_state.max_depth,
                st.session_state.distance_unit,
                st.session_state.total_distance,
                st.session_state.interpolated_coords if st.session_state.use_coords_for_distance else None
            )
            win = get_window_indices(x_full, y_full,
                                     st.session_state.depth_min,
                                     st.session_state.depth_max,
                                     st.session_state.distance_min,
                                     st.session_state.distance_max)
            win_data = st.session_state.processed_array[
                       win['depth_min_idx']:win['depth_max_idx'],
                       win['dist_min_idx']:win['dist_max_idx']]
            win_orig = st.session_state.original_array[
                       win['depth_min_idx']:win['depth_max_idx'],
                       win['dist_min_idx']:win['dist_max_idx']]
            x_win = x_full[win['dist_min_idx']:win['dist_max_idx']]
            y_win = y_full[win['depth_min_idx']:win['depth_max_idx']]

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Depth Range", f"{win['depth_min_val']:.1f}-{win['depth_max_val']:.1f} {st.session_state.depth_unit}")
            with col2:
                st.metric("Distance Range", f"{win['dist_min_val']:.1f}-{win['dist_max_val']:.1f} {st.session_state.distance_unit}")
            with col3:
                st.metric("Window Size", f"{win_data.shape[0]}×{win_data.shape[1]}")
            with col4:
                st.metric("Data Points", f"{win_data.size:,}")

            fig_win, (ax_orig, ax_proc) = plt.subplots(1, 2, figsize=(16, 6))
            im_orig = ax_orig.imshow(win_orig,
                                     extent=[x_win[0], x_win[-1], y_win[-1], y_win[0]],
                                     aspect='auto', cmap='seismic')
            ax_orig.set_xlabel(xlab)
            ax_orig.set_ylabel(ylab)
            ax_orig.set_title(f"Original Window")
            ax_orig.grid(True, alpha=0.3)
            plt.colorbar(im_orig, ax=ax_orig, label='Amplitude')

            im_proc = ax_proc.imshow(win_data,
                                     extent=[x_win[0], x_win[-1], y_win[-1], y_win[0]],
                                     aspect='auto', cmap='seismic')
            ax_proc.set_xlabel(xlab)
            ax_proc.set_ylabel(ylab)
            ax_proc.set_title(f"Processed Window")
            ax_proc.grid(True, alpha=0.3)
            plt.colorbar(im_proc, ax=ax_proc, label='Amplitude')
            plt.tight_layout()
            st.pyplot(fig_win)

            # Multiple windows if enabled
            if st.session_state.multiple_windows and hasattr(st.session_state, 'additional_windows'):
                st.subheader("Multiple Windows")
                n_add = len(st.session_state.additional_windows)
                cols = min(2, n_add+1)
                rows = (n_add+1 + cols -1) // cols
                fig_multi, axes = plt.subplots(rows, cols, figsize=(cols*8, rows*6))
                if rows*cols == 1:
                    axes = np.array([[axes]])
                elif rows == 1:
                    axes = axes.reshape(1, -1)
                elif cols == 1:
                    axes = axes.reshape(-1, 1)

                # main window
                ax = axes[0, 0]
                im = ax.imshow(win_data,
                              extent=[x_win[0], x_win[-1], y_win[-1], y_win[0]],
                              aspect='auto', cmap='seismic')
                ax.set_xlabel(xlab)
                ax.set_ylabel(ylab)
                ax.set_title(f"Window 1")
                ax.grid(True, alpha=0.3)
                plt.colorbar(im, ax=ax, label='Amplitude')

                # additional windows
                for idx, w in enumerate(st.session_state.additional_windows):
                    i = (idx+1) // cols
                    j = (idx+1) % cols
                    if i >= rows or j >= cols:
                        continue
                    win_info = get_window_indices(x_full, y_full,
                                                  w['depth_min'], w['depth_max'],
                                                  w['distance_min'], w['distance_max'])
                    win_data_add = st.session_state.processed_array[
                                   win_info['depth_min_idx']:win_info['depth_max_idx'],
                                   win_info['dist_min_idx']:win_info['dist_max_idx']]
                    x_add = x_full[win_info['dist_min_idx']:win_info['dist_max_idx']]
                    y_add = y_full[win_info['depth_min_idx']:win_info['depth_max_idx']]
                    axes[i, j].imshow(win_data_add,
                                      extent=[x_add[0], x_add[-1], y_add[-1], y_add[0]],
                                      aspect='auto', cmap='seismic')
                    axes[i, j].set_xlabel(xlab)
                    axes[i, j].set_ylabel(ylab)
                    axes[i, j].set_title(f"Window {idx+2}")
                    axes[i, j].grid(True, alpha=0.3)
                plt.tight_layout()
                st.pyplot(fig_multi)

            # trace and slice analysis
            st.subheader("Trace and Slice Analysis")
            col_t1, col_t2 = st.columns(2)
            with col_t1:
                trace_in_win = st.number_input("Select Trace in Window", 0, win_data.shape[1]-1,
                                               win_data.shape[1]//2, key="win_trace")
                actual_trace = win['dist_min_idx'] + trace_in_win
                trace_amp = win_data[:, trace_in_win]
                fig_trace, ax_t = plt.subplots(figsize=(10, 6))
                ax_t.plot(y_win, trace_amp, 'b-')
                ax_t.fill_between(y_win, 0, trace_amp, alpha=0.3, color='blue')
                ax_t.set_xlabel(ylab)
                ax_t.set_ylabel("Amplitude")
                ax_t.set_title(f"Trace {actual_trace}")
                ax_t.grid(True, alpha=0.3)
                ax_t.invert_xaxis()
                st.pyplot(fig_trace)
                trace_df = pd.DataFrame({ylab: y_win, 'Amplitude': trace_amp})
                st.download_button("📥 Download Trace CSV", trace_df.to_csv(index=False),
                                   f"trace_{actual_trace}.csv", "text/csv")
            with col_t2:
                depth_in_win = st.slider("Select Depth Slice", 0, win_data.shape[0]-1,
                                         win_data.shape[0]//2, key="win_depth")
                actual_depth = y_win[depth_in_win]
                slice_amp = win_data[depth_in_win, :]
                fig_slice, ax_s = plt.subplots(figsize=(10, 6))
                ax_s.plot(x_win, slice_amp, 'r-')
                ax_s.fill_between(x_win, 0, slice_amp, alpha=0.3, color='red')
                ax_s.set_xlabel(xlab)
                ax_s.set_ylabel("Amplitude")
                ax_s.set_title(f"Depth Slice at {actual_depth:.2f}")
                ax_s.grid(True, alpha=0.3)
                st.pyplot(fig_slice)
                slice_df = pd.DataFrame({xlab: x_win, 'Amplitude': slice_amp})
                st.download_button("📥 Download Slice CSV", slice_df.to_csv(index=False),
                                   f"depth_slice_{actual_depth:.2f}.csv", "text/csv")

    # ---- Coordinate View ----
    with tabs[3]:
        st.subheader("Coordinate-Based Visualization")
        if st.session_state.interpolated_coords is None:
            st.warning("No coordinates imported. Upload CSV with Easting, Northing, Elevation.")
        else:
            c = st.session_state.interpolated_coords
            # Pole CSV upload (now in this tab)
            pole_csv = st.file_uploader("Upload Electric Pole CSV (Easting, Northing, Name)",
                                        type=['csv'], key="pole_csv")
            if pole_csv:
                pole_data = process_pole_csv(pole_csv, c, max_dist_threshold=10.0)
                if pole_data:
                    st.session_state.pole_data = pole_data
                    st.info(f"Loaded {len(pole_data['names'])} poles near line")
            else:
                pole_data = st.session_state.pole_data if hasattr(st.session_state, 'pole_data') else None

            # stats
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Profile Length", f"{c['distance'][-1]:.1f} m")
                st.metric("Elevation Change", f"{c['elevation'].max() - c['elevation'].min():.1f} m")
            with col2:
                st.metric("Easting Range", f"{np.ptp(c['easting']):.1f} m")
                st.metric("Northing Range", f"{np.ptp(c['northing']):.1f} m")
            with col3:
                avg_spacing = np.mean(np.diff(c['distance']))
                st.metric("Avg Trace Spacing", f"{avg_spacing:.2f} m")
                bearing = np.degrees(np.arctan2(c['northing'][-1]-c['northing'][0],
                                                c['easting'][-1]-c['easting'][0]))
                st.metric("Profile Bearing", f"{bearing:.1f}°")
            with col4:
                slope = (c['elevation'][-1] - c['elevation'][0]) / c['distance'][-1]
                st.metric("Average Slope", f"{slope*100:.1f}%")
                st.metric("Data Points", f"{len(c['easting'])}")

            # create coordinate plots
            from mpl_toolkits.mplot3d import Axes3D
            fig_coords, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

            # plan view
            sc1 = ax1.scatter(c['easting'], c['northing'], c=c['distance'], cmap='viridis', s=20)
            ax1.set_xlabel('Easting (m)')
            ax1.set_ylabel('Northing (m)')
            ax1.set_title('Plan View')
            ax1.grid(True, alpha=0.3)
            ax1.axis('equal')
            plt.colorbar(sc1, ax=ax1, label='Distance (m)')
            if pole_data:
                for i in range(len(pole_data['easting'])):
                    if 'TS' in pole_data['names'][i]:
                        color = 'red'; marker = '^'
                    elif 'TL' in pole_data['names'][i]:
                        color = 'purple'; marker = '^'
                    else:
                        color = 'orange'; marker = 'o'
                    ax1.scatter(pole_data['easting'][i], pole_data['northing'][i],
                                c=color, marker=marker, s=80, edgecolor='black', linewidth=1)

            # elevation profile
            ax2.plot(c['distance'], c['elevation'], 'g-', linewidth=2)
            ax2.fill_between(c['distance'], c['elevation'].min(), c['elevation'], alpha=0.3, color='green')
            if pole_data:
                for i, d in enumerate(pole_data['projected_distances']):
                    # interpolate elevation at d
                    e = np.interp(d, c['distance'], c['elevation'])
                    if 'TS' in pole_data['names'][i]:
                        color = 'red'; marker = '^'
                    elif 'TL' in pole_data['names'][i]:
                        color = 'purple'; marker = '^'
                    else:
                        color = 'orange'; marker = 'o'
                    ax2.scatter(d, e, c=color, marker=marker, s=80, edgecolor='black', linewidth=1)
                    ax2.text(d, e+0.5, pole_data['names'][i], fontsize=8, ha='center')
            ax2.set_xlabel('Distance (m)')
            ax2.set_ylabel('Elevation (m)')
            ax2.set_title('Elevation Profile')
            ax2.grid(True, alpha=0.3)

            # 3D view
            ax3 = fig_coords.add_subplot(2, 2, 3, projection='3d')
            ax3.plot(c['easting'], c['northing'], c['elevation'], 'b-', alpha=0.7)
            sc3 = ax3.scatter(c['easting'], c['northing'], c['elevation'], c=c['distance'], cmap='viridis', s=20)
            if pole_data:
                for i in range(len(pole_data['easting'])):
                    e = np.interp(pole_data['projected_distances'][i], c['distance'], c['elevation'])
                    if 'TS' in pole_data['names'][i]:
                        color = 'red'; marker = '^'
                    elif 'TL' in pole_data['names'][i]:
                        color = 'purple'; marker = '^'
                    else:
                        color = 'orange'; marker = 'o'
                    ax3.scatter(pole_data['easting'][i], pole_data['northing'][i], e,
                                c=color, marker=marker, s=80, edgecolor='black', linewidth=1)
            ax3.set_xlabel('Easting (m)')
            ax3.set_ylabel('Northing (m)')
            ax3.set_zlabel('Elevation (m)')
            ax3.set_title('3D Survey Line')
            plt.colorbar(sc3, ax=ax3, label='Distance (m)')

            # GPR data with coordinate axis
            aspect_val = get_aspect_ratio(st.session_state.aspect_mode,
                                          st.session_state.aspect_ratio,
                                          st.session_state.processed_array.shape)
            if st.session_state.depth_unit != "samples":
                depth_axis = np.linspace(0, st.session_state.max_depth,
                                         st.session_state.processed_array.shape[0])
            else:
                depth_axis = np.arange(st.session_state.processed_array.shape[0])
            im4 = ax4.imshow(st.session_state.processed_array,
                            extent=[c['distance'][0], c['distance'][-1],
                                    depth_axis[-1], depth_axis[0]],
                            aspect=aspect_val, cmap='seismic', alpha=0.9)
            ax4.set_xlabel('Distance (m)')
            ax4.set_ylabel(f'Depth ({st.session_state.depth_unit})')
            ax4.set_title('GPR Data with Coordinate Scaling')
            ax4.grid(True, alpha=0.2)
            plt.colorbar(im4, ax=ax4, label='Amplitude')
            # overlay elevation on twin axis
            ax4_twin = ax4.twinx()
            ax4_twin.plot(c['distance'], c['elevation'], 'g-', linewidth=2, alpha=0.6, label='Elevation')
            ax4_twin.set_ylabel('Elevation (m)', color='green')
            ax4_twin.tick_params(axis='y', labelcolor='green')

            plt.tight_layout()
            st.pyplot(fig_coords)

            # Elevation-adjusted GPR plot
            st.subheader("GPR Section with Topography")
            fig_elev, ax_elev = plt.subplots(figsize=(14, 6))
            X, Y = np.meshgrid(c['distance'], depth_axis)
            Y_elev = c['elevation'] - depth_axis[:, np.newaxis]
            mesh = ax_elev.pcolormesh(X, Y_elev, st.session_state.processed_array,
                                      cmap='seismic', shading='auto', alpha=0.9,
                                      vmin=vmin_plot, vmax=vmax_plot)
            ax_elev.plot(c['distance'], c['elevation'], 'k-', linewidth=1, alpha=0.8, label='Surface')
            ax_elev.fill_between(c['distance'], Y_elev.min(), c['elevation'], alpha=0.1, color='gray')
            if pole_data:
                for i in range(len(pole_data['easting'])):
                    e = np.interp(pole_data['projected_distances'][i], c['distance'], c['elevation'])
                    if 'TS' in pole_data['names'][i]:
                        color = 'red'; marker = '^'
                    elif 'TL' in pole_data['names'][i]:
                        color = 'purple'; marker = '^'
                    else:
                        color = 'orange'; marker = 'o'
                    ax_elev.scatter(pole_data['projected_distances'][i], e,
                                    c=color, marker=marker, s=100, edgecolor='black', linewidth=1, alpha=0.9, zorder=10)
                    ax_elev.text(pole_data['projected_distances'][i], e+1,
                                 pole_data['names'][i], fontsize=6, ha='center')
            ax_elev.set_xlabel('Distance (m)')
            ax_elev.set_ylabel('Elevation (m)')
            ax_elev.set_title('GPR Section with Topography')
            ax_elev.grid(True, alpha=0.2)
            plt.colorbar(mesh, ax=ax_elev, label='Amplitude')
            ax_elev.set_ylim(Y_elev.min(), c['elevation'].max() + 5)
            plt.tight_layout()
            st.pyplot(fig_elev)

            # Export coordinates
            st.subheader("Export Interpolated Coordinates")
            coord_df = pd.DataFrame({
                'Trace_Index': c['trace_indices'],
                'Distance_m': c['distance'],
                'Easting_m': c['easting'],
                'Northing_m': c['northing'],
                'Elevation_m': c['elevation']
            })
            csv_coords = coord_df.to_csv(index=False)
            st.download_button("📥 Download as CSV", csv_coords,
                               "interpolated_coordinates.csv", "text/csv")

            if pole_data:
                st.subheader("Electric Pole Information")
                pole_df = pd.DataFrame({
                    'Name': pole_data['names'],
                    'Easting': pole_data['easting'],
                    'Northing': pole_data['northing'],
                    'Distance along profile (m)': pole_data['projected_distances'],
                    'Distance from line (m)': pole_data['min_distances']
                })
                st.dataframe(pole_df.sort_values('Distance along profile (m)'))

    # ---- FFT Analysis ----
    with tabs[4]:
        st.subheader("Frequency Analysis (FFT)")
        n_traces = st.session_state.processed_array.shape[1]
        col1, col2, col3 = st.columns(3)
        with col1:
            trace_fft = st.number_input("Select Trace", 0, n_traces-1, n_traces//2, key="fft_trace")
        with col2:
            samp_rate = st.number_input("Sampling Rate (MHz)", 100, 5000, 1000, 100, key="fft_samp")
        with col3:
            fft_mode = st.selectbox("Mode", ["Single Trace", "Average of All Traces", "Trace Range", "Windowed Traces"])

        if fft_mode == "Trace Range":
            tr_start = st.number_input("Start Trace", 0, n_traces-1, 0, key="fft_start")
            tr_end = st.number_input("End Trace", 0, n_traces-1, n_traces-1, key="fft_end")
        if fft_mode == "Windowed Traces" and not st.session_state.use_custom_window:
            st.warning("Enable Custom Plot Window for windowed FFT.")
            fft_data = None
        else:
            if fft_mode == "Single Trace":
                trace_data = st.session_state.processed_array[:, trace_fft]
            elif fft_mode == "Average of All Traces":
                trace_data = np.mean(st.session_state.processed_array, axis=1)
            elif fft_mode == "Trace Range":
                trace_data = np.mean(st.session_state.processed_array[:, tr_start:tr_end+1], axis=1)
            elif fft_mode == "Windowed Traces":
                x_full, y_full, _, _, _, _ = scale_axes(
                    st.session_state.processed_array.shape,
                    st.session_state.depth_unit,
                    st.session_state.max_depth,
                    st.session_state.distance_unit,
                    st.session_state.total_distance,
                    st.session_state.interpolated_coords if st.session_state.use_coords_for_distance else None
                )
                win = get_window_indices(x_full, y_full,
                                         st.session_state.depth_min,
                                         st.session_state.depth_max,
                                         st.session_state.distance_min,
                                         st.session_state.distance_max)
                trace_data = np.mean(st.session_state.processed_array[:, win['dist_min_idx']:win['dist_max_idx']], axis=1)
            else:
                trace_data = None

            if trace_data is not None:
                freq, mag = calculate_fft(trace_data, samp_rate)
                fig_fft, (ax_lin, ax_log) = plt.subplots(1, 2, figsize=(16, 6))
                ax_lin.plot(freq, mag, 'b-', linewidth=2)
                ax_lin.fill_between(freq, 0, mag, alpha=0.3, color='blue')
                ax_lin.set_xlabel("Frequency (MHz)")
                ax_lin.set_ylabel("Amplitude")
                ax_lin.set_title("Linear Scale")
                ax_lin.grid(True, alpha=0.3)
                ax_lin.set_xlim([0, samp_rate/2])

                ax_log.semilogy(freq, mag, 'r-', linewidth=2)
                ax_log.fill_between(freq, 1e-5, mag, alpha=0.3, color='red')
                ax_log.set_xlabel("Frequency (MHz)")
                ax_log.set_ylabel("Amplitude (log)")
                ax_log.set_title("Log Scale")
                ax_log.grid(True, alpha=0.3)
                ax_log.set_xlim([0, samp_rate/2])
                plt.tight_layout()
                st.pyplot(fig_fft)

                # stats
                peak_idx = np.argmax(mag)
                peak_freq = freq[peak_idx]
                half_power = mag[peak_idx] / np.sqrt(2)
                mask = mag >= half_power
                if np.any(mask):
                    low = freq[mask][0]
                    high = freq[mask][-1]
                    bw = high - low
                else:
                    low = high = bw = 0
                cola, colb, colc, cold = st.columns(4)
                with cola:
                    st.metric("Peak Frequency", f"{peak_freq:.1f} MHz")
                with colb:
                    st.metric("Peak Amplitude", f"{mag[peak_idx]:.3e}")
                with colc:
                    st.metric("Bandwidth (-3dB)", f"{bw:.1f} MHz")
                with cold:
                    st.metric("Center Freq", f"{(low+high)/2:.1f} MHz")

    # ---- Gain Analysis ----
    with tabs[5]:
        st.subheader("Gain Analysis")
        n_samp = st.session_state.original_array.shape[0]
        # compute gain profile
        gain_profile = np.zeros(n_samp)
        for i in range(n_samp):
            orig = st.session_state.original_array[i, :]
            proc = st.session_state.processed_array[i, :]
            mask = np.abs(orig) > 1e-10
            if np.any(mask):
                gain_profile[i] = np.median(np.abs(proc[mask]) / np.abs(orig[mask]))
            else:
                gain_profile[i] = 1.0
        # get depth axis
        if st.session_state.depth_unit != "samples":
            depth_ax = np.linspace(0, st.session_state.max_depth, n_samp)
            depth_label = f"Depth ({st.session_state.depth_unit})"
        else:
            depth_ax = np.arange(n_samp)
            depth_label = "Sample Number"
        fig_gain, ax_g = plt.subplots(figsize=(10, 6))
        ax_g.plot(gain_profile, depth_ax, 'b-', linewidth=2, label='Gain Factor')
        ax_g.fill_betweenx(depth_ax, 1, gain_profile, alpha=0.3, color='blue')
        ax_g.set_xlabel("Gain Factor (multiplier)")
        ax_g.set_ylabel(depth_label)
        ax_g.set_title("Gain Applied vs Depth")
        ax_g.grid(True, alpha=0.3)
        ax_g.legend()
        ax_g.invert_yaxis()
        st.pyplot(fig_gain)
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Min Gain", f"{gain_profile.min():.2f}x")
        with col2:
            st.metric("Max Gain", f"{gain_profile.max():.2f}x")
        with col3:
            st.metric("Mean Gain", f"{gain_profile.mean():.2f}x")

    # ---- Deconvolution Analysis ----
    with tabs[6]:
        st.subheader("Deconvolution Analysis")
        if not hasattr(st.session_state, 'deconvolution_applied') or not st.session_state.deconvolution_applied:
            st.warning("Enable 'Apply Deconvolution' in sidebar to use this feature.")
        else:
            col_left, col_right = st.columns(2)
            with col_left:
                st.markdown("### Estimated Wavelet")
                if st.session_state.estimated_wavelet is not None:
                    w = st.session_state.estimated_wavelet
                    fig_w, (ax_w1, ax_w2) = plt.subplots(2, 1, figsize=(8, 6))
                    ax_w1.plot(w, 'b-', linewidth=2)
                    ax_w1.fill_between(range(len(w)), 0, w, alpha=0.3, color='blue')
                    ax_w1.set_xlabel("Sample")
                    ax_w1.set_ylabel("Amplitude")
                    ax_w1.set_title(f"Wavelet (length {len(w)})")
                    ax_w1.grid(True, alpha=0.3)

                    f_w, m_w = calculate_fft(w, 1000)
                    ax_w2.semilogy(f_w, m_w, 'r-', linewidth=2)
                    ax_w2.fill_between(f_w, 1e-5, m_w, alpha=0.3, color='red')
                    ax_w2.set_xlabel("Frequency (MHz)")
                    ax_w2.set_ylabel("Amplitude (log)")
                    ax_w2.set_title("Wavelet Spectrum")
                    ax_w2.grid(True, alpha=0.3)
                    ax_w2.set_xlim([0, 500])
                    plt.tight_layout()
                    st.pyplot(fig_w)

            with col_right:
                st.markdown("### Quality Metrics")
                trace_idx = st.slider("Select Trace", 0, st.session_state.processed_array.shape[1]-1,
                                      st.session_state.processed_array.shape[1]//2, key="deconv_trace_ana")
                orig = st.session_state.original_array[:, trace_idx]
                if st.session_state.deconvolved_array is not None:
                    dec = st.session_state.deconvolved_array[:, trace_idx]
                else:
                    dec = st.session_state.processed_array[:, trace_idx]
                corr = np.corrcoef(orig, dec)[0,1]
                energy_ratio = np.sum(dec**2) / (np.sum(orig**2)+1e-10)
                k_orig = np.mean((orig - np.mean(orig))**4) / (np.std(orig)**4 + 1e-10)
                k_dec = np.mean((dec - np.mean(dec))**4) / (np.std(dec)**4 + 1e-10)
                sparsity = np.sum(np.abs(dec) > 0.1 * np.max(np.abs(dec))) / len(dec)
                st.metric("Correlation", f"{corr:.3f}")
                st.metric("Energy Ratio", f"{energy_ratio:.3f}")
                st.metric("Kurtosis (orig/dec)", f"{k_orig:.2f} / {k_dec:.2f}")
                st.metric("Sparsity", f"{sparsity:.3f}")

                fig_t, ax_t = plt.subplots(figsize=(10,4))
                ax_t.plot(orig, 'b-', alpha=0.7, label='Original')
                ax_t.plot(dec, 'r-', alpha=0.7, label='Deconvolved')
                ax_t.set_xlabel("Sample")
                ax_t.set_ylabel("Amplitude")
                ax_t.set_title(f"Trace {trace_idx}")
                ax_t.legend()
                ax_t.grid(True, alpha=0.3)
                st.pyplot(fig_t)

            # Residuals
            if st.session_state.deconvolved_array is not None:
                resid = st.session_state.original_array - st.session_state.deconvolved_array
                st.markdown("### Residual Analysis")
                colr1, colr2 = st.columns(2)
                with colr1:
                    resid_mean = np.mean(resid, axis=1)
                    resid_std = np.std(resid, axis=1)
                    fig_r, ax_r = plt.subplots(figsize=(10,5))
                    ax_r.plot(resid_mean, 'g-', label='Mean Residual')
                    ax_r.fill_between(range(len(resid_mean)),
                                      resid_mean - resid_std,
                                      resid_mean + resid_std,
                                      alpha=0.3, color='green', label='±1σ')
                    ax_r.axhline(0, color='k', linestyle='--', alpha=0.5)
                    ax_r.set_xlabel("Sample")
                    ax_r.set_ylabel("Amplitude")
                    ax_r.set_title("Residual Statistics")
                    ax_r.legend()
                    ax_r.grid(True, alpha=0.3)
                    st.pyplot(fig_r)
                with colr2:
                    flat_resid = resid.flatten()
                    q1, q99 = np.percentile(flat_resid, [1,99])
                    filtered = flat_resid[(flat_resid >= q1) & (flat_resid <= q99)]
                    fig_h, ax_h = plt.subplots(figsize=(10,5))
                    ax_h.hist(filtered, bins=100, density=True, alpha=0.7, color='purple', edgecolor='black')
                    mu, std = norm.fit(filtered)
                    x = np.linspace(filtered.min(), filtered.max(), 100)
                    pdf = norm.pdf(x, mu, std)
                    ax_h.plot(x, pdf, 'k-', linewidth=2, label=f'Gaussian fit\nμ={mu:.3f}, σ={std:.3f}')
                    ax_h.set_xlabel("Residual Amplitude")
                    ax_h.set_ylabel("Density")
                    ax_h.set_title("Residual Distribution")
                    ax_h.legend()
                    ax_h.grid(True, alpha=0.3)
                    st.pyplot(fig_h)
                    st.metric("Mean Residual", f"{np.mean(flat_resid):.3e}")
                    st.metric("Std Residual", f"{np.std(flat_resid):.3e}")

    # ---- Export ----
    with tabs[7]:
        st.subheader("Export Data")
        col_ex1, col_ex2, col_ex3, col_ex4 = st.columns(4)
        # Save images as PNG
        if col_ex1.button("💾 Save Full Image", use_container_width=True):
            x_full, y_full, xlab, ylab, _, _ = scale_axes(
                st.session_state.processed_array.shape,
                st.session_state.depth_unit,
                st.session_state.max_depth,
                st.session_state.distance_unit,
                st.session_state.total_distance,
                st.session_state.interpolated_coords if st.session_state.use_coords_for_distance else None
            )
            fig, ax = plt.subplots(figsize=(12,8))
            ax.imshow(st.session_state.processed_array,
                     extent=[x_full[0], x_full[-1], y_full[-1], y_full[0]],
                     aspect='auto', cmap='seismic')
            ax.set_xlabel(xlab)
            ax.set_ylabel(ylab)
            if hasattr(st.session_state, 'deconvolution_applied') and st.session_state.deconvolution_applied:
                ax.set_title(f"GPR Data - {gain_type} Gain + {st.session_state.deconv_method}")
            else:
                ax.set_title(f"GPR Data - {gain_type} Gain")
            plt.tight_layout()
            plt.savefig("gpr_data_full.png", dpi=300, bbox_inches='tight')
            st.success("Saved as 'gpr_data_full.png'")

        if st.session_state.use_custom_window:
            if col_ex2.button("💾 Save Windowed Image", use_container_width=True):
                x_full, y_full, xlab, ylab, _, _ = scale_axes(
                    st.session_state.processed_array.shape,
                    st.session_state.depth_unit,
                    st.session_state.max_depth,
                    st.session_state.distance_unit,
                    st.session_state.total_distance,
                    st.session_state.interpolated_coords if st.session_state.use_coords_for_distance else None
                )
                win = get_window_indices(x_full, y_full,
                                         st.session_state.depth_min,
                                         st.session_state.depth_max,
                                         st.session_state.distance_min,
                                         st.session_state.distance_max)
                win_data = st.session_state.processed_array[
                           win['depth_min_idx']:win['depth_max_idx'],
                           win['dist_min_idx']:win['dist_max_idx']]
                x_win = x_full[win['dist_min_idx']:win['dist_max_idx']]
                y_win = y_full[win['depth_min_idx']:win['depth_max_idx']]
                fig, ax = plt.subplots(figsize=(10,6))
                ax.imshow(win_data,
                         extent=[x_win[0], x_win[-1], y_win[-1], y_win[0]],
                         aspect='auto', cmap='seismic')
                ax.set_xlabel(xlab)
                ax.set_ylabel(ylab)
                ax.set_title(f"GPR Window\nDepth: {win['depth_min_val']:.1f}-{win['depth_max_val']:.1f} {st.session_state.depth_unit}\n"
                             f"Distance: {win['dist_min_val']:.1f}-{win['dist_max_val']:.1f} {st.session_state.distance_unit}")
                plt.tight_layout()
                plt.savefig("gpr_data_windowed.png", dpi=300, bbox_inches='tight')
                st.success("Saved as 'gpr_data_windowed.png'")

        # CSV export
        x_full, _, _, _, _, _ = scale_axes(
            st.session_state.processed_array.shape,
            st.session_state.depth_unit,
            st.session_state.max_depth,
            st.session_state.distance_unit,
            st.session_state.total_distance,
            st.session_state.interpolated_coords if st.session_state.use_coords_for_distance else None
        )
        csv_full = pd.DataFrame(st.session_state.processed_array,
                                columns=[f"{xi:.2f}" for xi in x_full])
        csv_full_str = csv_full.to_csv(index=False)
        col_ex3.download_button("📥 Download Full CSV", csv_full_str,
                                "gpr_data_full.csv", "text/csv", use_container_width=True)

        if st.session_state.use_custom_window:
            x_full, y_full, _, _, _, _ = scale_axes(
                st.session_state.processed_array.shape,
                st.session_state.depth_unit,
                st.session_state.max_depth,
                st.session_state.distance_unit,
                st.session_state.total_distance,
                st.session_state.interpolated_coords if st.session_state.use_coords_for_distance else None
            )
            win = get_window_indices(x_full, y_full,
                                     st.session_state.depth_min,
                                     st.session_state.depth_max,
                                     st.session_state.distance_min,
                                     st.session_state.distance_max)
            win_data = st.session_state.processed_array[
                       win['depth_min_idx']:win['depth_max_idx'],
                       win['dist_min_idx']:win['dist_max_idx']]
            x_win = x_full[win['dist_min_idx']:win['dist_max_idx']]
            csv_win = pd.DataFrame(win_data, columns=[f"{xi:.2f}" for xi in x_win])
            csv_win_str = csv_win.to_csv(index=False)
            col_ex4.download_button("📥 Download Window CSV", csv_win_str,
                                    "gpr_data_window.csv", "text/csv", use_container_width=True)

        # Deconvolved export
        if hasattr(st.session_state, 'deconvolution_applied') and st.session_state.deconvolution_applied:
            st.subheader("Export Deconvolved Data")
            col_d1, col_d2 = st.columns(2)
            if st.session_state.deconvolved_array is not None:
                csv_deconv = pd.DataFrame(st.session_state.deconvolved_array,
                                          columns=[f"{xi:.2f}" for xi in x_full])
                csv_deconv_str = csv_deconv.to_csv(index=False)
                col_d1.download_button("📥 Download Deconvolved CSV", csv_deconv_str,
                                       "gpr_data_deconvolved.csv", "text/csv", use_container_width=True)
            if hasattr(st.session_state, 'deconv_params'):
                settings_json = json.dumps({
                    'method': st.session_state.deconv_method,
                    'parameters': st.session_state.deconv_params
                }, indent=2)
                col_d2.download_button("📝 Download Settings", settings_json,
                                       "deconvolution_settings.json", "application/json", use_container_width=True)

        # Mute settings
        if hasattr(st.session_state, 'mute_applied') and st.session_state.mute_applied:
            st.subheader("Export Mute Settings")
            mute_settings = {
                'line_reversed': st.session_state.line_reversed if hasattr(st.session_state, 'line_reversed') else False,
                'mute_zones': st.session_state.mute_zones,
                'mute_strength': mute_strength if 'mute_strength' in locals() else 100
            }
            mute_json = json.dumps(mute_settings, indent=2)
            st.download_button("📝 Download Mute Settings", mute_json,
                               "mute_settings.json", "application/json", use_container_width=True)

# ------------------------------------------------------------------------------
# Initial message
# ------------------------------------------------------------------------------
elif not dzt_file:
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.info("""
        👈 **Upload a DZT file to begin processing**

        **New Advanced Deconvolution Features:**
        - Six deconvolution methods (Wiener, Predictive, Spiking, MED, Homomorphic, Bayesian)
        - Wavelet estimation and visualization
        - Quality metrics and residual analysis

        **Existing Features:**
        - Line reversal and trace muting
        - Near-surface amplitude correction
        - Coordinate import and interpolation
        - Custom windowing and aspect ratio control

        **Quick Start:**
        1. Upload DZT file
        2. Enable deconvolution in sidebar
        3. Select method and adjust parameters
        4. Process data and analyze results
        """)

# ------------------------------------------------------------------------------
# Footer
# ------------------------------------------------------------------------------
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666;'>"
    "📡 <b>GPR Data Processor v7.0</b> | Advanced Deconvolution Suite | "
    "Built with Streamlit & readgssi"
    "</div>",
    unsafe_allow_html=True
)
