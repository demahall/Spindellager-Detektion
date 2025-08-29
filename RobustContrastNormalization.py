import numpy as np

def robust_contrast_normalization(image, cutoff_percentage):
    """
    Performs robust contrast normalization on an image.

    Parameters:
        image: 2D numpy array (grayscale)
        cutoff_percentage: tuple of two floats, e.g., (0.5, 0.5)

    Returns:
        normalized_image: contrast enhanced image
        stretchlim_values: tuple (low, high) used for rescaling
    """
    if not isinstance(cutoff_percentage, (list, tuple)) or len(cutoff_percentage) != 2:
        raise ValueError("cutoff_percentage must be a 2-element list or tuple")

    # Ensure image is float64 in [0,1]
    if image.dtype != np.float64:
        image_float = image / 255.0
    else:
        image_float = image.copy()

    # Compute histogram and CDF
    hist, bin_edges = np.histogram(image_float, bins=256, range=(0, 1))
    cdf = np.cumsum(hist) / np.prod(image_float.shape)

    #Determine max and min Values of cdf
    cdf_min = np.min(cdf)
    cdf_max = np.max(cdf)

    # Compute percentage cutoffs
    percent_cutoff_lower = cutoff_percentage[0] / 100 * (cdf_max - cdf_min)
    percent_cutoff_upper = cutoff_percentage[1] / 100 * (cdf_max - cdf_min)

    # Compute stretch limits in CDF domain
    lower_cdf_val = cdf_min + percent_cutoff_lower
    upper_cdf_val = cdf_max - percent_cutoff_upper


    # Map CDF thresholds back to intensity values
    # (we find where the CDF crosses the thresholds)
    low_idx = np.searchsorted(cdf, lower_cdf_val)
    high_idx = np.searchsorted(cdf, upper_cdf_val)

    low_intensity = bin_edges[low_idx]
    high_intensity = bin_edges[min(high_idx, len(bin_edges)-1)]

    stretchlim_values = (low_intensity, high_intensity)

    # Avoid division by zero
    if np.isclose(low_intensity, high_intensity, atol=1e-5):
        normalized_image = np.zeros_like(image_float)
    else:
        # Rescale intensities to [0,1]
        scaled = (image_float - low_intensity) / (high_intensity - low_intensity)
        scaled = np.clip(scaled, 0, 1)
        normalized_image = (scaled * 255).astype(np.uint8)

    return normalized_image, stretchlim_values