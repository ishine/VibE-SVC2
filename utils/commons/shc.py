import numpy as np
from scipy.stats import mode


def subharmonics_correction(f0_contour, uv, scale_type='mode'):
    THRESHOLD_MIN = 0.25
    eps = 1e-6
    lf0 = np.log2(f0_contour + eps)
    f0_diff = lf0[1:] - lf0[: len(lf0) - 1]

    masked_lf0 = np.copy(lf0)
    masked_lf0[masked_lf0 == 0.0] = 0.0

    for i in range(len(f0_diff)):
        if np.abs(f0_diff[i]) > THRESHOLD_MIN:  # If exceed threshold
            if uv[i + 1] == 1.0 and uv[i] == 1.0:  # And voiced segment
                f0_diff[i] = 0.0  # Remove sub-harmonics

    corr_f0 = np.zeros_like(lf0)
    corr_f0[0] = lf0[0]
    for i in range(1, len(corr_f0)):
        if uv[i] == 1.0 and uv[i - 1] == 1.0:
            corr_f0[i] = f0_diff[i - 1] + corr_f0[i - 1]  # Voiced segment
        else:
            corr_f0[i] = corr_f0[i - 1]  # Unvoiced segment

    # Get Subharmonics scaling factor
    uv_recon_f0 = np.copy(corr_f0)
    uv_recon_f0[uv == 0.0] = np.log2(eps)

    # Get mode value of difference between original log F0 and corrected log F0
    diff_voiced_f0 = masked_lf0[uv != 0.0] - uv_recon_f0[uv != 0.0]
    if scale_type == 'mean':
        scaler = np.mean(diff_voiced_f0)
    else:  # 'mode'
        scaler = mode(diff_voiced_f0).mode

    # Convert to constant scale
    corr_f0 = 2 ** (corr_f0 + scaler) - eps
    return corr_f0, f0_diff
