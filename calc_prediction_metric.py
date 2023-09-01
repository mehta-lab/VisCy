#!/home/ankitr/conda/envs/viscy/bin/python

import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.measure import pearson_corr_coeff

# Calculate Mean Squared Error
def calc_MSE(img, img_pred):
    m = img.shape[0]
    n = img.shape[1]
    mse = np.sum((img - img_pred)**2)/(m*n)

    return mse

# Calculate Max dynamic range
def calc_img_MAX(img):
    max_img = img.max() - img.min()
    return max_img

# Calculate Peak signal-to-noise ratio
def calc_PSNR(img, img_pred, calc_MSE, calc_img_MAX):
    max_img = calc_img_MAX(img)
    mse = calc_MSE(img, img_pred)
    psnr = (20 * np.log10(max_img)) - (10 * np.log10(mse))

    return psnr

# Calculate Structural Similarity
def calc_SSIM(img, img_pred):
    return ssim(img, img_pred, data_range=(img_pred.max() - img_pred.min()))

# Calculate Pearson Correlation Coefficient
def calc_PearsonsCorr(img, img_pred):
    return pearson_corr_coeff(img, img_pred)

# Main function
def main():
    img = np.random.random((64,64))
    img_pred = img - np.random.random((64,64))*1e-1
    print(f"PSNR: {calc_PSNR(img, img_pred, calc_MSE, calc_img_MAX):.2f}")
    print(f"SSIM: {calc_SSIM(img, img_pred):.2f}")
    print(f"Pearson's Corr: {calc_PearsonsCorr(img, img_pred)[0]:.2f}")

main()