# =====================================================
# code authored by Faizal Nujumudeen
# Presidency University, Bengaluru
# =====================================================
    
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from math import log10
from skimage.metrics import structural_similarity as ssim

# =====================================================
# Configuration
# =====================================================
IMAGE_PATH = "Taj.jpg"
SECRET_PATH = "secret1.txt"
OUTPUT_DIR = "results"
T = 4
HEADER_BITS = 32

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =====================================================
# Load Image (safe integer domain)
# =====================================================
img_u8 = cv2.imread(IMAGE_PATH, cv2.IMREAD_GRAYSCALE)
if img_u8 is None:
    raise IOError("Input image not found.")

img = img_u8.astype(np.int16)
h, w = img.shape

# =====================================================
# Load Secret and Add Length Header
# =====================================================
with open(SECRET_PATH, "r", encoding="utf-8") as f:
    secret = f.read()

secret_bits = ''.join(format(ord(c), '08b') for c in secret)
payload_length = len(secret_bits)

length_header = format(payload_length, f'0{HEADER_BITS}b')
full_bitstream = length_header + secret_bits

bitstream = np.array([int(b) for b in full_bitstream], dtype=np.uint8)

# =====================================================
# Embedding (AXP-RDH)
# =====================================================
marked = img.copy()
errors = []
bit_idx = 0

for i in range(1, h):
    for j in range(1, w):
        if bit_idx >= len(bitstream):
            break

        left = int(img[i, j-1])
        up = int(img[i-1, j])
        pred = left ^ up

        cur = int(img[i, j])
        e = cur - pred

        if abs(e) <= T:
            e_p = 2 * e + int(bitstream[bit_idx])
            new_val = pred + e_p

            if 0 <= new_val <= 255:
                marked[i, j] = new_val
                errors.append(e)
                bit_idx += 1
    if bit_idx >= len(bitstream):
        break

embedded_bits = bit_idx

cv2.imwrite(
    os.path.join(OUTPUT_DIR, "embedded.png"),
    np.clip(marked, 0, 255).astype(np.uint8)
)

# =====================================================
# Extraction + Perfect Recovery
# =====================================================
recovered = marked.copy()
extracted_bits = []

for i in range(1, h):
    for j in range(1, w):
        if len(extracted_bits) >= embedded_bits:
            break

        left = int(recovered[i, j-1])
        up = int(recovered[i-1, j])
        pred = left ^ up

        e_p = int(marked[i, j]) - pred

        if (e_p & 1) in (0, 1):
            e = e_p >> 1
            if abs(e) <= T:
                extracted_bits.append(e_p & 1)
                recovered[i, j] = pred + e
    if len(extracted_bits) >= embedded_bits:
        break

cv2.imwrite(
    os.path.join(OUTPUT_DIR, "recovered.png"),
    np.clip(recovered, 0, 255).astype(np.uint8)
)

# =====================================================
# Recover Payload Length and Secret
# =====================================================
length_bits = extracted_bits[:HEADER_BITS]
recovered_length = int(''.join(str(b) for b in length_bits), 2)

payload_bits = extracted_bits[HEADER_BITS:HEADER_BITS + recovered_length]
bit_string = ''.join(str(b) for b in payload_bits)

chars = [
    chr(int(bit_string[i:i+8], 2))
    for i in range(0, len(bit_string), 8)
]
recovered_secret = ''.join(chars)

# =====================================================
# Metrics
# =====================================================
orig = img.astype(np.int16)
emb = marked.astype(np.int16)
rec = recovered.astype(np.int16)

mse_emb = np.mean((orig - emb) ** 2)
mse_rec = np.mean((orig - rec) ** 2)

psnr_emb = 10 * log10(255 * 255 / mse_emb)
psnr_rec = float("inf") if mse_rec == 0 else 10 * log10(255 * 255 / mse_rec)

ssim_emb = ssim(orig, emb, data_range=255)
ssim_rec = ssim(orig, rec, data_range=255)

perfect_recovery = np.array_equal(orig, rec)

# =====================================================
# MSE / PSNR / SSIM Subplot
# =====================================================
plt.figure(figsize=(10, 4))

plt.subplot(1, 3, 1)
plt.bar(["Embedded", "Recovered"], [mse_emb, mse_rec])
plt.title("MSE")

plt.subplot(1, 3, 2)
plt.bar(["Embedded", "Recovered"], [psnr_emb, psnr_rec])
plt.title("PSNR (dB)")

plt.subplot(1, 3, 3)
plt.bar(["Embedded", "Recovered"], [ssim_emb, ssim_rec])
plt.title("SSIM")

plt.suptitle("Quality Metrics Comparison")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "MSE_PSNR_SSIM_subplot.png"))
plt.close()

# =====================================================
# Prediction Error Histogram
# =====================================================
plt.figure()
plt.hist(errors, bins=50)
plt.xlabel("Prediction Error")
plt.ylabel("Frequency")
plt.title("Prediction Error Histogram")
plt.savefig(os.path.join(OUTPUT_DIR, "prediction_error_histogram.png"))
plt.close()

# =====================================================
# Save Analysis
# =====================================================
with open(os.path.join(OUTPUT_DIR, "analysis.txt"), "w") as f:
    f.write("AXP-RDH Experimental Analysis (Header-Based)\n")
    f.write("===========================================\n")
    f.write(f"Image size           : {h} x {w}\n")
    f.write(f"Threshold (T)        : {T}\n")
    f.write(f"Embedded bits        : {embedded_bits}\n")
    f.write(f"Payload bits         : {payload_length}\n")
    f.write(f"Payload (bpp)        : {payload_length/(h*w):.6f}\n")
    f.write(f"MSE (Embedded)       : {mse_emb:.6f}\n")
    f.write(f"MSE (Recovered)      : {mse_rec:.6f}\n")
    f.write(f"PSNR (Embedded)      : {psnr_emb:.4f} dB\n")
    f.write(f"PSNR (Recovered)     : {psnr_rec}\n")
    f.write(f"SSIM (Embedded)      : {ssim_emb:.6f}\n")
    f.write(f"SSIM (Recovered)     : {ssim_rec:.6f}\n")
    f.write(f"Perfect recovery     : {perfect_recovery}\n\n")
    f.write("Recovered Secret:\n")
    f.write(recovered_secret)

print("AXP-RDH with header completed successfully.")

# =====================================================
# "If you want to shine like a sun, first burn like a sun" - Dr. APJ Abdul Kalam.
# Success is a continuous process
# =====================================================
