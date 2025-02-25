import sys
import numpy as np
import cv2


def cross_correlation(img, kernel):
    
    # Get dimensions of the image and kernel
    image_height, image_width = img.shape  
    kernel_height, kernel_width = kernel.shape
    
    # Padding: Zero-padding around the image
    imagePadded = np.zeros((image_height + kernel_height - 1, image_width + kernel_width - 1))
    for i in range(image_height):
        for j in range(image_width):
            imagePadded[i + (kernel_height - 1) // 2, j + (kernel_width - 1) // 2] = img[i, j]  
    
    # Create an empty output image to store the correlation result
    output_img = np.zeros_like(img)
    
    # Perform cross-correlation
    for i in range(image_height):
        for j in range(image_width):
            window = imagePadded[i:i + kernel_height, j:j + kernel_width]
            output_img[i, j] = np.sum(window * kernel)  

    return output_img

def convolution(img, kernel):
    flipped_kernel = np.flip(kernel)  # Flip the kernel horizontally and vertically
    return cross_correlation(img, flipped_kernel)

def gaussian_kernel(sigma, height, width):
    center_y, center_x = height // 2, width // 2
    kernel = np.zeros((height, width))

    for i in range(height):
        for j in range(width):
            # Calculate the Gaussian function for each pixel
            kernel[i, j] = (1 / (2 * np.pi * sigma ** 2)) * np.exp(-((i - center_y) ** 2 + (j - center_x) ** 2) / (2 * sigma ** 2))
    
    # Normalize the kernel so that the sum of all elements equals 1
    return kernel / np.sum(kernel)

def low_pass(img, sigma, size):
    kernel = gaussian_kernel(sigma, size, size)
    return convolution(img, kernel)

def high_pass(img, sigma, size):
    low_pass_img = low_pass(img, sigma, size)
    return img - low_pass_img

def create_hybrid_image(img1, img2, sigma1, size1, high_low1, sigma2, size2, high_low2, mixin_ratio, scale_factor):
    high_low1 = high_low1.lower()
    high_low2 = high_low2.lower()

    if img1.dtype == np.uint8:
        img1 = img1.astype(np.float32) / 255.0
        img2 = img2.astype(np.float32) / 255.0

    if high_low1 == 'low':
        img1 = low_pass(img1, sigma1, size1)
    else:
        img1 = high_pass(img1, sigma1, size1)

    if high_low2 == 'low':
        img2 = low_pass(img2, sigma2, size2)
    else:
        img2 = high_pass(img2, sigma2, size2)

    img1 *= (1 - mixin_ratio)
    img2 *= mixin_ratio
    hybrid_img = (img1 + img2) * scale_factor
    return (hybrid_img * 255).clip(0, 255).astype(np.uint8)



# Load images in grayscale
img1 = cv2.imread(r"C:\Users\User\Documents\4TH YEAR (2ND SEMESTER)\CMSC 174\Cirunay_lab02_left.png", cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread(r"C:\Users\User\Documents\4TH YEAR (2ND SEMESTER)\CMSC 174\Cirunay_lab02_right.png", cv2.IMREAD_GRAYSCALE)


# Set filter parameters
sigma1, size1, low_freq = 10, 15, 'low'   
sigma2, size2, high_freq = 10, 15, 'high'   
mixin_ratio, scale = 0.5, 1.0      

# Apply low-pass and high-pass filters
low_pass_img = low_pass(img1, sigma1, size1)
high_pass_img = high_pass(img2, sigma2, size2)

# Create hybrid image
hybrid_img = create_hybrid_image(img1, img2, sigma1, size1, low_freq, sigma2, size2, high_freq, mixin_ratio, scale)

# Display images
cv2.imshow("Low-Pass Image", low_pass_img)
cv2.imshow("High-Pass Image", high_pass_img)
cv2.imshow("Hybrid Image", hybrid_img)

# Keep the images open until a key is pressed
cv2.waitKey(0)
cv2.destroyAllWindows()



