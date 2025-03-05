import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load and resize images
image_files = ['Shell001.png', 'Shell002.png', 'Shell003.png']
images = [cv2.imread(img) for img in image_files]

# Resize while maintaining aspect ratio
target_width = 800  #target width
resized_images = []

for img in images:
    if img is not None:
        h, w = img.shape[:2] 
        scale = target_width / w 
        new_size = (target_width, int(h * scale))  
        resized_img = cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)
        resized_images.append(resized_img)

# Convert to grayscale and apply Gaussian blur
grayscale_images = [cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (5, 5), 0) for img in resized_images ]

# Adaptive Thresholding Function
def adaptive_threshold(image, block_size=19, C=4):  
    binary = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, block_size, C)  
    
    kernel_size = np.ones((3, 3), np.uint8)  
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_size)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_size)
    
    return binary

# Contour Detection and Filtering Function
def apply_contours(original, binary):

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered_contours = sorted(contours, key=cv2.contourArea, reverse=True)

    # Draw contours 
    contoured_image = original.copy()
    cv2.drawContours(contoured_image, [filtered_contours[0]], -1, (0, 255, 0), 5)  
    cv2.drawContours(contoured_image, [filtered_contours[1]], -1, (0, 0, 255), 5) 

    return contoured_image, filtered_contours[:2]  

# Measuring the shell Area 
def compute_area(contours_sort):
    shell_contour = contours_sort[0]
    coin_contour = contours_sort[1]

    shell_area_pixels = cv2.contourArea(shell_contour)
    coin_area_pixels = cv2.contourArea(coin_contour)

    five_peso_diameter = 26.76  
    five_peso_radius = five_peso_diameter / 2
    five_peso_area = np.pi * (five_peso_radius ** 2)  # mm²

    approx_shell_area = shell_area_pixels * (five_peso_area / coin_area_pixels)
    shell_area_inch = approx_shell_area * 0.00155

    return shell_area_inch

# Process all images
binary_images = [adaptive_threshold(img) for img in grayscale_images]
contoured_images, shell_areas = [], []

for i in range(len(resized_images )):
    contoured_img, contours = apply_contours(resized_images [i], binary_images[i])
    
    if len(contours) == 2:
        shell_area = compute_area(contours)
        shell_areas.append(shell_area)
    else:
        shell_areas.append(None)  

    contoured_images.append(contoured_img)

# Print the approximate shell areas in square inches
for i, area in enumerate(shell_areas):
    if area:
        print(f"Approximate Shell Area for Image {i+1}: {area:.5f} in²")
    else:
        print(f"Approximate Shell Area for Image {i+1}: Not Detected")

# Display images in a landscape layout
fig, axes = plt.subplots(2, 3, figsize=(12, 6))

for i in range(len(resized_images )):
    axes[0, i].imshow(cv2.cvtColor(binary_images[i], cv2.COLOR_GRAY2RGB))
    axes[0, i].set_title(f"Binary Image {i+1}")
    axes[0, i].axis("off")

    axes[1, i].imshow(cv2.cvtColor(contoured_images[i], cv2.COLOR_BGR2RGB))
    axes[1, i].set_title(f"Contoured Image {i+1}")
    axes[1, i].axis("off")

plt.tight_layout()
plt.show() 