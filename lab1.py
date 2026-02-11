import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image

# =========================
# NumPy Exercises
# =========================

print(np.zeros(10))
print(np.ones(10))
print(np.full(10, 5))
print(np.arange(10, 51))
print(np.arange(10, 51, 2))
print(np.arange(9).reshape(3, 3))
print(np.identity(3))
print(np.random.rand(1))
print(np.random.standard_normal(25))
print(np.linspace(0, 1, 20))

# =========================
# Indexing
# =========================

arr_2d = np.array([
    [5, 10, 15],
    [20, 25, 30],
    [35, 40, 45]
])

print(arr_2d[1,1])      # 25
print(arr_2d[1])        # second row
print(arr_2d[:,2])      # third column
print(arr_2d[1:,1:])    # bottom-right 2x2

# =========================
# Task 2: Image Loading
# =========================

img = cv2.imread('images/cameraman.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

plt.imshow(img)
plt.title("OpenCV Image")
plt.show()

img2 = Image.open('images/lena_gray_256.png')

plt.imshow(img2, cmap='gray')
plt.title("PIL Image")
plt.show()

# =========================
# Task 3: Save Images
# =========================

cv2.imwrite('new_image.jpg', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
img2.save('new_image2.jpg')

# =========================
# Task 4: Display as Array
# =========================

print(img.shape)
print(img)

img_array = np.array(img2)
print(img_array.shape)
print(img_array)
