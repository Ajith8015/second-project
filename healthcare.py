import cv2
import numpy as np

# Load X-ray image in grayscale
image = cv2.imread('handfracture.png', cv2.IMREAD_GRAYSCALE)
image = cv2.resize(image,(500,500))
if image is None:
    print("Image not found.")
    exit()

# Step 1: Apply Gaussian Blur to reduce noise
blurred = cv2.GaussianBlur(image, (5, 5), 0)

# Step 2: Use Canny Edge Detection to find edges
edges = cv2.Canny(blurred, threshold1=50, threshold2=150)

# Step 3: Dilate edges to close gaps
kernel = np.ones((5,5), np.uint8)
dilated = cv2.dilate(edges, kernel, iterations=1)

# Step 4: Find contours in the edge map
contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Convert grayscale to BGR to draw colored contours
output = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

# Step 5: Filter and highlight possible fractures based on contour properties
for cnt in contours:
    area = cv2.contourArea(cnt)
    length = cv2.arcLength(cnt, True)
    
    # Heuristic: fractures tend to be thin and long edges (small area but relatively long perimeter)
    if area < 1000 and length > 200:
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(output, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(output, "Possible Fracture", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

# Show images
cv2.imshow("Original X-ray", image)
cv2.imshow("Edges", edges)
cv2.imshow("Fracture Detection", output)
cv2.waitKey(0)
cv2.destroyAllWindows()


