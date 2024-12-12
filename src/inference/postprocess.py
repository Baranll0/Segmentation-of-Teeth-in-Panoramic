import cv2
import numpy as np
from imutils import perspective
from scipy.spatial import distance as dist

def midpoint(ptA, ptB):
    return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)

def refine_teeth_segmentation(binary_mask):
    """
    Apply advanced morphological operations to refine teeth segmentation and separate closely connected components.

    Parameters:
        binary_mask (numpy array): Binary mask from the model prediction.

    Returns:
        refined_mask (numpy array): Refined binary mask after processing.
    """
    # Define kernels for morphological operations
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    # Fill gaps and holes using dilation and closing
    dilated = cv2.dilate(binary_mask, kernel_dilate, iterations=3)
    closed = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel_dilate, iterations=2)

    # Erode to separate connected regions
    eroded = cv2.erode(closed, kernel_erode, iterations=2)

    # Use distance transform to better separate connected components
    dist_transform = cv2.distanceTransform(eroded, cv2.DIST_L2, 5)
    _, refined_mask = cv2.threshold(dist_transform, 0.4 * dist_transform.max(), 255, cv2.THRESH_BINARY)
    refined_mask = refined_mask.astype(np.uint8)

    # Remove small noise using connected component analysis
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(refined_mask, connectivity=8)

    final_mask = np.zeros_like(binary_mask, dtype=np.uint8)
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area > 200:  # Threshold to remove small noise
            final_mask[labels == i] = 255

    return final_mask

def CCA_Analysis(orig_image, predict_image, erode_iteration=1, open_iteration=1):
    """
    Perform CCA (Connected Component Analysis) and advanced morphological operations on predicted mask.

    Parameters:
        orig_image (numpy array): Original grayscale image.
        predict_image (numpy array): Predicted binary mask.
        erode_iteration (int): Number of erosion iterations.
        open_iteration (int): Number of opening (erosion followed by dilation) iterations.

    Returns:
        image_with_boxes (numpy array): Original image with bounding boxes drawn.
        teeth_count (int): Count of detected components meeting area threshold.
    """
    # Refine the binary mask
    refined_mask = refine_teeth_segmentation(predict_image)

    # Define kernels for additional morphological operations
    kernel1 = np.ones((5, 5), dtype=np.uint8)

    # Apply opening to clean up noise
    image = cv2.morphologyEx(refined_mask, cv2.MORPH_OPEN, kernel1, iterations=open_iteration)

    # Find connected components
    num_labels, labels = cv2.connectedComponents(image, connectivity=8)

    # Create output image
    image_with_boxes = cv2.cvtColor(orig_image, cv2.COLOR_GRAY2BGR)

    teeth_count = 0

    for label in range(1, num_labels):
        # Create mask for current label
        mask = (labels == label).astype(np.uint8) * 255

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue

        cnt = contours[0]
        area = cv2.contourArea(cnt)

        # Filter components by area threshold
        if area > 2000:  # Minimum area threshold
            teeth_count += 1

            # Draw bounding box
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            cv2.drawContours(image_with_boxes, [box], -1, (0, 255, 0), 2)

            # Compute midpoints and draw dimensions
            (tl, tr, br, bl) = box
            (tltrX, tltrY) = midpoint(tl, tr)
            (blbrX, blbrY) = midpoint(bl, br)
            (tlblX, tlblY) = midpoint(tl, bl)
            (trbrX, trbrY) = midpoint(tr, br)

            cv2.line(image_with_boxes, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)), (255, 0, 0), 1)
            cv2.line(image_with_boxes, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)), (255, 0, 0), 1)

    return image_with_boxes, teeth_count