# Import required modules
import cv2 #pip install opencv
import numpy as np #pip install numpy
import os
import glob

# Define the dimensions of the checkerboard
CHECKERBOARD = (6, 9)

# Termination criteria for corner refinement
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Vector for 3D points
threedpoints = []

# Vector for 2D points
twodpoints = []

# 3D points real-world coordinates
objectp3d = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objectp3d[0, :, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)

prev_img_shape = None

# Specify the path to the folder containing checkerboard images
image_folder_path = r'C:\Users\Janhavi\Downloads\images'

# Extracting path of individual images stored in the specified directory
images = glob.glob(os.path.join(image_folder_path, '*.jpg'))

# Check if images were found in the specified path
if not images:
    print(f"No images found in the directory: {image_folder_path}")
else:
    for filename in images:
        image = cv2.imread(filename)
        grayColor = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(
            grayColor, CHECKERBOARD,
            cv2.CALIB_CB_ADAPTIVE_THRESH +
            cv2.CALIB_CB_FAST_CHECK +
            cv2.CALIB_CB_NORMALIZE_IMAGE
        )

        # If desired number of corners can be detected, refine and display them
        if ret:
            threedpoints.append(objectp3d)

            # Refining pixel coordinates for given 2D points
            corners2 = cv2.cornerSubPix(
                grayColor, corners, (11, 11), (-1, -1), criteria
            )

            twodpoints.append(corners2)

            # Draw and display the corners
            image = cv2.drawChessboardCorners(image, CHECKERBOARD, corners2, ret)

            cv2.imshow('img', image)
            cv2.waitKey(0)

    cv2.destroyAllWindows()

    h, w = image.shape[:2]

    # Perform camera calibration
    ret, matrix, distortion, r_vecs, t_vecs = cv2.calibrateCamera(
        threedpoints, twodpoints, grayColor.shape[::-1], None, None
    )

    # Displaying the calibration results
    print("Camera matrix:")
    print(matrix)

    print("\nDistortion coefficient:")
    print(distortion)

    print("\nRotation Vectors:")
    print(r_vecs)

    print("\nTranslation Vectors:")
    print(t_vecs)
