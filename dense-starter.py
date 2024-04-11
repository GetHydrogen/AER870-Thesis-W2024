import cv2 as cv
import numpy as np
import pyrealsense2 as rs

# Parameters for Shi-Tomasi corner detection
feature_params = dict(maxCorners=200, qualityLevel=0.2, minDistance=2, blockSize=7)
# Parameters for Lucas-Kanade optical flow
lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))
# Variable for color to draw optical flow track
color = (0, 255, 0)

# Initialize RealSense camera
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)  # Adjust resolution and frame rate as needed
pipeline.start(config)

# Variable for storing the previous frame
prev_frame = None

try:
    while True:
        # Wait for a new frame from the camera
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        # Convert RealSense frame to numpy array
        frame = np.asanyarray(color_frame.get_data())

        # Convert the frame to grayscale
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        # If it's not the first frame, track optical flow
        if prev_frame is not None:
            # Calculates sparse optical flow by Lucas-Kanade method
            next, status, error = cv.calcOpticalFlowPyrLK(prev_gray, gray, prev, None, **lk_params)
            # Selects good feature points for previous position
            good_old = prev[status == 1].astype(int)
            # Selects good feature points for next position
            good_new = next[status == 1].astype(int)
            # Draws the optical flow tracks
            for i, (new, old) in enumerate(zip(good_new, good_old)):
                # Returns a contiguous flattened array as (x, y) coordinates for new point
                a, b = new.ravel()
                # Returns a contiguous flattened array as (x, y) coordinates for old point
                c, d = old.ravel()
                # Draws line between new and old position with green color and 2 thickness
                frame = cv.line(frame, (a, b), (c, d), color, 2)
                # Draws filled circle (thickness of -1) at new position with green color and radius of 3
                frame = cv.circle(frame, (a, b), 3, color, -1)

        # Display the frame with optical flow tracks
        cv.imshow("sparse optical flow", frame)

        # Update previous frame and previous feature points
        prev_frame = frame.copy()
        prev_gray = gray.copy()
        prev = cv.goodFeaturesToTrack(prev_gray, mask=None, **feature_params)

        # Check for exit key
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # Stop streaming and close OpenCV windows
    pipeline.stop()
    cv.destroyAllWindows()
