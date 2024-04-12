import numpy as np
import cv2
import pyrealsense2 as rs

# Initialize a global list to store velocities
global_velocities = []

def run_tracking(pipeline, fx, fy, fps, Z):
    global global_velocities  # Declare the use of the global variable

    # Parameters for Lucas-Kanade optical flow
    lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
    color = np.random.randint(0, 255, (100, 3))

    # Wait for the first valid frame to start tracking
    while True:
        first_frame = pipeline.wait_for_frames().get_color_frame()
        if first_frame:
            old_frame = np.asanyarray(first_frame.get_data())
            old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
            p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
            if p0 is not None:
                break
        print("Waiting for valid frames...")

    mask = np.zeros_like(old_frame)

    while True:
        frame = pipeline.wait_for_frames().get_color_frame()
        if not frame:
            continue  # Skip this iteration if frame is not available

        frame_image = np.asanyarray(frame.get_data())
        frame_gray = cv2.cvtColor(frame_image, cv2.COLOR_BGR2GRAY)

        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
        if p1 is None or st is None:
            raise TypeError("Optical flow computation failed or no good points.")

        good_new = p1[st == 1]
        good_old = p0[st == 1]

        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()
            mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), color[i].tolist(), 2)
            frame_image = cv2.circle(frame_image, (int(a), int(b)), 5, color[i].tolist(), -1)
            
            # Calculate displacement in pixels
            delta_x, delta_y = a - c, b - d
            # Convert displacement to meters
            Delta_X = (delta_x * Z) / fx
            Delta_Y = (delta_y * Z) / fy
            # Calculate velocity in meters/second
            velocity_m_s = fps * np.sqrt(Delta_X**2 + Delta_Y**2)
            global_velocities.append(velocity_m_s)  # Append velocity to the global list
            
            cv2.putText(frame_image, f'Vel: {velocity_m_s:.2f} m/s', (int(a), int(b)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        img = cv2.add(frame_image, mask)
        cv2.imshow('Frame', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        old_gray = frame_gray.copy()
        p0 = good_new.reshape(-1, 1, 2)

    # After stopping the display, you can convert the list to a numpy array if needed
    velocity_array = np.array(global_velocities)
    print("Global velocities stored in array:", velocity_array)

def main():
    # Camera matrix and parameters
    fx = 385.5986022949219
    fy = 385.1598815917969
    fps = 30  # Frame rate of the camera
    Z = 2.0   # Assume a depth of 2 meters - adjust based on your application

    # Configure and start the RealSense pipeline
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipeline.start(config)

    try:
        while True:
            try:
                run_tracking(pipeline, fx, fy, fps, Z)
            except TypeError as e:
                print("Caught TypeError, restarting tracking:", e)
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        print("Final velocities:", global_velocities)  # Output the global velocity list

if __name__ == "__main__":
    main()
