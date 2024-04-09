#%%
import pyrealsense2 as rs
import numpy as np
import cv2 as cv2

pipe = rs.pipeline()
cfg  = rs.config()

cfg.enable_stream(rs.stream.color, 640,480, rs.format.bgr8, 30)

pipe.start(cfg)

while True:
    frame = pipe.wait_for_frames()
    color_frame = frame.get_color_frame()

    color_image = np.asanyarray(color_frame.get_data())

    gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

    cv2.imshow('rgb', color_image)

    if cv2.waitKey(1) == ord('q'):
        break

pipe.stop()
print("Hello World")
#%%
import numpy as np
import cv2
import pyrealsense2 as rs

def main():
    # Configure depth and color streams
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    # Start streaming
    pipeline.start(config)

    # Create a Lucas-Kanade optical flow object
    lk_params = dict(winSize=(15, 15),
                     maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    
    # Create some random colors
    color = np.random.randint(0, 255, (100, 3))

    # Take first frame and find corners in it
    ret, old_frame = pipeline.wait_for_frames()
    old_frame = np.asanyarray(old_frame.get_data())
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7))

    # Create a mask image for drawing purposes
    mask = np.zeros_like(old_frame)

    while True:
        # Wait for a new frame
        ret, frame = pipeline.wait_for_frames()
        frame = np.asanyarray(frame.get_data())
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Calculate optical flow
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

        # Select good points
        good_new = p1[st == 1]
        good_old = p0[st == 1]

        # Draw the tracks and velocity vectors
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel().astype(int)
            c, d = old.ravel().astype(int)
            mask = cv2.line(mask, (a, b), (c, d), color[i].tolist(), 2)
            frame = cv2.circle(frame, (a, b), 5, color[i].tolist(), -1)
            # Calculate velocity
            velocity = np.sqrt((a - c)**2 + (b - d)**2)
            # Display velocity
            cv2.putText(frame, f'{velocity:.2f}', (a, b), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        img = cv2.add(frame, mask)

        # Display the resulting frame
        cv2.imshow('Frame', img)

        # Exit if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Now update the previous frame and previous points
        old_gray = frame_gray.copy()
        p0 = good_new.reshape(-1, 1, 2)

    # Release the camera and close OpenCV windows
    pipeline.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()


