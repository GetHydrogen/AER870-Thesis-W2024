import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import pyrealsense2 as rs
import csv
import os


videoFolder = "/home/avl/AER870-Thesis-W2024/Videos"
plotFolder = "/home/avl/AER870-Thesis-W2024/Plots"
logFolder = "/home/avl/AER870-Thesis-W2024/Logs"

TRIAL_ID = len(os.listdir(videoFolder))+1

fx = 385.5986022949219
fy = 385.1598815917969
fps = 30
Z = 2.0

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)


def LucasKanade_Sparse(fps, Z):
    velocities_data = []
    lk_hp = dict(winSize=(15, 15), maxLevel=2, criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))
    features_hp = dict(maxCorners=10, qualityLevel=0.3, minDistance=7, blockSize=7)
    color = np.random.randint(0, 255, (10, 3))

    video_filename = f"Videos/LKSP_{TRIAL_ID}.avi"

    fourcc = cv.VideoWriter_fourcc(*'XVID')
    out = cv.VideoWriter(video_filename, fourcc, fps, (640, 480))

    while True:
        first_frame = pipeline.wait_for_frames().get_color_frame()
        if first_frame:
            old_frame = np.asanyarray(first_frame.get_data())
            old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)
            p0 = cv.goodFeaturesToTrack(old_gray, mask=None, **features_hp)
            if p0 is not None:
                break

    mask = np.zeros_like(old_frame)

    while True:
        frame = pipeline.wait_for_frames().get_color_frame()
        if not frame:
            continue

        frame_image = np.asanyarray(frame.get_data())
        frame_gray = cv.cvtColor(frame_image, cv.COLOR_BGR2GRAY)

        p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_hp)
        if p1 is None or st is None:
            print("Error: Lost Features... Attempting Reorientation.")
            frame = pipeline.wait_for_frames().get_color_frame()
            frame_image = np.asanyarray(frame.get_data())
            frame_gray = cv.cvtColor(frame_image, cv.COLOR_BGR2GRAY)
            p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_hp)
            continue

        good_new = p1[st == 1]
        good_old = p0[st == 1]

        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()
            mask = cv.line(mask, (int(a), int(b)), (int(c), int(d)), color[i].tolist(), 2)
            frame_image = cv.circle(frame_image, (int(a), int(b)), 5, color[i].tolist(), -1)
            
            # Pixels per frame
            dx_ppf = a - c 
            dy_ppf = b - d
            # Meters per frame
            dx_mpf = (dx_ppf * Z) / fx
            dy_mpf = (dy_ppf * Z) / fy
            
            velocity = fps * np.sqrt(dx_mpf**2 + dy_mpf**2)
            velocities_data.append(velocity)

            cv.putText(frame_image, f'Vel: {velocity:.2f} m/s', (int(a), int(b)), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        img = cv.add(frame_image, mask)
        out.write(img)  # Write frame to video file
        cv.imshow('Frame', img)
        
        
        key = cv.waitKey(1)
        if key & 0xFF == ord('q'):
            break  # Exit the loop if 'q' is pressed

        old_gray = frame_gray.copy()
        p0 = good_new.reshape(-1, 1, 2)

    out.release()  # Release the video writer
    cv.destroyAllWindows()

    return velocities_data


def Farneback_Dense(fps, Z):
    velocities_data = []
    video_filename = f"Videos/FBDS_{TRIAL_ID}.avi"
    fourcc = cv.VideoWriter_fourcc(*'XVID')
    out = cv.VideoWriter(video_filename, fourcc, 5, (640, 480), isColor=True)

    try:
        for _ in range(30):
            pipeline.wait_for_frames()

        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        first_frame = np.asanyarray(color_frame.get_data())
        prev_gray = cv.cvtColor(first_frame, cv.COLOR_BGR2GRAY)
        mask = np.zeros_like(first_frame)
        mask[..., 1] = 255

        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()

            if not color_frame:
                continue

            frame = np.asanyarray(color_frame.get_data())
            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

            flow = cv.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            magnitude, angle = cv.cartToPolar(flow[..., 0], flow[..., 1])
            mask[..., 0] = angle * 180 / np.pi / 2
            mask[..., 2] = cv.normalize(magnitude, None, 0, 255, cv.NORM_MINMAX)
            rgb = cv.cvtColor(mask, cv.COLOR_HSV2BGR)

            ppf = np.mean(magnitude)
            
            px = (fx+fy) / 2
            velocity = (ppf / px) * Z * fps

            velocities_data.append(velocity)
            print(velocity)

            cv.putText(rgb, f"Velocity: {velocity:.2f} m/s", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv.LINE_AA)
            cv.imshow('Dense Optical Flow', rgb)
            
            out.write(rgb)

            prev_gray = gray

            if cv.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        pipeline.stop()
        cv.destroyAllWindows()
        out.release()

        return velocities_data
    
def logFiles(velocities_data, method, trial_id):

    logFilename = f"Logs/{method}_{trial_id}.csv"
    plotFilename = f"Plots/{method}_{trial_id}.png"
    
    with open(logFilename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Frame', 'Velocity (m/s)'])
        for i, velocity in enumerate(velocities_data):
            writer.writerow([i, velocity])

    plt.figure()
    plt.plot(velocities_data)
    plt.ylabel("Velocity (m/s)")
    plt.savefig(plotFilename)

    return