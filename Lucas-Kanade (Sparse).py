import numpy as np
import cv2
import matplotlib.pyplot as plt
import pyrealsense2 as rs
import csv
import os

velocities_data = []
video_counter = 1

def run_tracking(pipeline, fx, fy, fps, Z):
    video_counter = len(os.listdir('Videos/'))+1

    lk_hp = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    features_hp = dict(maxCorners=10, qualityLevel=0.3, minDistance=7, blockSize=7)
    color = np.random.randint(0, 255, (10, 3))

    video_filename = f'Videos/LkSp{video_counter}.avi'

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(video_filename, fourcc, fps, (640, 480))

    while True:
        first_frame = pipeline.wait_for_frames().get_color_frame()
        if first_frame:
            old_frame = np.asanyarray(first_frame.get_data())
            old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
            p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **features_hp)
            if p0 is not None:
                break

    mask = np.zeros_like(old_frame)

    while True:
        frame = pipeline.wait_for_frames().get_color_frame()
        if not frame:
            continue

        frame_image = np.asanyarray(frame.get_data())
        frame_gray = cv2.cvtColor(frame_image, cv2.COLOR_BGR2GRAY)

        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_hp)
        if p1 is None or st is None:
            raise TypeError("New references")

        good_new = p1[st == 1]
        good_old = p0[st == 1]

        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()
            mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), color[i].tolist(), 2)
            frame_image = cv2.circle(frame_image, (int(a), int(b)), 5, color[i].tolist(), -1)
            
            # Pixels per frame
            dx_ppf = a - c 
            dy_ppf = b - d
            # Meters per frame
            dx_mpf = (dx_ppf * Z) / fx
            dy_mpf = (dy_ppf * Z) / fy
            
            velocity = fps * np.sqrt(dx_mpf**2 + dy_mpf**2)
            velocities_data.append(velocity)
            
            cv2.putText(frame_image, f'Vel: {velocity:.2f} m/s', (int(a), int(b)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        img = cv2.add(frame_image, mask)
        out.write(img)  # Write frame to video file
        cv2.imshow('Frame', img)
        
        
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            break  # Exit the loop if 'q' is pressed

        old_gray = frame_gray.copy()
        p0 = good_new.reshape(-1, 1, 2)

    # Saving velocities_data as CSV
    with open('Logs/LkSp_data.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Velocity (m/s)'])
        for velocity in velocities_data:
            writer.writerow([velocity])

    out.release()  # Release the video writer

    video_counter += 1

    print("Final velocities:", velocities_data)
    plt.figure()
    plt.plot(velocities_data)
    plt.title("Drone Velocity Scatters")
    plt.ylabel("Velocity (m/s)")
    plt.savefig('LkSp_plot.png')

    #plt.show()
    cv2.destroyAllWindows()


def main():
    fx = 385.5986022949219
    fy = 385.1598815917969
    fps = 30
    Z = 2.0

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipeline.start(config)

    try:
        while True:
            try:
                run_tracking(pipeline, fx, fy, fps, Z)
            except TypeError as e:
                print("Tracking feature out of frame,", e)
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
