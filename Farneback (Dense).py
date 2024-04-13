import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import pyrealsense2 as rs
import csv

#Intel Realsense D455 Intrinsic
fx = 385.5986022949219
fy = 385.1598815917969
fps = 30

#Depth
Z = 2.0


pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

pipeline.start(config)

velocities_data = []

# Define the codec and create VideoWriter object
fourcc = cv.VideoWriter_fourcc(*'XVID')
out = cv.VideoWriter('FbDs.avi', fourcc, 5, (640, 480), isColor=True)

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

    with open('FbDs_data.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Frame', 'Velocity (m/s)'])
        for i, velocity in enumerate(velocities_data):
            writer.writerow([i, velocity])

    plt.figure()
    plt.plot(velocities_data)
    #plt.title("Drone Velocity Scatters")
    plt.ylabel("Velocity (m/s)")
    plt.savefig('FbDs_plot.png')
    #plt.show()
