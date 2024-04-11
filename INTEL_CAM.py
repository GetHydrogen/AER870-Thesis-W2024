import numpy as np
import cv2
import pyrealsense2 as rs

def main():
    K = np.array([639.0849609375, 0.0, 644.4653930664062, 0.0, 639.0849609375, 364.2340393066406, 0.0, 0.0, 1.0]).reshape(3, 3)
    focal_length_x = K[0, 0]
    sensor_width = 6.4


    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    pipeline.start(config)

    lk_params = dict(winSize=(15, 15),
                     maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    
    color = np.random.randint(0, 255, (100, 3))

    ret, old_frame = pipeline.wait_for_frames()
    old_frame = np.asanyarray(old_frame.get_data())
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7))

    mask = np.zeros_like(old_frame)

    while True:
        ret, frame = pipeline.wait_for_frames()
        frame = np.asanyarray(frame.get_data())
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

        good_new = p1[st == 1]
        good_old = p0[st == 1]

        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel().astype(int)
            c, d = old.ravel().astype(int)
            mask = cv2.line(mask, (a, b), (c, d), color[i].tolist(), 2)
            frame = cv2.circle(frame, (a, b), 5, color[i].tolist(), -1)

            velocity_pixel_frame = np.sqrt((a - c)**2 + (b - d)**2)
            velocity_real_world = (velocity_pixel_frame * sensor_width) / focal_length_x

            cv2.putText(frame, f'{velocity_real_world:.2f} m/s', (a, b), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        img = cv2.add(frame, mask)


        cv2.imshow('Frame', img)


        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


        old_gray = frame_gray.copy()
        p0 = good_new.reshape(-1, 1, 2)

    pipeline.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()


