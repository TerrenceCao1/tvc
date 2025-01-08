import cv2
import math
import numpy as np
from collections import deque
import argparse


def calculate_ground_fov_km(altitude_km, fov_degrees):
    """Calculate the ground field of view in kilometers."""
    fov_radians = math.radians(fov_degrees)
    return 2 * altitude_km * math.tan(fov_radians / 2)


def calculate_scaling_factor(ground_fov_km, frame_width):
    """Calculate the scaling factor (km per pixel)."""
    return ground_fov_km / frame_width


def calculate_expected_distance_per_frame(speed_kmh, fps):
    """Calculate the expected ground distance traveled per frame."""
    return (speed_kmh / 3600) * (1 / fps)


def calculate_observed_distance(avg_movement, scaling_factor):
    """Calculate the observed ground distance based on pixel movement."""
    return avg_movement * scaling_factor


class ISSSpeedEstimator:
    def __init__(self, video_path, video_fps=0, mask_path=None, visualize=True, debug=True):
        self.video_path = video_path
        self.mask_path = mask_path
        self.visualize = visualize
        self.debug = debug
        self.video_fps = video_fps

        # ISS constants
        self.ISS_ALTITUDE_KM = 408
        self.CAMERA_FOV_DEGREES = 45
        self.ISS_SPEED_KMH = 27580

        # Parameters for optical flow
        self.feature_params = dict(maxCorners=200, qualityLevel=0.01, minDistance=5, blockSize=7)
        self.lk_params = dict(winSize=(15, 15), maxLevel=2,
                              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

        # Feature tracking thresholds
        self.min_points_threshold = 200
        self.add_points_fraction = 0.2
        self.screen_margin = 20

        # Data buffers for speed smoothing
        self.movement_30_frame_avg = deque(maxlen=30)
        self.movement_100_frame_avg = deque(maxlen=100)
        self.smoothed_speed = None

        # Other attributes
        self.cap = None
        self.mask = None
        self.old_gray = None
        self.p0 = None
        self.scaling_factor = None
        self.fps = None

    def initialize(self):
        """Initialize the video capture and calculate scaling factors."""
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            raise RuntimeError(f"Error: Could not open video at {self.video_path}.")

        ret, old_frame = self.cap.read()
        if not ret:
            raise RuntimeError("Error: Could not read the first frame.")

        self.old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

        if self.mask_path:
            self.mask = cv2.imread(self.mask_path, cv2.IMREAD_GRAYSCALE)
            if self.mask is None:
                raise RuntimeError(f"Error: Could not read mask image at {self.mask_path}.")
            self.mask = cv2.resize(self.mask, (self.old_gray.shape[1], self.old_gray.shape[0]))

        ground_fov_km = calculate_ground_fov_km(self.ISS_ALTITUDE_KM, self.CAMERA_FOV_DEGREES)
        self.scaling_factor = calculate_scaling_factor(ground_fov_km, self.old_gray.shape[1])
        if self.video_fps <= 0:
            self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        else:
            self.fps = self.video_fps
        

        if self.fps <= 0:
            raise RuntimeError("Error: Invalid FPS retrieved from video metadata. Please verify the video source.")

        if self.debug:
            print(f"Initialization: Scaling Factor: {self.scaling_factor:.5f}, FPS: {self.fps:.2f}")

        self.p0 = cv2.goodFeaturesToTrack(self.old_gray, mask=self.mask, **self.feature_params)

    def process_optical_flow(self, frame_gray):
        """Calculate optical flow and filter valid points."""
        p1, st, err = cv2.calcOpticalFlowPyrLK(self.old_gray, frame_gray, self.p0, None, **self.lk_params)
        if p1 is not None:
            good_new = p1[st == 1]
            good_old = self.p0[st == 1]

            h, w = frame_gray.shape
            good_new_filtered = []
            good_old_filtered = []

            for new, old in zip(good_new, good_old):
                x_new, y_new = new.ravel()
                if self.screen_margin < x_new < w - self.screen_margin and self.screen_margin < y_new < h - self.screen_margin:
                    good_new_filtered.append(new)
                    good_old_filtered.append(old)

            return np.array(good_new_filtered), np.array(good_old_filtered)
        return None, None

    def update_feature_points(self):
        """Add new feature points if existing points are insufficient."""
        if self.p0 is None or len(self.p0) < self.min_points_threshold:
            new_points = cv2.goodFeaturesToTrack(self.old_gray, mask=self.mask, **self.feature_params)
            if new_points is not None:
                num_points_to_add = int(self.add_points_fraction * self.min_points_threshold)
                self.p0 = np.vstack((self.p0, new_points[:num_points_to_add])) if self.p0 is not None else new_points[:num_points_to_add]

    def calculate_speed(self, avg_movement):
        """Calculate and smooth the speed based on average movement."""
        current_speed = avg_movement * self.scaling_factor * self.fps * 3600
        if self.smoothed_speed is None:
            self.smoothed_speed = current_speed
        else:
            self.smoothed_speed = 0.05 * current_speed + 0.95 * self.smoothed_speed
        return current_speed

    def run(self):
        """Process video frames to estimate ISS speed."""
        frame_count = 0

        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break

            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            if self.p0 is not None and len(self.p0) > 0:
                good_new, good_old = self.process_optical_flow(frame_gray)
                if good_new is not None and len(good_new) > 0:
                    avg_movement = np.linalg.norm(good_new - good_old, axis=1).mean()
                    self.movement_30_frame_avg.append(avg_movement)
                    self.movement_100_frame_avg.append(avg_movement)

                    # Calculate speeds
                    current_speed = self.calculate_speed(avg_movement)
                    expected_distance_per_frame_km = calculate_expected_distance_per_frame(self.ISS_SPEED_KMH, self.fps)
                    observed_distance_per_frame_km = calculate_observed_distance(avg_movement, self.scaling_factor)

                    if self.debug:
                        print(f"Frame {frame_count}: Avg Movement: {avg_movement:.2f}, Scaling Factor: {self.scaling_factor:.5f}, FPS: {self.fps:.2f}, "
                              f"Observed Dist/Frame: {observed_distance_per_frame_km:.5f}, Expected Dist/Frame: {expected_distance_per_frame_km:.5f}, "
                              f"Smoothed Speed: {self.smoothed_speed:.2f} km/h")

                    if self.visualize:
                        for new, old in zip(good_new, good_old):
                            a, b = new.ravel()
                            c, d = old.ravel()
                            frame = cv2.line(frame, (int(a), int(b)), (int(c), int(d)), (0, 255, 0), 2)
                            frame = cv2.circle(frame, (int(a), int(b)), 5, (0, 0, 255), -1)

                    self.p0 = good_new.reshape(-1, 1, 2)

            self.update_feature_points()
            self.old_gray = frame_gray.copy()
            frame_count += 1

            if self.visualize:
                resized_frame = cv2.resize(frame, (frame.shape[1] // 2, frame.shape[0] // 2))
                cv2.imshow('ISS Speed Estimation', resized_frame)
                if cv2.waitKey(30) & 0xFF == ord('q'):
                    break

        self.cap.release()
        cv2.destroyAllWindows()


def main():
    """Parse arguments and run the ISS speed estimator."""
    parser = argparse.ArgumentParser(description="ISS Speed Estimation")
    parser.add_argument('--video', type=str, required=True, help="Path to the video file.")
    parser.add_argument('--mask', type=str, help="Path to the mask image.")
    args = parser.parse_args()

    estimator = ISSSpeedEstimator(video_path=args.video, video_fps=30, mask_path=args.mask, visualize=True, debug=True)
    estimator.initialize()
    estimator.run()


if __name__ == "__main__":
    main()
