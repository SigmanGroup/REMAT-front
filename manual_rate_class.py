import cv2
import PySimpleGUI as sg
import numpy as np


class FrameAnalyzer:

    def __init__(self):
        self.frames = []
        self.timestamps = []
        self.clicked_points = []
        self.current_frame_index = 0
        self.step_size_constant = 5
        self.step_size = 150
        self.show_all_points = False
        self.reference_line = ((375, 166), (375, 254))
        self.pixel_to_mm_ratio = 0.113636364
        self.drawing_line = False
        self.reference_points = []
        self.panning = False
        self.panning_enabled = False
        self.start_event_frame = None
        self.end_event_frame = None
        self.event_frames = []
        self.current_event_frame_index = 0
        self.event_frames_set = False

    def calculate_equidistant_frames(self):
        """Calculates 10 equidistant frames between the start and end frames"""
        if self.start_event_frame is not None and self.end_event_frame is not None:
            step = (self.end_event_frame - self.start_event_frame) // 9
            self.event_frames = [self.start_event_frame + i * step for i in range(10)]
            self.event_frames_set = True
            self.current_event_frame_index = 0
            self.current_frame_index = self.event_frames[self.current_event_frame_index]  # Move to the 'i' frame

    def reset_event_frames(self):
        """Resets the event frames"""
        self.start_event_frame = None
        self.end_event_frame = None
        self.event_frames_set = False
        self.event_frames = []
        self.current_event_frame_index = 0

    def add_frame(self, timestamp, frame):
        self.timestamps.append(timestamp)
        self.frames.append(frame)

    def import_and_analyze(self, timestamp_frame_tuple_list):
        for time, frame in timestamp_frame_tuple_list:
            self.timestamps.append(time)
            self.frames.append(frame)
        return self.analyze(), self.clicked_points, self.timestamps

    def reimport_and_reanalyze(self, timestamp_frame_tuple_list, clicked_points):
        for time, frame in timestamp_frame_tuple_list:
            self.timestamps.append(time)
            self.frames.append(frame)
        if np.ndim(clicked_points) == 0:
            pass
        else:
            self.clicked_points = clicked_points
            if not self.clicked_points == []:
                self.start_event_frame = self.clicked_points[0][0]
                self.end_event_frame = self.clicked_points[-1][0]
                self.calculate_equidistant_frames()
        return self.analyze(), self.clicked_points, self.timestamps

    def handle_mouse_click(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.panning_enabled:
                self.panning = True
            elif not self.panning:
                if self.drawing_line:
                    self.reference_points.append((x, y))
                    if len(self.reference_points) == 2:
                        self.draw_reference_line(*self.reference_points[0], *self.reference_points[1])
                        self.drawing_line = False
                else:
                    self.clicked_points.append((self.current_frame_index, x, y))
                    self.display_frame(self.frames[self.current_frame_index].copy(), self.current_frame_index)
        elif event == cv2.EVENT_LBUTTONUP:
            self.panning = False
        elif event == cv2.EVENT_RBUTTONDOWN:
            if not self.panning:
                for point in self.clicked_points:
                    frame_num, px, py = point
                    if (frame_num == self.current_frame_index or self.show_all_points) and abs(px - x) <= 5 and abs(
                            py - y) <= 5:
                        self.clicked_points.remove(point)
                        break
                self.display_frame(self.frames[self.current_frame_index].copy(), self.current_frame_index)


    def display_frame(self, frame, frame_index):
        if cv2.getWindowProperty('Frame', 0) == -1:  # check if the window is closed
            # If closed, recreate the window
            cv2.namedWindow('Frame', cv2.WINDOW_AUTOSIZE | cv2.WINDOW_GUI_NORMAL)
            cv2.setMouseCallback('Frame', self.handle_mouse_click)
        for point_index, (frame_num, x, y) in enumerate(self.clicked_points):
            if self.show_all_points or frame_num == frame_index:
                cv2.putText(frame, f"{point_index + 1}", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
                cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)
        cv2.putText(frame, f"Frame: {frame_index}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, f"Step: {self.step_size}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        if self.reference_line is not None:
            cv2.line(frame, *self.reference_line, (255, 0, 0), 1)
        cv2.imshow("Frame", frame)

    def remove_points_on_current_frame(self):
        self.clicked_points = [(frame_num, x, y) for frame_num, x, y in self.clicked_points if
                               frame_num != self.current_frame_index]

    def remove_all_points_with_confirmation(self):
        layout = [[sg.Text("Are you sure you want to delete all points?")], [sg.Button("OK"), sg.Button("Cancel")]]
        window = sg.Window("Confirmation", layout)
        while True:
            event, values = window.read()
            if event == "OK":
                self.clicked_points = []
                break
            elif event == "Cancel" or event == sg.WINDOW_CLOSED:
                break
        window.close()

    def calculate_velocity(self):
        if len(self.clicked_points) < 2 or self.pixel_to_mm_ratio is None:
            return None

        # Sort the points by frame number
        sorted_points = sorted(self.clicked_points, key=lambda p: p[0])

        # Group points by frame and average y-coordinate for each frame
        from collections import defaultdict
        y_averages = defaultdict(list)
        for frame_num, _, y in sorted_points:
            y_averages[frame_num].append(y)

        averaged_points = [(frame_num, sum(y_values) / len(y_values)) for frame_num, y_values in y_averages.items()]

        # Calculate the differences in y and time between each pair of averaged points
        differences = []
        for i in range(1, len(averaged_points)):
            frame_num1, y1 = averaged_points[i - 1]
            frame_num2, y2 = averaged_points[i]

            delta_y = abs(y2 - y1) * self.pixel_to_mm_ratio  # Convert pixels to mm
            delta_t = abs(self.timestamps[frame_num2] - self.timestamps[frame_num1])

            if delta_t != 0:  # Avoid division by zero
                rate = delta_y / delta_t
                differences.append(rate)

        if differences:
            # Calculate and return the average rate
            avg_rate = sum(differences) / len(differences)
            avg_rate = round(avg_rate, 2)
        else:
            avg_rate = None

        return avg_rate

    def draw_reference_line(self, x1, y1, x2, y2):
        self.reference_line = ((x1, y1), (x2, y2))
        self.pixel_to_mm_ratio = 10 / np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        self.reference_points = []  # Clear the reference points after the line is drawn
        self.display_frame(self.frames[self.current_frame_index].copy(), self.current_frame_index)

    def analyze(self):
        cv2.namedWindow("Frame", cv2.WINDOW_AUTOSIZE | cv2.WINDOW_GUI_NORMAL)
        cv2.setMouseCallback("Frame", self.handle_mouse_click)

        try:
            while True:
                # Check if window is open
                if cv2.getWindowProperty('Frame', cv2.WND_PROP_VISIBLE) < 1:
                    # Break the loop if window is closed
                    break

                self.display_frame(self.frames[self.current_frame_index].copy(), self.current_frame_index)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('p'):  # Use 'p' to toggle panning
                    self.panning = not self.panning

                elif key == ord('r'):
                    if self.show_all_points:
                        self.remove_all_points_with_confirmation()
                    else:
                        self.remove_points_on_current_frame()

                elif key == ord('s'):
                    self.show_all_points = not self.show_all_points

                # this doesn't work with when calling from the main gui
                # elif key == ord('v'):
                #     print(self.clicked_points)

                elif key == ord('l') and len(self.reference_points) < 2:
                    self.drawing_line = True

                elif key in [ord(str(i)) for i in range(10)]:
                    if key == ord('0'):
                        # TODO: change to multiply the step size by 10?
                        self.step_size = self.step_size * 10
                    else:
                        # TODO: change increments to be exponential?
                        self.step_size = int(chr(key)) * self.step_size_constant

                elif key == ord('i'):
                    self.start_event_frame = self.current_frame_index
                    if self.end_event_frame is not None:
                        self.calculate_equidistant_frames()
                elif key == ord('o'):
                    self.end_event_frame = self.current_frame_index
                    if self.start_event_frame is not None:
                        self.calculate_equidistant_frames()
                elif key == ord('u'):
                    self.reset_event_frames()

                # Limit frame navigation to loop if event frames are set
                if self.event_frames_set:
                    if key == ord('a'):
                        self.current_event_frame_index = (self.current_event_frame_index - 1) % len(self.event_frames)
                        self.current_frame_index = self.event_frames[self.current_event_frame_index]
                    elif key == ord('d'):
                        self.current_event_frame_index = (self.current_event_frame_index + 1) % len(self.event_frames)
                        self.current_frame_index = self.event_frames[self.current_event_frame_index]
                else:
                    if key == ord('a') and self.current_frame_index - self.step_size >= 0:
                        self.current_frame_index -= self.step_size
                    elif key == ord('d') and self.current_frame_index + self.step_size < len(self.frames):
                        self.current_frame_index += self.step_size

        finally:
            cv2.destroyWindow("Frame")

        return self.calculate_velocity()


if __name__ == "__main__":
    print("Please run the main script.")