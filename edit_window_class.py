import PySimpleGUI as sg
import manual_rate_class as mrc
import cv2
import numpy as np
import time
import re
import os.path
import pandas as pd
# import sys


class TableGui:
    # def __init__(self, df, dcpd_list, inhibitor_list, folder_name=None, file_path=None):
    def __init__(self, df, dcpd_list, inhibitor_list, folder_name=None):
        self.df = df
        self.display_df_columns = ['DCPD', 'ENB', 'Inhibitor_name', 'DHF', 'FROMP_worked', 'Front_rate_mm/s', 'Note']
        self.display_df = self.df[self.display_df_columns]
        self.display_df = self.display_df.round({'Front_rate_mm/s': 2})
        self.data = [self.display_df.columns.values.tolist()] + self.display_df.values.tolist()
        self.edit_columns = ['DCPD', 'Inhibitor_name',  'DHF', 'Note']
        self.layout = self.make_layout()
        self.main_window = sg.Window('Table', self.layout)
        self.edit_window = None
        self.row_idx = None
        # self.current_directory = os.path.dirname(os.path.abspath(__file__))
        self.current_directory = os.getcwd()
        # self.current_directory = os.path.dirname(os.path.abspath(sys.argv[0]))
        self.dcpd_list = dcpd_list
        self.inhibitor_list = inhibitor_list
        self.display_df_initial = self.display_df.copy()
        self.changed = False
        if folder_name is None:
            self.folder_path = self.current_directory
        else:
            self.folder_name = folder_name
            self.folder_path = os.path.join(self.current_directory, self.folder_name)
        # if file_path is None:
        #     pass
        # else:
        #     self.folder_path = os.path.dirname(file_path)
        #     self.current_directory = self.folder_path

    def get_incremented_filename(self, filename):
        basename, extension = os.path.splitext(filename)
        # If the filename already contains a sequence number, strip it out and start incrementing from there.
        match = re.match(r"^(.*?)-(\d+)$", basename)
        if match:
            basename = match.group(1)
            seq = int(match.group(2))
        else:
            seq = 1  # Start the sequence from 1

        # If the file without a sequence number does not exist, return the original filename
        if not os.path.exists(os.path.join(self.folder_path, f"{basename}{extension}")):
            return filename

        # If the file exists, keep incrementing the sequence number until we find a filename that doesn't exist.
        while os.path.exists(os.path.join(self.folder_path, f"{basename}-{seq}{extension}")):
            seq += 1

        # Return the incremented filename.
        return f"{basename}-{seq}{extension}"

    def make_layout(self):
        # col_widths = [{"key": "-COL{}-".format(i), "size": 20} for i in range(len(self.display_df.columns))]
        col_widths = [15 for _ in range(len(self.display_df.columns))]
        return [[sg.Table(values=self.data[1:], headings=self.display_df.columns.tolist(),
                          display_row_numbers=True, auto_size_columns=False,
                          col_widths=col_widths, key='-TABLE-', justification='center', enable_events=True)],
                [sg.Button('Edit'), sg.Button('Recalculate Front Rate'), sg.Button('Exit')]]

    def make_edit_window(self, row, columns):
        layout = []
        for i, value in enumerate(row):
            col_key = self.display_df.columns[i]
            if col_key in columns:
                if col_key == 'DCPD':  # Check if the column name is 'DCPD'
                    # If the column name is 'DCPD', create a dropdown menu with the specified options
                    options = self.dcpd_list
                    layout.append([sg.Text(self.display_df.columns[i]),
                                   sg.DropDown(options, default_value=value, key=col_key)])
                elif col_key == 'Inhibitor_name':  # Check if the column name is 'Inhibitor_name'
                    # If the column name is 'Inhibitor_name', create a dropdown menu with the specified options
                    options = self.inhibitor_list
                    layout.append([sg.Text(self.display_df.columns[i]),
                                   sg.DropDown(options, default_value=value, key=col_key)])
                elif col_key == 'DHF':  # Check if the column name is 'DHF'
                    layout.append([sg.Text('%Inhibitor'),
                                   sg.Input(default_text=value, key=col_key)])
                # checks if there is a boolean value in the column and will format the edit window accordingly
                elif isinstance(value, bool):
                    # if boolean, create a drop down menu with True and False options
                    options = ['True', 'False']
                    default_value = str(value)  # Convert boolean to string
                    layout.append([sg.Text(self.display_df.columns[i]),
                                   sg.DropDown(options, default_value=default_value, key=col_key)])
                else:
                    # if not boolean, create a text box with the current value
                    layout.append([sg.Text(self.display_df.columns[i]), sg.Input(default_text=value, key=col_key)])
        layout.append([sg.Button('Save'), sg.Button('Cancel')])
        return sg.Window('Edit Row', layout, modal=True)

    def run(self):
        while True:
            if self.edit_window is None:
                event, values = self.main_window.read(timeout=24)
                if event in (sg.WINDOW_CLOSED, 'Exit'):
                    break
                # elif event == 'Edit' and values['-TABLE-']:
                #     self.row_idx = values['-TABLE-'][0]
                #     self.row_values = self.data[self.row_idx + 1]
                #     self.edit_window = self.make_edit_window()
                #     time.sleep(0.5)

                elif event == 'Edit' and values['-TABLE-']:
                    self.row_idx = values['-TABLE-'][0]
                    self.row_values = self.data[self.row_idx + 1]

                    # Get the editable columns for the selected row
                    editable_columns = [self.display_df.columns[i] for i in range(len(self.row_values)) if
                                        self.display_df.columns[i] in self.edit_columns]

                    self.edit_window = self.make_edit_window(self.row_values, editable_columns)
                    time.sleep(0.5)

                elif event == 'Recalculate Front Rate':
                    response = sg.popup_yes_no('Did the sample FROMP?', title='Query', keep_on_top=True)
                    fromp_worked = True if response == 'Yes' else False

                    self.row_idx = values['-TABLE-'][0]

                    selected_row = self.df.iloc[self.row_idx]  # Access the full df for the selected row

                    self.df.loc[self.row_idx, 'FROMP_worked'] = fromp_worked

                    if fromp_worked:
                        edit_analyzer = mrc.FrameAnalyzer()
                        video_filename = selected_row['video_filename']
                        video_path = os.path.join(self.folder_path, video_filename)
                        cap = cv2.VideoCapture(video_path)
                        if not cap.isOpened():
                            print(f"Cannot open video at {video_path}")
                            exit()
                        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                        fps = cap.get(cv2.CAP_PROP_FPS)
                        try:
                            video_frame_timestamps = selected_row['frame_timestamps']
                        except KeyError:
                            video_frame_timestamps = None

                        if video_frame_timestamps is None or str(video_frame_timestamps).lower() == 'nan':
                            video_frame_timestamps = np.linspace(0, total_frames, total_frames) / fps
                        else:
                            video_frame_timestamps = np.array(video_frame_timestamps)

                        video_clicked_points = selected_row['clicked_points']
                        if str(video_clicked_points).lower() == 'nan':
                            video_clicked_points = pd.NA

                        timestamp_frame_tuple_list = []

                        for timestamp in video_frame_timestamps:
                            ret, frame = cap.read()
                            if not ret:
                                break
                            timestamp_frame_tuple_list.append((timestamp, frame))

                        front_rate, clicked_points, frame_timestamps = \
                            edit_analyzer.reimport_and_reanalyze(timestamp_frame_tuple_list, video_clicked_points)
                        if front_rate is None or clicked_points is None:
                            self.df.loc[self.row_idx, 'Front_rate_mm/s'] = pd.NA
                            self.df.loc[self.row_idx, 'clicked_points'] = pd.NA
                        else:
                            self.df.loc[self.row_idx, 'Front_rate_mm/s'] = front_rate
                            self.df['clicked_points'] = self.df['clicked_points'].astype(object)
                            self.df.loc[self.row_idx, 'clicked_points'] = [clicked_points]

                        cap.release()
                        del edit_analyzer
                        time.sleep(0.5)
                    else:
                        self.df.loc[self.row_idx, 'Front_rate_mm/s'] = pd.NA
                        self.df.loc[self.row_idx, 'clicked_points'] = pd.NA

                    # Update the data variable used to populate the table
                    self.display_df = self.df[self.display_df_columns]
                    self.data = [self.display_df.columns.values.tolist()] + self.display_df.values.tolist()

                    # Update the table in the main window
                    self.main_window['-TABLE-'].update(values=self.data[1:])  # Update values without the header row

            else:
                event, values = self.edit_window.read(timeout=24)
                if event in (sg.WINDOW_CLOSED, 'Cancel'):
                    self.edit_window.close()
                    self.edit_window = None
                elif event == 'Save':
                    change_filenames = False
                    edited_values = {}  # Store edited values in a dictionary

                    for col_key in self.edit_columns:
                        if col_key in values:
                            edited_value = values[col_key]
                            original_value = self.row_values[self.display_df_columns.index(col_key)]

                            if edited_value != original_value:
                                edited_values[col_key] = edited_value

                                if col_key in ['DCPD', 'DHF']:
                                    change_filenames = True

                    # Check 'Inhibitor_name' and 'DHF' values to ensure consistency
                    if 'Inhibitor_name' in edited_values or 'DHF' in edited_values:
                        inhibitor_name = edited_values.get('Inhibitor_name',
                                                           self.df.loc[self.row_idx, 'Inhibitor_name'])
                        dhf = edited_values.get('DHF', self.df.loc[self.row_idx, 'DHF'])

                        if dhf == "0" and inhibitor_name != 'None':
                            edited_values['Inhibitor_name'] = 'None'

                        if inhibitor_name == 'None' and dhf != "0":
                            edited_values['DHF'] = "0"

                    # Update the original DataFrame with edited values
                    if edited_values:
                        self.df.loc[self.row_idx, list(edited_values.keys())] = list(edited_values.values())

                        # Check if 'Inhibitor_name' is 'None' and set 'DHF' to '0' if it is
                        if 'Inhibitor_name' in edited_values and edited_values['Inhibitor_name'] == 'None':
                            self.df.loc[self.row_idx, 'DHF'] = "0"

                    if change_filenames:
                        # Calculate and update ENB value
                        self.df.loc[self.row_idx, 'ENB'] = 100 - int(self.df.loc[self.row_idx, 'DCPD'])

                        # Updating filenames
                        video_filename_old = self.df.loc[self.row_idx, 'video_filename']
                        temperature_filename_old = self.df.loc[self.row_idx, 'temperature_filename']

                        pattern = re.compile(
                            r"(?P<identifier>[\w\-]+)_DCPD(?P<dcpd>\d+)_DHF(?P<dhf>\d+)(?:-(?P<run>\d+))?(?:\.csv|\.mp4)?")

                        match = pattern.match(video_filename_old)

                        if match:
                            identifier = match.group("identifier")
                            dhf_value = self.df.loc[self.row_idx, 'DHF']

                            if dhf_value.strip().lower() in ['', 'none', 'n/a', 'na', 'nan']:
                                self.df.loc[self.row_idx, 'DHF'] = 0

                            dcpd_value = str(self.df.loc[self.row_idx, 'DCPD']).zfill(2)
                            dhf_value = str(self.df.loc[self.row_idx, 'DHF']).zfill(3)

                            video_filename_old_full = os.path.join(self.folder_path, video_filename_old)
                            temperature_filename_old_full = os.path.join(self.folder_path, temperature_filename_old)

                            video_filename_new = self.get_incremented_filename(
                                f"{identifier}_DCPD{dcpd_value}_DHF{dhf_value}.mp4")

                            temperature_filename_new = self.get_incremented_filename(
                                f"{identifier}_DCPD{dcpd_value}_DHF{dhf_value}.csv")

                            video_filename_new_full = os.path.join(self.folder_path, video_filename_new)
                            temperature_filename_new_full = os.path.join(self.folder_path, temperature_filename_new)
                            os.rename(video_filename_old_full, video_filename_new_full)
                            os.rename(temperature_filename_old_full, temperature_filename_new_full)

                            # Update DataFrame with new filenames
                            self.df.loc[self.row_idx, 'video_filename'] = video_filename_new
                            self.df.loc[self.row_idx, 'temperature_filename'] = temperature_filename_new

                    # Update display DataFrame, data, and the table in the main window
                    self.display_df = self.df[self.display_df_columns]
                    self.data = [self.display_df.columns.values.tolist()] + self.display_df.values.tolist()
                    self.main_window['-TABLE-'].update(values=self.data[1:])

                    # Close the edit window
                    self.edit_window.close()
                    self.edit_window = None

        if self.edit_window:  # Make sure self.edit_window is not None before trying to close it
            self.edit_window.close()
        self.main_window.close()
        if not self.display_df.equals(self.display_df_initial):
            print("File edited")
        return self.df


if __name__ == '__main__':
    print("Please run the main script.")