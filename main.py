import cv2
import PySimpleGUI as sg
import time
import os
import re  # for savefile naming
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
from Phidget22.Devices.TemperatureSensor import *
import numpy as np
import pandas as pd
import manual_rate_class as mrc
import edit_window_class as ewc
import pickle

version = f'v2023-09-28'

if getattr(sys, 'frozen', False):
    # The application is running as a compiled executable
    exe_dir = os.path.dirname(os.path.abspath(sys.executable))
else:
    # The application is running as a script
    exe_dir = os.path.dirname(os.path.abspath(__file__))


# TODO: incorporate pathlib to handle file paths
# from pathlib import Path

# TODO: add a defaults file to store default values ex) Focus, inhibitor list, dcpd list, etc.
# TODO: add ability to restore camera connection with -CHECK- without restarting program
# TODO: add delete ability to edit window?

class VideoRecorder:
    def __init__(self, focus=210):
        self.start_time = None
        self.recording = False
        self.filename = 'output.mp4'
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FOURCC,
                     cv2.VideoWriter_fourcc(*"MJPG"))  # realtime compression of video from source also H.264 ("H264")  
        self.cap.set(cv2.CAP_PROP_FOCUS, focus)
        self.previous_frame = None
        self.timestamp_frame_tuple_list = []
        self.camera_present = True

        if not self.cap.isOpened():
            self.camera_present = False
            # note: self.cap.isOpened() returns False if no camera is connected. Try/except won't catch this error
            sg.Popup('Camera not found. Please check connection and restart program.')
            # exit()

    def start_recording(self, frame):
        print(f"recording start: {self.filename}")
        self.recording = True

    def stop_recording(self, reduce_frame_rate=True, target_fps=30):
        print(f"recording stop: {self.filename}")
        start_time = self.timestamp_frame_tuple_list[0][0]
        end_time = self.timestamp_frame_tuple_list[-1][0]
        elapsed_time_seconds = end_time - start_time
        frame_count = len(self.timestamp_frame_tuple_list)
        fps = frame_count / elapsed_time_seconds
        height, width, _ = self.timestamp_frame_tuple_list[0][1].shape

        if reduce_frame_rate:
            # reduces frame rate (subsample) to target_fps
            reduced_timestamp_frame_tuple_list = []
            video_writer = cv2.VideoWriter(self.filename, cv2.VideoWriter_fourcc(*"mp4v"), target_fps, (width, height))
            frame_interval = int(round(fps / target_fps))  # Calculate the frame interval for subsampling
            for n, frame_tuple in enumerate(self.timestamp_frame_tuple_list):
                timestamp, frame = frame_tuple
                if n % frame_interval == 0:
                    reduced_timestamp_frame_tuple_list.append(frame_tuple)
                    video_writer.write(frame)
                self.timestamp_frame_tuple_list = reduced_timestamp_frame_tuple_list.copy()
        else:
            video_writer = cv2.VideoWriter(self.filename, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
            for timestamp, frame in self.timestamp_frame_tuple_list:
                video_writer.write(frame)

        video_writer.release()
        self.recording = False

    def record_frame(self, frame):
        target_fps = 20000
        correction_factor = 20
        sleep_duration = 1 / (target_fps + correction_factor)
        if self.recording:
            self.timestamp_frame_tuple_list.append((time.time(), frame))
            # TODO: cleanup this sleep
            time.sleep(sleep_duration)

    def focus(self, focus):
        self.cap.set(cv2.CAP_PROP_FOCUS, focus)


class TempRecorder:
    def __init__(self):
        self.time_and_temp_tuple_list = []
        self.recording = False
        self.filename = 'output.csv'
        self.temperature_present = True
        # checks to see if temperature sensor is connected
        try:
            ch = TemperatureSensor()
            ch.openWaitForAttachment(1000)
            ch.close()
            self.temperature_present = True
        except:
            self.temperature_present = False
            sg.Popup('Temperature sensor not found. Please check connection and try again.')
            # exit()

    def start_recording(self):
        self.time_and_temp_tuple_list = []
        self.recording = True

    def stop_recording(self):
        df_temperature = pd.DataFrame(self.time_and_temp_tuple_list, columns=['Time', 'Temperature'])
        df_temperature.to_csv(self.filename, index=False)
        self.recording = False
        return df_temperature['Temperature'].tolist()

    def record_temp(self):
        if self.temperature_present:
            try:
                ch = TemperatureSensor()
                ch.openWaitForAttachment(1000)
                temperature = ch.getTemperature()
                ch.close()
            except:
                print('Temperature sensor not found. Please check connection.')
                temperature = np.nan
                self.temperature_present = False
        else:
            temperature = np.nan  # if no temperature sensor is present, record NA
        temperature_time = time.time()
        self.time_and_temp_tuple_list.append((temperature_time, temperature))
        return temperature_time, temperature
    
    def check_connection(self):
        try:
            ch = TemperatureSensor()
            ch.openWaitForAttachment(1000)
            ch.close()
            if self.temperature_present == False:
                self.temperature_present = True
                print('Temperature sensor found.')
        except:
            pass

class DataTable:
    def __init__(self, dcpd_list):
        self.result_list = []
        self.dcpd_list = dcpd_list
        dcpd_list_str = [str(number) for number in dcpd_list]
        self.table_header_list = ['Inhibitor name', 'Inh. mol%']
        self.table_header_list.extend(dcpd_list_str)

        # Calculate the column widths
        TABLE_TOTAL_COLUMN_WIDTHS = 62  # don't change this
        name_column_width = 18
        inh_mol_percent_column_width = 8
        remaining_width = TABLE_TOTAL_COLUMN_WIDTHS - name_column_width - inh_mol_percent_column_width
        remaining_columns = len(self.table_header_list) - 2  # minus 2 for name and inh mol percent columns
        remaining_columns_width_ea = remaining_width // remaining_columns

        # column_widths is a list of the widths of each column
        self.column_widths = [name_column_width, inh_mol_percent_column_width]
        self.column_widths.extend([remaining_columns_width_ea] * remaining_columns)

        # Initialize the DataFrame
        self.df = pd.DataFrame(columns=self.table_header_list)
        self.table_data = self.df.values.tolist()

    def update_data(self, df, DCPD_LIST):
        # takes the df and updates the table data
        df_main = df.copy()

        # Function to check for presence of each element from DCPD_LIST
        def check_dcpd_list_presence(row):
            presence_dict = dict.fromkeys(DCPD_LIST, 0)  # Initialize presence_dict with zeros
            # sg.popup(print(row['DCPD'])) #nan
            if isinstance(row['DCPD'], list):
                for dcpd_val in row['DCPD']:
                    if dcpd_val in presence_dict:
                        presence_dict[dcpd_val] += 1
                return list(presence_dict.values())

        # Group by two columns: 'name' and 'cat', and aggregate the 'DCPD' values into a list
        # prints dfmain columns
        # sg.Popup(print(df_main.columns))
        if 'Inhibitor_name' in df_main.columns:
            grouped_df = df_main.groupby(['Inhibitor_name', 'DHF'])['DCPD'].agg(list).reset_index()

            # Calculate the presence of each element from the DCPD_LIST for each unique pair
            grouped_df['presence'] = grouped_df.apply(check_dcpd_list_presence, axis=1)

            # Concatenate the 'name', 'cat', and 'presence' columns
            result_df = pd.concat(
                [grouped_df[['Inhibitor_name', 'DHF']], pd.DataFrame(grouped_df['presence'].to_list(), columns=DCPD_LIST)],axis=1)
            self.result_list = result_df.values.tolist()

    def get_table_data(self):
        return self.result_list


def get_incremented_filename(self):
    filename, filetype = os.path.splitext(self)
    seq = 0
    # continue from existing sequence number if any
    rex = re.search(r"^(.*)-(\d+)$", filename)
    if rex:
        filename = rex[1]
        seq = int(rex[2])

    while os.path.exists(self):
        seq += 1
        self = f"{filename}-{seq}{filetype}"
    return self


def open_new_file(df_filename, df):
    new_filename = sg.popup_get_file('Open a new file', file_types=(("pkl Files", "*.pkl"), ("All Files", "*.*")))
    if new_filename is not None:
        new_directory = os.path.dirname(new_filename)  # get the directory of the new file
        if not new_filename.endswith('.pkl'):
            new_filename = new_filename + '.pkl'
        if os.path.exists(new_filename):
            os.chdir(new_directory)  # change the working directory to the new directory
            try:
                with open(new_filename, 'rb') as f:
                    df = pickle.load(f)
                print('file opened')
                df_filename = os.path.splitext(os.path.basename(new_filename))[0]
                return df_filename, df
            except pd.errors.ParserError:
                sg.popup_error('Error: Unable to read the file as pkl.')
        else:
            df_filename = os.path.splitext(os.path.basename(new_filename))[0]
            return df_filename, df
    else:
        return df_filename, df


# def print_df(df):
#     # outofdate, it is from before the table was added
#     print_df = df.groupby(['DCPD', 'DHF']).size().reset_index(name='Count')
#     print_df = print_df.sort_values(by=['DCPD', 'DHF'], ascending=False)
#     print(print_df.to_string(index=False, justify='center'))


def save_df(df, df_filename):
    with open(f"{df_filename}.pkl", 'wb') as f:
        pickle.dump(df, f)
    df_export = df[['DCPD', 'ENB', 'DHF', 'T_max', 'Front_rate_mm/s', 'FROMP_worked', "Note"]].sort_values(
        by=['DCPD', 'DHF'])
    df_export.to_csv(f"{df_filename}.csv", index=False)


def draw_figure(canvas, figure):
    figure_canvas_agg = FigureCanvasTkAgg(figure, canvas)
    figure_canvas_agg.draw()
    figure_canvas_agg.get_tk_widget().pack(side='top', fill='both', expand=1)
    return figure_canvas_agg


def main():
    sg.theme('DarkAmber')
    # TODO: call these from the defaults file
    DEFAULT_FOCUS = 210
    DCPD_LIST = [95, 80, 70, 60, 50, 40, 30, 20, 0]
    INHIBITOR_LIST = ['DHF', 'iPrSi-8', 'other-add_note']
    # end todo

    default_dataframe = pd.DataFrame(
        columns=[
            'Name', 'DCPD', 'ENB', 'DHF', 'FROMP_worked', 'T_max', 'Front_rate_mm/s', 'T_list', 'video_filename',
            'temperature_filename', 'datetime', 'clicked_points', 'Note', 'Inhibitor_name'
        ]
    )

    # Create an instance of the DataTable
    data_table = DataTable(dcpd_list=DCPD_LIST)
    df_filename, df = open_new_file('default', default_dataframe)

    video_recorder = VideoRecorder(focus=DEFAULT_FOCUS)
    temperature_recorder = TempRecorder()
    analyzer = mrc.FrameAnalyzer()

    column_one = [
        [
            sg.Text('Filename'),
            sg.Text(df_filename, key='-FILENAME-', enable_events=True)
        ],
        [
            sg.Text('%DCPD'), sg.DropDown(DCPD_LIST, key='-%DCPD_DROPDOWN-', default_value='95'),
            sg.Text('Inhibitor'), sg.DropDown(INHIBITOR_LIST, key='-INHIBITOR_DROPDOWN-', default_value='DHF'),
            sg.Text('%Inhibitor'), sg.Input(key='-DHF-', size=(4, 1), default_text='0'),
        ],
        [
            # main button row
            sg.Button('Record', key='-RECORD-'),
            sg.Button('Open', key='-OPEN-'),
            sg.Button('Edit Runs', key='-EDIT-'),
            sg.Button('Delete Last Run', key='-DELETE-'),
            sg.Button('Note Last Run', key='-NOTE-'),
            sg.Button('Check Connections', key='-CHECK-'),
            sg.Button('Exit', key='-EXIT-'),
        ],
        [
            # multiline terminal-like output
            sg.Output(size=(95, 6)) 
        ],
    ]

    column_two = [
        [
            # data table
            sg.Table(
                    values=data_table.get_table_data(), headings=data_table.table_header_list, num_rows=9,
                    display_row_numbers=False, justification='center', key='-TABLE-',
                    col_widths=data_table.column_widths, enable_events=True, auto_size_columns=False,
                    expand_x=True, expand_y=True
                ),
            # focus slider
            sg.Column(
                [
                    [sg.Text('Focus')],
                    [sg.Slider((0, 255), DEFAULT_FOCUS, 5, orientation="v", size=(8, 14),
                               key="-FOCUS SLIDER-", enable_events=True)]
                ]
            )
        ],
    ]

    layout = [
        [
            sg.Image(filename='', key='-IMAGE-'),
            sg.Canvas(size=(640, 480), key='-PLOT-')
        ],
        # Bottom Row of Gui is two columns
        [
            # column 1
            sg.Column(column_one, vertical_alignment='top', size=(630, 200)),
            # column 2
            sg.Column(column_two, vertical_alignment='top', size=(640, 200))
        ],
    ]

    window = sg.Window(f"FROMP App demo {version}", layout, location=(800, 400), finalize=True)

    canvas_elem = window['-PLOT-']
    canvas = canvas_elem.TKCanvas

    if len(df) != 0:
        # load the initial data if there is any
        data_table.update_data(df, DCPD_LIST)
        window['-TABLE-'].update(values=data_table.get_table_data())

    # draw the initial scatter plot
    fig, ax = plt.subplots()
    ax.grid(True)
    fig_agg = draw_figure(canvas, fig)
    x, y = [], []

    record = False  # required to have record button work for start and stop

    while True:
        event, values = window.read(timeout=1)


        if video_recorder.camera_present:
            ret, frame = video_recorder.cap.read()
            imgbytes = cv2.imencode('.png', frame)[1].tobytes()
            window['-IMAGE-'].update(data=imgbytes)

        if event == sg.WINDOW_CLOSED or event == '-EXIT-':
            break

        elif event == '-FOCUS SLIDER-':
            video_recorder.focus(values["-FOCUS SLIDER-"])

        elif event == '-RECORD-':
            if video_recorder.camera_present == True:
                if not record:
                    timestamp = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())

                    # check if the user has entered a non-zero number for the inhibitor. If so, use the inhibitor name
                    is_non_zero_number = pd.notna(values['-DHF-']) and pd.to_numeric(values['-DHF-'], errors='coerce') != 0
                    if is_non_zero_number:
                        inhibitor_name = values['-INHIBITOR_DROPDOWN-']
                    else:
                        inhibitor_name = "None"

                    tmp_dict = {
                        'Name': df_filename,
                        'Inhibitor_name': inhibitor_name,
                        'DCPD': values['-%DCPD_DROPDOWN-'],
                        'ENB': 100 - values['-%DCPD_DROPDOWN-'],
                        'DHF': values['-DHF-'],
                        'datetime': timestamp
                    }

                    df_record = pd.DataFrame(tmp_dict, index=[0])
                    dhf_recording = df_record.at[0, 'DHF']
                    if pd.isna(dhf_recording) or str(dhf_recording).strip().lower() in ['', 'none', 'n/a', 'na']:
                        df_record.at[0, 'DHF'] = 0
                    dcpd_recording = str(df_record.at[0, 'DCPD']).zfill(2)
                    dhf_recording = str(df_record.at[0, 'DHF']).zfill(3)
                    video_recorder.filename = get_incremented_filename(
                        f"{df_filename}_DCPD{dcpd_recording}_DHF{dhf_recording}.mp4"
                    )
                    temperature_recorder.filename = get_incremented_filename(
                        f"{df_filename}_DCPD{dcpd_recording}_DHF{dhf_recording}.csv"
                    )

                    tmp_dict.update({
                        'video_filename': video_recorder.filename,
                        'temperature_filename': temperature_recorder.filename
                    })
                    df_record = df_record.assign(**tmp_dict)

                    temperature_recorder.start_recording()
                    video_recorder.start_recording(frame)

                elif record:
                    video_recorder.stop_recording()
                    t_list = temperature_recorder.stop_recording()
                    response = sg.popup_yes_no('Did the sample FROMP?', title='Query', keep_on_top=True)
                    fromp_worked = True if response == 'Yes' else False
                    if fromp_worked:
                        front_rate, clicked_points, frame_timestamps = analyzer.import_and_analyze(
                            video_recorder.timestamp_frame_tuple_list)
                    else:
                        frame_timestamps = [timestamp for timestamp, frame in video_recorder.timestamp_frame_tuple_list]
                        front_rate, clicked_points = pd.NA, pd.NA
                    video_recorder.timestamp_frame_tuple_list = []

                    tmp_dict = {
                        'FROMP_worked': fromp_worked,
                        'T_max': np.max(t_list),
                        'Front_rate_mm/s': front_rate,
                        'T_list': [t_list],
                        'clicked_points': [clicked_points],
                        'frame_timestamps': [frame_timestamps],
                        'Note': ''
                    }

                    df_record = df_record.assign(**tmp_dict)
                    df = pd.concat([df, df_record], ignore_index=True)
                    save_df(df, df_filename)
                    print(f'Front_rate_mm/s: {front_rate}')

                    del analyzer
                    analyzer = mrc.FrameAnalyzer()

                    data_table.update_data(df, DCPD_LIST)

                    window['-TABLE-'].update(values=data_table.get_table_data())

                record = not record
                x, y = [], []

        elif event == '-OPEN-':
            df_filename, df = open_new_file(df_filename, df)
            window['-FILENAME-'].update(df_filename)
            # TODO: print df_tracker to gui count and count_dhf and dcpd ...
            data_table.update_data(df, DCPD_LIST)
            window['-TABLE-'].update(values=data_table.get_table_data())
            # print('file opened')

        elif event == '-EDIT-':
            if len(df) == 0:
                sg.popup('Cannot edit an empty dataframe', title='Error', keep_on_top=True)
                continue
            else:
                gui = ewc.TableGui(df.copy(), dcpd_list=DCPD_LIST, inhibitor_list=INHIBITOR_LIST)
                df = gui.run()
                save_df(df, df_filename)
                data_table.update_data(df, DCPD_LIST)
                window['-TABLE-'].update(values=data_table.get_table_data())
                del gui

        elif event == '-NOTE-':
            if len(df) == 0:
                sg.popup('Cannot add a note to an empty dataframe', title='Error', keep_on_top=True)
                continue
            else:
                note = sg.popup_get_text('Enter a note', title='Note', keep_on_top=True)
                df.at[df.index[-1], 'Note'] = note
                save_df(df, df_filename)
                data_table.update_data(df, DCPD_LIST)
                window['-TABLE-'].update(values=data_table.get_table_data())
                print(f'Note added: {note}')      

        elif event == '-DELETE-':
            if len(df) > 0:
                response = sg.popup_yes_no('Are you sure you want to delete the last run?', title='Query', keep_on_top=True)
                if response == 'Yes':
                    # deletes the data files associated with the last row of the dataframe
                    column_names_files_to_be_deleted = ['video_filename', 'temperature_filename']
                    for column_name in column_names_files_to_be_deleted:
                        filename_to_be_deleted = df.iloc[-1][column_name]
                        if os.path.exists(filename_to_be_deleted):
                            os.remove(filename_to_be_deleted)
                    # removes the last row of the dataframe
                    if len(df) == 1:                 
                        df = default_dataframe
                        save_df(df, df_filename)
                        print('Last run deleted')
                        window['-TABLE-'].update(values=[])
                    else:
                        df = df[:-1]
                        save_df(df, df_filename)
                        print('Last run deleted')
                        data_table.update_data(df, DCPD_LIST)
                        window['-TABLE-'].update(values=data_table.get_table_data())
            else:
                sg.popup('No rows in the dataframe to delete', title='Error', keep_on_top=True)

        elif event == '-CHECK-':
            if not video_recorder.camera_present:
                print('Camera not found. Please check connection and restart program.')
                # causes program to crash
                # del video_recorder
                # video_recorder = VideoRecorder(focus=DEFAULT_FOCUS)
            if not temperature_recorder.temperature_present:
                del temperature_recorder
                temperature_recorder = TempRecorder()

        if video_recorder.camera_present:
            # allows for the video to be recorded for 4 frames before the temperature is recorded.
            # This significantly speeds up the recording process (FPS).
            for n in range(1, 5):
                video_recorder.record_frame(frame)

        if len(video_recorder.timestamp_frame_tuple_list) % 3 == 0:
            x_new, y_new = temperature_recorder.record_temp()
            x.append(x_new)
            x_normalized = np.array(x) - x[0]  # required to create a new array to avoid error
            y.append(y_new)
            ax.clear()
            ax.plot(x_normalized, y)
            ax.set_xlabel('Time (s)')  # Set x-axis label
            ax.set_ylabel('Temperature (°C)')  # Set y-axis label
            fig_agg.draw()


    window.close()
    video_recorder.cap.release()


if __name__ == "__main__":
    # main()
    import traceback

    try:
        main()
    except Exception as e:
        print(f"An error occurred: {e}")
        traceback.print_exc()
