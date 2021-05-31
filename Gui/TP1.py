import dearpygui.core as dpg
import dearpygui.simple as sdpg

from psola import get_periods, psola2

import pyaudio
import librosa

import pytsmod as tsm

import numpy as np

count = 0

class GuiBuilder:
    def __init__(self, width, height, theme='Dark'):
        self.theme = theme
        self.width = width
        self.height = height

    def make_gui(self):

        dpg.set_main_window_size(self.width, self.height)
        dpg.set_theme(self.theme)
        Menu().generate()

    @staticmethod
    def run_gui():
        dpg.start_dearpygui(primary_window="Main Window")


class Menu:
    def __init__(self):
        self.files_list = []
        self.stream1 = None
        self.stream2 = None

        self.count1 = 0

        self.sample1Original = None
        self.sample2Original = None

        self.sample1 = None
        self.sample2 = None

        self.scale1 = 1
        self.stretch1 = False

        self.Shift1 = 0

        self.frame_var = 0

        self.p = pyaudio.PyAudio()

    # Modify the class 'Menu' to get the required structure of menu bar
    def theme_setting(self, sender, data):
        dpg.set_theme(data)

    def addFile(self, sender, data):
        is_file_unique = True

        for f in self.files_list:
            if f['name'] == data[1]:
                is_file_unique = False

        if is_file_unique:
            self.files_list.append({'path': data[0], 'name': data[1]})
        else:
            dpg.set_value('##warnings', "There are files with the same name.")

        if len(self.files_list) == 1:
            sdpg.set_item_label('Song label 1', label='Song 1: ' + self.files_list[0]['name'])
        else:
            sdpg.set_item_label('Song label 2', label='Song 2: ' + self.files_list[1]['name'])


    def selectFile(self, sender, data):
        dpg.open_file_dialog(self.addFile)


    def playSound(self, sender, data):
        if data == 1:
            if self.stream1 is not None and not self.stream1.is_stopped():
                self.stream1.stop_stream()
                self.stream1.close()
                self.count1 = 0

            self.sample1Original, fs = librosa.load(self.files_list[0]['path'] + '/' + self.files_list[0]['name'], sr=44100, dtype=np.float64)
            self.sample1 = self.sample1Original
            
            def callback(in_data, frame_count, time_info, status):

                track1_frame = self.sample1[frame_count * self.count1: frame_count * (self.count1 + 1)]

                final = tsm.wsola(track1_frame, self.scale1)
                final = final[:frame_count]
                print(len(final))
                ret_data = track1_frame.astype(np.float32).tostring()

                self.count1 += 1
                return (ret_data, pyaudio.paContinue)

            self.stream1 = self.p.open(format=pyaudio.paFloat32,
                                 channels=1,
                                 rate=int(44100),
                                 output=True,
                                 stream_callback=callback,
                                 frames_per_buffer=2 ** 16)
            self.stream1.start_stream()

        else:
            #sample, fs = librosa.load(self.files_list[1]['path'] + '/' + self.files_list[1]['name'], sr=44100, dtype=np.float64)
            pass




    def pauseSound(self, sender, data):
        pass

    def stretchSound(self, sender, data):
        self.scale1 = dpg.get_value('Stretch 1')
        #self.stretch1 = True
        #self.sample1 = tsm.wsola(self.sample1Original, self.scale1)
        print('Finished!!!')

    def pitchShift(self, sender, data):
        self.Shift1 = dpg.get_value('Shift 1')
        peaks = get_periods(self.sample1)
        self.sample1 = psola2(self.sample1, peaks, self.Shift1)


    def generate(self):
        with sdpg.window("Main Window"):
            with sdpg.menu_bar(name='Main Menu'):
                with sdpg.menu(name='Menu Section 1'):
                    dpg.add_menu_item('item 1')
                    # add_separator()  # optional line to separate items in the menu
                    dpg.add_menu_item('item 2')
                with sdpg.menu(name='Menu Section 2'):
                    with sdpg.menu('Theme'):
                        dpg.add_menu_item('Light##Theme 1', callback=self.theme_setting, callback_data='Light')
                        dpg.add_menu_item('Dark##Theme 2', callback=self.theme_setting, callback_data='Dark')

            dpg.add_button("Add file", width=250, callback=self.selectFile, label='Select Song')
            with sdpg.child('Song1', width=310, height=600):
                dpg.add_text('')
                dpg.add_same_line(xoffset=-200)
                dpg.add_label_text('Song label 1', show=True, label='Song 1: None')

                dpg.add_spacing(count=10)
                dpg.add_button("Play song 1", width=250, callback=self.playSound, callback_data=1, label='Play')
                dpg.add_button("Pause song 1", width=250, callback=self.pauseSound, callback_data=1, label='Pause')

                dpg.add_spacing(count=5)

                dpg.add_slider_float('Stretch 1', default_value=1.0, min_value=0.5, max_value=1.5, format='%0.2f', callback=self.stretchSound)#, callback_data=1)

                dpg.add_spacing(count=5)

                dpg.add_slider_float('Shift 1', default_value=1.0, min_value=0.5, max_value=1.5, format='%0.2f', callback=self.pitchShift)#, callback_data=1)


            dpg.add_same_line()
            with sdpg.child('Song2', width=310, height=600):
                dpg.add_text('')
                dpg.add_same_line(xoffset=-200)
                dpg.add_label_text('Song label 2', show=True, label='Song 2: None')

                dpg.add_spacing(count=10)
                dpg.add_button("Play song 2", width=250, callback=self.playSound, callback_data=2, label='Play')
                dpg.add_button("Pause song 2", width=250, callback=self.pauseSound, callback_data=2, label='Pause')

                dpg.add_spacing(count=5)

                dpg.add_slider_float('Stretch 2', default_value=1.0, min_value=0.9, max_value=1.1, format='%0.2f', callback=self.stretchSound, callback_data=2)

                dpg.add_spacing(count=5)

                dpg.add_slider_float('Shift 2', default_value=1.0, min_value=0.9, max_value=1.1, format='%0.2f', callback=self.pitchShift, callback_data=2)


class Tab:
    def __init__(self, tab_name, parent):
        self.tab_name = tab_name
        self.parent = parent

    def generate(self):
        with sdpg.tab(name=self.tab_name, parent=self.parent):
            dpg.add_text(f'Content of {self.tab_name}')


if __name__ == '__main__':
    template = GuiBuilder(650, 650, theme='Light')
    template.make_gui()
    template.run_gui()

