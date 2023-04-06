import asugrmodule
import PySimpleGUI as sg
import numpy as np
from cv2 import cv2
from random import choice
from mediapipe.python.solutions import hands as mphands
from mediapipe.python.solutions import drawing_utils

sg.theme('Default1')

# Otwarcie urządzenia przechwytującego obraz
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError("Cannot open webcam")

# Obiekty dla metody MediaPipe
hands = mphands.Hands(max_num_hands=1, min_detection_confidence=0.7, static_image_mode=True)
mpDraw = drawing_utils

# Utworzenie subtractora dla metody odejmowania tła
subtractor = asugrmodule.create_subtractor(cap)
reference = subtractor.getBackgroundImage()

# Zmienna określająca metodę
method = 0

# Wczytanie podpowiedzi
tooltips = asugrmodule.get_tooltips('recorder')

# Layout aplikacji graficznej
subtractor_settings = [
    [sg.Text('varThreshold:', justification='center', tooltip=tooltips['var_threshold']),
     sg.Slider((1, 20), default_value=3, key='varThreshold', orientation='horizontal',
               disable_number_display=False, enable_events=True, expand_x=True,
               tooltip=tooltips['var_threshold'])],
    [sg.Button('Reset subtractor', key='subtractorReset', tooltip=tooltips['subtractor_reset'])],
    [sg.Checkbox('Grayscale Histogram', key='grayhist', default=False,
                 enable_events=True, tooltip=tooltips['grayhist'])]
]
hsv_settings = [
    [sg.Frame('Lower',
              [[sg.Text('H', expand_x=True),
                sg.Text('S', expand_x=True),
                sg.Text('V', expand_x=True)],
               [sg.Slider((0, 179), default_value=0, key='lH', orientation='v'),
                sg.Slider((0, 255), default_value=50, key='lS', orientation='v'),
                sg.Slider((0, 255), default_value=90, key='lV', orientation='v')]
               ], visible=True, vertical_alignment='top',
              element_justification='center', tooltip=tooltips['hsv_lower']),
     sg.Frame('Upper',
              [[sg.Text('H', expand_x=True),
                sg.Text('S', expand_x=True),
                sg.Text('V', expand_x=True)],
               [sg.Slider((0, 179), default_value=30, key='uH', orientation='v'),
                sg.Slider((0, 255), default_value=150, key='uS', orientation='v'),
                sg.Slider((0, 255), default_value=160, key='uV', orientation='v')]
               ], visible=True, vertical_alignment='top',
              element_justification='center', tooltip=tooltips['hsv_upper'])],
    [sg.Push(), sg.Text('Domyślne: '),
     sg.Button('Skin', key='defaultSkin', tooltip=tooltips['default_skin']),
     sg.Button('Blue', key='defaultBlue', tooltip=tooltips['default_blue'])]
]
mp_settings = []
main_column = [
    [sg.Combo(['Subtractor', 'HSV', 'MediaPipe Hands'],
              expand_x=True, key='methodCombo', default_value='Subtractor',
              enable_events=True, readonly=True, tooltip=tooltips['method_combo'])],
    [sg.Text("Frame", justification='center', font='Any 12 bold', expand_x=True)],
    [sg.Checkbox("Widok całej klatki", default=False, key="view_switch", enable_events=True),
     sg.Push(),
     sg.Checkbox("Ustawienia metody", default=True, key="settings_bool",
                 enable_events=True, tooltip=tooltips['method_setting'])],
    [sg.Image(key='capture', tooltip=tooltips['frame'], size=(300, 300))]
]
cropped_column = [
    [sg.Text('Mask', justification='center', font='Any 12 bold', expand_x=True)],
    [sg.Image(key='mask', size=(300, 300), tooltip=tooltips['mask'])],
    [sg.HorizontalSeparator()]
]
record_column = [
    [sg.Text('0/200', key='progress_text', expand_x=True, justification='center', tooltip=tooltips['progress'])],
    [sg.ProgressBar(max_value=200, orientation='h', size=(-1, 20), expand_x=True,
                    key='progress')],
    [sg.Text('Klatek:', size=(8, 1), tooltip=tooltips['frames_count']),
     sg.InputText(key='frames_count', size=(15, 1), default_text='200')],
    [sg.Text('Sztuczne:', size=(8, 1), tooltip=tooltips['artificial_frames']),
     sg.InputText(key='artificial_frames', size=(15, 1), default_text='0')],
    [sg.Text('Nazwa:', size=(8, 1), tooltip=tooltips['name']), sg.InputText(key='gesture_name', size=(15, 1))],
    [sg.Text('Lokacja:', size=(8, 1), tooltip=tooltips['location']), sg.InputText(key='saving_location', size=(15, 1)),
     sg.FolderBrowse("Przeglądaj...", key='folder_browse')],
    [sg.Button('Nagraj gest', key='record_button', tooltip=tooltips['record_button']),
     sg.Button('Zapisz', key='save_button', tooltip=tooltips['save_button']), sg.Push(),
     sg.Button('Exit', tooltip=tooltips['exit'])],
]
histogram_column = [
    [sg.Text('Histogram', justification='center', font='Any 12 bold', expand_x=True)],
    [sg.Image(key='plot', size=(300, 300), tooltip=tooltips['histogram'])]
]
settings = [[sg.Column(cropped_column)],
            [sg.Column(subtractor_settings, key='subtractorSettings'),
             sg.Column(hsv_settings, key='hsvSettings', visible=False),
             sg.Column(mp_settings, key='mpSettings', visible=False)]]
layout = [
    [sg.Column(main_column + record_column),
     sg.Column(settings, key='settings_column'),
     sg.Column(histogram_column, key='histogram_column', visible=False)],
]

# Obiekt okna aplikacji
window = sg.Window('Gesture Recorder', layout, location=(300, 100), finalize=True,
                   use_default_focus=False)

# Zmienne wykorzystywane w procesie nagrywania gestu
gesture_array = []
currentCount = 200
maxCount = 200

while True:
    event, values = window.read(timeout=1)
    # Obsługa zdarzeń wywoływanych przez użytkownika
    if event == sg.WIN_CLOSED or event == 'Exit':
        break
    # Wybór metody
    elif event == 'methodCombo':
        if values['methodCombo'] == 'Subtractor':
            window['subtractorSettings'].update(visible=True)
            window['hsvSettings'].update(visible=False)
            window['mpSettings'].update(visible=False)
            method = 0
        elif values['methodCombo'] == 'HSV':
            window['subtractorSettings'].update(visible=False)
            window['hsvSettings'].update(visible=True)
            window['mpSettings'].update(visible=False)
            method = 1
        elif values['methodCombo'] == 'MediaPipe Hands':
            window['subtractorSettings'].update(visible=False)
            window['hsvSettings'].update(visible=False)
            window['mpSettings'].update(visible=True)
            method = 2
    # Zresetowanie subtractora
    elif event == 'subtractorReset':
        subtractor = asugrmodule.create_subtractor(cap)
        reference = subtractor.getBackgroundImage()
    # Określenie zmiennej varThreshold dla subtractora
    elif event == 'varThreshold':
        subtractor.setVarThreshold(values['varThreshold'] * 10)
    # Wartości domyślne dla metody HSV
    elif event == 'defaultSkin':
        window['lH'].update(0)
        window['lS'].update(50)
        window['lV'].update(90)
        window['uH'].update(30)
        window['uS'].update(150)
        window['uV'].update(160)
    elif event == 'defaultBlue':
        window['lH'].update(90)
        window['lS'].update(60)
        window['lV'].update(70)
        window['uH'].update(120)
        window['uS'].update(155)
        window['uV'].update(170)
    # Ustawienia metody
    elif event == 'settings_bool':
        window['settings_column'].update(visible=values['settings_bool'])
        if values['grayhist']:
            window['grayhist'].update(value=False)
            window['histogram_column'].update(visible=False)
    # Ustawienia widoki klatki
    elif event == 'view_switch':
        if values['view_switch']:
            window['capture'].set_size((640, 480))
        else:
            window['capture'].set_size((300, 300))
    # Histogram dla metody odejmowania tła
    elif event == 'grayhist':
        window['histogram_column'].update(visible=values['grayhist'])
    # Uruchomienie nagrywania gestu
    elif event == 'record_button' or event == 'F3:114':
        try:
            maxCount = int(values['frames_count'])
        except ValueError:
            maxCount = 200
        currentCount = 0
        gesture_array = []
    # Zapisywanie gestu
    elif event == 'save_button':
        # Dane o geście są zapisywane, jeśli zostały spełnione wszystkie warunki
        if len(gesture_array) == maxCount and values['gesture_name'] != '' and values['saving_location'] != '':
            artificial_array = []
            try:
                artificialCount = int(values['artificial_frames'])
            except ValueError:
                artificialCount = 0
            # Tworzenie sztucznych klatek
            for i in range(0, artificialCount):
                random_mask = np.floor(np.random.random((28, 28)) * 256)
                random_mask = np.reshape(random_mask, (28, 28))
                random_mask = random_mask.astype(np.uint8)
                random_mask = cv2.GaussianBlur(random_mask, (3, 3), 0)
                _, random_mask = cv2.threshold(random_mask, 91, 255, cv2.THRESH_BINARY)
                random_choice = choice(gesture_array)
                random_gesture = cv2.bitwise_and(random_choice,
                                                 random_choice,
                                                 mask=random_mask)
                artificial_array.append(random_gesture)
            nparray = np.array(gesture_array + artificial_array)
            # Zapis gestu
            try:
                np.save(values['saving_location'] + '/' + values['gesture_name'], nparray)
                sg.Popup("Zapisano {}-elementowy zbiór jako \n{}/{}".format(len(nparray), values['saving_location'],
                                                                            values['gesture_name']))
            except OSError:
                sg.Popup("Nie udało się zapisać zbioru (błędna nazwa/lokacja?)")
        else:
            sg.Popup('Nie można zapisać\n(pola nie są wypełnione lub gest nie został nagrany)')

    # Wczytanie kolejnej klatki
    _, frame = cap.read()
    cropped = np.zeros((300, 300, 3), np.uint8)
    cv2.copyTo(frame[0:300, 0:300], None, cropped)
    cv2.rectangle(frame, (0, 0), (300, 300), (255, 0, 255), 2)

    fgmask = []
    # Generowanie binarnej maski w zależności od metody
    if method == 0:
        image = asugrmodule.prepare_image(cropped, reference)
        if values['grayhist']:
            my_plot = asugrmodule.grayscale_histogram(image, reference)
            my_plotbytes = cv2.imencode('.png', my_plot)[1].tobytes()
            window['plot'].update(data=my_plotbytes)
        fgmask = asugrmodule.mask_subtractor(image, subtractor, True, True)
    elif method == 1:
        lH, lS, lV, uH, uS, uV = values['lH'], values['lS'], values['lV'], values['uH'], values['uS'], values['uV']
        image = asugrmodule.prepare_image(cropped)
        fgmask = asugrmodule.mask_hsv(image, [lH, lS, lV], [uH, uS, uV], True)
    elif method == 2:
        fgmask = asugrmodule.mask_mediapipe(cropped, hands, mphands, mpDraw, True)

    # Wyświetlenie obrazu w interfejsie graficznym
    if values['view_switch']:
        framebytes = cv2.imencode('.png', frame)[1].tobytes()
    else:
        framebytes = cv2.imencode('.png', cropped)[1].tobytes()
    maskbytes = cv2.imencode('.png', fgmask)[1].tobytes()
    window['capture'].update(data=framebytes)
    window['mask'].update(data=maskbytes)

    # Nagrywanie gestu przez określoną liczbę klatek
    if currentCount < maxCount:
        fgmask = asugrmodule.rotate_mask(fgmask)
        fgmask = asugrmodule.feature_extract(fgmask)
        fgmask = cv2.resize(fgmask, (28, 28))
        gesture_array.append(fgmask)
        currentCount += 1
        window['progress_text'].update("{}/{}".format(currentCount, maxCount))
        window['progress'].update(current_count=currentCount, max=maxCount)

window.close()
