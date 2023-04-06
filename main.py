import requests
from requests.auth import HTTPBasicAuth
from requests.exceptions import ConnectionError
from multiprocessing import Process, SimpleQueue

# Funkcja pomocnicza służąca do zrównoleglenia wysyłania zapytania o autoryzację do serwera
def login_request(url: str, username: str, key: str, queue: SimpleQueue):
    http_basic = HTTPBasicAuth(username, key)
    try:
        http_response = requests.get(url, auth=http_basic)
        queue.put(http_response.status_code)
    except ConnectionError:
        queue.put(-1)


if __name__ == '__main__':
    import asugrmodule
    import PySimpleGUI as sg
    import numpy as np
    from cv2 import cv2
    from keras.models import load_model
    from mediapipe.python.solutions import hands as mphands, drawing_utils

    sg.theme('Default1')

    # Wczytanie pliku konfiguracyjnego
    config = asugrmodule.get_config()

    # Zmienna określająca metodę
    method = 0

    # Kolejka dla równoległości zapytań
    login_response_queue = SimpleQueue()

    # Obiekty dla metody MediaPipe
    hands = mphands.Hands(max_num_hands=1, min_detection_confidence=0.7, static_image_mode=True)
    mpDraw = drawing_utils

    # Otwarcie urządzenia przechwytującego obraz
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise IOError("Cannot open webcam")

    # Utworzenie subtractora dla metody odejmowania tła
    subtractor = asugrmodule.create_subtractor(cap)
    reference = subtractor.getBackgroundImage()

    # Wczytanie podpowiedzi
    tooltips = asugrmodule.get_tooltips('main')

    # Layout aplikacji graficznej
    subtractor_settings = [
        [sg.Text('varThreshold:', justification='center', tooltip=tooltips['var_threshold']),
         sg.Slider((1, 20), default_value=3, key='varThreshold', orientation='horizontal',
                   disable_number_display=False, enable_events=True, expand_x=True,
                   tooltip=tooltips['var_threshold'])],
        [sg.Button('Zresetuj subtractor', key='subtractorReset', tooltip=tooltips['subtractor_reset'])],
        [sg.Checkbox('Histogram', key='grayhist', default=False, enable_events=True, tooltip=tooltips['grayhist'])]
    ]
    hsv_settings = [
        [sg.Frame('Dolna',
                  [[sg.Text('H', expand_x=True),
                    sg.Text('S', expand_x=True),
                    sg.Text('V', expand_x=True)],
                   [sg.Slider((0, 179), default_value=0, key='lH', orientation='v'),
                    sg.Slider((0, 255), default_value=50, key='lS', orientation='v'),
                    sg.Slider((0, 255), default_value=90, key='lV', orientation='v')]
                   ], visible=True, vertical_alignment='top',
                  element_justification='center', tooltip=tooltips['hsv_lower']),
         sg.Frame('Górna',
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
        [sg.Button('Zatrzymaj rozpoznawanie', key='stopRecognition', disabled=False)],
        [sg.HorizontalSeparator()]
    ]
    login_column = [
        [sg.ProgressBar(max_value=30, orientation='h', size=(-1, 20), expand_x=True, key='progress')],
        [sg.Text('-', justification='center', expand_x=True, key='progressText', tooltip=tooltips['progress'])],
        [sg.Text('Sekwencja:', tooltip=tooltips['sequence']),
         sg.Text('', key='sequence', auto_size_text=True, size=(25, -1), expand_x=True, tooltip=tooltips['sequence'])],
        [sg.Button('Cofnij', key='erase_from_sequence', tooltip=tooltips['erase'])],
        [sg.Text('Kombinacja:'), sg.Text('', key='combo_sequence', size=(25, -1))],
        [sg.Button('Zresetuj sekwencje', key='sequenceReset', tooltip=tooltips['reset_sequence'])],
        [sg.Text('', key='status_info', text_color='red')],
        [sg.Text('Login:'), sg.InputText(tooltip=tooltips['login'], key='login_input',
                                         enable_events=True, size=(20, 1))],
        [sg.Text('', key='server_info')],
        [sg.Button('Zarejestruj', key='register', tooltip=tooltips['register']),
         sg.Push(),
         sg.Button('Ustawienia', key='settings_button', tooltip=tooltips['main_app_setting']), sg.Button('Exit')],
    ]
    histogram_column = [
        [sg.Text('Histogram', justification='center', font='Any 12 bold', expand_x=True)],
        [sg.Image(key='plot', tooltip=tooltips['histogram'], size=(300, 300))]
    ]
    settings = [[sg.Column(cropped_column)],
                [sg.Column(subtractor_settings, key='subtractorSettings'),
                 sg.Column(hsv_settings, key='hsvSettings', visible=False),
                 sg.Column(mp_settings, key='mpSettings', visible=False)]]
    layout = [
        [sg.Column(main_column + login_column),
         sg.Column(settings, key='settings_column'),
         sg.Column(histogram_column, key='histogram_column', visible=False)],
    ]

    # Obiekt okna aplikacji
    window = sg.Window('Gesture Recognition Authorization', layout, location=(300, 100), finalize=True)

    # Zmienne wykorzystywane w procesie tworzenia sekwencji oraz autoryzacji
    predictionIndex = -1
    predict_count = 0
    write_count = 0
    gestureSequence = []
    comboSequence = []
    # Zmienna try_to_login służy do sprawdzania, czy należy wykonać zapytanie o autoryzację
    try_to_login = False
    procs = []
    stop_recognition = 0

    # Wczytanie konfiguracji
    loading_errors = []
    try:
        model = load_model(config['MAIN']['model_path'])
    except OSError:
        loading_errors.append("Nie udało się załadować modelu")
        model = None
    try:
        labels = np.load(config['MAIN']['model_path'] + '\\labels.npy')
    except FileNotFoundError:
        loading_errors.append("Nie udało się załadować etykiet")
        labels = None
    try:
        max_predict_count = int(config['MAIN']['max_predict_count'])
    except ValueError:
        max_predict_count = 20
    try:
        max_write_count = int(config['MAIN']['max_write_count'])
    except ValueError:
        max_write_count = 20

    window['status_info'].update("\n".join(val for val in loading_errors))

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
        # Reset sekwencji
        elif event == 'sequenceReset':
            gestureSequence = []
            window['sequence'].update('')
        # Usuwanie pojedynczej kombinacji z sekewncji
        elif event == 'erase_from_sequence':
            if len(gestureSequence) > 0:
                gestureSequence = gestureSequence[:-1]
                tmp = ''
                for gesture in gestureSequence:
                    tmp += '['
                    if labels is not None:
                        tmp += ', '.join(labels[val] for val in gesture)
                    else:
                        tmp += ', '.join(str(val) for val in gesture)
                    tmp += '] '
                window['sequence'].update(tmp)
        elif event == 'login_input':
            if len(values['login_input']) > 2:
                try_to_login = True
        # Zatrzymanie rozpoznawania gestów
        elif event == 'stopRecognition':
            stop_recognition = (stop_recognition + 1) % 2
            if stop_recognition:
                window['stopRecognition'].update('Wznów rozpoznawanie')
                window['progressText'].update('-')
                window['progress'].update(current_count=0)
                predict_count = 0
            else:
                window['stopRecognition'].update('Zatrzymaj recognition')
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
        # Obsługa rejestracji użytkownika
        elif event == 'register':
            # Sprawdzenie, czy spełnione zostały warunki
            if len(values['login_input']) > 2 or len(gestureSequence) > 1:
                password = asugrmodule.sequence_to_password(gestureSequence)
                basic = HTTPBasicAuth(values['login_input'], password)
                try:
                    # Wykonanie zapytania i obsługa odpowiedzi
                    response = requests.post(config['MAIN']['server_address'], auth=basic)
                    if response.status_code == 200:
                        sg.Popup("Zarejestrowano użytkownika o nazwie {}".format(values['login_input']),
                                 non_blocking=True)
                        gestureSequence = []
                        window['login_input'].update(value='')
                        window['sequence'].update(value='')
                    elif response.status_code == 402:
                        sg.Popup("Użytkownik o podanej nazwie już istnieje", non_blocking=True)
                except ConnectionError:
                    sg.Popup("Błąd połączenia z serwerem", non_blocking=True)
            else:
                sg.Popup("Za krótki login lub sekwencja gestów", non_blocking=True)
        # Zmiana ustawień konfiguracyjnych oraz ponowne wczytanie konfiguracji
        elif event == 'settings_button':
            window.disable()
            asugrmodule.main_settings_window(config)
            window.enable()
            config = asugrmodule.get_config(config)
            window['status_info'].update(value='')
            loading_errors = []
            try:
                model = load_model(config['MAIN']['model_path'])
            except OSError:
                loading_errors.append("Nie udało się załadować modelu")
                model = None
            try:
                labels = np.load(config['MAIN']['model_path'] + '\\labels.npy')
            except FileNotFoundError:
                loading_errors.append("Nie udało się załadować etykiet")
                labels = None
            try:
                max_predict_count = int(config['MAIN']['max_predict_count'])
            except ValueError:
                max_predict_count = 20
            try:
                max_write_count = int(config['MAIN']['max_write_count'])
            except ValueError:
                max_write_count = 20

            window['status_info'].update("\n".join(val for val in loading_errors))

        # Wykonanie zapytania o autoryzację, jeśli zostały spełnione wszystkie warunki
        if len(gestureSequence) > 1 and len(values['login_input']) > 2 and try_to_login:
            try_to_login = False
            password = asugrmodule.sequence_to_password(gestureSequence)
            proc = Process(target=login_request, args=(config['MAIN']['server_address'],
                                                       values['login_input'],
                                                       password,
                                                       login_response_queue))
            proc.start()
            procs.append(proc)

        # Odczytywanie wiadomości zwrotnej od serwera
        while login_response_queue.empty() is False:
            status_code = login_response_queue.get()
            if status_code == 200:
                sg.Popup('Podano prawidłowe dane', keep_on_top=True)
            elif status_code == -1:
                window['server_info'].update('Błąd połączenia z serwerem')
        is_alive_array = np.array([], dtype=bool)
        for proc in procs:
            is_alive_array = np.append(is_alive_array, proc.is_alive())
        for val in np.argwhere(is_alive_array == 0):
            try:
                procs.pop(val[0])
            except IndexError:
                pass
        if len(procs) > 0 and window['server_info'].get() != 'Odpytuje serwer...':
            window['server_info'].update('Odpytuje serwer...')
        elif len(procs) == 0 and \
                window['server_info'].get() != '' and \
                window['server_info'].get() != 'Błąd połączenia z serwerem':
            window['server_info'].update('')

        # Wczytanie kolejnej klatki
        _, frame = cap.read()
        cropped = np.zeros((300, 300, 3), np.uint8)
        cv2.copyTo(frame[0:300, 0:300], None, cropped)
        cv2.rectangle(frame, (0, 0), (300, 300), (255, 0, 255), 2)

        fgmask = []
        # Generowanie binarnej maski w zależności od metody
        if cropped is not None:
            if method == 0:
                image = asugrmodule.prepare_image(cropped, reference)
                if values['grayhist']:
                    my_plot = asugrmodule.grayscale_histogram(image, reference)
                    my_plotbytes = cv2.imencode('.png', my_plot)[1].tobytes()
                    window['plot'].update(data=my_plotbytes)
                fgmask = asugrmodule.mask_subtractor(image, subtractor, True, True)
            elif method == 1:
                lH, lS, lV, uH, uS, uV = values['lH'], values['lS'], values['lV'], values['uH'], values['uS'], values[
                    'uV']
                image = asugrmodule.prepare_image(cropped)
                fgmask = asugrmodule.mask_hsv(image, [lH, lS, lV], [uH, uS, uV], True)
            elif method == 2:
                fgmask = asugrmodule.mask_mediapipe(cropped, hands, mphands, mpDraw, True)

        # Wyświetlenie obrazu w interfejsie graficznym
        if values['view_switch']:
            framebytes = cv2.imencode('.png', frame)[1].tobytes()
        else:
            framebytes = cv2.imencode('.png', cropped)[1].tobytes()
        window['capture'].update(data=framebytes)
        maskbytes = cv2.imencode('.png', fgmask)[1].tobytes()
        window['mask'].update(data=maskbytes)

        # Obróbka maski
        fgmask = asugrmodule.rotate_mask(fgmask)
        fgmask = asugrmodule.feature_extract(fgmask)

        # Dokonywanie predykcji na binarnej masce
        if not stop_recognition and fgmask.any() != 0 and model is not None:
            resized = cv2.resize(fgmask, (28, 28))
            prediction_array = np.array(resized) / 255.0
            prediction_array = np.reshape(prediction_array, (1, 28, 28))

            predictions = model.predict(prediction_array,
                                        verbose=0,
                                        use_multiprocessing=True)

            # Odczytywanie predykcji o największym prawdopodobieństwie
            if predictions[0][np.argmax(predictions)] > 0.6:
                write_count = 0
                if labels is not None:
                    window['progressText'].update(labels[np.argmax(predictions)] + "(" + "{:.2f}"
                                                  .format(predictions[0][np.argmax(predictions)] * 100) + "%)")
                else:
                    window['progressText'].update(str(np.argmax(predictions)) + "(" + "{:.2f}"
                                                  .format(predictions[0][np.argmax(predictions)] * 100) + "%)")
                # Zliczanie klatek potrzebnych do zapisania danego gestu do kombinacji
                if predictionIndex == np.argmax(predictions):
                    predict_count += 1
                    window['progress'].update(predict_count, max=max_predict_count)
                    if predict_count == max_predict_count:
                        predict_count = 0
                        comboSequence.append(predictionIndex)
                        if labels is not None:
                            window['combo_sequence'].update("".join([labels[i] + " " for i in comboSequence]))
                        else:
                            window['combo_sequence'].update("".join([str(i) + " " for i in comboSequence]))
                else:
                    predictionIndex = np.argmax(predictions)
                    window['progressText'].update('-')
                    window['progress'].update(current_count=0)
                    predict_count = 0
        else:
            window['progressText'].update('-')
            window['progress'].update(current_count=0)
            predict_count = 0
        # Zliczanie klatek potrzebnych do zapisania kombinacji do sekwencji
        if len(comboSequence) > 0:
            write_count += 1
            if write_count == max_write_count:
                gestureSequence.append(comboSequence)
                comboSequence = []
                if len(gestureSequence) > 1:
                    try_to_login = True
                tmp = ''
                for gesture in gestureSequence:
                    tmp += '['
                    if labels is not None:
                        tmp += ', '.join(labels[val] for val in gesture)
                    else:
                        tmp += ', '.join(str(val) for val in gesture)
                    tmp += '] '
                window['sequence'].update(tmp)
                window['combo_sequence'].update('')

    window.close()
