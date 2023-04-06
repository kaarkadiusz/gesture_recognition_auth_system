import configparser
import os

import numpy as np
from cv2 import cv2

# Stworzenie subtractora dla metody odejmowania tłą
def create_subtractor(capture):
    new_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, detectShadows=False, varThreshold=25)

    _, frame = capture.read()
    cropped = frame[0:300, 0:300]

    cropped = prepare_image(cropped)
    new_subtractor.apply(cropped, learningRate=-1)

    return new_subtractor

# Przeprowadzenie operacji morfologicznych na binarnej masce
def do_morphological(binary_image):
    kernel_ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    morphological = cv2.dilate(binary_image, kernel_ellipse, iterations=1)
    morphological = cv2.erode(morphological, np.ones((11, 11), np.uint8), iterations=1)
    morphological = cv2.dilate(morphological, kernel_ellipse, iterations=1)
    morphological = cv2.GaussianBlur(morphological, (5, 5), 0)
    morphological = cv2.dilate(morphological, kernel_ellipse, iterations=2)
    morphological = cv2.erode(morphological, np.ones((5, 5), np.uint8))
    morphological = cv2.GaussianBlur(morphological, (5, 5), 0)

    _, thresh = cv2.threshold(morphological, 127, 255, 0)

    return thresh

# Wykadrowanie maski
def feature_extract(mask, minsize=2000):
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) > 0:
        c = max(contours, key=cv2.contourArea)
    else:
        return np.zeros(mask.shape, np.uint8)

    if cv2.contourArea(c) > minsize:
        rect = cv2.boundingRect(c)

        result = mask[rect[1]:rect[1] + rect[3], rect[0]:rect[0] + rect[2]]
        h, w = result.shape
        val_x = int((h - w) / 2) if h > w else 0
        val_y = int((w - h) / 2) if w > h else 0
        result = cv2.copyMakeBorder(result, val_y, val_y, val_x, val_x, cv2.BORDER_CONSTANT, None, 0)
        return cv2.resize(result, mask.shape)
    else:
        return np.zeros(mask.shape, np.uint8)

# Wczytanie ustawień konfiguracyjnych
def get_config(config: configparser.ConfigParser = None):
    import configparser
    default_config = configparser.ConfigParser()
    default_config['MAIN'] = {
        'model_path': 'default/default_recognition_model',
        'server_address': 'http://localhost:8000/',
        'max_predict_count': '20',
        'max_write_count': '20',
    }
    if config is None:
        config = configparser.ConfigParser()
        try:
            config.read_file(open('config.ini'))
        except FileNotFoundError:
            with open('config.ini', 'w') as configfile:
                default_config.write(configfile)
            return default_config
    for section in default_config.sections():
        if config.has_section(section):
            for key in default_config[section]:
                if config.has_option(section, key) and config[section][key] != '':
                    default_config[section][key] = config[section][key]
    return default_config

# Wczytanie podpowiedzi
def get_tooltips(app_type):
    tooltips = {}
    if app_type == 'recorder' or 'main':
        tooltips['var_threshold'] = 'Ustawia próg wariancji dla dopasowania obecna klatka - model tła.'
        tooltips['subtractor_reset'] = 'Resetuje model tła.'
        tooltips['grayhist'] = 'Włącza podgląd histogramu porównującego jasność obecnej klatki z modelem tła.'
        tooltips['hsv_lower'] = 'Dolna granica dla koloru w przestrzeni kolorów HSV.'
        tooltips['hsv_upper'] = 'Górna granica dla koloru w przestrzeni kolorów HSV.'
        tooltips['default_skin'] = 'Wartości domyślne dla koloru białoskórego.'
        tooltips['default_blue'] = 'Wartości domyślne dla koloru niebieskiego.'
        tooltips['method_combo'] = 'Wybierz metodę generowania binarnej maski\n' \
                                   'Subtractor - odejmowanie tła\n' \
                                   'HSV - kluczowanie koloru w przestrzeni kolorów HSV\n' \
                                   'MediaPipe Hands - rozwiązanie Google'
        tooltips['frame'] = 'Obecna klatka. Pod uwagę brany jest region zaznaczony kwadratem'
        tooltips['mask'] = 'Wygenerowana binarna maska.'
        tooltips['method_setting'] = 'Włącza podgląd maski oraz ustawienia metody.'
        tooltips['histogram'] = 'Histogram porównujący jasność obecnej klatki z modelem tła.\n' \
                                'Obecna klatka - kolor niebieski\n' \
                                'Model tła - kolor biały\n' \
                                'Porównanie jest wynikiem korelacji między histogramami'

    if app_type == 'model_creator':
        tooltips['gestures'] = 'Wybór katalogu zawierającego dane o gestach.\n' \
                               'Katalog powinien zawierać wyłącznie pliki gestów (*.npy).\n' \
                               'Należy pamiętać, że nazwa pliku będzie oznaczać nazwę gestu w pliku etykiet.'
        tooltips['test_bool'] = 'Zaznacz, jeśli chcesz aby 20% wszystkich danych zostało przeznaczonych na testowanie ' \
                                'modelu.'
        tooltips['compile_train_button'] = 'Pozwala skompilować i wytrenować model na wcześniej podanych gestach.\n' \
                                           'Musi zostać wypełnione pole \'Gesty\'.\n' \
                                           'Trafność jest obliczana z ewaluacji zbioru testowego.\n' \
                                           'Jeśli nie utworzono zbioru testowego - trafność jest obliczana z całego procesu uczącego '
        tooltips['location'] = 'Wybór katalogu, w którym zostanie zapisany model.\n' \
                               'Katalog powinien być pusty.'
        tooltips['save_button'] = 'Pozwala zapisać model.\nModel musi być wcześniej skompilowany oraz musi zostać ' \
                                  'wypełnione pole \'Lokacja\'.'
    elif app_type == 'recorder':
        tooltips['progress'] = 'Postęp nagrywania gestu.'
        tooltips['frames_count'] = 'Przez ile klatek zbierać dane.\n' \
                                   'Musi być liczbą całkowitą\n' \
                                   'Domyślnie: 200'
        tooltips['artificial_frames'] = 'Ile klatek stworzyć sztucznie wykorzystując klatki nagrane.\n' \
                                        'Tym klatkom dodawany jest szum\n' \
                                        'Musi być liczbą całkowitą\n' \
                                        'Domyślnie: 0'
        tooltips['name'] = 'Nazwa jaka ma zostać nadana gestowi.\n' \
                           'Ta sama nazwa zostanie nadana plikowi a co za tym idzie ta nazwa' \
                           'zostanie wpisana do pliku z etykietami.'
        tooltips['location'] = 'Katalog, w którym ma zostać zapisany gest.'
        tooltips['record_button'] = 'Rozpoczyna nagrywanie gestu. Postęp można śledzić na pasku postępu.'
        tooltips['save_button'] = 'Zapisuje gest o podanej nazwe w podanej lokacji.\n' \
                                  'Musi zostać wypełnione pole \'Nazwa\' i \'Lokacja\'.'
    elif app_type == 'main':
        tooltips['progress'] = 'Rozpoznawany gest z prawdopodobieństwem.'
        tooltips['erase'] = 'Usuń najnowszą kombinację z sekwencji.'
        tooltips['reset_sequence'] = 'Wyczyść całą sekwencję.'
        tooltips['sequence'] = 'Sekewncja kombinacji gestów używana do autoryzacji.'
        tooltips['login'] = 'Login użytkownika używany do autoryzacji.'
        tooltips['register'] = 'Wysłanie zapytania o rejestrację użytkownika korzystając z podanych' \
                               'danych (login i sekwencja)'
        tooltips['main_app_setting'] = 'Otwórz okno ustawień.'
    tooltips['exit'] = 'Zakończ działanie programu.'

    return tooltips

# Funckja rysująca histogramy obecnej klatki i obrazu referencyjnego w skali szarości
def grayscale_histogram(image, reference=None):
    if reference is None:
        reference = np.zeros(image.shape, np.uint8)
    grayscale_reference = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
    grayscale_reference_hist = cv2.calcHist([grayscale_reference], [0], None, [256], [0, 256])
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    grayscale_image_hist = cv2.calcHist([grayscale_image], [0], None, [256], [0, 256])
    comparison = cv2.compareHist(grayscale_reference_hist, grayscale_image_hist, cv2.HISTCMP_CORREL)
    grayscale_reference_hist_norm = grayscale_reference_hist / max(grayscale_reference_hist)
    grayscale_blurred_hist_norm = grayscale_image_hist / max(grayscale_image_hist)

    my_plot = np.full((300, 300, 3), 0, np.uint8)
    for i in range(27):
        cv2.line(my_plot, (i * 10, 0), (i * 10, 300), (100, 100, 100))
    for i in range(0, 255):
        pt1 = (i, 299 - int(200 * grayscale_reference_hist_norm[i]))
        pt2 = ((i + 1), 299 - int(200 * grayscale_reference_hist_norm[i + 1]))
        pt3 = (i, 299 - int(200 * grayscale_blurred_hist_norm[i]))
        pt4 = ((i + 1), 299 - int(200 * grayscale_blurred_hist_norm[i + 1]))

        cv2.line(my_plot, pt1, pt2, (255, 255, 255))
        cv2.line(my_plot, pt3, pt4, (255, 0, 0))
    cv2.putText(my_plot, str(comparison), (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

    return my_plot

# Wczytanie danych o gestach
def load_data(path):
    from random import shuffle

    x_train = []
    y_train = []
    labels = []
    k = 0

    for file in os.listdir(path):
        xarray = np.load(path + '/' + str(file))
        yarray = np.full(xarray.shape[0], k)
        labels.append(str(file)[:-4])
        k = k + 1

        x_train.append(xarray)
        y_train.append(yarray)

    x_train = np.array(x_train)
    x_train = x_train.reshape((x_train.shape[0] * x_train.shape[1], 28, 28))
    y_train = np.array(y_train)
    y_train = y_train.reshape((y_train.shape[0] * y_train.shape[1]))

    tmp = list(zip(x_train, y_train))
    shuffle(tmp)
    x_train, y_train = zip(*tmp)
    x_train = np.asarray(x_train)
    y_train = np.asarray(y_train)

    return x_train, y_train, labels

# Stworzenie okna ustawień dla aplikacji głównej
def main_settings_window(config: configparser.ConfigParser):
    import PySimpleGUI as sg
    tooltips = {}
    tooltips['model_location'] = 'Katalog modelu TensorFlow rozpoznającego gesty, ' \
                                 'otrzymany w procesie kreacji modelu.'
    tooltips['server_address'] = 'Adres serwera HTTP używanego do wysyłania zapytań o autoryzację'
    tooltips['max_predict_frames'] = 'Ilość klatek, po których pokazywany przez użytkownika gest ' \
                                     'jest zapisywany w kombinacji.'
    tooltips['max_write_frames'] = 'Ilość klatek, po których kombinacja jest zapisywana do sekwencji. '
    layout = [
        [sg.Text('Lokacja modelu:', tooltip=tooltips['model_location'])],
        [sg.InputText(key='model_path', tooltip=tooltips['model_location'], default_text=config['MAIN']['model_path']),
         sg.FolderBrowse("Przeglądaj...", key='folder_browse')],
        [sg.Text('Adres serwera:', tooltip=tooltips['server_address'])],
        [sg.InputText(key='server_address', tooltip=tooltips['server_address'], default_text=config['MAIN']['server_address'])],
        [sg.Text('Klatek do zapisania gestu:', tooltip=tooltips['max_predict_frames'])],
        [sg.InputText(key='max_predict_count', default_text=config['MAIN']['max_predict_count'],
                      tooltip=tooltips['max_predict_frames'])],
        [sg.Text('Klatek do zapisania kombinacji:', tooltip=tooltips['max_write_frames'])],
        [sg.InputText(key='max_write_count', default_text=config['MAIN']['max_write_count'],
                      tooltip=tooltips['max_write_frames'])],
        [sg.Text('', key='status_text')],
        [sg.Button('Zapisz'), sg.Button('Exit')]
    ]

    settings_window = sg.Window('Settings', layout, location=(400, 200), finalize=True,
                                keep_on_top=True)

    while True:
        event, values = settings_window.read()
        if event == sg.WIN_CLOSED or event == 'Exit':
            break
        elif event == 'Zapisz':
            try:
                max_predict_count = int(values['max_predict_count'])
            except ValueError:
                max_predict_count = 20
            try:
                max_write_count = int(values['max_write_count'])
            except ValueError:
                max_write_count = 20
            config['MAIN'] = {
                'model_path': values['model_path'],
                'server_address': values['server_address'],
                'max_predict_count': max_predict_count,
                'max_write_count': max_write_count
            }
            with open('config.ini', 'w') as configfile:
                config.write(configfile)
            sg.Popup("Zapisano", location=(settings_window.current_location()[0]+30,
                                           settings_window.current_location()[1]+30),
                     keep_on_top=True)

    settings_window.close()

# Generowanie binarnej maski metodą odejomawnia tłą
def mask_subtractor(frame, subtractor, iscropped=False, isprepared=False, learningRate=0.0):
    if not iscropped:
        cropped = frame[0:300, 0:300]
    else:
        cropped = frame

    if not isprepared:
        blurred = cv2.GaussianBlur(cropped, (5, 5), 0)
        image = cv2.fastNlMeansDenoisingColored(blurred, None, 127, 127, 7, 3)
    else:
        image = frame

    binary_mask = subtractor.apply(image, learningRate=learningRate)
    morphological = do_morphological(binary_mask)

    return morphological

# Generowanie binarnej maski metodą HSV
def mask_hsv(frame, lowerBound=None, upperBound=None, iscropped=False):
    if upperBound is None:
        upperBound = [15, 255, 255]
    if lowerBound is None:
        lowerBound = [2, 50, 50]
    if not iscropped:
        cropped = frame[0:300, 0:300]
    else:
        cropped = frame
    blur = cv2.blur(cropped, (3, 3))
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    binary_mask = cv2.inRange(hsv, np.array(lowerBound), np.array(upperBound))
    morphological = do_morphological(binary_mask)

    return morphological

# Generowanie binarnej maski metodą MediaPipe
def mask_mediapipe(frame, hands, mpHands, mpDraw, iscropped=False):
    if not iscropped:
        cropped = frame[0:300, 0:300]
    else:
        cropped = frame

    mask = np.zeros(cropped.shape, np.uint8)
    y, x, c = cropped.shape
    croppedrgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
    result = hands.process(croppedrgb)
    if result.multi_hand_landmarks:
        landmarks = []
        for handslms in result.multi_hand_landmarks:
            for i in [0, 1, 2, 5, 9, 13, 17]:
                lmx = int(handslms.landmark[i].x * x)
                lmy = int(handslms.landmark[i].y * y)
                landmarks.append([lmx, lmy])
            mpDraw.draw_landmarks(mask, handslms, mpHands.HAND_CONNECTIONS,
                                  None,
                                  mpDraw.DrawingSpec((255, 255, 255), 28, 10))
            landmarks = np.array(landmarks)
            cv2.fillPoly(mask, [landmarks], (255, 255, 255))
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    return mask

# Wyrównywanie jasności na podstawie porównania histogramów w skali szarości
def match_brightness(image, reference, scope=10):
    grayscale_reference = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    grayscale_reference_hist = cv2.calcHist([grayscale_reference], [0], None, [256], [0, 256])
    grayscale_image_hist = cv2.calcHist([grayscale_image], [0], None, [256], [0, 256])

    comparison = cv2.compareHist(grayscale_reference_hist, grayscale_image_hist, cv2.HISTCMP_CORREL)
    copy1 = cv2.add(grayscale_image, (1, 1, 1, 0))
    copy2 = cv2.add(grayscale_image, (-1, -1, -1, 0))
    copy1_hist = cv2.calcHist([copy1], [0], None, [256], [0, 256])
    copy2_hist = cv2.calcHist([copy2], [0], None, [256], [0, 256])
    comparison1 = cv2.compareHist(grayscale_reference_hist, copy1_hist, cv2.HISTCMP_CORREL)
    comparison2 = cv2.compareHist(grayscale_reference_hist, copy2_hist, cv2.HISTCMP_CORREL)

    if comparison1 > comparison:
        r = range(2, scope + 1, 1)
        comparison = comparison1
        i_best = 1
    elif comparison2 > comparison:
        r = range(-2, -scope - 1, -1)
        comparison = comparison2
        i_best = -1
    else:
        return image

    for i in r:
        copy1 = cv2.add(grayscale_image, (i, i, i, 0))
        copy1_hist = cv2.calcHist([copy1], [0], None, [256], [0, 256])
        comparison1 = cv2.compareHist(grayscale_reference_hist, copy1_hist, cv2.HISTCMP_CORREL)
        if comparison1 > comparison:
            comparison = comparison1
            i_best = i
        else:
            new_image = cv2.add(image, (i_best, i_best, i_best, 0))
            return new_image

    return image

# Przygotowywanie obrazu (odszumianie, wyrównywanie jasności)
def prepare_image(image, reference=None):
    denoised = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 5, 3)
    blurred = cv2.GaussianBlur(denoised, (5, 5), 0)
    if reference is not None:
        brightness_matched = match_brightness(blurred, reference)
        return brightness_matched
    return blurred

# Obrót maski w celu jej normalizacji
def rotate_mask(src):
    rotated = np.zeros((300, 300), np.uint8)
    contours, _ = cv2.findContours(src.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) != 0:
        c = max(contours, key=cv2.contourArea)
        cv2.drawContours(rotated, [c], 0, 255, -1)

        if len(c) > 5:
            _, _, angle = cv2.fitEllipse(c)
            if angle > 90:
                angle = 180 - angle
                angle = -angle
            rot_matrix = cv2.getRotationMatrix2D((150, 150), angle, 1.0)
            rotated = cv2.warpAffine(rotated, rot_matrix, (300, 300), flags=cv2.INTER_CUBIC)

    return rotated

# Tłumaczenie sekwencji gestów na ciąg znaków
def sequence_to_password(sequence):
    password = ''
    for combo in sequence:
        password += '['
        password += ','.join(str(val) for val in combo)
        password += ']'
    return password
