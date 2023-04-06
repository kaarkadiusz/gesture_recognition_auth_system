import PySimpleGUI as sg
import numpy as np
from keras.models import Sequential
from keras import layers
from os import listdir

import asugrmodule

sg.theme('Default1')

# Wczytanie podpowiedzi
tooltips = asugrmodule.get_tooltips('model_creator')

# Layout aplikacji graficznej
main_column = [
    [sg.Text('Gesty:', tooltip=tooltips['gestures']), sg.Push(), sg.InputText(key='gestures_location'),
     sg.FolderBrowse("Przeglądaj...", key='folder_browse')],
    [sg.Checkbox('Tworzyć zbiór testowy?', default=True, key='test_bool', tooltip=tooltips['test_bool'])],
    [sg.Button('Kompiluj i trenuj model', key='compile_train_button', tooltip=tooltips['compile_train_button']),
     sg.Text('', key='status_text')],
    [sg.Text('Lokacja:', tooltip=tooltips['location']), sg.Push(), sg.InputText(key='model_name'),
     sg.FolderBrowse("Przeglądaj...", key='folder_browse')],
    [sg.Button('Zapisz model', key='save_button',
               tooltip=tooltips['save_button']), sg.Push(),
     sg.Button('Exit', tooltip=tooltips['exit'])]
]
layout = [
    [sg.Column(main_column)]
]

# Obiekt okna aplikacji
window = sg.Window('Model Creator', layout, location=(300, 100), finalize=True, return_keyboard_events=True)

# Zmienne wykorzystywane w procesie tworzenia modelu i pliku etykiet
model = None
labels = None

while True:
    event, values = window.read(timeout=1)
    # Obsługa zdarzeń wywoływanych przez użytkownika
    if event == sg.WIN_CLOSED or event == 'Exit':
        break
    # Obsługa trenowania
    elif event == 'compile_train_button' and values['gestures_location'] != '':
        window.disable()
        window['status_text'].update(value='Przygotowywanie danych...')
        window.refresh()
        # Wczytanie danych o gestach
        x_train, y_train, labels = asugrmodule.load_data(values['gestures_location'])
        x_train = x_train / 255.0
        ngestures = len(labels)
        # Utworzenie zbioru testowego
        split_point = int(len(x_train) * 0.8)
        if values['test_bool']:
            x_test = x_train[split_point:]
            y_test = y_train[split_point:]
            x_train = x_train[:split_point]
            y_train = y_train[:split_point]
        else:
            x_test = x_train[split_point:]
            y_test = y_train[split_point:]
        window['status_text'].update(value='Tworzenie modelu...')
        window.refresh()
        # Definicja modelu i jego warstw
        model = Sequential([
            layers.Conv2D(input_shape=(28, 28, 1), kernel_size=5,
                          filters=8, strides=1,
                          activation='relu',
                          kernel_initializer='variance_scaling'),
            layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
            layers.Conv2D(kernel_size=5,
                          filters=16, strides=1,
                          activation='relu',
                          kernel_initializer='variance_scaling'),
            layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(ngestures, activation='softmax')
        ])
        window['status_text'].update(value='Konfiguracja...')
        window.refresh()
        # Konfiguracja modelu pod trening
        model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
        window['status_text'].update(value='Uczenie...')
        window.refresh()
        # Trenowanie modelu
        history = model.fit(x_train, y_train, epochs=2)
        if values['test_bool']:
            window['status_text'].update(value='Ewaluacja...')
            window.refresh()
            _, accuracy = model.evaluate(x_test, y_test)
        else:
            accuracy = history.history['accuracy'][-1]
        # Wypisanie informacji o trafności
        window['status_text'].update(value='Trafność: ' + str(accuracy))
        window.enable()
    # Zapisanie modelu oraz pliku etykiet
    elif event == 'save_button' and model is not None and labels is not None \
            and values['model_name'] != '':
        try:
            directory_length = len(listdir(values['model_name']))
        except FileNotFoundError:
            directory_length = 0
        if directory_length != 0:
            choice = sg.Popup("Podany folder nie jest pusty\nKontynuowanie dopisze model do katalogu.",
                              custom_text=('Kontynuuj', 'Anuluj'))
            if choice == 'Kontynuuj':
                model.save(values['model_name'])
                np.save(values['model_name'] + '\\labels', np.array(labels))
                sg.Popup("Zapisano jako: " + values['model_name'])
        else:
            model.save(values['model_name'])
            np.save(values['model_name'] + '\\labels', np.array(labels))
            sg.Popup("Zapisano jako: " + values['model_name'])

window.close()
