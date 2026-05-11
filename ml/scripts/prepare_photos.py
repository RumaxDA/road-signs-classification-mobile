from PIL import Image
import os

def stworz_siatke_zdjec_hd(lista_zdjec, sciezka_wyjsciowa, docelowa_szerokosc_ikony=1200):
    """
    Łączy listę 4 zdjęć w siatkę 2x2 w wysokiej rozdzielczości, zachowując proporcje.
    """
    if len(lista_zdjec) != 4:
        print("Błąd: Funkcja wymaga dokładnie 4 zdjęć!")
        return

    # 1. Pobieramy wymiary z pierwszego zdjęcia, aby zachować oryginalne proporcje
    img0 = Image.open(lista_zdjec[0])
    oryginalna_szer_img, oryginalna_wys_img = img0.size
    
    # 2. Obliczamy nową wysokość na podstawie docelowej szerokości, żeby nie spłaszczyć wykresu
    wspolczynnik = docelowa_szerokosc_ikony / oryginalna_szer_img
    docelowa_wysokosc_ikony = int(oryginalna_wys_img * wspolczynnik)

    # 3. Tworzymy wielkie białe płótno (2 kolumny, 2 wiersze)
    nowa_szerokosc = docelowa_szerokosc_ikony * 2
    nowa_wysokosc = docelowa_wysokosc_ikony * 2
    nowy_obraz = Image.new('RGB', (nowa_szerokosc, nowa_wysokosc), (255, 255, 255))

    # Pozycje (X, Y) dla kolejnych zdjęć
    pozycje = [
        (0, 0),                                               # Lewy górny
        (docelowa_szerokosc_ikony, 0),                        # Prawy górny
        (0, docelowa_wysokosc_ikony),                         # Lewy dolny
        (docelowa_szerokosc_ikony, docelowa_wysokosc_ikony)   # Prawy dolny
    ]

    # 4. Wklejamy obrazy z wykorzystaniem algorytmu LANCZOS (najlepszy do tekstów)
    for i, plik in enumerate(lista_zdjec):
        img = Image.open(plik)
        img = img.resize((docelowa_szerokosc_ikony, docelowa_wysokosc_ikony), Image.Resampling.LANCZOS)
        nowy_obraz.paste(img, pozycje[i])

    # 5. Zapisujemy z parametrem quality=95, aby uniknąć kompresji JPG
    nowy_obraz.save(sciezka_wyjsciowa, quality=95)
    print(f"✓ Siatka HD zapisana jako: {sciezka_wyjsciowa}")

# --- Użycie ---
# pliki = [
#     'new_plots/Custom_CNN/cnn_32/CNN_32_confidence_distribution.png', 
#     'new_plots/Custom_CNN/cnn_48/CNN_48_confidence_distribution.png', 
#     'new_plots/Custom_CNN/cnn_96/CNN_96_confidence_distribution.png', 
#     'new_plots/Custom_CNN/cnn_224/CNN_224_confidence_distribution.png'
# ]
pliki = [
    'new_plots/EfficientNet/tl_32/EfficientNetB0_32_confidence_distribution.png', 
    'new_plots/EfficientNet/tl_48/EfficientNetB0_48_confidence_distribution.png', 
    'new_plots/EfficientNet/tl_96/EfficientNetB0_96_confidence_distribution.png', 
    'new_plots/EfficientNet/tl_224/EfficientNetB0_224_confidence_distribution.png'
]

stworz_siatke_zdjec_hd(pliki, "siatka_confidence_2x2.jpg")