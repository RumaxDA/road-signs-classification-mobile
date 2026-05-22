from PIL import Image

def stworz_siatke_bez_napisow(lista_zdjec, sciezka_wyjsciowa):
    if not lista_zdjec:
        print("Błąd: Lista zdjęć jest pusta.")
        return

    docelowa_szerokosc_ikony = 800  
    
    # Otwórz pierwsze zdjęcie, żeby ustalić proporcje
    img0 = Image.open(lista_zdjec[0])
    wspolczynnik = docelowa_szerokosc_ikony / img0.size[0]
    docelowa_wysokosc_ikony = int(img0.size[1] * wspolczynnik)

    nowa_szerokosc = docelowa_szerokosc_ikony * len(lista_zdjec)
    # Płótno musi pomieścić wszystkie zdjęcia z listy ułożone w pionie
    nowa_wysokosc = docelowa_wysokosc_ikony 
    
    nowy_obraz = Image.new('RGB', (nowa_szerokosc, nowa_wysokosc), (255, 255, 255))

    for i, plik in enumerate(lista_zdjec):
        # Zdjęcia układamy pionowo, więc obliczamy przesunięcie na osi Y
        # mnożąc indeks przez wysokość (a nie szerokość) obrazka
        x_pozycja = i * docelowa_szerokosc_ikony

        img = Image.open(plik)
        img = img.resize((docelowa_szerokosc_ikony, docelowa_wysokosc_ikony), Image.Resampling.LANCZOS)
        
        # Wklejamy zdjęcie na współrzędnej X=0 i wyliczonej współrzędnej Y
        nowy_obraz.paste(img, (x_pozycja, 0))

    nowy_obraz.save(sciezka_wyjsciowa, quality=95)
    print(f"✓ Gotowe! Obraz zapisany jako: {sciezka_wyjsciowa}")

# --- Użycie ---
pliki = ['ml/zdjecia/przejscie.png', 'ml/zdjecia/images.png']
stworz_siatke_bez_napisow(pliki, "znaki_przejscia.png")