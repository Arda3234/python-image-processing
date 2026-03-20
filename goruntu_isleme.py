import cv2
import numpy as np
import matplotlib.pyplot as plt


def goster(basliklar, resimler):
    n = len(resimler)
    fig, eksenler = plt.subplots(1, n, figsize=(5 * n, 5))
    if n == 1:
        eksenler = [eksenler]
    for ax, baslik, resim in zip(eksenler, basliklar, resimler):
        harita = 'gray' if len(resim.shape) == 2 else None
        ax.imshow(resim, cmap=harita)
        ax.set_title(baslik, fontsize=12, fontweight='bold')
        ax.axis('off')
    plt.tight_layout()
    plt.show()


def test_goruntusu_olustur():
    kanvas = np.zeros((400, 400, 3), dtype=np.uint8)
    kanvas[50:150, 50:150]   = [255, 0, 0]
    kanvas[50:150, 200:300]  = [0, 255, 0]
    kanvas[200:300, 50:150]  = [0, 0, 255]
    kanvas[200:300, 200:300] = [255, 255, 0]
    cv2.circle(kanvas, (200, 200), 60, (255, 128, 0), -1)
    cv2.putText(kanvas, "Goruntu Isleme", (60, 370),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    rgb = cv2.cvtColor(kanvas, cv2.COLOR_BGR2RGB)
    goster(["Test Görüntüsü"], [rgb])
    return kanvas


def gri_ve_kanallar(goruntu):
    gri = cv2.cvtColor(goruntu, cv2.COLOR_BGR2GRAY)
    b, g, r = cv2.split(goruntu)
    goster(
        ["Orijinal", "Gri", "Mavi", "Yeşil", "Kırmızı"],
        [cv2.cvtColor(goruntu, cv2.COLOR_BGR2RGB), gri, b, g, r]
    )


def bulaniklik_filtreleri(goruntu):
    gurultulu = goruntu.copy()
    tuz_biber = np.random.randint(0, 2, goruntu.shape[:2], dtype=np.uint8)
    gurultulu[tuz_biber == 1] = [255, 255, 255]

    gaussian  = cv2.GaussianBlur(gurultulu, (7, 7), sigmaX=0)
    median    = cv2.medianBlur(gurultulu, 5)
    bilateral = cv2.bilateralFilter(goruntu.copy(), 9, 75, 75)

    goster(
        ["Gürültülü", "Gaussian", "Median", "Bilateral"],
        [cv2.cvtColor(g, cv2.COLOR_BGR2RGB)
         for g in [gurultulu, gaussian, median, bilateral]]
    )


def kenar_tespiti(goruntu):
    gri = cv2.cvtColor(goruntu, cv2.COLOR_BGR2GRAY)

    canny = cv2.Canny(gri, 50, 150)

    sobel_x = cv2.Sobel(gri, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gri, cv2.CV_64F, 0, 1, ksize=3)
    sobel   = cv2.magnitude(sobel_x, sobel_y).astype(np.uint8)

    laplacian = np.uint8(np.absolute(cv2.Laplacian(gri, cv2.CV_64F)))

    goster(["Gri", "Canny", "Sobel", "Laplacian"], [gri, canny, sobel, laplacian])


def morfolojik_islemler(goruntu):
    gri = cv2.cvtColor(goruntu, cv2.COLOR_BGR2GRAY)
    _, ikili = cv2.threshold(gri, 127, 255, cv2.THRESH_BINARY)
    kernel = np.ones((5, 5), np.uint8)

    goster(
        ["İkili", "Erozyon", "Genişleme", "Açma", "Kapama"],
        [
            ikili,
            cv2.erode(ikili, kernel),
            cv2.dilate(ikili, kernel),
            cv2.morphologyEx(ikili, cv2.MORPH_OPEN, kernel),
            cv2.morphologyEx(ikili, cv2.MORPH_CLOSE, kernel),
        ]
    )


def renk_maskeleme(goruntu):
    hsv   = cv2.cvtColor(goruntu, cv2.COLOR_BGR2HSV)
    maske = cv2.inRange(hsv, np.array([20, 100, 100]), np.array([30, 255, 255]))
    sonuc = cv2.bitwise_and(goruntu, goruntu, mask=maske)

    goster(
        ["Orijinal", "Sarı Maskesi", "İzole Sarı"],
        [
            cv2.cvtColor(goruntu, cv2.COLOR_BGR2RGB),
            maske,
            cv2.cvtColor(sonuc, cv2.COLOR_BGR2RGB),
        ]
    )


def histogram_esitleme(goruntu):
    gri         = cv2.cvtColor(goruntu, cv2.COLOR_BGR2GRAY)
    esitlenmis  = cv2.equalizeHist(gri)
    clahe_sonuc = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gri)

    fig, eksenler = plt.subplots(2, 3, figsize=(15, 8))
    for ax, img, baslik in zip(eksenler[0],
                                [gri, esitlenmis, clahe_sonuc],
                                ["Orijinal", "Histogram Eşitleme", "CLAHE"]):
        ax.imshow(img, cmap='gray')
        ax.set_title(baslik, fontweight='bold')
        ax.axis('off')

    for ax, img, baslik in zip(eksenler[1],
                                [gri, esitlenmis, clahe_sonuc],
                                ["Orijinal Hist.", "Eşitleme Sonrası", "CLAHE Sonrası"]):
        ax.hist(img.ravel(), 256, [0, 256], color='steelblue', alpha=0.8)
        ax.set_title(baslik, fontweight='bold')
        ax.set_xlim([0, 256])

    plt.tight_layout()
    plt.show()


def kontur_tespiti(goruntu):
    gri = cv2.cvtColor(goruntu, cv2.COLOR_BGR2GRAY)
    _, ikili = cv2.threshold(gri, 100, 255, cv2.THRESH_BINARY)
    konturlar, _ = cv2.findContours(ikili, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    sonuc = goruntu.copy()
    cv2.drawContours(sonuc, konturlar, -1, (0, 255, 0), 3)

    for i, k in enumerate(konturlar, 1):
        print(f"  Kontur #{i}: Alan={cv2.contourArea(k):.0f}px²  "
              f"Çevre={cv2.arcLength(k, True):.0f}px")

    goster(
        ["Orijinal", "İkili", "Konturlar"],
        [cv2.cvtColor(g, cv2.COLOR_BGR2RGB)
         for g in [goruntu, cv2.cvtColor(ikili, cv2.COLOR_GRAY2BGR), sonuc]]
    )


if __name__ == "__main__":
    # Kendi görüntünü kullanmak için:
    # goruntu = cv2.imread("resim.jpg")

    goruntu = test_goruntusu_olustur()
    gri_ve_kanallar(goruntu)
    bulaniklik_filtreleri(goruntu)
    kenar_tespiti(goruntu)
    morfolojik_islemler(goruntu)
    renk_maskeleme(goruntu)
    histogram_esitleme(goruntu)
    kontur_tespiti(goruntu)
