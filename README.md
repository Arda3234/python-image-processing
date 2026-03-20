# 🖼️ python-image-processing

OpenCV ve NumPy kullanarak temel ve orta seviye görüntü işleme örnekleri.

## İçerik

| # | Fonksiyon | Konu |
|---|-----------|------|
| 1 | `test_goruntusu_olustur` | NumPy ile sıfırdan görüntü oluşturma |
| 2 | `gri_ve_kanallar` | Gri tonlama & RGB kanal ayrıştırma |
| 3 | `bulaniklik_filtreleri` | Gaussian, Median, Bilateral filtreler |
| 4 | `kenar_tespiti` | Canny, Sobel, Laplacian |
| 5 | `morfolojik_islemler` | Erozyon, Genişleme, Açma, Kapama |
| 6 | `renk_maskeleme` | HSV renk uzayında maskeleme |
| 7 | `histogram_esitleme` | Histogram eşitleme & CLAHE |
| 8 | `kontur_tespiti` | Kontur bulma, alan ve çevre hesaplama |

## Kurulum

```bash
pip install opencv-python numpy matplotlib
```

## Kullanım

```bash
python goruntu_isleme.py
```

Kendi görüntünü kullanmak için `__main__` bloğundaki satırı düzenle:

```python
goruntu = cv2.imread("resim.jpg")
```

## Gereksinimler

- Python 3.8+
- opencv-python
- numpy
- matplotlib
