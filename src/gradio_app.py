# Güncellenecek
import gradio as gr
from PIL import Image
import numpy as np

def segment_image(input_image):
    """
    Bu fonksiyon, bir görüntü alır ve segmentasyon modelini kullanarak segmentasyon maskesini döndürür.
    Burada, yer tutucu olarak rastgele bir maske oluşturuyoruz.
    Gerçek modelinizi buraya entegre etmelisiniz.
    """

    # PIL Image'ı NumPy dizisine dönüştür
    img_array = np.array(input_image)

    # Rastgele maske oluştur (gerçek modelinizi burada kullanın)
    # Burada rastgele bir maske oluşturuyoruz.
    # Gerçekte, modelinizin çıktısını kullanmalısınız.
    mask = np.random.randint(0, 256, size=img_array.shape[:2], dtype=np.uint8)

    # NumPy dizisini PIL Image'a dönüştür ve döndür
    segmented_image = Image.fromarray(mask)

    return segmented_image


iface = gr.Interface(
    fn=segment_image,
    inputs=gr.Image(type="pil"), # Girdi tipi "pil" olarak belirtilmelidir
    outputs=gr.Image(type="pil"), # Çıktı tipi "pil" olarak belirtilmelidir
    title="Görüntü Segmentasyonu",
    description="Bir görüntü yükleyin ve segmentasyon maskesini alın."
)


iface.launch()