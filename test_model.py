import cv2
import matplotlib.pyplot as plt
from PIL import Image
from lang_efficient_sam import LangEfficientSAM
from utils.draw_image import draw_image


def main():
    image_path = './images/fruits.jpg'
    image = Image.open(image_path).convert("RGB")
    image2 = cv2.imread(image_path)
    image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)

    model = LangEfficientSAM()

    masks, boxes, phrases, logits = model.predict(image, "apple")

    plt.figure(figsize=(20, 20))
    plt.imshow(draw_image(image2, masks, boxes, phrases, alpha=0.4))
    plt.show()


if __name__ == '__main__':
    main()
