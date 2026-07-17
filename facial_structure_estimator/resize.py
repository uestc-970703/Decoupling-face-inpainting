from PIL import Image

image = Image.open('./images_512/13050_results.png')

resize = image.resize((256, 256))
resize.save('./images_512/13050_results.png')