from PIL import Image
import numpy as np

prefix = 'img'

file = open('output.txt', 'w')

array = []

for i in range(100):
    imageArray = []
    if i < 10:
        file_name = prefix + '0' + str(i) + '.jpg'
    else:
        file_name = prefix + str(i) + '.jpg'

    path = 'img/' + file_name
    img = Image.open(path)
    img = img.convert('L')
    data = np.array(img)

    for j in data:
        for k in j:
            imageArray.append(k)

    array.append(imageArray)
file.write(str(array))
file.close()
