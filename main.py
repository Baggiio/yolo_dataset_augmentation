import os
import albumentations as A
import cv2 as cv
import sys

print("Para que o algoritmo funcione corretamente, é necessário que as imagens e as labels estejam na pasta 'images', e que o arquivo 'data.yaml' esteja na pasta raiz.")
print("Certifique-se que está tudo correto. Deseja continuar? (s/n)")
answer = input()
if answer == "n":
    exit()

try:
    with open("data.yaml", "r") as yaml_file:
        labels = yaml_file.readlines()
        label_names = labels[5]
        label_names = label_names[8:-2]
        label_names = label_names.replace("'", "")
        label_names = label_names.split(", ")
except:
    print("Arquivo 'data.yaml' não encontrado.")
    exit()

if os.path.exists("transformed_images") == False:
    os.mkdir("transformed_images")

# Declare an augmentation pipeline
transform = A.Compose([
    A.Affine(scale=(0.5, 1.5), rotate=(-10, 10), shear=(-3, 3), p=0.5),
    A.RandomCrop(width=1008, height=756),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.3, p=1.0),
], bbox_params=A.BboxParams(format='yolo', min_area=4000, min_visibility=0.2))

image_list = os.listdir('images/')
image_names = []
for image in image_list:
    image_names.append(image[:-4])

augment_factor = int(input("Insira em quantas vezes deseja aumentar o dataset: "))

toolbar_width = len(image_names)

# setup toolbar
sys.stdout.write("[%s]" % (" " * toolbar_width))
sys.stdout.flush()
sys.stdout.write("\b" * (toolbar_width+1))

# Read an image with OpenCV and convert it to the RGB colorspace
for image_name in image_names:

    image = cv.imread(f"images/{image_name}.jpg")
    # image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

    with open (f"images/{image_name}.txt", "r") as f:
        lines = f.readlines()
        bboxes = []
        for line in lines:
            line = line.split()
            bboxes.append([float(line[1]), float(line[2]), float(line[3]), float(line[4]), int(line[0])])

    for i in range(augment_factor):
        # Augment an image
        transformed = transform(image=image, bboxes=bboxes)
        transformed_image = transformed["image"]
        transformed_bboxes = transformed["bboxes"]

        transformed_image = cv.resize(transformed_image, (1280, 960))

        # Save the image
        with open (f"transformed_images/{image_name}_{i}.txt", "w") as f:
            for j in range(len(transformed_bboxes)):
                

                if j == len(transformed_bboxes) - 1:
                    f.write(f"{label_names[transformed_bboxes[j][4]]} {transformed_bboxes[j][0]} {transformed_bboxes[j][1]} {transformed_bboxes[j][2]} {transformed_bboxes[j][3]}")
                else:
                    f.write(f"{label_names[transformed_bboxes[j][4]]} {transformed_bboxes[j][0]} {transformed_bboxes[j][1]} {transformed_bboxes[j][2]} {transformed_bboxes[j][3]}\n")

        cv.imwrite(f"transformed_images/{image_name}_{i}.jpg", transformed_image)

    sys.stdout.write("-")
    sys.stdout.flush()

sys.stdout.write("]\n")