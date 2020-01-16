
This folder contains the images and annotations to train the model.

`data` folder structure:
```
data/
    train/
        Blacky/
            imgs.jpg
            ...
        Niche/
            imgs.jpg
            ...
        cats-annotations.json
    validation/
        Blacky/
            imgs.jpg
            ...
        Niche/
            imgs.jpg
            ...
        cats-annotations.json
```

The `cats-annotations.json` file in `train` folder has annotations for the training images. The file in `validation` folder has the annotations for the validation images.

Download the dataset from: https://drive.google.com/open?id=1o9zyd51QCqWG3DlArQnfH4q4clMVQmTY (zip file ~39 MB)

Annotation tool: [VIA](http://www.robots.ox.ac.uk/~vgg/software/via/)