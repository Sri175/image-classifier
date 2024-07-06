# Helmet Detection Image Classifier

This project is an image classifier that categorizes images into two classes: with helmet and without helmet. The classifier is built using TensorFlow and Keras.

## Project Structure

Helmet-Detection-Image-Classifier/
├── data/
│ ├── with helmet/
│ └── without helmet/
├── logs/
├── README.md
├── requirements.txt
└── classifier.py




## Data Preparation

1. Place your image data in the `data/` directory, with subdirectories `with helmet` and `without helmet`.

2. The script will check and remove any files that are not valid image formats (jpeg, jpg, bmp, png).
3. data could be downloaded from this link 

https://drive.google.com/drive/folders/1po0LNhvdAFPjWTiCTsJ_P4k715qUXLrH?usp=sharing



## Training the Model

1. Run the `classifier.py` script to preprocess the data, build, and train the model:

    ```bash
    python classifier.py
    ```

2. The training logs will be saved in the `logs/` directory, which can be visualized using TensorBoard.

## Testing the Model

1. Load an image to test:

    ```python
    import cv2
    import tensorflow as tf
    import numpy as np
    from matplotlib import pyplot as plt

    img = cv2.imread('path/to/image.jpg')
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.show()

    resize = tf.image.resize(img, (256, 256))
    plt.imshow(resize.numpy().astype(int))
    plt.show()

    yhat = model.predict(np.expand_dims(resize / 255, 0))
    if yhat < 0.5:
        print("Photo is with helmet")
    else:
        print("Photo is without helmet")
    ```

## Results

Here are some example results:

![Example with Helmet](logs/example_with_helmet.png)
![Example without Helmet](logs/example_without_helmet.png)

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
