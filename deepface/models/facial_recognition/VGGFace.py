# built-in dependencies
from typing import List

# 3rd party dependencies
import numpy as np

# project dependencies
from deepface.commons import package_utils, weight_utils
from deepface.modules import verification
from deepface.models.FacialRecognition import FacialRecognition
from deepface.commons.logger import Logger

logger = Logger()

# ---------------------------------------

tf_version = package_utils.get_tf_major_version()
if tf_version == 1:
    from keras.models import Model, Sequential
    from keras.layers import (
        Convolution2D,
        ZeroPadding2D,
        MaxPooling2D,
        Flatten,
        Dropout,
        Activation,
    )
else:
    from tensorflow.keras.models import Model, Sequential
    from tensorflow.keras.layers import (
        Convolution2D,
        ZeroPadding2D,
        MaxPooling2D,
        Flatten,
        Dropout,
        Activation,
    )

# ---------------------------------------

WEIGHTS_URL = (
    "https://github.com/serengil/deepface_models/releases/download/v1.0/vgg_face_weights.h5"
)


# pylint: disable=too-few-public-methods
class VggFaceClient(FacialRecognition):
    """
    VGG-Face model class
    """

    def __init__(self):
        self.model = load_model()
        self.model_name = "VGG-Face"
        self.input_shape = (224, 224)
        self.output_shape = 4096

    def forward(self, img: np.ndarray) -> List[float]:
        """
        Generates embeddings using the VGG-Face model.
            This method incorporates an additional normalization layer.

        Args:
            img (np.ndarray): pre-loaded image in BGR
        Returns
            embeddings (list): multi-dimensional vector
        """
        # model.predict causes memory issue when it is called in a for loop
        # embedding = model.predict(img, verbose=0)[0].tolist()

        # having normalization layer in descriptor troubles for some gpu users (e.g. issue 957, 966)
        # instead we are now calculating it with traditional way not with keras backend
        embedding = super().forward(img)
        if isinstance(embedding, list) and len(embedding) > 0 and isinstance(embedding[0], list):
            embedding = verification.l2_normalize(embedding, axis=1)
        else:
            embedding = verification.l2_normalize(embedding)
        return embedding.tolist()


def base_model() -> Sequential:
    """
    Base model of VGG-Face being used for classification - not to find embeddings
    Returns:
        model (Sequential): model was trained to classify 2622 identities
    """
    model = Sequential()
    model.add(ZeroPadding2D((1, 1), input_shape=(224, 224, 3)))
    model.add(Convolution2D(64, (3, 3), activation="relu"))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(64, (3, 3), activation="relu"))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, (3, 3), activation="relu"))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, (3, 3), activation="relu"))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, (3, 3), activation="relu"))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, (3, 3), activation="relu"))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, (3, 3), activation="relu"))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, (3, 3), activation="relu"))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, (3, 3), activation="relu"))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, (3, 3), activation="relu"))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, (3, 3), activation="relu"))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, (3, 3), activation="relu"))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, (3, 3), activation="relu"))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(Convolution2D(4096, (7, 7), activation="relu"))
    model.add(Dropout(0.5))
    model.add(Convolution2D(4096, (1, 1), activation="relu"))
    model.add(Dropout(0.5))
    model.add(Convolution2D(2622, (1, 1)))
    model.add(Flatten())
    model.add(Activation("softmax"))

    return model


def load_model(
    url=WEIGHTS_URL,
) -> Model:
    """
    Final VGG-Face model being used for finding embeddings
    Returns:
        model (Model): returning 4096 dimensional vectors
    """

    model = base_model()

    weight_file = weight_utils.download_weights_if_necessary(
        file_name="vgg_face_weights.h5", source_url=url
    )

    model = weight_utils.load_model_weights(model=model, weight_file=weight_file)

    # 2622d dimensional model
    # vgg_face_descriptor = Model(inputs=model.layers[0].input, outputs=model.layers[-2].output)

    # 4096 dimensional model offers 6% to 14% increasement on accuracy!
    # - softmax causes underfitting
    # - added normalization layer to avoid underfitting with euclidean
    # as described here: https://github.com/serengil/deepface/issues/944
    base_model_output = Flatten()(model.layers[-5].output)
    # keras backend's l2 normalization layer troubles some gpu users (e.g. issue 957, 966)
    # base_model_output = Lambda(lambda x: K.l2_normalize(x, axis=1), name="norm_layer")(
    #     base_model_output
    # )
    vgg_face_descriptor = Model(inputs=model.layers[0].input, outputs=base_model_output)

    return vgg_face_descriptor
