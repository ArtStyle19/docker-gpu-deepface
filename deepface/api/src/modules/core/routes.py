# built-in dependencies
from typing import Union

# 3rd party dependencies
from flask import Blueprint, request, jsonify
import numpy as np

# project dependencies
from deepface import DeepFace
from deepface.api.src.modules.core import service
from deepface.commons import image_utils
from deepface.commons.logger import Logger

# from deepface.commons import functions
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
# from deepface.basemodels import VGGFace
import os  # 👈 Add this line
# from deepface.commons.models import Facenet512

logger = Logger()

blueprint = Blueprint("routes", __name__)

# pylint: disable=no-else-return, broad-except


@blueprint.route("/")
def home():
    return f"<h1>Welcome to DeepFace API v{DeepFace.__version__}!</h1>"


def extract_image_from_request(img_key: str) -> Union[str, np.ndarray]:
    """
    Extracts an image from the request either from json or a multipart/form-data file.

    Args:
        img_key (str): The key used to retrieve the image data
            from the request (e.g., 'img1').

    Returns:
        img (str or np.ndarray): Given image detail (base64 encoded string, image path or url)
            or the decoded image as a numpy array.
    """

    # Check if the request is multipart/form-data (file input)
    if request.files:
        # request.files is instance of werkzeug.datastructures.ImmutableMultiDict
        # file is instance of werkzeug.datastructures.FileStorage
        file = request.files.get(img_key)

        if file is None:
            raise ValueError(f"Request form data doesn't have {img_key}")

        if file.filename == "":
            raise ValueError(f"No file uploaded for '{img_key}'")

        img = image_utils.load_image_from_file_storage(file)

        return img
    # Check if the request is coming as base64, file path or url from json or form data
    elif request.is_json or request.form:
        input_args = request.get_json() or request.form.to_dict()

        if input_args is None:
            raise ValueError("empty input set passed")

        # this can be base64 encoded image, and image path or url
        img = input_args.get(img_key)

        if not img:
            raise ValueError(f"'{img_key}' not found in either json or form data request")

        return img

    # If neither JSON nor file input is present
    raise ValueError(f"'{img_key}' not found in request in either json or form data")


@blueprint.route("/represent", methods=["POST"])
def represent():
    input_args = (request.is_json and request.get_json()) or (
        request.form and request.form.to_dict()
    )

    try:
        img = extract_image_from_request("img")
    except Exception as err:
        return {"exception": str(err)}, 400

    obj = service.represent(
        img_path=img,
        model_name=input_args.get("model_name", "VGG-Face"),
        detector_backend=input_args.get("detector_backend", "opencv"),
        enforce_detection=input_args.get("enforce_detection", True),
        align=input_args.get("align", True),
        anti_spoofing=input_args.get("anti_spoofing", False),
        max_faces=input_args.get("max_faces"),
    )

    logger.debug(obj)

    return obj


@blueprint.route("/verify", methods=["POST"])
def verify():
    input_args = (request.is_json and request.get_json()) or (
        request.form and request.form.to_dict()
    )

    try:
        img1 = extract_image_from_request("img1")
    except Exception as err:
        return {"exception": str(err)}, 400

    try:
        img2 = extract_image_from_request("img2")
    except Exception as err:
        return {"exception": str(err)}, 400

    verification = service.verify(
        img1_path=img1,
        img2_path=img2,
        model_name=input_args.get("model_name", "VGG-Face"),
        detector_backend=input_args.get("detector_backend", "opencv"),
        distance_metric=input_args.get("distance_metric", "cosine"),
        align=input_args.get("align", True),
        enforce_detection=input_args.get("enforce_detection", True),
        anti_spoofing=input_args.get("anti_spoofing", False),
    )

    logger.debug(verification)

    return verification


@blueprint.route("/analyze", methods=["POST"])
def analyze():
    input_args = (request.is_json and request.get_json()) or (
        request.form and request.form.to_dict()
    )

    try:
        img = extract_image_from_request("img")
    except Exception as err:
        return {"exception": str(err)}, 400

    actions = input_args.get("actions", ["age", "gender", "emotion", "race"])
    # actions is the only argument instance of list or tuple
    # if request is form data, input args can either be text or file
    if isinstance(actions, str):
        actions = (
            actions.replace("[", "")
            .replace("]", "")
            .replace("(", "")
            .replace(")", "")
            .replace('"', "")
            .replace("'", "")
            .replace(" ", "")
            .split(",")
        )

    demographies = service.analyze(
        img_path=img,
        actions=actions,
        detector_backend=input_args.get("detector_backend", "opencv"),
        enforce_detection=input_args.get("enforce_detection", True),
        align=input_args.get("align", True),
        anti_spoofing=input_args.get("anti_spoofing", False),
    )
    

    logger.debug(demographies)

    return demographies


@blueprint.route("/find", methods=["POST"])
def find():
    input_args = (request.is_json and request.get_json()) or (
        request.form and request.form.to_dict()
    )

    try:
        img = extract_image_from_request("img")
    except Exception as err:
        return {"exception": str(err)}, 400

    db_path = input_args.get("db_path")
    if not db_path:
        return {"exception": "'db_path' parameter is required"}, 400

    try:
        results = DeepFace.find(
            img_path=img,
            db_path=db_path,
            # model_name=input_args.get("model_name", "VGG-Face"),
            model_name=input_args.get("model_name", "DeepFace"),
            # detector_backend=input_args.get("detector_backend", "opencv"),
            detector_backend=input_args.get("detector_backend", "mediapipe"),
            distance_metric=input_args.get("distance_metric", "cosine"),
            enforce_detection=input_args.get("enforce_detection", True),
            align=input_args.get("align", True)
        )
        
        
        if isinstance(results, list) and len(results) > 0:
            results_json = results[0].to_dict(orient="records")
            return jsonify({"results": results_json})
        else:
            return jsonify({"results": []})
    except Exception as err:
        return {"exception": str(err)}, 500







@blueprint.route("/search_pkl", methods=["POST"])
def search_pkl_route():
    input_args = (request.is_json and request.get_json()) or (
        request.form and request.form.to_dict()
    )

    try:
        img = extract_image_from_request("img")
    except Exception as err:
        return {"exception": str(err)}, 400

    pkl_path = input_args.get("pkl_path")
    if not pkl_path or not os.path.exists(pkl_path):
        return {"exception": "'pkl_path' not found or invalid"}, 400

    try:
        response = service.search_pkl(
            img_path=img,
            pkl_path=pkl_path,
            model_name=input_args.get("model_name", "Facenet512"),
            detector_backend=input_args.get("detector_backend", "mediapipe"),
            distance_metric=input_args.get("distance_metric", "cosine"),
            enforce_detection=input_args.get("enforce_detection", True),
            align=input_args.get("align", True),
            top_k=int(input_args.get("top_k", 5)),
            threshold=float(input_args.get("threshold", 0.7))
        )
        return jsonify(response)
    except Exception as err:
        return {"exception": str(err)}, 500

