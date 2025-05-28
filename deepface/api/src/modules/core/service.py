# built-in dependencies
import traceback
from typing import Optional, Union

import tensorflow as tf
tf.config.optimizer.set_jit(False)                # 🔴 Required!
tf.config.run_functions_eagerly(True)             # 🟡 Optional but helps stability

import faiss
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from typing import Union
import traceback


# 3rd party dependencies
import numpy as np

# project dependencies
from deepface import DeepFace
from deepface.commons.logger import Logger

logger = Logger()


# pylint: disable=broad-except


def represent(
    img_path: Union[str, np.ndarray],
    model_name: str,
    detector_backend: str,
    enforce_detection: bool,
    align: bool,
    anti_spoofing: bool,
    max_faces: Optional[int] = None,
):
    try:
        result = {}
        embedding_objs = DeepFace.represent(
            img_path=img_path,
            model_name=model_name,
            detector_backend=detector_backend,
            enforce_detection=enforce_detection,
            align=align,
            anti_spoofing=anti_spoofing,
            max_faces=max_faces,
        )
        result["results"] = embedding_objs
        return result
    except Exception as err:
        tb_str = traceback.format_exc()
        logger.error(str(err))
        logger.error(tb_str)
        return {"error": f"Exception while representing: {str(err)} - {tb_str}"}, 400


def verify(
    img1_path: Union[str, np.ndarray],
    img2_path: Union[str, np.ndarray],
    model_name: str,
    detector_backend: str,
    distance_metric: str,
    enforce_detection: bool,
    align: bool,
    anti_spoofing: bool,
):
    try:
        obj = DeepFace.verify(
            img1_path=img1_path,
            img2_path=img2_path,
            model_name=model_name,
            detector_backend=detector_backend,
            distance_metric=distance_metric,
            align=align,
            enforce_detection=enforce_detection,
            anti_spoofing=anti_spoofing,
        )
        return obj
    except Exception as err:
        tb_str = traceback.format_exc()
        logger.error(str(err))
        logger.error(tb_str)
        return {"error": f"Exception while verifying: {str(err)} - {tb_str}"}, 400


def analyze(
    img_path: Union[str, np.ndarray],
    actions: list,
    detector_backend: str,
    enforce_detection: bool,
    align: bool,
    anti_spoofing: bool,
):
    try:
        result = {}
        demographies = DeepFace.analyze(
            img_path=img_path,
            actions=actions,
            detector_backend=detector_backend,
            enforce_detection=enforce_detection,
            align=align,
            silent=True,
            anti_spoofing=anti_spoofing,
        )
        result["results"] = demographies
        return result
    except Exception as err:
        tb_str = traceback.format_exc()
        logger.error(str(err))
        logger.error(tb_str)
        return {"error": f"Exception while analyzing: {str(err)} - {tb_str}"}, 400


def find(
    img_path: Union[str, np.ndarray],
    db_path: str,
    model_name: str,
    detector_backend: str,
    distance_metric: str,
    enforce_detection: bool,
    align: bool,
    anti_spoofing: bool,
):
    try:
        results = DeepFace.find(
            img_path=img_path,
            db_path=db_path,
            model_name=model_name,
            detector_backend=detector_backend,
            distance_metric=distance_metric,
            enforce_detection=enforce_detection,
            align=align,
            anti_spoofing=anti_spoofing,
        )
        return {"results": results}
    except Exception as err:
        tb_str = traceback.format_exc()
        logger.error(str(err))
        logger.error(tb_str)
        return {"error": f"Exception while finding: {str(err)} - {tb_str}"}, 400








def search_pkl(
    img_path: Union[str, np.ndarray],
    pkl_path: str,
    model_name: str,
    detector_backend: str,
    distance_metric: str,
    enforce_detection: bool,
    align: bool,
    top_k: int = 5,
    threshold: float = 0.7,
):
    try:
        import faiss
        import pickle
        import numpy as np
        from deepface import DeepFace

        # Load the pickle file (should be a list of dicts)
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)

        # ✅ Robust parsing for list of dicts or list of lists
        embeddings = []
        identities = []

        for row in data:
            if isinstance(row, dict) and "embedding" in row and "identity" in row:
                embeddings.append(row["embedding"])
                identities.append(row["identity"])
            elif isinstance(row, (list, tuple)) and len(row) == 2:
                embeddings.append(row[0])
                identities.append(row[1])
            else:
                continue  # or raise an error if preferred

        embeddings = np.array(embeddings).astype("float32")

        # Build FAISS index
        dim = embeddings.shape[1]
        if distance_metric == "cosine":
            faiss.normalize_L2(embeddings)
            index = faiss.IndexFlatIP(dim)
        else:
            index = faiss.IndexFlatL2(dim)

        index.add(embeddings)

        # Represent input image
        target = DeepFace.represent(
            img_path=img_path,
            model_name=model_name,
            detector_backend=detector_backend,
            enforce_detection=enforce_detection,
            align=align,
        )[0]["embedding"]

        target = np.array(target).reshape(1, -1).astype("float32")
        if distance_metric == "cosine":
            faiss.normalize_L2(target)

        # Search
        D, I = index.search(target, top_k)

        # Format results with threshold
        results = []
        for rank, (idx, score) in enumerate(zip(I[0], D[0])):
            if distance_metric == "cosine":
                matched = score >= threshold
            else:
                matched = score <= threshold

            identity = identities[idx] if matched else "Unknown"

            results.append({
                "identity": identity,
                "score": float(score),
                "rank": rank + 1
            })

        return {"results": results}

    except Exception as err:
        import traceback
        tb_str = traceback.format_exc()
        return {
            "error": f"Exception during FAISS search: {str(err)}",
            "traceback": tb_str
        }, 400

