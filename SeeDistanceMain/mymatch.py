import cv2 as cv
import pandas as pd


def _features_orb(edges):
    """Extract ORB keypoints and descriptors."""
    orb = cv.ORB_create()
    keypoints, descriptors = orb.detectAndCompute(edges, None)
    return keypoints, descriptors


def _features_sift(edges):
    """Extract SIFT keypoints and descriptors."""
    sift = cv.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(edges, None)
    return keypoints, descriptors


def _features_kaze(edges):
    """Extract KAZE keypoints and descriptors."""
    kaze = cv.KAZE_create()
    keypoints, descriptors = kaze.detectAndCompute(edges, None)
    return keypoints, descriptors


def _match(descriptors1, descriptors2):
    """Match descriptors using brute-force matching."""
    bf = cv.BFMatcher(cv.NORM_L2, crossCheck=True)
    matches = bf.match(descriptors1, descriptors2)
    return sorted(matches, key=lambda x: x.distance)


def _edge_preprocess(image):
    """Convert an image to an edge map."""
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    edges = cv.Canny(gray, threshold1=50, threshold2=150)
    return edges


def _matches_to_df(matches, keypoints1, keypoints2):
    """Convert matched keypoints into a pandas DataFrame."""
    matched_features = []
    for match in matches:
        idx1 = match.queryIdx
        idx2 = match.trainIdx

        x1, y1 = keypoints1[idx1].pt
        x2, y2 = keypoints2[idx2].pt

        matched_features.append(
            {
                "Image1_X": x1,
                "Image1_Y": y1,
                "Image2_X": x2,
                "Image2_Y": y2,
                "Distance": match.distance,
            }
        )

    return pd.DataFrame(matched_features)


def _get_feature_extractor(model: str):
    """Return the feature extractor method for the selected model."""
    model = model.lower()

    if model == "orb":
        return _features_orb
    if model == "sift":
        return _features_sift
    if model == "kaze":
        return _features_kaze

    raise ValueError(f"Unsupported model: {model}")


class FeatureMatcher:
    """Feature matching pipeline for image pair comparison."""

    def __init__(self, model: str) -> None:
        """Initialize the feature matcher.
        model: Name of the feature model to use.
            Supported values are "orb", "sift", and "kaze"."""
        self.results = None
        self.extractor = _get_feature_extractor(model)

    def run(self, image1, image2):
        """Run the selected matching model.
        Args:
            image1: First input image.
            image2: Second input image.
        Returns:
            A pandas DataFrame with matched feature coordinates.
        """
        edges1 = _edge_preprocess(image1)
        edges2 = _edge_preprocess(image2)

        keypoints1, descriptors1 = self.extractor(edges1)
        keypoints2, descriptors2 = self.extractor(edges2)

        matches = _match(descriptors1, descriptors2)
        self.results = _matches_to_df(matches, keypoints1, keypoints2)
        return self.results