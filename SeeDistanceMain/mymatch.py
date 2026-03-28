import cv2 as cv
import pandas as pd


class FeatureMatcher:
    """Feature matching pipeline for image pair comparison."""

    def __init__(self) -> None:
        """Initialize the feature matcher."""
        self.results = None

    def _edge_preprocess(self, image):
        """Convert an image to an edge map."""
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        edges = cv.Canny(gray, threshold1=50, threshold2=150)
        return edges

    def _features_orb(self, edges):
        """Extract ORB keypoints and descriptors."""
        orb = cv.ORB_create()
        keypoints, descriptors = orb.detectAndCompute(edges, None)
        return keypoints, descriptors

    def _features_sift(self, edges):
        """Extract SIFT keypoints and descriptors."""
        sift = cv.SIFT_create()
        keypoints, descriptors = sift.detectAndCompute(edges, None)
        return keypoints, descriptors

    def _features_kaze(self, edges):
        """Extract KAZE keypoints and descriptors."""
        kaze = cv.KAZE_create()
        keypoints, descriptors = kaze.detectAndCompute(edges, None)
        return keypoints, descriptors

    def _match(self, descriptors1, descriptors2):
        """Match descriptors using brute-force matching."""
        bf = cv.BFMatcher(cv.NORM_L2, crossCheck=True)
        matches = bf.match(descriptors1, descriptors2)
        return sorted(matches, key=lambda x: x.distance)

    def _visualize_images_matched_features(self, image1, image2, keypoints1, keypoints2, matches):
        """Display matched features in a window."""
        matched_image = cv.drawMatches(
            image1,
            keypoints1,
            image2,
            keypoints2,
            matches[:50],
            None,
            flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
        )
        cv.namedWindow("Matched Features", cv.WINDOW_NORMAL)
        cv.imshow("Matched Features", matched_image)
        cv.waitKey(0)
        cv.destroyAllWindows()

    def _save_images_matched_features(self, image1, image2, keypoints1, keypoints2, matches, name="matchedFeatures.jpg"):
        """Save matched features visualization to disk."""
        matched_image = cv.drawMatches(
            image1,
            keypoints1,
            image2,
            keypoints2,
            matches[:50],
            None,
            flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
        )
        cv.imwrite(name, matched_image)

    def _matches_to_df(self, matches, keypoints1, keypoints2):
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

    def _run_orb(self, image1, image2):
        """Run the ORB matching pipeline."""
        edges1 = self._edge_preprocess(image1)
        edges2 = self._edge_preprocess(image2)

        keypoints1, descriptors1 = self._features_orb(edges1)
        keypoints2, descriptors2 = self._features_orb(edges2)

        matches = self._match(descriptors1, descriptors2)
        return self._matches_to_df(matches, keypoints1, keypoints2)

    def _run_sift(self, image1, image2):
        """Run the SIFT matching pipeline."""
        edges1 = self._edge_preprocess(image1)
        edges2 = self._edge_preprocess(image2)

        keypoints1, descriptors1 = self._features_sift(edges1)
        keypoints2, descriptors2 = self._features_sift(edges2)

        matches = self._match(descriptors1, descriptors2)
        return self._matches_to_df(matches, keypoints1, keypoints2)

    def _run_kaze(self, image1, image2):
        """Run the KAZE matching pipeline."""
        edges1 = self._edge_preprocess(image1)
        edges2 = self._edge_preprocess(image2)

        keypoints1, descriptors1 = self._features_kaze(edges1)
        keypoints2, descriptors2 = self._features_kaze(edges2)

        matches = self._match(descriptors1, descriptors2)
        return self._matches_to_df(matches, keypoints1, keypoints2)

    def run(self, model: str, image1, image2):
        """Run the selected matching model.

        Args:
            model: Name of the feature model to use. Supported values are
                "orb", "sift", and "kaze".
            image1: First input image.
            image2: Second input image.

        Returns:
            A pandas DataFrame with matched feature coordinates.
        """
        model = model.lower()

        if model == "orb":
            self.results = self._run_orb(image1, image2)
        elif model == "sift":
            self.results = self._run_sift(image1, image2)
        elif model == "kaze":
            self.results = self._run_kaze(image1, image2)
        else:
            raise ValueError(f"Unsupported model: {model}")

        return self.results