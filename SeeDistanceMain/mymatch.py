import cv2 as cv
import pandas as pd


def edge_preprocess(image):
    """

    :param image:
    :return:
    """
    # Convert to grayscale
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    edges = cv.Canny(gray, threshold1=50, threshold2=150)

    return edges

def features_ORB(edges):
    """

    :param edges:
    :return:
    """
    # ORB feature detector
    orb = cv.ORB_create()

    # Detect keypoints and descriptors
    keypoints, descriptors = orb.detectAndCompute(edges, None)
    return keypoints, descriptors

def features_SIFT(edges):
    # Create SIFT detector
    sift = cv.SIFT_create()

    # Detect keypoints and descriptors
    keypoints, descriptors = sift.detectAndCompute(edges, None)
    return keypoints, descriptors

def features_KAZE(edges):
    kaze = cv.KAZE_create()

    # Detect keypoints and descriptors
    keypoints, descriptors = kaze.detectAndCompute(edges, None)
    return keypoints, descriptors

def match(descriptors1, descriptors2):
    # Match features using BFMatcher
    bf = cv.BFMatcher(cv.NORM_L2, crossCheck=True)
    matches = bf.match(descriptors1, descriptors2)

    # Sort matches by distance (lower is better)
    matches = sorted(matches, key=lambda x: x.distance)
    return matches


def visualize_images_matched_features(image1, image2, keypoints1, keypoints2, matches):
    # Draw the top matches
    matched_image = cv.drawMatches(image1, keypoints1, image2, keypoints2, matches[:50], None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    # Display the result
    cv.namedWindow('Matched Features', cv.WINDOW_NORMAL)
    cv.imshow('Matched Features', matched_image)
    cv.waitKey(0)
    cv.destroyAllWindows()

def save_images_matched_features(image1, image2, keypoints1, keypoints2, matches, name='matchedFeatures.jpg'):
    # Draw the top matches
    matched_image = cv.drawMatches(image1, keypoints1, image2, keypoints2, matches[:50], None,
                                   flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    # Display the result
    cv.imwrite(name, matched_image)

def matches_to_df(matches, keypoints1, keypoints2):
    # Extract matched feature coordinates
    matched_features = []
    for match in matches:
        # Query and train indices
        idx1 = match.queryIdx
        idx2 = match.trainIdx

        # Get keypoint coordinates
        x1, y1 = keypoints1[idx1].pt  # Coordinates in image1
        x2, y2 = keypoints2[idx2].pt  # Coordinates in image2

        # Append to list
        matched_features.append({
            "Image1_X": x1,
            "Image1_Y": y1,
            "Image2_X": x2,
            "Image2_Y": y2,
            "Distance": match.distance
        })

    # Convert to Pandas DataFrame
    df = pd.DataFrame(matched_features)
    return df