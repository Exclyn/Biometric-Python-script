import cv2
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from enhance import image_enhance
from skimage.morphology import skeletonize, thin

os.chdir("app\\database")

def removedot(invertThin):
    temp0 = np.array(invertThin[:])
    temp0 = np.array(temp0)
    temp1 = temp0/255
    temp2 = np.array(temp1)
    temp3 = np.array(temp2)

    enhanced_img = np.array(temp0)
    filter0 = np.zeros((10,10))
    W,H = temp0.shape[:2]
    filtersize = 6

    for i in range(W - filtersize):
        for j in range(H - filtersize):
            filter0 = temp1[i:i + filtersize,j:j + filtersize]

            flag = 0
            if sum(filter0[:,0]) == 0:
                flag += 1
            if sum(filter0[:,filtersize - 1]) == 0:
                flag += 1
            if sum(filter0[0,:]) == 0:
                flag += 1
            if sum(filter0[filtersize - 1,:]) == 0:
                flag += 1
            if flag > 3:
                temp2[i:i + filtersize, j:j + filtersize] = np.zeros((filtersize, filtersize))

    return temp2

def get_descriptors(img):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img = clahe.apply(img)
    img = image_enhance.image_enhance(img)
    img = np.array(img, dtype=np.uint8)
    # Threshold
    ret, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    # Normalize to 0 and 1 range
    img[img == 255] = 1
    # Thinning
    skeleton = skeletonize(img)
    skeleton = np.array(skeleton, dtype=np.uint8)
    skeleton = removedot(skeleton)
    # Harris corners
    harris_corners = cv2.cornerHarris(img, 3, 3, 0.04)
    harris_normalized = cv2.normalize(harris_corners, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32FC1)
    threshold_harris = 125
    # Extract keypoints
    keypoints = []
    for x in range(0, harris_normalized.shape[0]):
        for y in range(0, harris_normalized.shape[1]):
            if harris_normalized[x][y] > threshold_harris:  
                keypoints.append(cv2.KeyPoint(y, x, 1)) 
    # SIFT MORE ACCURATE ALOGORITHM BUT SLOWERṆ
    sift = cv2.SIFT_create()
    _, des_sift = sift.compute(img, keypoints)
    return keypoints, des_sift
    # orb = cv2.ORB_create()
    # _,des_orb = orb.compute(img, keypoints)
    # return keypoints, des_orb
    
def init():
    image_name = get_descriptors()
    

def main():
    image_name = sys.argv[1] 
    img1 = cv2.imread(image_name, cv2.IMREAD_GRAYSCALE)
    kp1, des1 = get_descriptors(img1)

    # Iterate through all fingerprint images in the database directory
    database_dir = os.getcwd()
    database_images = os.listdir(database_dir)
    found_match = False
    match_image = None
    best_match_score = 0
    for database_image in database_images:
        database_path = os.path.join(database_dir, database_image)
        db_img = cv2.imread(database_path, cv2.IMREAD_GRAYSCALE)
        db_kp, db_des = get_descriptors(db_img)
        ######

        # Matching between descriptors using Brute-Force Matcher
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        matches = sorted(bf.match(des1, db_des), key=lambda match: match.distance)

        top_n_matches = 10  # Adjust this according to your needs
        matches = matches [:top_n_matches]

        flann = cv2.FlannBasedMatcher_create()
        matches_flann = flann.knnMatch(des1, db_des, k=2)

        good_matches = []
        for m, n in matches_flann:
            if m.distance < 0.7 * n.distance:
                 good_matches.append(m)

        all_matches = matches + good_matches
        all_matches = sorted(all_matches, key=lambda match: match.distance)

        # Calculate the score and perform fingerprint matching based on the combined matches
        score_threshold = 60
        score = len(all_matches)

        if score > best_match_score:
            best_match_score = score
            match_image = database_image
        
        if score >= score_threshold:
            found_match = True
            
    
    if found_match:
        print("Fingerprint matches with", match_image)
            
            
    else :
        print("Fingerprint does not match.")


if __name__ == "__main__":
    try:
        main()
    except IndexError:
        print("No image name provided as a command-line argument.")
    except AttributeError:
        raise