# Fingerprint Matching System

This project provides a fingerprint matching system using computer vision techniques. It takes a sample fingerprint image as input and compares it against a database of existing fingerprints to find a match. The matching process is based on extracting and comparing minutiae points from the fingerprints.

---

## Table of Contents
* [Workflow](#workflow)
* [Features](#features)
* [Dependencies](#dependencies)
* [Setup & Installation](#setup--installation)
* [Directory Structure](#directory-structure)
* [How to Run](#how-to-run)
* [Code Explanation](#code-explanation)

---

## Workflow

The fingerprint matching process follows these key steps:

1.  **Image Enhancement**: The input fingerprint image is enhanced using **CLAHE** (Contrast Limited Adaptive Histogram Equalization) and a custom enhancement function to improve the clarity of the ridges and valleys.
2.  **Binarization & Thinning**: The enhanced image is converted to a binary format using **Otsu's thresholding**. It is then thinned using **skeletonization** to create a single-pixel-wide representation of the fingerprint ridges.
3.  **Feature Extraction**:
    * **Harris Corner Detection** is used to identify key minutiae points (corners and bifurcations) on the thinned fingerprint image.
    * **SIFT (Scale-Invariant Feature Transform)** is then used to create a descriptor for each keypoint, capturing the local features around it.
4.  **Matching**:
    * The SIFT descriptors of the input image are compared against the descriptors of every image in the database.
    * A combination of **Brute-Force (BF) Matcher** and **FLANN-based Matcher** is used to find robust matches.
    * A **ratio test** is applied to the FLANN matches to filter out ambiguous matches.
5.  **Scoring & Decision**:
    * A score is calculated based on the number of good matches found between the input image and a database image.
    * If the score exceeds a predefined threshold (`60`), the system declares a match and identifies the corresponding image from the database.

---

## Features

* **Image Enhancement**: Uses CLAHE and custom filters for better feature extraction.
* **Fingerprint Thinning**: Implements skeletonization to reduce ridge thickness to a single pixel.
* **Minutiae Detection**: Employs Harris Corner Detection to locate key feature points.
* **Robust Descriptors**: Uses the SIFT algorithm for creating scale and rotation-invariant feature descriptors.
* **Hybrid Matching**: Combines Brute-Force and FLANN matchers for improved accuracy.

---

## Dependencies

The script requires the following Python libraries:

* **OpenCV**: `opencv-python`
* **scikit-image**: `scikit-image`
* **NumPy**: `numpy`
* **Matplotlib**: `matplotlib`

It also depends on a local module named `enhance.py`, which should be in the same directory.

---

## Setup & Installation

1.  **Clone the repository**:
    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```
2.  **Create a virtual environment**:
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows, use `.venv\Scripts\activate`
    ```
3.  **Install the required packages**:
    ```bash
    pip install opencv-python scikit-image numpy matplotlib
    ```

---

## How to Run

The script is intended to be run from the command line, with the path to the input fingerprint image provided as an argument.

1.  Place the fingerprint images you want in your database inside the `app/database/` directory.
2.  Navigate to the project's root directory in your terminal.
3.  Run the script with the following command:

    ```bash
    python app.py path/to/your/input_fingerprint.jpg
    ```
--
