#include <stdio.h>
#include <iostream>
#include <iomanip>
#include <deque>
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/imgproc.hpp"

using namespace cv;
using namespace std;

void addImageToComposition(Mat& composedImage, Mat& image, const bool isGrayscale, const int quarter) {
    const auto width = composedImage.size().width;
    const auto height = composedImage.size().height;
    const auto halfWidth = width / 2;
    const auto halfHeight = height / 2;

    resize(image, image, Size(halfWidth, halfHeight));
    if (isGrayscale) {
        cvtColor(image, image, COLOR_GRAY2RGB);
    }

    Range rowsRange, colsRange;
    if (quarter == 1) {
        rowsRange = Range(0, halfHeight);
        colsRange = Range(halfWidth, width);
    } else if (quarter == 2) {
        rowsRange = Range(0, halfHeight);
        colsRange = Range(0, halfWidth);
    } else if (quarter == 3) {
        rowsRange = Range(halfHeight, height);
        colsRange = Range(0, halfWidth);
    } else if (quarter == 4) {
        rowsRange = Range(halfHeight, height);
        colsRange = Range(halfWidth, width);
    }

    image.copyTo(Mat(composedImage, rowsRange, colsRange));
}

void uniteRectangles(vector<Rect>& rectangles) {
    vector<char> useRectangles(rectangles.size(), 1);
    while (true) {
        auto wasChange = false;
        for (int i = 0; i < rectangles.size(); ++i) {
            if (!useRectangles[i]) {
                continue;
            }
            for (int j = i + 1; j < rectangles.size(); ++j) {
                if (useRectangles[j] && ((rectangles[i] & rectangles[j]).area() > 0)) {
                    rectangles[i] |= rectangles[j];
                    useRectangles[j] = false;
                    wasChange = true;
                }
            }
        }
        if (!wasChange) {
            break;
        }
    }
    vector<Rect> resultRectangles;
    for (int i = 0; i < rectangles.size(); ++i) {
        if (useRectangles[i]) {
            resultRectangles.push_back(rectangles[i]);
        }
    }
    rectangles = resultRectangles;
}

void processVideo() {
    VideoCapture capture(0);
    Mat colorFrame, frame, prevFrame;
    Ptr<Feature2D> detector = ORB::create();
    std::vector<KeyPoint> prevKeypoints, keypoints;
    Mat prevDescriptors, descriptors;
    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");

    const double firstTwoCloseMatchesDiff = 0.8;
    const double maxMatchHeightDiff = 20;
    const double maxMatchHorizontalDiff = 70;
    const int binaryThreshold = 70;
    Mat erodeElement = getStructuringElement(MORPH_RECT, Size(4, 4));
    Mat dilateElement = getStructuringElement(MORPH_RECT, Size(24, 24));

    while (true) {
        capture.read(colorFrame);
        resize(colorFrame, colorFrame, Size(colorFrame.size().width / 2, colorFrame.size().height / 2));
        frame = colorFrame.clone();
        cvtColor(frame, frame, CV_BGR2GRAY);

        if (prevFrame.rows == 0) {
            prevFrame = frame.clone();
        }

        detector->detectAndCompute(prevFrame, noArray(), prevKeypoints, prevDescriptors);
        detector->detectAndCompute(frame, noArray(), keypoints, descriptors);

        vector<vector<DMatch>> matches;
        matcher->knnMatch(prevDescriptors, descriptors, matches, 2);

        vector<Point2f> prevFilteredPoints, filteredPoints;
        for (int i = 0; i < matches.size(); ++i) {
            if (matches[i][0].distance > firstTwoCloseMatchesDiff * matches[i][1].distance) {
                continue;
            }
            const auto& prevPoint = prevKeypoints[matches[i][0].queryIdx].pt;
            const auto& point = keypoints[matches[i][0].trainIdx].pt;
            if (fabs(point.y - prevPoint.y) > maxMatchHeightDiff || fabs(point.x - prevPoint.x) > maxMatchHorizontalDiff) {
                continue;
            }
            prevFilteredPoints.push_back(prevPoint);
            filteredPoints.push_back(point);
        }

        Mat vectorsImage = colorFrame.clone();
        vector<double> horizontalDiffs;
        for (int i = 0; i < filteredPoints.size(); ++i) {
            arrowedLine(vectorsImage, prevFilteredPoints[i], filteredPoints[i], Scalar(0, 0, 255), 2);
            horizontalDiffs.push_back(filteredPoints[i].x - prevFilteredPoints[i].x);
        }
        vector<double> mean;
        vector<double> stdDev;
        meanStdDev(horizontalDiffs, mean, stdDev);
        cout << "mean: " << setprecision(3) << setw(4) << int(mean[0]) << " stdDev: " << stdDev[0] << endl;


        Mat mask(frame.size(), CV_8U, Scalar(255));
        if (filteredPoints.size() >= 5) {
            Mat homography = findHomography(prevFilteredPoints, filteredPoints, CV_RANSAC);
            warpPerspective(prevFrame, prevFrame, homography, prevFrame.size());
            warpPerspective(mask, mask, homography, mask.size());
        }

        Mat diff;
        subtract(frame, prevFrame, diff, mask);

        Mat movement;
        threshold(diff, movement, binaryThreshold, 255, THRESH_BINARY);
        erode(movement, movement, erodeElement);
        dilate(movement, movement, dilateElement, Point(-1, -1), 2);

        Moments movementMoments = moments(movement, true);
        circle(
                colorFrame,
                Point(movementMoments.m10 / movementMoments.m00, movementMoments.m01 / movementMoments.m00),
                10,
                Scalar(0, 255, 0),
                -1
        );

        Mat contoursImage = movement.clone();

        vector<vector<Point>> contours;
        vector<Vec4i> hierarchy;
        findContours(contoursImage, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);

        vector<Rect> rectangles;
        contoursImage = Mat::zeros(contoursImage.size(), CV_8UC3);
        for(int i = 0; i < contours.size(); ++i) {
            drawContours(contoursImage, contours, i, Scalar(255, 0, 0), 2, 8, hierarchy, 0);
            const auto rect = boundingRect(contours[i]);
            rectangles.push_back(rect);
        }

        uniteRectangles(rectangles);

        for (const auto& rect : rectangles) {
            rectangle(colorFrame, rect, Scalar(0, 255, 0), 3);
        }

        Mat composedImage(colorFrame.rows * 2, colorFrame.cols * 2, colorFrame.type(), Scalar(0, 0, 0));

        addImageToComposition(composedImage, vectorsImage, false, 2);
        addImageToComposition(composedImage, diff, true, 1);
        addImageToComposition(composedImage, movement, true, 3);
        addImageToComposition(composedImage, colorFrame, false, 4);

        imshow("Scene", composedImage);

        prevFrame = frame;

        int c = waitKey(30);
        if (c == 27)
            break;
    }
    capture.release();
}

int main( int argc, char** argv ) {
    processVideo();
    destroyAllWindows();

    return EXIT_SUCCESS;
}
