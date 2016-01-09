#include <stdio.h>
#include <iostream>
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/imgproc.hpp"

using namespace cv;
using namespace std;

void processVideo() {
    VideoCapture capture(0);
    Mat frame, prevFrame;
    Ptr<Feature2D> detector = ORB::create();
    std::vector<KeyPoint> prevKeypoints, keypoints;
    Mat prevDescriptors, descriptors;
    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");
    while(true) {
        capture.read(frame);
        resize(frame, frame, Size(frame.size().width / 2, frame.size().height / 2));
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
            if (matches[i][0].distance < 0.8 * matches[i][1].distance) {
                prevFilteredPoints.push_back(prevKeypoints[matches[i][0].queryIdx].pt);
                filteredPoints.push_back(keypoints[matches[i][0].trainIdx].pt);
            }
        }

        Mat mask(frame.size(), CV_8U, Scalar(255));
        if (filteredPoints.size() >= 4) {
            Mat homography = findHomography(prevFilteredPoints, filteredPoints, CV_RANSAC);
            warpPerspective(prevFrame, prevFrame, homography, prevFrame.size());
            warpPerspective(mask, mask, homography, mask.size());
        }

        Mat diff;
        subtract(frame, prevFrame, diff, mask);

        imshow("Original", frame);
        imshow("Diff", diff);

        prevFrame = frame;

        int c = waitKey(30);
        if( c == 27 || c == 'q' || c == 'Q' )
            break;
    }
    capture.release();
}

int main( int argc, char** argv ) {
    processVideo();
    destroyAllWindows();

    return EXIT_SUCCESS;
}
