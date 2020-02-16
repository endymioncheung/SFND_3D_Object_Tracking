#include <iostream>
#include <numeric>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "structIO.hpp"
#include "dataStructures.h"

using namespace std;

void loadCalibrationData(cv::Mat &P_rect_00, cv::Mat &R_rect_00, cv::Mat &RT)
{
    RT.at<double>(0,0) = 7.533745e-03; RT.at<double>(0,1) = -9.999714e-01; RT.at<double>(0,2) = -6.166020e-04; RT.at<double>(0,3) = -4.069766e-03;
    RT.at<double>(1,0) = 1.480249e-02; RT.at<double>(1,1) = 7.280733e-04; RT.at<double>(1,2) = -9.998902e-01; RT.at<double>(1,3) = -7.631618e-02;
    RT.at<double>(2,0) = 9.998621e-01; RT.at<double>(2,1) = 7.523790e-03; RT.at<double>(2,2) = 1.480755e-02; RT.at<double>(2,3) = -2.717806e-01;
    RT.at<double>(3,0) = 0.0; RT.at<double>(3,1) = 0.0; RT.at<double>(3,2) = 0.0; RT.at<double>(3,3) = 1.0;
    
    R_rect_00.at<double>(0,0) = 9.999239e-01; R_rect_00.at<double>(0,1) = 9.837760e-03; R_rect_00.at<double>(0,2) = -7.445048e-03; R_rect_00.at<double>(0,3) = 0.0;
    R_rect_00.at<double>(1,0) = -9.869795e-03; R_rect_00.at<double>(1,1) = 9.999421e-01; R_rect_00.at<double>(1,2) = -4.278459e-03; R_rect_00.at<double>(1,3) = 0.0;
    R_rect_00.at<double>(2,0) = 7.402527e-03; R_rect_00.at<double>(2,1) = 4.351614e-03; R_rect_00.at<double>(2,2) = 9.999631e-01; R_rect_00.at<double>(2,3) = 0.0;
    R_rect_00.at<double>(3,0) = 0; R_rect_00.at<double>(3,1) = 0; R_rect_00.at<double>(3,2) = 0; R_rect_00.at<double>(3,3) = 1;
    
    P_rect_00.at<double>(0,0) = 7.215377e+02; P_rect_00.at<double>(0,1) = 0.000000e+00; P_rect_00.at<double>(0,2) = 6.095593e+02; P_rect_00.at<double>(0,3) = 0.000000e+00;
    P_rect_00.at<double>(1,0) = 0.000000e+00; P_rect_00.at<double>(1,1) = 7.215377e+02; P_rect_00.at<double>(1,2) = 1.728540e+02; P_rect_00.at<double>(1,3) = 0.000000e+00;
    P_rect_00.at<double>(2,0) = 0.000000e+00; P_rect_00.at<double>(2,1) = 0.000000e+00; P_rect_00.at<double>(2,2) = 1.000000e+00; P_rect_00.at<double>(2,3) = 0.000000e+00;

}

void showLidarTopview(std::vector<LidarPoint> &lidarPoints, cv::Size worldSize, cv::Size imageSize)
{
    // Create top view image (black background)
    const cv::Scalar black = cv::Scalar(0, 0, 0);
    cv::Mat topviewImg(imageSize, CV_8UC3, black);

    // Plot Lidar points into image
    for (auto it = lidarPoints.begin(); it != lidarPoints.end(); ++it)
    {
        // World coordinates
        float x_world = (*it).x; // world position in m with x facing forward from sensor
        float y_world = (*it).y; // world position in m with y facing left from sensor

        // Top view coordinates
        int y = (-x_world * imageSize.height / worldSize.height) + imageSize.height;
        int x = (-y_world * imageSize.height / worldSize.height) + imageSize.width / 2;

        float zw = (*it).z; // world position in m with y facing left from sensor
        if(zw > -1.40){       

            float val = it->x;
            float maxVal = worldSize.height;
            int red = min(255, (int)(255 * abs((val - maxVal) / maxVal)));
            int green = min(255, (int)(255 * (1 - abs((val - maxVal) / maxVal))));
            cv::circle(topviewImg, cv::Point(x, y), 5, cv::Scalar(0, green, red), -1);
        }
    }

    // Plot distance markers
    float lineSpacing = 2.0; // gap between distance markers
    int nMarkers = floor(worldSize.height / lineSpacing);
    const cv::Scalar blue = cv::Scalar(255, 0, 0);
    for (size_t i = 0; i < nMarkers; ++i)
    {
        int y = (-(i * lineSpacing) * imageSize.height / worldSize.height) + imageSize.height;
        cv::line(topviewImg, cv::Point(0, y), cv::Point(imageSize.width, y), blue);
    }

    // Display image
    string windowName = "Top-View Perspective of LiDAR data";
    cv::namedWindow(windowName, 2);
    cv::imshow(windowName, topviewImg);
    cv::waitKey(0); // wait for key to be pressed
}

void show3DObjects(std::vector<BoundingBox> &boundingBoxes, cv::Size worldSize, cv::Size imageSize, bool bWait)
{
    // Create top view image (white background)
    const cv::Scalar white = cv::Scalar(255, 255, 255);
    cv::Mat topviewImg(imageSize, CV_8UC3, white);

    // Overlay bounding boxes onto image
    for (auto it1 = boundingBoxes.begin(); it1 != boundingBoxes.end(); ++it1)
    {
        // Create randomized color for current 3D object
        int num_unique_ID = 150;
        cv::RNG rng(it1->boxID);
        cv::Scalar currColor = cv::Scalar(rng.uniform(0,num_unique_ID), rng.uniform(0,num_unique_ID), rng.uniform(0,num_unique_ID));

        // Plot Lidar points into top view image
        int top = 1e8, bottom = 0.0;
        int left = 1e8, right = 0.0;
        float x_world_min = 1e8, y_world_min = 1e8, y_world_max = -1e8;

        for (auto it2 = it1->lidarPoints.begin(); it2 != it1->lidarPoints.end(); ++it2)
        {
            // World coordinates
            float x_world = (*it2).x; // world position in m with x facing forward from sensor
            float y_world = (*it2).y; // world position in m with y facing left from sensor
            
            // Find the real-world coordinate of the closest x direction and the width of the object in [m]
            x_world_min = x_world_min < x_world ? x_world_min : x_world;
            y_world_min = y_world_min < y_world ? y_world_min : y_world;
            y_world_max = y_world_max > y_world ? y_world_max : y_world;

            // Top view coordinates in [pixels]
            int y = (-x_world * imageSize.height / worldSize.height) + imageSize.height;
            int x = (-y_world * imageSize.height / worldSize.height) + imageSize.width / 2;

            // Find enclosing rectangle
            top     = y > top ? top : y;
            bottom  = y < bottom ? bottom: y;
            left    = x > left ? left : x;
            // left    = x < left ? x: left;
            right   = x < right ? right : x;
            // right   = x > right ? x : right;

            cv::circle(topviewImg, cv::Point(x, y), 4, currColor, -1);
            
        }

        // Draw enclosing rectangle
        cv::Scalar black = cv::Scalar(0, 0, 0);
        cv::rectangle(topviewImg, cv::Point(left,top), cv::Point(right,bottom), black,2);

        // Augment object with some key data
        char str1[200], str2[200];
        sprintf(str1, "id=%d, #pts=%d", it1->boxID, (int)it1->lidarPoints.size());
        putText(topviewImg, str1, cv::Point2f(left-250, bottom+50), cv::FONT_ITALIC, 2, currColor);

        sprintf(str2, "xmin=%2.2f m, yw=%2.2f m", x_world_min, y_world_max- y_world_min);
        putText(topviewImg, str2, cv::Point2f(left-250, bottom+50+75), cv::FONT_ITALIC, 2, currColor);
    }

    // Plot distance markers
    float lineSpacing = 2.0; // gap between distance markers
    int nMarkers = floor(worldSize.height / lineSpacing);
    const cv::Scalar blue = cv::Scalar(255, 0, 0);
    for (size_t i = 0; i < nMarkers; ++i)
    {
        int y = (-(i * lineSpacing) * imageSize.height / worldSize.height) + imageSize.height;
        cv::line(topviewImg, cv::Point(0, y), cv::Point(imageSize.width, y), blue);
    }

    // Display image
    string windowName = "3D Objects";
    cv::namedWindow(windowName, 1);
    cv::imshow(windowName, topviewImg);
    
    if(bWait)
        cv::waitKey(0); // wait for key to be pressed
}

void clusterLidarWithROI(std::vector<BoundingBox> &boundingBoxes, std::vector<LidarPoint> &lidarPoints)
{
    // Store calibration data in OpenCV matrices
    cv::Mat P_rect_xx(3,4,cv::DataType<double>::type); // 3x4 projection matrix after rectification
    cv::Mat R_rect_xx(4,4,cv::DataType<double>::type); // 3x3 rectifying rotation to make image planes co-planar
    cv::Mat RT(4,4,cv::DataType<double>::type); // rotation matrix and translation vector
    loadCalibrationData(P_rect_xx, R_rect_xx, RT);

    // Loop over all Lidar points and associate them to a 2D bounding box
    cv::Mat X(4, 1, cv::DataType<double>::type);
    cv::Mat Y(3, 1, cv::DataType<double>::type);

    for (auto it1 = lidarPoints.begin(); it1 != lidarPoints.end(); ++it1)
    {
        // Assemble vector for matrix-vector-multiplication
        X.at<double>(0, 0) = it1->x;
        X.at<double>(1, 0) = it1->y;
        X.at<double>(2, 0) = it1->z;
        X.at<double>(3, 0) = 1;

        // Project Lidar point into camera
        Y = P_rect_xx * R_rect_xx * RT * X;
        cv::Point pt;
        // Convert homogeneous coordinates to pixel coordinates
        pt.x = Y.at<double>(0, 0) / Y.at<double>(0, 2);
        pt.y = Y.at<double>(1, 0) / Y.at<double>(0, 2);

        double shrinkFactor = 0.10;
        vector<vector<BoundingBox>::iterator> enclosingBoxes; // Pointers to all bounding boxes which enclose the current Lidar point
        for (vector<BoundingBox>::iterator it2 = boundingBoxes.begin(); it2 != boundingBoxes.end(); ++it2)
        {
            // Shrink current bounding box slightly to avoid having too many outlier points around the edges
            cv::Rect smallerBox;
            smallerBox.x = (*it2).roi.x + shrinkFactor * (*it2).roi.width / 2.0;
            smallerBox.y = (*it2).roi.y + shrinkFactor * (*it2).roi.height / 2.0;
            smallerBox.width = (*it2).roi.width * (1 - shrinkFactor);
            smallerBox.height = (*it2).roi.height * (1 - shrinkFactor);

            // Check whether point is within current bounding box
            if (smallerBox.contains(pt))
            {
                // Add enclosing box to vector
                enclosingBoxes.push_back(it2);
            }
        } // EOF loop over all bounding boxes

        // Check whether point has been enclosed by one or multiple boxes
        if (enclosingBoxes.size() == 1)
        {
            // Add Lidar point to bounding box
            enclosingBoxes[0]->lidarPoints.push_back(*it1);
        }

    } // EOF loop over all Lidar points
}

void runClustering()
{
    // Read Lidar points and bounding boxes
    std::vector<LidarPoint> lidarPoints;
    const char*  path_lidarPoints    = "../dat/C53A3_currLidarPts.dat";
    readLidarPts(path_lidarPoints, lidarPoints);

    std::vector<BoundingBox> boundingBoxes;
    const char*  path_boundingBoxes  = "../dat/C53A3_currBoundingBoxes.dat";
    readBoundingBoxes(path_boundingBoxes, boundingBoxes);

    // Group Lidar points to bounding boxes
    clusterLidarWithROI(boundingBoxes, lidarPoints);

    // World size in [m] and image size in [px]
    const cv::Size worldSize = cv::Size(10.0, 25.0);
    const cv::Size imageSize = cv::Size(1000, 2000);

    // Display the isolated 3D objects in top view prospective
    // for (auto it = boundingBoxes.begin(); it != boundingBoxes.end(); ++it)
    // {
    //     if (it->lidarPoints.size() > 0)
    //     {
    //         // Show Lidar top view
    //         showLidarTopview(it->lidarPoints, worldSize, imageSize);
    //     }
    // }

    // Show 3D objects
    const bool bWait = true;
    show3DObjects(boundingBoxes,worldSize,imageSize,bWait);
}

int main()
{
    runClustering();
    return 0;
}