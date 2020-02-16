#include <iostream>
#include <numeric>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "structIO.hpp"

using namespace std;

void showLidarTopview()
{
    std::vector<LidarPoint> lidarPoints;
    readLidarPts("../dat/C51_LidarPts_0000.dat", lidarPoints);

    // Width and height (driving direction) of sensor field in [m]
    cv::Size worldSize(10.0, 20.0);
    // Corresponding width and height of the top view image in [pixel]
    cv::Size imageSize(1000, 2000);

    // Create topview image with black color by default
    cv::Mat topviewImg(imageSize, CV_8UC3, cv::Scalar(0, 0, 0)); 

    // Plot Lidar points into image
    for (auto it = lidarPoints.begin(); it != lidarPoints.end(); ++it)
    {
        // Sensor position in [meters]
        float x_world = it->x; // world position in m with x facing forward from sensor
        float y_world = it->y; // world position in m with y facing left from sensor

        // Image in the top view coordinate system
        int y = (-x_world * imageSize.height / worldSize.height) + imageSize.height;
        int x = (-y_world * imageSize.height / worldSize.height) + imageSize.width / 2;
        
        // TODO: 
        // 1. Change the color of the Lidar points such that 
        // x_world = 0.0m corresponds to red while x_world = 20.0m is shown as green

        // Only consider the Lidar points above ground
        float z_world = it->z; // height above the ground
        double minZ = -1.40; // [m]
        if (z_world > minZ)
        {
            // Current Lidar point in x-direction [pixels]
            float val = it->x;
            // Max Lidar detection range in x-direction [pixels]
            float maxVal = worldSize.height; 
            // Scale the red to green based on the max Lidar range
            int red   = min(255, (int)(255 * abs((maxVal - val/ maxVal))));
            int green = min(255, (int)(255 * (1 - abs((maxVal - val)/ maxVal))));
            // Overlay the Lidar points on blank image
            cv::circle(topviewImg, cv::Point(x, y), 5, cv::Scalar(0, green, red), -1);
        }

        // 2. Remove all Lidar points on the road surface while preserving 
        // measurements on the obstacles in the scene.
    }

    // Plot distance markers
    float lineSpacing = 2.0; // gap between distance markers in [m]
    int nMarkers = floor(worldSize.height / lineSpacing);
    // Draw horziontal lines for every line space
    for (size_t i = 0; i < nMarkers; ++i)
    {
        // Paint a blue line that runs horizontally across the top of the image
        int y = (-(i * lineSpacing) * imageSize.height / worldSize.height) + imageSize.height;
        cv::line(topviewImg, cv::Point(0, y), cv::Point(imageSize.width, y), cv::Scalar(255, 0, 0));
    }

    // Display image
    string windowName = "Top-View Perspective of LiDAR data";
    cv::namedWindow(windowName, 2);
    cv::imshow(windowName, topviewImg);
    cv::waitKey(0); // wait for key to be pressed
}

int main()
{
    showLidarTopview();
}