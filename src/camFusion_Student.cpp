
#include <iostream>
#include <algorithm>
#include <numeric>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "camFusion.hpp"
#include "dataStructures.h"

using namespace std;


// Create groups of Lidar points whose projection into the camera falls into the same bounding box
void clusterLidarWithROI(std::vector<BoundingBox> &boundingBoxes, std::vector<LidarPoint> &lidarPoints, float shrinkFactor, cv::Mat &P_rect_xx, cv::Mat &R_rect_xx, cv::Mat &RT)
{
    // loop over all Lidar points and associate them to a 2D bounding box
    cv::Mat X(4, 1, cv::DataType<double>::type);
    cv::Mat Y(3, 1, cv::DataType<double>::type);

    for (auto it1 = lidarPoints.begin(); it1 != lidarPoints.end(); ++it1)
    {
        // assemble vector for matrix-vector-multiplication
        X.at<double>(0, 0) = it1->x;
        X.at<double>(1, 0) = it1->y;
        X.at<double>(2, 0) = it1->z;
        X.at<double>(3, 0) = 1;

        // project Lidar point into camera
        Y = P_rect_xx * R_rect_xx * RT * X;
        cv::Point pt;
        pt.x = Y.at<double>(0, 0) / Y.at<double>(0, 2); // pixel coordinates
        pt.y = Y.at<double>(1, 0) / Y.at<double>(0, 2);

        vector<vector<BoundingBox>::iterator> enclosingBoxes; // pointers to all bounding boxes which enclose the current Lidar point
        for (vector<BoundingBox>::iterator it2 = boundingBoxes.begin(); it2 != boundingBoxes.end(); ++it2)
        {
            // shrink current bounding box slightly to avoid having too many outlier points around the edges
            cv::Rect smallerBox;
            smallerBox.x = (*it2).roi.x + shrinkFactor * (*it2).roi.width / 2.0;
            smallerBox.y = (*it2).roi.y + shrinkFactor * (*it2).roi.height / 2.0;
            smallerBox.width = (*it2).roi.width * (1 - shrinkFactor);
            smallerBox.height = (*it2).roi.height * (1 - shrinkFactor);

            // check wether point is within current bounding box
            if (smallerBox.contains(pt))
            {
                enclosingBoxes.push_back(it2);
            }

        } // eof loop over all bounding boxes

        // check wether point has been enclosed by one or by multiple boxes
        if (enclosingBoxes.size() == 1)
        { 
            // add Lidar point to bounding box
            enclosingBoxes[0]->lidarPoints.push_back(*it1);
        }

    } // eof loop over all Lidar points
}


void show3DObjects(std::vector<BoundingBox> &boundingBoxes, cv::Size worldSize, cv::Size imageSize, bool bWait)
{
    // create topview image
    cv::Mat topviewImg(imageSize, CV_8UC3, cv::Scalar(255, 255, 255));

    for(auto it1=boundingBoxes.begin(); it1!=boundingBoxes.end(); ++it1)
    {
        // create randomized color for current 3D object
        cv::RNG rng(it1->boxID);
        cv::Scalar currColor = cv::Scalar(rng.uniform(0,150), rng.uniform(0, 150), rng.uniform(0, 150));

        // plot Lidar points into top view image
        int top=1e8, left=1e8, bottom=0.0, right=0.0; 
        float xwmin=1e8, ywmin=1e8, ywmax=-1e8;
        for (auto it2 = it1->lidarPoints.begin(); it2 != it1->lidarPoints.end(); ++it2)
        {
            // world coordinates
            float xw = (*it2).x; // world position in m with x facing forward from sensor
            float yw = (*it2).y; // world position in m with y facing left from sensor
            xwmin = xwmin<xw ? xwmin : xw;
            ywmin = ywmin<yw ? ywmin : yw;
            ywmax = ywmax>yw ? ywmax : yw;

            // top-view coordinates
            int y = (-xw * imageSize.height / worldSize.height) + imageSize.height;
            int x = (-yw * imageSize.width / worldSize.width) + imageSize.width / 2;

            // find enclosing rectangle
            top = top<y ? top : y;
            left = left<x ? left : x;
            bottom = bottom>y ? bottom : y;
            right = right>x ? right : x;

            // draw individual point
            cv::circle(topviewImg, cv::Point(x, y), 4, currColor, -1);
        }

        // draw enclosing rectangle
        cv::rectangle(topviewImg, cv::Point(left, top), cv::Point(right, bottom),cv::Scalar(0,0,0), 2);

        // augment object with some key data
        char str1[200], str2[200];
        sprintf(str1, "id=%d, #pts=%d", it1->boxID, (int)it1->lidarPoints.size());
        putText(topviewImg, str1, cv::Point2f(left-250, bottom+50), cv::FONT_ITALIC, 2, currColor);
        sprintf(str2, "xmin=%2.2f m, yw=%2.2f m", xwmin, ywmax-ywmin);
        putText(topviewImg, str2, cv::Point2f(left-250, bottom+125), cv::FONT_ITALIC, 2, currColor);  
    }

    // plot distance markers
    float lineSpacing = 2.0; // gap between distance markers
    int nMarkers = floor(worldSize.height / lineSpacing);
    for (size_t i = 0; i < nMarkers; ++i)
    {
        int y = (-(i * lineSpacing) * imageSize.height / worldSize.height) + imageSize.height;
        cv::line(topviewImg, cv::Point(0, y), cv::Point(imageSize.width, y), cv::Scalar(255, 0, 0));
    }

    // display image
    string windowName = "3D Objects";
    cv::namedWindow(windowName, 1);
    cv::imshow(windowName, topviewImg);

    if(bWait)
    {
        cv::waitKey(0); // wait for key to be pressed
    }
}


// Associate a given bounding box with the keypoints it contains
void clusterKptMatchesWithROI(BoundingBox &boundingBox, std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, std::vector<cv::DMatch> &kptMatches)
{
    // ...
}


// Compute time-to-collision (TTC) based on keypoint correspondences in successive images
void computeTTCCamera(std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, 
                      std::vector<cv::DMatch> kptMatches, double frameRate, double &TTC, cv::Mat *visImg)
{
    // ...
}

// Compute time-to-collision (TTC) based on
void computeTTCLidar(std::vector<LidarPoint> &lidarPointsPrev,
                     std::vector<LidarPoint> &lidarPointsCurr, double frameRate, double &TTC)
{
    // ...
}

// Matching bounding boxes
void matchBoundingBoxes(std::vector<cv::DMatch> &matches, std::map<int, int> &bbBestMatches, DataFrame &prevFrame, DataFrame &currFrame)
{
    // Find which bounding boxes key points are enclosed both 
    // on the previous and current frame by using the key of the std::multimap

    // Placeholder for storing the bounding box ID for current and previous frame as multimap
    multimap<int, int> mmap {};

    // Highest bounding box ID in previous frame
    int max_prev_boxID = 0;
    for (auto match : matches)
    {
        // `cv::DMatch` has two relevant attributes: `queryIdx` and `trainIdx` indices
        // which will be used to locate the keypoints in the previous frame using `queryIdx` 
        // and current frame using `trainIdx`

        // Get key points from current and past frame   
        cv::KeyPoint prevKp = prevFrame.keypoints[match.queryIdx];
        cv::KeyPoint currKp = currFrame.keypoints[match.trainIdx];
        
        // Set bounding box ID for both previous and current frame  
        // as invalid index if no keypoints are found in the enclosed bounding box
        int prev_boxID = -1;
        int curr_boxID = -1;

        // For each bounding box in the previous frame and current frame,
        // update the bounding box ID if the keypoint is enclosed within region of interest (ROI)
        for (auto bbox : prevFrame.boundingBoxes)
        {
            if (bbox.roi.contains(prevKp.pt))
                prev_boxID = bbox.boxID;
        }
        max_prev_boxID = prev_boxID > max_prev_boxID ? prev_boxID : max_prev_boxID;

        for (auto bbox : currFrame.boundingBoxes)
        {
            if (bbox.roi.contains(currKp.pt))
                curr_boxID = bbox.boxID;
        }
        
        // Save the bounding box ID of potential match candidates to multimap
        // that allows mapping of potential multiple bounding boxes 
        // in previous frame map to the same bounding box in current frame
        mmap.insert({curr_boxID, prev_boxID});
    }

    // Create the list of bounding box IDs for current frame
    // for constructing the best matching bounding box match pair `bbBestMatches`
    vector<int> currFrameBoxIDs;
    for (auto box : currFrame.boundingBoxes)
        currFrameBoxIDs.push_back(box.boxID);

    // Loop through each boxID in the current frame
    // and get the most frequent value of the associated boxID for the previous frame
    for (int currFrameBoxID : currFrameBoxIDs)
    {
        // Count the greatest number of matches in the multimap where each element is {key=curr_boxID, val=prev_boxID}
        // Get all elements from the multimap that has the key value matches `currFrameBoxID`
        auto prev_boxIDs_range = mmap.equal_range(currFrameBoxID);

        // Initalize the counter of bounding boxes in previous frame with zeros
        std::vector<int> prev_bbCounter(max_prev_boxID+1, 0);

        // Loop through all the bounding boxes in the previous frame
        // that matches to the bounding box in the current frame
        for (auto it = prev_boxIDs_range.first; it != prev_boxIDs_range.second; ++it)
        {
            // Increment counter of matching bounding box in previous frame
            // when the keypoint is within that bounding box
            if (it->second != -1) 
                prev_bbCounter[it->second] += 1;
        }
        
        // Find the bounding box ID in previous frame with the highest number of keypoint correspondences
        // (i.e. index of the bounding box in the previous frame that has the highest keypoint counts)
        int prevFrameBoxID = distance(prev_bbCounter.begin(), max_element(prev_bbCounter.begin(), prev_bbCounter.end()));

        // Insert the best matching bounding box pair {prevFrameBoxID, currFrameBoxID}
        // for each bounding box in current frame
        bbBestMatches.insert({prevFrameBoxID, currFrameBoxID});
    }
}