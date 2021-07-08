
#include <numeric>
#include "matching2D.hpp"

using namespace std;

// Find best matches for keypoints in two camera images based on several matching methods
void matchDescriptors(std::vector<cv::KeyPoint> &kPtsSource, std::vector<cv::KeyPoint> &kPtsRef, cv::Mat &descSource, cv::Mat &descRef,
                      std::vector<cv::DMatch> &matches, std::string descriptorType, std::string matcherType, std::string selectorType)
{
    // configure matcher
    bool crossCheck = false;
    cv::Ptr<cv::DescriptorMatcher> matcher;
    std::vector<int> matchCountList;

    if (matcherType.compare("MAT_BF") == 0)
    {
        int normType = descriptorType.compare("DES_BINARY") == 0 ? cv::NORM_HAMMING : cv::NORM_L2;
        matcher = cv::BFMatcher::create(normType, crossCheck);
        std::cout << "     BF matcher with cross-check = " << crossCheck << std::endl;
    }
    else if (matcherType.compare("MAT_FLANN") == 0)
    {
        if(descSource.type() != CV_32F)
        {
            // convert binary descriptors to floating point
            descSource.convertTo(descSource, CV_32F);
            descRef.convertTo(descRef, CV_32F);
        }
        matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
    }

    // perform matching task
    if (selectorType.compare("SEL_NN") == 0)
    { // nearest neighbor (best match)
        double t = (double)cv::getTickCount();
        matcher->match(descSource, descRef, matches); // Finds the best match for each descriptor in desc1
        t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
        std::cout << "    Nearest Neighbor matching with n = " << matches.size() << " matches in " << 1000 * t / 1.0 << " ms." << std::endl;
    }
    else if (selectorType.compare("SEL_KNN") == 0)
    {   
        // k nearest neighbors (k=2)
        std::vector<std::vector<cv::DMatch>> knnMatches;
        double t = (double)cv::getTickCount();
        matcher->knnMatch(descSource, descRef, knnMatches, 2);
        t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
        std::cout << "     K-Nearest Neighbor matching with n = " << knnMatches.size() << " matches in " << 1000 * t / 1.0 << " ms." << std::endl;

        // filter matches using descriptor distance ratio test
        double minDistRatio = 0.8;
        for(auto it = knnMatches.begin(); it < knnMatches.end(); ++it)
        {
            // calculate minimum distance ratio between source [0] and reference [1] matches
            if((*it)[0].distance < minDistRatio * (*it)[1].distance)
            {
                matches.push_back((*it)[0]);
            }
        }
        std::cout << "     Descriptor distance ratio filtering removed " << knnMatches.size() - matches.size() 
                  << " keypoints." << std::endl;
    
    }
}

// Use one of several types of state-of-art descriptors to uniquely identify keypoints
// BRISK, BRIEF, ORB, FREAK, AKAZE, SIFT
void descKeypoints(vector<cv::KeyPoint> &keypoints, cv::Mat &img, cv::Mat &descriptors, string descriptorType)
{
    // select appropriate descriptor
    cv::Ptr<cv::DescriptorExtractor> extractor;
    // BRISK
    if (descriptorType.compare("BRISK") == 0)
    {

        int threshold = 30;        // FAST/AGAST detection threshold score.
        int octaves = 3;           // detection octaves (use 0 to do single scale)
        float patternScale = 1.0f; // apply this scale to the pattern used for sampling the neighbourhood of a keypoint.

        extractor = cv::BRISK::create(threshold, octaves, patternScale);
    }
    // BRIEF
    else if (descriptorType.compare("BRIEF") == 0)
    {
        extractor = cv::xfeatures2d::BriefDescriptorExtractor::create();
    }
    // ORB
    else if (descriptorType.compare("ORB") == 0)
    {
        extractor = cv::ORB::create();
    }
    // FREAK
    else if (descriptorType.compare("FREAK") == 0)
    {
        extractor = cv::xfeatures2d::FREAK::create();
    }
    // AKAZE
    else if (descriptorType.compare("AKAZE") == 0)
    {
        extractor = cv::AKAZE::create();
    }
    // SIFT
    else
    {
        extractor = cv::SIFT::create();
    }

    // perform feature extraction
    double t = (double)cv::getTickCount();
    extractor->compute(img, keypoints, descriptors);
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    std::cout << "     "<< descriptorType << " descriptor extraction in " << 1000 * t / 1.0 << " ms" << endl;
}

// Detect keypoints in image using the traditional Shi-Thomasi detector
void detKeypointsShiTomasi(vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis)
{
    // compute detector parameters based on image size
    int blockSize = 4;       //  size of an average block for computing a derivative covariation matrix over each pixel neighborhood
    double maxOverlap = 0.0; // max. permissible overlap between two features in %
    double minDistance = (1.0 - maxOverlap) * blockSize;
    int maxCorners = img.rows * img.cols / max(1.0, minDistance); // max. num. of keypoints

    double qualityLevel = 0.01; // minimal accepted quality of image corners
    double k = 0.04;

    // Apply corner detection
    double t = (double)cv::getTickCount();
    vector<cv::Point2f> corners;
    cv::goodFeaturesToTrack(img, corners, maxCorners, qualityLevel, minDistance, cv::Mat(), blockSize, false, k);

    // add corners to result vector
    for (auto it = corners.begin(); it != corners.end(); ++it)
    {

        cv::KeyPoint newKeyPoint;
        newKeyPoint.pt = cv::Point2f((*it).x, (*it).y);
        newKeyPoint.size = blockSize;
        keypoints.push_back(newKeyPoint);
    }
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    std::cout << "     Shi-Tomasi detection with n =" << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms." << std::endl;

    // visualize results
    if (bVis)
    {
        cv::Mat visImage = img.clone();
        cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        string windowName = "Shi-Tomasi Corner Detector Results";
        cv::namedWindow(windowName, 6);
        imshow(windowName, visImage);
        cv::waitKey(0);
    }
}

// Detect keypoints in an image using the Harris Method
void detKeypointsHarris(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis)
{
    // Detector parameters
    int blockSize = 2; // for every pixel, a blockSize × blockSize neighborhood is considered
    int aperatureSize = 3; // aperture parameter for Sobel operator (must be odd)
    int minResponse = 100; // minimum value for a corner in the 8bit scaled response matrix
    double k = 0.04; // Harris parameter (see equation for details)
    double maxOverlap = 0.05; // used for NMS

    // Detect Harris corners and normalize output 
    cv::Mat dst, dstNorm, dstNormScaled;
    dst = cv::Mat::zeros(img.size(), CV_32FC1);

    double t = (double)cv::getTickCount();
    cv::cornerHarris(img, dst, blockSize, aperatureSize, k, cv::BORDER_DEFAULT);
    cv::normalize(dst, dstNorm, 0, 255, cv::NORM_MINMAX, CV_32FC1, cv::Mat());
    cv::convertScaleAbs(dstNorm, dstNormScaled);

    // locate local maxima in the Harris response matrix 
    // and perform a non-maximum suppression (NMS) in a local  
    // neighborhood around each maximum.

    for(int j = 0; j < dstNorm.rows; j++)
    {
        for(int i = 0; i < dstNorm.cols; i++)
        {
            // response output from normalized Harris Detector
            int response = (int)dstNorm.at<float>(j, i); // (row, col)
            if(response > minResponse)
            {
                // store keypoint (pt), size, and response in newKeyPointObj
                cv::KeyPoint newKeyPoint;
                newKeyPoint.pt = cv::Point2f(i, j);
                newKeyPoint.size = 2 * aperatureSize;
                newKeyPoint.response = response;

                bool isOverlap = false;

                // perform NMS to reduce the number of keypoints displayed
                for(auto it = keypoints.begin(); it < keypoints.end(); ++it)
                {
                    // calculate overlap. No overlap == 0
                    double keyPtOverlap = cv::KeyPoint::overlap(newKeyPoint, *it);
                    if(keyPtOverlap > maxOverlap)
                        {
                            isOverlap = true;
                            // if overlap check response
                            if(newKeyPoint.response > it->response)
                            {
                                *it = newKeyPoint;
                                break;
                            }
                        }
                }
                // if no overlap add new keypoint 
                if(!isOverlap)
                    keypoints.push_back(newKeyPoint);
            }
        } // end of loop over cols
    } // end of loop over rows

    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    std::cout << "     Harris detection with n =" << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms." << std::endl;

    // visualize results
    if(bVis)
    {
        string windowName = "Harris Corner Detection Results";
        cv::namedWindow(windowName, 6); 
        cv::Mat visImg = img.clone();
        cv::drawKeypoints(img, keypoints, visImg, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        cv::imshow(windowName, visImg);
        cv::waitKey(0);
    }
}

// Modern Keypoint Detectors. FAST, BRISK, ORB, AKAZE, SIFT, FREAK
void detKeypointsModern(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, std::string detectorType, bool bVis)
{
    // FAST detector
    if(detectorType.compare("FAST") == 0)
    {
        // FAST parameters
        int threshold = 30; // difference between intensity of the central pixel and pixels of a circle around this pixel
        bool NMS = true;
        cv::Ptr<cv::FastFeatureDetector> detector = cv::FastFeatureDetector::create(threshold, NMS); 
        
        double t =(double)cv::getTickCount();
        detector->detect(img, keypoints, img);
        t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
        std::cout << "     FAST feature detector with n = " << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms." << std::endl;
    }
    // BRISK detector
    else if(detectorType.compare("BRISK") == 0)
    {
        cv::Ptr<cv::FeatureDetector> detector = cv::BRISK::create();
        
        double t =(double)cv::getTickCount();
        detector->detect(img, keypoints);
        t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
        std::cout << "     BRISK feature detector with n = " << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms." << std::endl;
    }
    // ORB detector
    else if(detectorType.compare("ORB") == 0)
    {
        cv::Ptr<cv::FeatureDetector> detector = cv::ORB::create();
        double t = (double)cv::getTickCount();
        detector->detect(img, keypoints);
        t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
        std::cout << "     ORB feature detector with n = " << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms." << std::endl;
    }
    // SIFT detector
    else if(detectorType.compare("SIFT") == 0)
    {
        cv::Ptr<cv::FeatureDetector> detector = cv::SIFT::create();
        double t = (double)cv::getTickCount();
        detector->detect(img, keypoints);
        t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
        std::cout << "     SIFT feature detector with n = " << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms." << std::endl;
    }
    // AKAZE detector. NOTE -> ONLY can use AKAZE descriptor excractor and descriptors with AKAZE keypoints
    else
    {
        cv::Ptr<cv::FeatureDetector> detector = cv::AKAZE::create();
        double t = (double)cv::getTickCount();
        detector->detect(img, keypoints);
        t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
        std::cout << "     AKAZE feature detector with n = " << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms." << std::endl;
    }
   
    // visualize results
    if(bVis)
    {
        string windowName = detectorType + " Keypoint Detector";
        cv::namedWindow(windowName, 1);
        cv::Mat visImg = img.clone();
        cv::drawKeypoints(img, keypoints, visImg, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        cv::imshow(windowName, visImg);
        cv::waitKey(0);
    }
}