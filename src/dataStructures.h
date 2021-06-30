
#ifndef dataStructures_h
#define dataStructures_h

#include <vector>
#include <map>
#include <opencv2/core.hpp>
#include <vector>
#include <math.h>

struct LidarPoint { // single lidar point in space
    double x,y,z,r; // x,y,z in [m], r is point reflectivity
};

struct BoundingBox { // bounding box around a classified object (contains both 2D and 3D data)
    
    int boxID; // unique identifier for this bounding box
    int trackID; // unique identifier for the track to which this bounding box belongs
    
    cv::Rect roi; // 2D region-of-interest in image coordinates
    int classID; // ID based on class file provided to YOLO framework
    double confidence; // classification trust

    std::vector<LidarPoint> lidarPoints; // Lidar 3D points which project into 2D image roi
    std::vector<cv::KeyPoint> keypoints; // keypoints enclosed by 2D roi
    std::vector<cv::DMatch> kptMatches; // keypoint matches enclosed by 2D roi
};

struct DataFrame { // represents the available sensor information at the same time instance
    
    cv::Mat cameraImg; // camera image
    
    std::vector<cv::KeyPoint> keypoints; // 2D keypoints within camera image
    cv::Mat descriptors; // keypoint descriptors
    std::vector<cv::DMatch> kptMatches; // keypoint matches between previous and current frame
    std::vector<LidarPoint> lidarPoints;

    std::vector<BoundingBox> boundingBoxes; // ROI around detected objects in 2D image coordinates
    std::map<int,int> bbMatches; // bounding box matches between previous and current frame
};

// node for KdTree
struct Node
{
    std::vector<double> point;
    int id;
    Node* left;
    Node* right;

    // constructor with initialization
    Node(std::vector<double> arr, int setID) : point(arr), id(setID), left(NULL), right(NULL) {}

    ~Node()
    {
        delete left;
        delete right; 
    }
};

// kdTree
struct KdTree
{
    KdTree() : root(NULL){}

    ~KdTree()
    {
        delete root;
    }

    Node* root;

    // methods
    void insert(std::vector<double> point, int id)
    {
        insertRecursively(&root, 0, point, id);
    }

    void insertRecursively(Node** node, unsigned int depth, std::vector<double> point, int id)
    {
        // if no root node, create a new node
        if(*node == NULL)
            *node = new Node(point, id);
        else
        {
            unsigned int currentDepth = depth % 3;
            // if less tan go left
            if(point[currentDepth] < ((*node)->point[currentDepth]))
                insertRecursively(&(*node)->left, depth+1, point, id);
            // if greater than go right
            else    
                insertRecursively(&(*node)->right, depth+1, point, id);
        }
    }

    // return a list of point ids in the tree that are within distance of target
    std::vector<int> search(std::vector<double> target, float distanceTolerance)
    {
        std::vector<int> ids;
        searchRecursively(target, &root, 0, distanceTolerance, ids);
        return ids;
    }

    void searchRecursively(std::vector<double> target, Node** node, unsigned int depth, float distanceTolerance, std::vector<int> &ids)
    {
        if(*node != NULL)
        {
            // check x, y, and z and calculate distance from 
            if((*node)->point[0] >= (target[0] - distanceTolerance) && (*node)->point[0] <= (target[0] + distanceTolerance)
                && (*node)->point[1] >= (target[1] - distanceTolerance) && (*node)->point[1] <= (target[1] + distanceTolerance)
                && (*node)->point[2] >= (target[2] - distanceTolerance) && (*node)->point[2] <= (target[2] + distanceTolerance))
            {
                double distance = sqrt(((*node)->point[0] - target[0])*((*node)->point[0] - target[0])
                                      + ((*node)->point[1] - target[1])*((*node)->point[1] - target[1])
                                      + ((*node)->point[2] - target[2])*((*node)->point[2] - target[2]));
                if(distance <= distanceTolerance)
                    ids.push_back((*node)->id);
            }
            // check left or right
            if((target[depth % 3] - distanceTolerance) <= (*node)->point[depth % 3])
                searchRecursively(target, &((*node)->left), depth + 1, distanceTolerance, ids);
            if((target[depth % 3] + distanceTolerance) >= (*node)->point[depth % 3])
                searchRecursively(target, &((*node)->right), depth + 1, distanceTolerance, ids);
        }
        
    }
};
#endif /* dataStructures_h */
