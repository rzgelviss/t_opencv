// #include <iostream>
// #include "opencv2/core.hpp"
// #ifdef HAVE_OPENCV_XFEATURES2D
// #include "opencv2/highgui.hpp"
// #include "opencv2/features2d.hpp"
// #include "opencv2/xfeatures2d.hpp"
// using namespace cv;
// using namespace cv::xfeatures2d;
// using std::cout;
// using std::endl;
// int main( int argc, char* argv[] )
// {
//     CommandLineParser parser( argc, argv, "{@input | ../b.jpg | input image}" );
//     Mat src = imread( samples::findFile( parser.get<String>( "@input" ) ), IMREAD_GRAYSCALE );
//     if ( src.empty() )
//     {
//         cout << "Could not open or find the image!\n" << endl;
//         cout << "Usage: " << argv[0] << " <Input image>" << endl;
//         return -1;
//     }
//     //-- Step 1: Detect the keypoints using SURF Detector
//     int minHessian = 400;
//     Ptr<SURF> detector = SURF::create( minHessian );
//     std::vector<KeyPoint> keypoints;
//     detector->detect( src, keypoints );
//     //-- Draw keypoints
//     Mat img_keypoints;
//     drawKeypoints( src, keypoints, img_keypoints );
//     //-- Show detected (drawn) keypoints
//     imshow("SURF Keypoints", img_keypoints );
//     waitKey();
//     return 0;
// }
// #else
// int main()
// {
//     std::cout << "This tutorial code needs the xfeatures2d contrib module to be run." << std::endl;
//     return 0;
// }
// #endif


#include <iostream>
#include "opencv2/core.hpp"
// #ifdef HAVE_OPENCV_XFEATURES2D
#include "opencv2/highgui.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"
using namespace cv;
using namespace cv::xfeatures2d;
using std::cout;
using std::endl;
const char* keys =
    "{ help h |                  | Print help message. }"
    "{ input1 | ../b.jpg          | Path to input image 1. }"
    "{ input2 | ../b.jpg | Path to input image 2. }";
int main( int argc, char* argv[] )
{
    CommandLineParser parser( argc, argv, keys );
    Mat img1 = imread( samples::findFile( parser.get<String>("input1") ), IMREAD_GRAYSCALE );
    Mat img2 = imread( samples::findFile( parser.get<String>("input2") ), IMREAD_GRAYSCALE );
    if ( img1.empty() || img2.empty() )
    {
        cout << "Could not open or find the image!\n" << endl;
        parser.printMessage();
        return -1;
    }
    //-- Step 1: Detect the keypoints using SURF Detector, compute the descriptors
    int minHessian = 400;
    Ptr<SURF> detector = SURF::create( minHessian );
    std::vector<KeyPoint> keypoints1, keypoints2;
    Mat descriptors1, descriptors2;
    detector->detectAndCompute( img1, noArray(), keypoints1, descriptors1 );
    detector->detectAndCompute( img2, noArray(), keypoints2, descriptors2 );
    //-- Step 2: Matching descriptor vectors with a brute force matcher
    // Since SURF is a floating-point descriptor NORM_L2 is used
    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(DescriptorMatcher::BRUTEFORCE);
    std::vector< DMatch > matches;
    matcher->match( descriptors1, descriptors2, matches );
    //-- Draw matches
    Mat img_matches;
    drawMatches( img1, keypoints1, img2, keypoints2, matches, img_matches );
    //-- Show detected matches
    imshow("Matches", img_matches );
    waitKey();
    return 0;
}
// #else
// int main()
// {
//     std::cout << "This tutorial code needs the xfeatures2d contrib module to be run." << std::endl;
//     return 0;
// }
// #endif


// #include <opencv2/imgproc.hpp>
// #include <opencv2/highgui.hpp>
// #include <iostream>
// using namespace std;
// using namespace cv;
// const float inlier_threshold = 2.5f; // Distance threshold to identify inliers with homography check
// const float nn_match_ratio = 0.8f;   // Nearest neighbor matching ratio
// int main(int argc, char* argv[])
// {
//     CommandLineParser parser(argc, argv,
//                              "{@img1 | ../b.jpg | input image 1}"
//                              "{@img2 | ../b.jpg | input image 2}"
//                              "{@homography | H1to3p.xml | homography matrix}");
//     Mat img1 = imread( samples::findFile( parser.get<String>("@img1") ), IMREAD_GRAYSCALE);
//     Mat img2 = imread( samples::findFile( parser.get<String>("@img2") ), IMREAD_GRAYSCALE);
//     Mat homography;
//     FileStorage fs( samples::findFile( parser.get<String>("@homography") ), FileStorage::READ);
//     fs.getFirstTopLevelNode() >> homography;
//     vector<KeyPoint> kpts1, kpts2;
//     Mat desc1, desc2;
//     Ptr<AKAZE> akaze = AKAZE::create();
//     akaze->detectAndCompute(img1, noArray(), kpts1, desc1);
//     akaze->detectAndCompute(img2, noArray(), kpts2, desc2);
//     BFMatcher matcher(NORM_HAMMING);
//     vector< vector<DMatch> > nn_matches;
//     matcher.knnMatch(desc1, desc2, nn_matches, 2);
//     vector<KeyPoint> matched1, matched2;
//     for(size_t i = 0; i < nn_matches.size(); i++) {
//         DMatch first = nn_matches[i][0];
//         float dist1 = nn_matches[i][0].distance;
//         float dist2 = nn_matches[i][1].distance;
//         if(dist1 < nn_match_ratio * dist2) {
//             matched1.push_back(kpts1[first.queryIdx]);
//             matched2.push_back(kpts2[first.trainIdx]);
//         }
//     }
//     vector<DMatch> good_matches;
//     vector<KeyPoint> inliers1, inliers2;
//     for(size_t i = 0; i < matched1.size(); i++) {
//         Mat col = Mat::ones(3, 1, CV_64F);
//         col.at<double>(0) = matched1[i].pt.x;
//         col.at<double>(1) = matched1[i].pt.y;
//         col = homography * col;
//         col /= col.at<double>(2);
//         double dist = sqrt( pow(col.at<double>(0) - matched2[i].pt.x, 2) +
//                             pow(col.at<double>(1) - matched2[i].pt.y, 2));
//         if(dist < inlier_threshold) {
//             int new_i = static_cast<int>(inliers1.size());
//             inliers1.push_back(matched1[i]);
//             inliers2.push_back(matched2[i]);
//             good_matches.push_back(DMatch(new_i, new_i, 0));
//         }
//     }
//     Mat res;
//     drawMatches(img1, inliers1, img2, inliers2, good_matches, res);
//     imwrite("akaze_result.png", res);
//     double inlier_ratio = inliers1.size() / (double) matched1.size();
//     cout << "A-KAZE Matching Results" << endl;
//     cout << "*******************************" << endl;
//     cout << "# Keypoints 1:                        \t" << kpts1.size() << endl;
//     cout << "# Keypoints 2:                        \t" << kpts2.size() << endl;
//     cout << "# Matches:                            \t" << matched1.size() << endl;
//     cout << "# Inliers:                            \t" << inliers1.size() << endl;
//     cout << "# Inliers Ratio:                      \t" << inlier_ratio << endl;
//     cout << endl;
//     imshow("result", res);
//     waitKey();
//     return 0;
// }