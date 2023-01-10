// #include "opencv2/highgui.hpp"
// #include "opencv2/imgproc.hpp"
// #include <iostream>
// using namespace cv;
// using namespace std;
// Mat src, src_gray;
// int thresh = 200;
// int max_thresh = 255;
// const char* source_window = "Source image";
// const char* corners_window = "Corners detected";
// void cornerHarris_demo( int, void* );
// int main( int argc, char** argv )
// {
//     CommandLineParser parser( argc, argv, "{@input | ../b.jpg | input image}" );
//     src = imread( samples::findFile( parser.get<String>( "@input" ) ) );
//     if ( src.empty() )
//     {
//         cout << "Could not open or find the image!\n" << endl;
//         cout << "Usage: " << argv[0] << " <Input image>" << endl;
//         return -1;
//     }
//     cvtColor( src, src_gray, COLOR_BGR2GRAY );
//     namedWindow( source_window );
//     createTrackbar( "Threshold: ", source_window, &thresh, max_thresh, cornerHarris_demo );
//     imshow( source_window, src );
//     cornerHarris_demo( 0, 0 );
//     waitKey();
//     return 0;
// }
// void cornerHarris_demo( int, void* )
// {
//     int blockSize = 2;
//     int apertureSize = 3;
//     double k = 0.04;
//     Mat dst = Mat::zeros( src.size(), CV_32FC1 );
//     cornerHarris( src_gray, dst, blockSize, apertureSize, k );
//     Mat dst_norm, dst_norm_scaled;
//     normalize( dst, dst_norm, 0, 255, NORM_MINMAX, CV_32FC1, Mat() );
//     convertScaleAbs( dst_norm, dst_norm_scaled );
//     for( int i = 0; i < dst_norm.rows ; i++ )
//     {
//         for( int j = 0; j < dst_norm.cols; j++ )
//         {
//             if( (int) dst_norm.at<float>(i,j) > thresh )
//             {
//                 circle( dst_norm_scaled, Point(j,i), 5,  Scalar(0), 2, 8, 0 );
//             }
//         }
//     }
//     namedWindow( corners_window );
//     imshow( corners_window, dst_norm_scaled );
// }

// #include "opencv2/imgcodecs.hpp"
// #include "opencv2/highgui.hpp"
// #include "opencv2/imgproc.hpp"
// #include <iostream>
// using namespace cv;
// using namespace std;
// Mat src, src_gray;
// int maxCorners = 23;
// int maxTrackbar = 100;
// RNG rng(12345);
// const char* source_window = "Image";
// void goodFeaturesToTrack_Demo( int, void* );
// int main( int argc, char** argv )
// {
//     CommandLineParser parser( argc, argv, "{@input |../b.jpg | input image}" );
//     src = imread( samples::findFile( parser.get<String>( "@input" ) ) );
//     if( src.empty() )
//     {
//         cout << "Could not open or find the image!\n" << endl;
//         cout << "Usage: " << argv[0] << " <Input image>" << endl;
//         return -1;
//     }
//     cvtColor( src, src_gray, COLOR_BGR2GRAY );
//     namedWindow( source_window );
//     createTrackbar( "Max corners:", source_window, &maxCorners, maxTrackbar, goodFeaturesToTrack_Demo );
//     imshow( source_window, src );
//     goodFeaturesToTrack_Demo( 0, 0 );
//     waitKey();
//     return 0;
// }
// void goodFeaturesToTrack_Demo( int, void* )
// {
//     maxCorners = MAX(maxCorners, 1);
//     vector<Point2f> corners;
//     double qualityLevel = 0.01;
//     double minDistance = 10;
//     int blockSize = 3, gradientSize = 3;
//     bool useHarrisDetector = false;
//     double k = 0.04;
//     Mat copy = src.clone();
//     goodFeaturesToTrack( src_gray,
//                          corners,
//                          maxCorners,
//                          qualityLevel,
//                          minDistance,
//                          Mat(),
//                          blockSize,
//                          gradientSize,
//                          useHarrisDetector,
//                          k );
//     cout << "** Number of corners detected: " << corners.size() << endl;
//     int radius = 4;
//     for( size_t i = 0; i < corners.size(); i++ )
//     {
//         circle( copy, corners[i], radius, Scalar(rng.uniform(0,255), rng.uniform(0, 256), rng.uniform(0, 256)), FILLED );
//     }
//     namedWindow( source_window );
//     imshow( source_window, copy );
// }



// #include "opencv2/highgui.hpp"
// #include "opencv2/imgproc.hpp"
// #include <iostream>
// using namespace cv;
// using namespace std;
// Mat src, src_gray;
// Mat myHarris_dst, myHarris_copy, Mc;
// Mat myShiTomasi_dst, myShiTomasi_copy;
// int myShiTomasi_qualityLevel = 50;
// int myHarris_qualityLevel = 50;
// int max_qualityLevel = 100;
// double myHarris_minVal, myHarris_maxVal;
// double myShiTomasi_minVal, myShiTomasi_maxVal;
// RNG rng(12345);
// const char* myHarris_window = "My Harris corner detector";
// const char* myShiTomasi_window = "My Shi Tomasi corner detector";
// void myShiTomasi_function( int, void* );
// void myHarris_function( int, void* );
// int main( int argc, char** argv )
// {
//     CommandLineParser parser( argc, argv, "{@input | ../b.jpg | input image}" );
//     src = imread( samples::findFile( parser.get<String>( "@input" ) ) );
//     if ( src.empty() )
//     {
//         cout << "Could not open or find the image!\n" << endl;
//         cout << "Usage: " << argv[0] << " <Input image>" << endl;
//         return -1;
//     }
//     cvtColor( src, src_gray, COLOR_BGR2GRAY );
//     int blockSize = 3, apertureSize = 3;
//     cornerEigenValsAndVecs( src_gray, myHarris_dst, blockSize, apertureSize );
//     /* calculate Mc */
//     Mc = Mat( src_gray.size(), CV_32FC1 );
//     for( int i = 0; i < src_gray.rows; i++ )
//     {
//         for( int j = 0; j < src_gray.cols; j++ )
//         {
//             float lambda_1 = myHarris_dst.at<Vec6f>(i, j)[0];
//             float lambda_2 = myHarris_dst.at<Vec6f>(i, j)[1];
//             Mc.at<float>(i, j) = lambda_1*lambda_2 - 0.04f*((lambda_1 + lambda_2) * (lambda_1 + lambda_2));
//         }
//     }
//     minMaxLoc( Mc, &myHarris_minVal, &myHarris_maxVal );
//     /* Create Window and Trackbar */
//     namedWindow( myHarris_window );
//     createTrackbar( "Quality Level:", myHarris_window, &myHarris_qualityLevel, max_qualityLevel, myHarris_function );
//     myHarris_function( 0, 0 );
//     cornerMinEigenVal( src_gray, myShiTomasi_dst, blockSize, apertureSize );
//     minMaxLoc( myShiTomasi_dst, &myShiTomasi_minVal, &myShiTomasi_maxVal );
//     /* Create Window and Trackbar */
//     namedWindow( myShiTomasi_window );
//     createTrackbar( "Quality Level:", myShiTomasi_window, &myShiTomasi_qualityLevel, max_qualityLevel, myShiTomasi_function );
//     myShiTomasi_function( 0, 0 );
//     waitKey();
//     return 0;
// }
// void myShiTomasi_function( int, void* )
// {
//     myShiTomasi_copy = src.clone();
//     myShiTomasi_qualityLevel = MAX(myShiTomasi_qualityLevel, 1);
//     for( int i = 0; i < src_gray.rows; i++ )
//     {
//         for( int j = 0; j < src_gray.cols; j++ )
//         {
//             if( myShiTomasi_dst.at<float>(i,j) > myShiTomasi_minVal + ( myShiTomasi_maxVal - myShiTomasi_minVal )*myShiTomasi_qualityLevel/max_qualityLevel )
//             {
//                 circle( myShiTomasi_copy, Point(j,i), 4, Scalar( rng.uniform(0,256), rng.uniform(0,256), rng.uniform(0,256) ), FILLED );
//             }
//         }
//     }
//     imshow( myShiTomasi_window, myShiTomasi_copy );
// }
// void myHarris_function( int, void* )
// {
//     myHarris_copy = src.clone();
//     myHarris_qualityLevel = MAX(myHarris_qualityLevel, 1);
//     for( int i = 0; i < src_gray.rows; i++ )
//     {
//         for( int j = 0; j < src_gray.cols; j++ )
//         {
//             if( Mc.at<float>(i,j) > myHarris_minVal + ( myHarris_maxVal - myHarris_minVal )*myHarris_qualityLevel/max_qualityLevel )
//             {
//                 circle( myHarris_copy, Point(j,i), 4, Scalar( rng.uniform(0,256), rng.uniform(0,256), rng.uniform(0,256) ), FILLED );
//             }
//         }
//     }
//     imshow( myHarris_window, myHarris_copy );
// }


#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>
using namespace cv;
using namespace std;
Mat src, src_gray;
int maxCorners = 10;
int maxTrackbar = 25;
RNG rng(12345);
const char* source_window = "Image";
void goodFeaturesToTrack_Demo( int, void* );
int main( int argc, char** argv )
{
    CommandLineParser parser( argc, argv, "{@input | ../b.jpg | input image}" );
    src = imread( samples::findFile( parser.get<String>( "@input" ) ) );
    if( src.empty() )
    {
        cout << "Could not open or find the image!\n" << endl;
        cout << "Usage: " << argv[0] << " <Input image>" << endl;
        return -1;
    }
    cvtColor( src, src_gray, COLOR_BGR2GRAY );
    namedWindow( source_window );
    createTrackbar( "Max corners:", source_window, &maxCorners, maxTrackbar, goodFeaturesToTrack_Demo );
    imshow( source_window, src );
    goodFeaturesToTrack_Demo( 0, 0 );
    waitKey();
    return 0;
}
void goodFeaturesToTrack_Demo( int, void* )
{
    maxCorners = MAX(maxCorners, 1);
    vector<Point2f> corners;
    double qualityLevel = 0.01;
    double minDistance = 10;
    int blockSize = 3, gradientSize = 3;
    bool useHarrisDetector = false;
    double k = 0.04;
    Mat copy = src.clone();
    goodFeaturesToTrack( src_gray,
                         corners,
                         maxCorners,
                         qualityLevel,
                         minDistance,
                         Mat(),
                         blockSize,
                         gradientSize,
                         useHarrisDetector,
                         k );
    cout << "** Number of corners detected: " << corners.size() << endl;
    int radius = 4;
    for( size_t i = 0; i < corners.size(); i++ )
    {
        circle( copy, corners[i], radius, Scalar(rng.uniform(0,255), rng.uniform(0, 256), rng.uniform(0, 256)), FILLED );
    }
    namedWindow( source_window );
    imshow( source_window, copy );
    Size winSize = Size( 5, 5 );
    Size zeroZone = Size( -1, -1 );
    TermCriteria criteria = TermCriteria( TermCriteria::EPS + TermCriteria::COUNT, 40, 0.001 );
    cornerSubPix( src_gray, corners, winSize, zeroZone, criteria );
    for( size_t i = 0; i < corners.size(); i++ )
    {
        cout << " -- Refined Corner [" << i << "]  (" << corners[i].x << "," << corners[i].y << ")" << endl;
    }
}