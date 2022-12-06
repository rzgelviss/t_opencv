// #include "opencv2/imgproc.hpp"
// #include "opencv2/imgcodecs.hpp"
// #include "opencv2/highgui.hpp"
// #include <iostream>
// using namespace cv;
// using std::cout;
// int threshold_value = 0;
// int threshold_type = 3;
// int const max_value = 255;
// int const max_type = 4;
// int const max_binary_value = 255;
// Mat src, src_gray, dst;
// const char* window_name = "Threshold Demo";
// const char* trackbar_type = "Type:\n 0: Binary \n 1:Binary Inverted \n 2:Truncate \n 3: To Zero \n 4:To Zero Inverted";
// const char* trackbar_value = "Value";
// static void Threshold_Demo(int, void*)
// {
//     /*0:Binary
//     1:Binary Inverted
//     2:Threshold Truncated
//     3:Threshold to Zero
//     4:Threshold to Zero Inverted
//     */
//    threshold(src_gray, dst, threshold_value, max_binary_value, threshold_type);
//    imshow(window_name, dst);
// }

// int main(int argc, char** argv)
// {
//     String imageName("../a.jpg");
//     if(argc > 1)
//     {
//         imageName = argv[1];
//     }
//     src = imread(samples::findFile(imageName), IMREAD_COLOR);
//     if(src.empty())
//     {
//         cout << "Cannot read the image:" << imageName << std::endl;
//         return -1;
//     }
//     cvtColor(src, src_gray, COLOR_BGR2GRAY);
//     namedWindow(window_name, WINDOW_AUTOSIZE);
//     createTrackbar(trackbar_type, window_name, &threshold_type, max_type, Threshold_Demo);
//     createTrackbar(trackbar_value, window_name, &threshold_value, max_value, Threshold_Demo);
//     Threshold_Demo(0, 0);
//     waitKey();
//     return 0;

// }

// #include "opencv2/imgproc.hpp"
// #include "opencv2/highgui.hpp"
// #include "opencv2/videoio.hpp"
// #include <iostream>
// using namespace cv;
// const int max_value_H = 360/2;
// const int max_value = 255;
// const String window_capture_name = "Video Capture";
// const String window_detection_name = "Object Detection";
// int low_H = 0, low_S = 0, low_V = 0;
// int high_H = max_value_H, high_S = max_value, high_V = max_value;
// static void on_low_H_thresh_trackbar(int, void *)
// {
//     low_H = min(high_H-1, low_H);
//     setTrackbarPos("Low H", window_detection_name, low_H);
// }
// static void on_high_H_thresh_trackbar(int, void *)
// {
//     high_H = max(high_H, low_H+1);
//     setTrackbarPos("High H", window_detection_name, high_H);
// }
// static void on_low_S_thresh_trackbar(int, void *)
// {
//     low_S = min(high_S-1, low_S);
//     setTrackbarPos("Low S", window_detection_name, low_S);
// }
// static void on_high_S_thresh_trackbar(int, void *)
// {
//     high_S = max(high_S, low_S+1);
//     setTrackbarPos("High S", window_detection_name, high_S);
// }
// static void on_low_V_thresh_trackbar(int, void *)
// {
//     low_V = min(high_V-1, low_V);
//     setTrackbarPos("Low V", window_detection_name, low_V);
// }
// static void on_high_V_thresh_trackbar(int, void *)
// {
//     high_V = max(high_V, low_V+1);
//     setTrackbarPos("High V", window_detection_name, high_V);
// }
// int main(int argc, char* argv[])
// {
//     VideoCapture cap(argc > 1 ? atoi(argv[1]) : 0);
//     namedWindow(window_capture_name);
//     namedWindow(window_detection_name);
//     // Trackbars to set thresholds for HSV values
//     createTrackbar("Low H", window_detection_name, &low_H, max_value_H, on_low_H_thresh_trackbar);
//     createTrackbar("High H", window_detection_name, &high_H, max_value_H, on_high_H_thresh_trackbar);
//     createTrackbar("Low S", window_detection_name, &low_S, max_value, on_low_S_thresh_trackbar);
//     createTrackbar("High S", window_detection_name, &high_S, max_value, on_high_S_thresh_trackbar);
//     createTrackbar("Low V", window_detection_name, &low_V, max_value, on_low_V_thresh_trackbar);
//     createTrackbar("High V", window_detection_name, &high_V, max_value, on_high_V_thresh_trackbar);
//     Mat frame, frame_HSV, frame_threshold;
//     while (true) {
//         cap >> frame;
//         if(frame.empty())
//         {
//             break;
//         }
//         // Convert from BGR to HSV colorspace
//         cvtColor(frame, frame_HSV, COLOR_BGR2HSV);
//         // Detect the object based on HSV Range Values
//         inRange(frame_HSV, Scalar(low_H, low_S, low_V), Scalar(high_H, high_S, high_V), frame_threshold);
//         // Show the frames
//         imshow(window_capture_name, frame);
//         imshow(window_detection_name, frame_threshold);
//         char key = (char) waitKey(30);
//         if (key == 'q' || key == 27)
//         {
//             break;
//         }
//     }
//     return 0;
// }


#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/videoio.hpp"
#include <iostream>
using namespace cv;
const int max_value_H = 360/2;
const int max_value = 255;
const String window_capture_name = "Video Capture";
const String window_detection_name = "Object Detection";
int low_H = 0, low_S = 0, low_V = 0;
int high_H = max_value_H, high_S = max_value, high_V = max_value;
static void on_low_H_thresh_trackbar(int, void*)
{
    low_H =  min(high_H-1, low_H);
    setTrackbarPos("Low H", window_detection_name, low_H);
}
static void on_high_H_thresh_trackbar(int, void *)
{
    high_H = max(high_H, low_H+1);
    setTrackbarPos("High H", window_detection_name, high_H);
}
static void on_low_S_thresh_trackbar(int, void*)
{
    low_S = min(high_S-1, low_S);
    setTrackbarPos("Low S", window_detection_name, low_S);
}
static void on_high_S_thresh_trackbar(int, void *)
{
    high_S = max(high_S, low_S+1);
    setTrackbarPos("High S", window_detection_name, high_S);
}
static void on_low_V_thresh_trackbar(int, void *)
{
    low_V = min(high_V-1, low_V);
    setTrackbarPos("Low V", window_detection_name, low_V);
}
static void on_high_V_thresh_trackbar(int, void *)
{
    high_V = max(high_V, low_V+1);
    setTrackbarPos("High V", window_detection_name, high_V);
}

int main(int argc, char* argv[])
{
    VideoCapture cap(argc>1?atoi(argv[1]):0);
    namedWindow(window_capture_name);
    namedWindow(window_detection_name);
    createTrackbar("Low H", window_detection_name, &low_H, max_value_H, on_low_H_thresh_trackbar);
    createTrackbar("High H", window_detection_name, &high_H, max_value_H, on_high_H_thresh_trackbar);
    createTrackbar("Low S", window_detection_name, &low_S, max_value, on_low_S_thresh_trackbar);
    createTrackbar("High S", window_detection_name, &high_S, max_value, on_high_S_thresh_trackbar);
    createTrackbar("Low V", window_detection_name, &low_V, max_value, on_low_V_thresh_trackbar);
    createTrackbar("High V", window_detection_name, &high_V, max_value, on_high_V_thresh_trackbar);
    Mat frame, frame_HSV, frame_threshold;
    while(true)
    {
        cap >>frame;;
        if(frame.empty())
        {
            break;
        }
        cvtColor(frame, frame_HSV, COLOR_BGR2HSV);
        inRange(frame_HSV, Scalar(low_H, low_S, low_V), Scalar(high_H, high_S, high_V), frame_threshold);
        imshow(window_capture_name, frame);
        imshow(window_detection_name, frame_threshold);
        char key = (char) waitKey(30);
        if(key == 'q' || key ==27)
        {
            break;
        }

    }
    return 0;
}