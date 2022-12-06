// #include "opencv2/imgproc.hpp"
// #include "opencv2/imgcodecs.hpp"
// #include "opencv2/highgui.hpp"
// #include <iostream>
// using namespace cv;
// using namespace std;
// int main( int argc, char** argv )
// {
//   cv::CommandLineParser parser(argc, argv,
//                                "{@input   |../a.jpg|input image}"
//                                "{ksize   k|1|ksize (hit 'K' to increase its value at run time)}"
//                                "{scale   s|1|scale (hit 'S' to increase its value at run time)}"
//                                "{delta   d|0|delta (hit 'D' to increase its value at run time)}"
//                                "{help    h|false|show help message}");
//   cout << "The sample uses Sobel or Scharr OpenCV functions for edge detection\n\n";
//   parser.printMessage();
//   cout << "\nPress 'ESC' to exit program.\nPress 'R' to reset values ( ksize will be -1 equal to Scharr function )";
//   // First we declare the variables we are going to use
//   Mat image,src, src_gray;
//   Mat grad;
//   const String window_name = "Sobel Demo - Simple Edge Detector";
//   int ksize = parser.get<int>("ksize");
//   int scale = parser.get<int>("scale");
//   int delta = parser.get<int>("delta");
//   int ddepth = CV_16S;
//   String imageName = parser.get<String>("@input");
//   // As usual we load our source image (src)
//   image = imread( samples::findFile( imageName ), IMREAD_COLOR ); // Load an image
//   // Check if image is loaded fine
//   if( image.empty() )
//   {
//     printf("Error opening image: %s\n", imageName.c_str());
//     return EXIT_FAILURE;
//   }
//   for (;;)
//   {
//     // Remove noise by blurring with a Gaussian filter ( kernel size = 3 )
//     GaussianBlur(image, src, Size(3, 3), 0, 0, BORDER_DEFAULT);
//     // Convert the image to grayscale
//     cvtColor(src, src_gray, COLOR_BGR2GRAY);
//     Mat grad_x, grad_y;
//     Mat abs_grad_x, abs_grad_y;
//     Sobel(src_gray, grad_x, ddepth, 1, 0, ksize, scale, delta, BORDER_DEFAULT);
//     Sobel(src_gray, grad_y, ddepth, 0, 1, ksize, scale, delta, BORDER_DEFAULT);
//     // converting back to CV_8U
//     convertScaleAbs(grad_x, abs_grad_x);
//     convertScaleAbs(grad_y, abs_grad_y);
//     addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad);
//     imshow(window_name, grad);
//     char key = (char)waitKey(0);
//     if(key == 27)
//     {
//       return EXIT_SUCCESS;
//     }
//     if (key == 'k' || key == 'K')
//     {
//       ksize = ksize < 30 ? ksize+2 : -1;
//     }
//     if (key == 's' || key == 'S')
//     {
//       scale++;
//     }
//     if (key == 'd' || key == 'D')
//     {
//       delta++;
//     }
//     if (key == 'r' || key == 'R')
//     {
//       scale =  1;
//       ksize = -1;
//       delta =  0;
//     }
//   }
//   return EXIT_SUCCESS;
// }


#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include <iostream>
using namespace cv;
Mat src, src_gray;
Mat dst, detected_edges;
int lowThreshold = 0;
const int max_lowThreshold = 100;
const int ratio = 3;
const int kernel_size = 3;
const char* window_name = "Edge Map";
static void CannyThreshold(int, void*)
{
    blur( src_gray, detected_edges, Size(3,3) );
    Canny( detected_edges, detected_edges, lowThreshold, lowThreshold*ratio, kernel_size );
    dst = Scalar::all(0);
    src.copyTo( dst, detected_edges);
    imshow( window_name, dst );
}
int main( int argc, char** argv )
{
  CommandLineParser parser( argc, argv, "{@input | ../a.jpg | input image}" );
  src = imread( samples::findFile( parser.get<String>( "@input" ) ), IMREAD_COLOR ); // Load an image
  if( src.empty() )
  {
    std::cout << "Could not open or find the image!\n" << std::endl;
    std::cout << "Usage: " << argv[0] << " <Input image>" << std::endl;
    return -1;
  }
  dst.create( src.size(), src.type() );
  cvtColor( src, src_gray, COLOR_BGR2GRAY );
  namedWindow( window_name, WINDOW_AUTOSIZE );
  createTrackbar( "Min Threshold:", window_name, &lowThreshold, max_lowThreshold, CannyThreshold );
  CannyThreshold(0, 0);
  waitKey(0);
  return 0;
}