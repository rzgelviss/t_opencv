#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include <iostream>

using namespace cv;
using namespace std;

Mat src, erosion_dst, dilation_dst, dst;

int erosion_elem = 0;
int erosion_size = 0;
int dilation_elem = 0;
int dilation_size = 0;
int const max_elem = 2;
int const max_kernel_size = 21;

void Erosion(int, void*);
void Dilation(int, void*);

int morph_elem = 0;
int morph_size = 0;
int morph_operator = 0;
int const max_operator = 4;
// int const max_elem = 2;
// int const max_kernel_size = 21;
const char* window_name = "Morphology Transformations Demo";

void Morphology_Operations(int, void*);
void show_wait_destroy(const char* winname, cv::Mat img) {
    imshow(winname, img);
    moveWindow(winname, 500, 0);
    waitKey(0);
    destroyWindow(winname);
}



int main(int argc, char** argv)
{

    const char* window_name = "Paramids Demo";
    cout <<"\n Zoom In-Out demo \n"
        "-----------------------------\n"
        "*[i] ->Zoom in \n"
        "*[o] ->Zoom out \n"
        "[ESC] ->Close program \n" << endl;
    const char* filename = argc>=2?argv[1]:"chicky_512.png";
    Mat src = imread(samples::findFile(filename));
    if(src.empty())
    {
        printf("Error opening image\n");
        printf("Program Srguments:[image_name --default chicky_512.png]\n");
        return EXIT_FAILURE;
    }

    for(;;)
    {
        imshow(window_name, src);
        char c = (char)waitKey(0);
        if(c==27)
        {
            break;
        }
        else if(c == 'i')
        {
            pyrUp(src, src ,Size(src.cols*2, src.rows*2));
            printf("** Zoom In:Image x2 \n");
        }
        else if(c=='o')
        {
            pyrDown(src, src, Size(src.cols/2, src.rows/2));
            printf("** Zoom Out:Image /2 \n");
        }
    }

    // cout << "\n Zoom In-Out demo \n "
    //         "------------------  \n"
    //         " * [i] -> Zoom in   \n"
    //         " * [o] -> Zoom out  \n"
    //         " * [ESC] -> Close program \n" << endl;
    // const char* filename = argc >=2 ? argv[1] : "chicky_512.png";
    // // Loads an image
    // Mat src = imread( samples::findFile( filename ) );
    // // Check if image is loaded fine
    // if(src.empty()){
    //     printf(" Error opening image\n");
    //     printf(" Program Arguments: [image_name -- default chicky_512.png] \n");
    //     return EXIT_FAILURE;
    // }
    // for(;;)
    // {
    //     imshow( window_name, src );
    //     char c = (char)waitKey(0);
    //     if( c == 27 )
    //     { break; }
    //     else if( c == 'i' )
    //     { pyrUp( src, src, Size( src.cols*2, src.rows*2 ) );
    //         printf( "** Zoom In: Image x 2 \n" );
    //     }
    //     else if( c == 'o' )
    //     { pyrDown( src, src, Size( src.cols/2, src.rows/2 ) );
    //         printf( "** Zoom Out: Image / 2 \n" );
    //     }
    // }

    // CommandLineParser parser(argc, argv, "{@input | notes.png | input image}");
    // Mat src = imread( samples::findFile( parser.get<String>("@input") ), IMREAD_COLOR);
    // if (src.empty())
    // {
    //     cout << "Could not open or find the image!\n" << endl;
    //     cout << "Usage: " << argv[0] << " <Input image>" << endl;
    //     return -1;
    // }

    // imshow("src", src);
    // Mat gray;
    // if(src.channels() == 3)
    // {
    //     cvtColor(src, gray, COLOR_BGR2GRAY);
    // }
    // else gray = src;
    // show_wait_destroy("gray", gray);
    // Mat bw;
    // adaptiveThreshold(~gray, bw, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 15, -2);
    // show_wait_destroy("binary", bw);
    // Mat horizontal = bw.clone();
    // Mat vertical = bw.clone();
    // int horizontal_size = horizontal.cols / 30;
    // Mat horizontalStructure = getStructuringElement(MORPH_RECT, Size(horizontal_size, 1));
    // erode(horizontal, horizontal, horizontalStructure, Point(-1, -1));
    // dilate(horizontal, horizontal, horizontalStructure, Point(-1, -1));
    // show_wait_destroy("horizontal", horizontal);
    // int vertical_size = vertical.rows /30 ;
    // Mat verticalStructure = getStructuringElement(MORPH_RECT, Size(1,vertical_size));
    // erode(vertical, vertical, verticalStructure, Point(-1, -1));
    // dilate(vertical, vertical, verticalStructure, Point(-1, -1));
    // show_wait_destroy("vertical", vertical);
    // bitwise_not(vertical, vertical);
    // show_wait_destroy("vertical_bit", vertical);
    // Mat edges;
    // adaptiveThreshold(vertical, edges, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 3, -2);
    // show_wait_destroy("edges", edges);
    // Mat kernel = Mat::ones(2,2, CV_8UC1);
    // dilate(edges, edges, kernel);
    // show_wait_destroy("dilate", edges);
    // Mat smooth;
    // vertical.copyTo(smooth);
    // blur(smooth, smooth, Size(2,2));
    // smooth.copyTo(vertical, edges);
    // show_wait_destroy("smooth-final", vertical);



    




    //     Mat input_image = (Mat_<uchar>(8, 8) <<
    //     0, 0, 0, 0, 0, 0, 0, 0,
    //     0, 255, 255, 255, 0, 0, 0, 255,
    //     0, 255, 255, 255, 0, 0, 0, 0,
    //     0, 255, 255, 255, 0, 255, 0, 0,
    //     0, 0, 255, 0, 0, 0, 0, 0,
    //     0, 0, 255, 0, 0, 255, 255, 0,
    //     0, 255, 0, 255, 0, 0, 255, 0,
    //     0, 255, 255, 255, 0, 0, 0, 0);
    // Mat kernel = (Mat_<int>(3, 3) <<
    //     0, 1, 0,
    //     1, -1, 1,
    //     0, 1, 0);
    // Mat output_image;
    // morphologyEx(input_image, output_image, MORPH_HITMISS, kernel);
    // const int rate = 50;
    // kernel = (kernel + 1) * 127;
    // kernel.convertTo(kernel, CV_8U);
    // resize(kernel, kernel, Size(), rate, rate, INTER_NEAREST);
    // imshow("kernel", kernel);
    // moveWindow("kernel", 0, 0);
    // resize(input_image, input_image, Size(), rate, rate, INTER_NEAREST);
    // imshow("Original", input_image);
    // moveWindow("Original", 0, 200);
    // resize(output_image, output_image, Size(), rate, rate, INTER_NEAREST);
    // imshow("Hit or Miss", output_image);
    // moveWindow("Hit or Miss", 500, 200);
    // waitKey(0);

    // CommandLineParser parser(argc, argv, "{@input | ../a.jpg | input image}");
    // src = imread(samples::findFile(parser.get<String>("@input")), IMREAD_COLOR);
    // if(src.empty())
    // {
    //     cout <<"Could not open or find the image!\n" <<endl;
    //     cout << "Usage: " << argv[0] <<"Input image>" <<endl;
    //     return -1;
    // }

    // namedWindow("Erosion Demo", WINDOW_AUTOSIZE);
    // namedWindow("Dilation Demo", WINDOW_AUTOSIZE);
    // moveWindow("Dilation Demo", src.cols, 0);

    // createTrackbar("Element:\n 0: Rect \n 1: Cross\n2:Ellipse", "Erosion Demo", &erosion_elem, max_elem, Erosion);
    // createTrackbar("Kernel size:\n 2n+1", "Erosion Demo", &erosion_size, max_kernel_size, Erosion);

    // createTrackbar( "Element:\n 0: Rect \n 1: Cross \n 2: Ellipse", "Dilation Demo",
    //     &dilation_elem, max_elem,
    //     Dilation );

    // createTrackbar( "Kernel size:\n 2n +1", "Dilation Demo",
    //         &dilation_size, max_kernel_size,
    //         Dilation );
    // Erosion( 0, 0 );
    // Dilation( 0, 0 );

    // namedWindow( window_name, WINDOW_AUTOSIZE ); // Create window
    // createTrackbar("Operator:\n 0: Opening - 1: Closing  \n 2: Gradient - 3: Top Hat \n 4: Black Hat", window_name, &morph_operator, max_operator, Morphology_Operations );
    // createTrackbar( "Element:\n 0: Rect - 1: Cross - 2: Ellipse", window_name,
    //                 &morph_elem, max_elem,
    //                 Morphology_Operations );
    // createTrackbar( "Kernel size:\n 2n +1", window_name,
    //                 &morph_size, max_kernel_size,
    //                 Morphology_Operations );
    // Morphology_Operations( 0, 0 );
    // waitKey(0);
    return 0;



}

void Morphology_Operations( int, void* )
{
  // Since MORPH_X : 2,3,4,5 and 6
  int operation = morph_operator + 2;
  Mat element = getStructuringElement( morph_elem, Size( 2*morph_size + 1, 2*morph_size+1 ), Point( morph_size, morph_size ) );
  morphologyEx( src, dst, operation, element );
  imshow( window_name, dst );
}

void Erosion(int, void*)
{
    int erosion_type = 0;
    if(erosion_elem == 0)
    {
        erosion_type = MORPH_RECT;
    }
    else if(erosion_elem==1)
    {
        erosion_type = MORPH_CROSS;
    }
    else if(erosion_elem == 2)
    {
        erosion_type = MORPH_ELLIPSE;
    }

    Mat element = getStructuringElement(erosion_type, Size(2*erosion_size +1, 2*erosion_size+1), Point(erosion_size, erosion_size));
    erode(src, erosion_dst, element);
    imshow("Erosion Demo", erosion_dst);
}

void Dilation(int, void*)
{
    int dilation_type = 0;
    if(dilation_elem ==0)
    {
        dilation_type = MORPH_RECT;
    }
    else if(dilation_elem==1)
    {
        dilation_type = MORPH_CROSS;
    }
    else if(dilation_elem==2)
    {
        dilation_type = MORPH_ELLIPSE;
    }
    
    Mat element = getStructuringElement( dilation_type,
                        Size( 2*dilation_size + 1, 2*dilation_size+1 ),
                        Point( dilation_size, dilation_size ) );
    dilate( src, dilation_dst, element );
    imshow( "Dilation Demo", dilation_dst );
}