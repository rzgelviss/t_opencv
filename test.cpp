#include <stdio.h>
#include <opencv2/opencv.hpp>
#include "include/test.h"
using namespace cv;


test::test(/*args*/)
{

}

test::~test()
{
}

void test::show_img(std::string path)
{
    Mat image;
    image = imread(path, 1 );
    if ( !image.data )
    {
        printf("No image data \n");
        return;
    }
    namedWindow("Display Image", WINDOW_AUTOSIZE );
    imshow("Display Image", image);
    waitKey(0);
}

// int main(int argc, char** argv )
// {
//     if ( argc != 2 )
//     {
//         printf("usage: DisplayImage.out <Image_Path>\n");
//         return -1;
//     }
//     test t1;
//     t1.show_img(argv[1]);
//     return 0;
// }