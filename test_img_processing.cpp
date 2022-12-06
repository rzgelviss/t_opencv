#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>

// #define w 400
using namespace cv;
using namespace std;

void MyEllipse(Mat img, double angle);
void MyFilledCircle(Mat img, Point center);
void MyPolygon(Mat img);
void MyLine(Mat img, Point start, Point end);

bool line(int h, int w, int h1, int w1, int h2, int w2)
{
    float k = (w2 - w1) / (h2 - h1);
    float b = w1 - k * h1;
    return (h*k +b) <= w;
}

void cut(Mat img, Point Point_A, Point Point_B, Mat &up, Mat &down)
{
    int h1 = Point_A.y, w1 = Point_A.x;
    int h2 = Point_B.y, w2 = Point_B.x;
    int height = img.rows, width = img.cols;
    up = cv::Mat::zeros(height, width, CV_8UC1);
    down = cv::Mat::zeros(height, width, CV_8UC1);
    cout << __LINE__ << endl;
    for(int i = 0; i<height; i++)
    {
        for(int j = 0; j<width;j++)
        {
            if(line(i, j, h1, w1, h2, w2))
            {
                up.at<uint8_t>(i,j) = img.at<uint8_t>(i,j);
                // cout << __LINE__ << " " << i << " "<< j << endl;
            }
            else
            {
                down.at<uint8_t>(i,j) = img.at<uint8_t>(i,j);
                // cout << __LINE__ << " "<< i << " "<< j<< endl;
            }

        }
    }

}

int main(void)
{

    //斜线分割
    string path = "../b.jpg";
    Mat img = imread(path, IMREAD_GRAYSCALE);
    cout << img.size << endl;
    Point point_A = Point(200, 200);
    Point point_B = Point(200, 400);
    cv::line(img, point_A, point_B, Scalar(255), 1, 8);
    cout << __LINE__ << endl;
    Mat up;
    Mat down;
    cut(img, point_A, point_B, up,down);
    cout << __LINE__ << endl;
    putText( img, "OpenCV forever!", point_A, FONT_HERSHEY_COMPLEX, 3,
        Scalar(0, 0, 255), 5, 8 );
    imshow("img", img);
    imshow("up", up);
    imshow("down", down);
    waitKey(0);
    destroyAllWindows();




    // char atom_window[] = "Drawing 1:Atop";
    // char rook_window[] = "Drawing 2:Rook";
    // Mat atom_image = Mat::zeros(w,w,CV_8UC3);
    // Mat rook_image = Mat::zeros(w,w,CV_8UC3);

    // MyEllipse(atom_image, 90);
    // MyEllipse(atom_image, 0);
    // MyEllipse(atom_image, 45);
    // MyEllipse(atom_image, -45);

    // MyPolygon(rook_image);

    // rectangle(rook_image, Point(0, 7*w/8), Point(w,w), Scalar(0, 255,255), FILLED, LINE_8);

    // MyLine(rook_image, Point(0, 15*w/16), Point(w, 15*w/16));
    // MyLine( rook_image, Point( w/4, 7*w/8 ), Point( w/4, w ) );
    // MyLine( rook_image, Point( w/2, 7*w/8 ), Point( w/2, w ) );
    // MyLine( rook_image, Point( 3*w/4, 7*w/8 ), Point( 3*w/4, w ) );

    // imshow(atom_window, atom_image);
    // moveWindow(atom_window, 0,200);
    // imshow(rook_window, rook_image);
    // moveWindow(rook_window, w, 200);

    // waitKey(0);
    // return(0);

}

// void MyEllipse(Mat img, double angle)
// {
//     int thickness = 2;
//     int lineType = 8;
//     ellipse(img, Point(w/2, w/2), Size(w/4, w/16), angle, 0, 360, Scalar(255, 0, 0), thickness, lineType);
// }

// void MyFilledCircle(Mat img, Point center)
// {
//     circle(img, center, w/32, Scalar(0, 0, 255), FILLED, LINE_8);
// }

// void MyPolygon( Mat img )
// {
//   int lineType = LINE_8;
//   Point rook_points[1][20];
//   rook_points[0][0]  = Point(    w/4,   7*w/8 );
//   rook_points[0][1]  = Point(  3*w/4,   7*w/8 );
//   rook_points[0][2]  = Point(  3*w/4,  13*w/16 );
//   rook_points[0][3]  = Point( 11*w/16, 13*w/16 );
//   rook_points[0][4]  = Point( 19*w/32,  3*w/8 );
//   rook_points[0][5]  = Point(  3*w/4,   3*w/8 );
//   rook_points[0][6]  = Point(  3*w/4,     w/8 );
//   rook_points[0][7]  = Point( 26*w/40,    w/8 );
//   rook_points[0][8]  = Point( 26*w/40,    w/4 );
//   rook_points[0][9]  = Point( 22*w/40,    w/4 );
//   rook_points[0][10] = Point( 22*w/40,    w/8 );
//   rook_points[0][11] = Point( 18*w/40,    w/8 );
//   rook_points[0][12] = Point( 18*w/40,    w/4 );
//   rook_points[0][13] = Point( 14*w/40,    w/4 );
//   rook_points[0][14] = Point( 14*w/40,    w/8 );
//   rook_points[0][15] = Point(    w/4,     w/8 );
//   rook_points[0][16] = Point(    w/4,   3*w/8 );
//   rook_points[0][17] = Point( 13*w/32,  3*w/8 );
//   rook_points[0][18] = Point(  5*w/16, 13*w/16 );
//   rook_points[0][19] = Point(    w/4,  13*w/16 );
//   const Point* ppt[1] = { rook_points[0] };
//   int npt[] = { 20 };
//   fillPoly( img,
//         ppt,
//         npt,
//         1,
//         Scalar( 255, 255, 255 ),
//         lineType );
// }

// void MyLine(Mat img, Point start, Point end)
// {
//     int thickness = 2;
//     int lineType = LINE_8;
//     line(img, start, end, Scalar(0, 0, 0), thickness, lineType);
// }
