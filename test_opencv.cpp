#include <stdio.h>
#include <opencv2/opencv.hpp>
#include "include/test.h"
#include <opencv2/core.hpp>
// #include <boost/thread/xtime.hpp>
#include <unistd.h>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <stdio.h>
#include <time.h>
#include <cstring>
using namespace cv;
using namespace std;

// static void help(char *progName)
// {
//      cout << endl
//           << "This program shows how to filter images with mask: the write it yourself and the"
//           << "filter2d way." << endl
//           << "Usage:" << endl
//           << progName << "[image_path --default a.jpg] [G --grayscale] " << endl
//           << endl;
// }

static void help(char **av)
{
     cout << endl
          << av[0] << " shows the usage of the OpenCV serialization functionality." << endl
          << "usage: " << endl
          << av[0] << " outputfile.yml.gz" << endl
          << "The output file may be either XML (xml) or YAML (yml/yaml). You can even compress it by "
          << "specifying this in its extension like xml.gz yaml.gz etc... " << endl
          << "With FileStorage you can serialize objects in OpenCV by using the << and >> operators" << endl
          << "For example: - create a class and have it serialized" << endl
          << "             - use it to read and write matrices." << endl;
}

void Sharpen(const Mat &myImage, Mat &Result)
{
     CV_Assert(myImage.depth() == CV_8U);
     const int nChannels = myImage.channels();
     cout << myImage.depth() << endl
          << nChannels << endl
          << myImage.size() << endl
          << myImage.type() << endl;
     Result.create(myImage.size(), myImage.type());
     for (int j = 1; j < myImage.rows - 1; ++j)
     {
          const uchar *previous = myImage.ptr<uchar>(j - 1);
          const uchar *current = myImage.ptr<uchar>(j);
          const uchar *next = myImage.ptr<uchar>(j + 1);
          uchar *output = Result.ptr<uchar>(j);
          for (int i = nChannels; i < nChannels * (myImage.cols - 1); ++i)
          {
               *output++ = saturate_cast<uchar>(5 * current[i] - current[i - nChannels] - current[i + nChannels] - previous[i] - next[i]);
          }
     }
     Result.row(0).setTo(Scalar(0));
     Result.row(Result.rows - 1).setTo(Scalar(0));
     Result.col(0).setTo(Scalar(0));
     Result.col(Result.cols - 1).setTo(Scalar(0));
}

// function getLocalTime(timezone) {
//       if (typeof timezone !== "number") {
//         return new Date();
//       }
//       var d = new Date();
//       var len = d.getTime();
//       var offset = d.getTimezoneOffset() * 60000;
//       var utcTime = len + offset;
//       return new Date(utcTime + 3600000 * timezone);
//     }

// CString GetTimeZoneNow() {          TIME_ZONE_INFORMATION   tzi;

// GetSystemTime(&tzi.StandardDate);

// GetTimeZoneInformation(&tzi);

// CString   strStandName   =   tzi.StandardName;

// CString   strDaylightName   =   tzi.DaylightName;

// int zone = tzi.Bias/ -60; //时区，如果是中国标准时间则得到8

// return strStandName; }

class MyData
{
private:
     /* data */
public:
     MyData() : A(0), X(0), id() {}
     explicit MyData(int) : A(97), X(CV_PI), id("mydata1234") {}
     void write(FileStorage &fs) const
     {
          fs << "{"
             << "A" << A << "X" << X << "id" << id << "}";
     }
     void read(const FileNode &node)
     {
          A = (int)node["A"];
          X = (double)node["X"];
          id = (string)node["id"];
     }

public:
     int A;
     double X;
     string id;
     // ~MyData();
};

static void read(const FileNode &node, MyData &x, const MyData &default_value = MyData())
{
     if (node.empty())
          x = default_value;
     else
          x.read(node);
}

static ostream &operator<<(ostream &out, const MyData &m)
{
     out << "{id = " << m.id << ",";
     out << "X = " << m.X << ", ";
     out << "A = " << m.A << "}";
     return out;
}



void conv_seq(Mat src, Mat &dst, Mat kernel)
{
     int rows = src.rows, cols = src.cols;
     dst = Mat(rows, cols, src.type());
     int sz = kernel.rows / 2;
     copyMakeBorder(src, src, sz, sz, sz, sz, BORDER_REPLICATE);
     for (int i = 0; i < rows; i++)
     {
          uchar *dptr = dst.ptr(i);
          for (int j = 0; j < cols; j++)
          {
               double value = 0;
               for (int k = -sz; k <= sz; k++)
               {
                    uchar *sptr = src.ptr(i + sz + k);
                    for (int l = -sz; l <= sz; l++)
                    {
                         value += kernel.ptr<double>(k + sz)[l + sz] * sptr[j + sz + l];
                    }
               }
               dptr[j] = saturate_cast<uchar>(value);
          }
     }
}

class parallelConvolution : public ParallelLoopBody
{
     private:
          Mat m_src, &m_dst;
          Mat m_kernel;
          int sz;
     public:
          parallelConvolution(Mat src, Mat &dst, Mat kernel):m_src(src), m_dst(dst), m_kernel(kernel)
          {
               sz = kernel.rows / 2;
          }
          virtual void operator()(const Range &range) const CV_OVERRIDE 
          {
               for(int r = range.start; r < range.end; r++)
               {
                    int i = r/m_src.cols, j = r%m_src.cols;
                    double value = 0;
                    for(int k = -sz; k<=sz; k++)
                    {
                         uchar *sptr = const_cast<uchar*>(m_src.ptr(i+sz+k));
                         for(int l = -sz; l<=sz;l++)
                         {
                              value += m_kernel.ptr<double>(k+sz)[l+sz] * sptr[j+sz+l];
                         }
                    }
                    m_dst.ptr(i)[j] = saturate_cast<uchar>(value);
               }
          }
};

// int main()
// int main(int argc, char **argv)
int main(int ac, char **av)
{


     

     //vectorizing your code using universal intrinsics
     // v_uint8 a;
     // v_int32x8 a;
     // int n = a.nlanes;
     // cout << n <<endl;
     // float ptr[4] = {1, 2, 3, 32};   // ptr is a pointer to a contiguous memory block of 32 floats

     // // Variable Sized Registers //
     // int x = v_float32x4().nlanes;  
     // v_float32x4(1, 2, 3, 4);




//      Mat src = imread("../a.jpg");
//      Mat& dst = src;
//      Mat kernel = Mat_<uchar>::eye(3, 3);
//      int rows = src.rows;
//      int cols = src.cols;
//      parallelConvolution obj(src, dst, kernel);
//     parallel_for_(Range(0, rows * cols), obj);
//     imshow("src", src);
//     imshow("dst", dst);
//     waitKey(0);
     //      //file input output using xml and yaml files
     //      if (ac != 2)
     //     {
     //         help(av);
     //         return 1;
     //     }
     //     string filename = av[1];
     //     { //write
     //         Mat R = Mat_<uchar>::eye(3, 3),
     //             T = Mat_<double>::zeros(3, 1);
     //         MyData m(1);
     //         FileStorage fs(filename, FileStorage::WRITE);
     //         // or:
     //         // FileStorage fs;
     //         // fs.open(filename, FileStorage::WRITE);
     //         fs << "iterationNr" << 100;
     //         fs << "strings" << "[";                              // text - string sequence
     //         fs << "image1.jpg" << "Awesomeness" << "../data/baboon.jpg";
     //         fs << "]";                                           // close sequence
     //         fs << "Mapping";                              // text - mapping
     //         fs << "{" << "One" << 1;
     //         fs <<        "Two" << 2 << "}";
     //         fs << "R" << R;                                      // cv::Mat
     //         fs << "T" << T;
     //      //    fs << "MyData" << m;                                // your own data structures
     //         fs.release();                                       // explicit close
     //         cout << "Write Done." << endl;
     //     }
     //     {//read
     //         cout << endl << "Reading: " << endl;
     //         FileStorage fs;
     //         fs.open(filename, FileStorage::READ);
     //         int itNr;
     //         //fs["iterationNr"] >> itNr;
     //         itNr = (int) fs["iterationNr"];
     //         cout << itNr;
     //         if (!fs.isOpened())
     //         {
     //             cerr << "Failed to open " << filename << endl;
     //             help(av);
     //             return 1;
     //         }
     //         FileNode n = fs["strings"];                         // Read string sequence - Get node
     //         if (n.type() != FileNode::SEQ)
     //         {
     //             cerr << "strings is not a sequence! FAIL" << endl;
     //             return 1;
     //         }
     //         FileNodeIterator it = n.begin(), it_end = n.end(); // Go through the node
     //         for (; it != it_end; ++it)
     //             cout << (string)*it << endl;
     //         n = fs["Mapping"];                                // Read mappings from a sequence
     //         cout << "Two  " << (int)(n["Two"]) << "; ";
     //         cout << "One  " << (int)(n["One"]) << endl << endl;
     //         MyData m;
     //         Mat R, T;
     //         fs["R"] >> R;                                      // Read cv::Mat
     //         fs["T"] >> T;
     //         fs["MyData"] >> m;                                 // Read your own structure_
     //         cout << endl
     //             << "R = " << R << endl;
     //         cout << "T = " << T << endl << endl;
     //         cout << "MyData = " << endl << m << endl << endl;
     //         //Show default behavior for non existing nodes
     //         cout << "Attempt to read NonExisting (should initialize the data structure with its default).";
     //         fs["NonExisting"] >> m;
     //         cout << endl << "NonExisting = " << endl << m << endl;
     //     }
     //     cout << endl
     //         << "Tip: Open up " << filename << " with a text editor to see the serialized data." << endl;

     // new Date().getTimezoneOffset()/60;

     // // 获取系统时间
     // 	time_t _rt = time(NULL);
     // // 系统时间转换为GMT时间
     // tm _gtm = *gmtime(&_rt);
     // // 系统时间转换为本地时间
     // tm _ltm = *localtime(&_rt);
     // printf("UTC:       %s", asctime(&_gtm));
     // printf("local:     %s", asctime(&_ltm));
     // // 再将GMT时间重新转换为系统时间
     // time_t _gt = mktime(&_gtm);
     // tm _gtm2 = *localtime(&_gt);
     // // 这时的_gt已经与实际的系统时间_rt有时区偏移了,计算两个值的之差就是时区偏的秒数,除60就是分钟
     // int offset = ((_rt - _gt ) + (_gtm2.tm_isdst ? 3600 : 0)) / 60;
     // printf(" offset (minutes) %d", offset);

     // unsigned char a[2] = {0xfe, 0x28};
     // cout << (int)a[0] << " " << (int)a[1] << endl;
     // cout << (uint)a[0] << " " << (uint)a[1] << endl;
     // int a = -2;
     // char b = a;
     // // cout << b << endl;
     // printf("%x\n", b);
     // 65536 - 65534 = 2;
     // cout << static_cast<uint>(a) << endl;
     // cout << (uint)a << endl;

     // help(*argv);
     // const char* filename = argc >=2 ? argv[1]:"../a.jpg";
     // Mat I = imread(samples::findFile(filename), IMREAD_GRAYSCALE);
     // if(I.empty())
     // {
     //      cout << "Error opening image" << endl;
     //      return EXIT_FAILURE;
     // }

     // Mat padded;
     // int m = getOptimalDFTSize(I.rows);
     // int n = getOptimalDFTSize(I.cols);
     // cout << "m, n" << m << "," << n << "row, col" << I.rows << ", " << I.cols<<endl;
     // cout << I.channels()<< endl;
     // copyMakeBorder(I, padded, 0, m-I.rows, 0, n-I.cols, BORDER_CONSTANT, Scalar::all(0));
     // // imshow("pad", padded);
     // // waitKey(0);
     // Mat planes[] = {Mat_<float>(padded), Mat::zeros(padded.size(), CV_32F)};
     // // imshow("planes", planes[0]);
     // // waitKey(0);
     // Mat complexI;
     // merge(planes, 2, complexI);
     // cout << "merge" <<endl;
     // dft(complexI, complexI);
     // cout << "dft" << endl;
     // // cout << complexI<<endl;
     // cout <<complexI.rows <<", " << complexI.cols << ", " << complexI.channels() <<  endl;
     // split(complexI, planes);
     // magnitude(planes[0], planes[1], planes[0]);
     // Mat magI = planes[0];
     // magI += Scalar::all(1);
     // log(magI, magI);
     // magI = magI(Rect(0, 0, magI.cols & -2, magI.rows & -2));
     // // imshow("magI", magI);
     // // waitKey(0);
     // int cx = magI.cols/2;
     // int cy = magI.rows/2;
     // Mat q0(magI, Rect(0, 0, cx, cy));
     // Mat q1(magI, Rect(cx, 0, cx, cy));
     // Mat q2(magI, Rect(0, cy,cx,cy));
     // Mat q3(magI, Rect(cx, cy, cx, cy));
     // Mat tmp;
     // q0.copyTo(tmp);
     // q3.copyTo(q0);
     // tmp.copyTo(q3);

     // q1.copyTo(tmp);
     // q2.copyTo(q1);
     // tmp.copyTo(q2);

     // normalize(magI, magI, 0, 1, NORM_MINMAX);
     // imshow("Input Image", I);
     // imshow("spectrum magnitude", magI);
     // waitKey();

     // Mat lookUpTable(1,256,CV_8U);
     // uchar* p = lookUpTable.ptr();
     // for(int i=0; i<256;i++)
     // {
     //      p[i] = saturate_cast<uchar>(pow(i/255.0, gamma_)*255.0);
     // }
     // Mat res = img.clone();
     // LUT(img, lookUpTable, res);

     // CommandLineParser parser(argc, argv, "{@input | a1.jpg | input image}");
     // Mat image = imread(samples::findFile(parser.get<String>("@input")));
     // if (image.empty())
     // {
     //      cout << "Could not open or find the image!\n"
     //           << endl;
     //      cout << "Usage: " << argv[0] << "Input image" << endl;
     //      return -1;
     // }

     // Mat new_image = Mat::zeros(image.size(), image.type());
     // cout << image.type() << endl;
     // double alpha = 1.0;
     // int beta = 0;
     // cout << "basic Linear Transforms" << endl;
     // cout << "----------" << endl;
     // cout << "Enter the alpha value[1.0-3.0]: ";
     // cin >> alpha;
     // cout << "Enter the beta[0-100]:";
     // cin >> beta;
     // for (int y = 0; y < image.rows; y++)
     // {
     //      for (int x = 0; x < image.cols; x++)
     //      {
     //           for (int c = 0; c < image.channels(); c++)
     //           {
     //                new_image.at<Vec3b>(y, x)[c] = saturate_cast<uchar>(alpha * image.at<Vec3b>(y, x)[c] + beta);
     //           }
     //      }
     // }
     // imshow("Original Image", image);
     // imshow("New Image", new_image);
     // waitKey();

     // const char* filename = "../a.jpg";
     // double alpha = 0.5; double beta; double input;
     // Mat src1, src2, dst;
     // cout << "Simple linear Blender" << endl;
     // cout << "----------" << endl;
     // cout << "Enter alpha [0.0-1.0]: ";
     // cin >> input;
     // if(input >=0 && input <=1)
     // {
     //      alpha = input;
     // }
     // src1 = imread(samples::findFile("../a.jpg"));
     // // cout << src1<<endl;
     // src2 = imread(samples::findFile("../a.jpg"));
     // // Rect r(10, 10, 10,10);
     // // src2 = src1(r);
     // // Mat D (src1, Rect(2, 3, 4, 4) );
     // // cout <<D<<endl;

     // if(src1.empty()) {cout << "Error loading src1" <<endl; return EXIT_FAILURE;}
     // if(src2.empty()) {cout << "Error loading src2" << endl; return EXIT_FAILURE;}
     // beta = (1.0 - alpha);
     // addWeighted(src1, alpha, src2, beta, 0.0, dst);
     // imshow("Linear blend", dst);
     // waitKey(0);

     // Mat img = imread(filename, IMREAD_COLOR);
     // imwrite("a1.jpg", img);
     // Scalar intensity = img.at<uchar>(300,400);
     // cout << intensity << endl;
     // Scalar intensity1 = img.at<uchar>(Point(300, 400));
     // cout << intensity1 << endl;
     // Vec3b intensity2 = img.at<Vec3b>(300, 400);
     // cout << intensity2<<endl;
     // uchar blue = intensity2.val[0];
     // cout << (int)blue << endl;
     // Vec3f intensity3 = img.at<Vec3f>(300, 400);
     // cout <<intensity3 <<endl;
     // float blue_ = intensity3.val[0];
     // cout << blue_ <<endl;
     // img.at<uchar>(300, 400) = 100;
     // cout <<intensity3 <<endl;
     // Scalar intensity4 = img.at<uchar>(300,400);
     // cout << intensity4 << endl;
     // std::vector<Point3f>points;

     // points.push_back(Point3f(1.0, 2.0, 3.0));
     // points.push_back(Point3f(1.0, 2.0, 3.0));
     // Mat PointsMat = Mat(points).reshape(1);
     // cout << PointsMat<<endl;
     // Mat img1 = PointsMat.clone();
     // Mat sobelx;
     // Sobel(img1, sobelx, CV_32F, 1, 0);
     // cout << sobelx<< endl;

     // img1 = Scalar(0);
     // cout << img1 << endl;
     // Rect r(0, 0, 1,2);
     // Mat smallImage = img1(r);
     // cout << smallImage<<endl;
     // Mat grey;
     // cvtColor(img, grey, COLOR_BGR2GRAY);
     // // cout << grey<<endl;
     // Mat dst;
     // grey.convertTo(dst, CV_32F);
     // // cout << dst <<endl;

     // Mat draw, sobel1;
     // Sobel(grey, sobel1, CV_32F, 1, 0);
     // double minVal, maxVal;
     // minMaxLoc(sobelx, &minVal, &maxVal);
     // sobel1.convertTo(draw, CV_8U, 255.0/(maxVal - minVal), -minVal*255.0/(maxVal-minVal));
     // namedWindow("image", WINDOW_AUTOSIZE);
     // imshow("image", draw);
     // waitKey();

     // help("test_opencv");
     // const char* filename = "../a.jpg";
     // Mat src, dst0, dst1;
     // if(!strcmp("G", "BGR"))
     //      {
     //           cout <<filename<<endl;
     //           src = imread(samples::findFile(filename), IMREAD_GRAYSCALE);
     //           // src = imread("../a.jpg");
     //           // cout << src << endl;
     //           // cout << format(src, Formatter::FMT_PYTHON) <<endl;
     //      }
     // else
     //      src = imread(samples::findFile(filename), IMREAD_COLOR);

     // if(src.empty())
     // {
     //      cerr << "Can't open image[" << filename << "]" <<endl;
     //      return EXIT_FAILURE;
     // }
     // namedWindow("Input", WINDOW_AUTOSIZE);
     // namedWindow("Output", WINDOW_AUTOSIZE);

     // imshow("Input", src);
     // double t = (double)getTickCount();

     // Sharpen(src, dst0);
     // t = ((double)getTickCount() -t)/getTickFrequency();
     // cout << "Hand written function time passed in seconds: " << t << endl;
     // imshow("Outputdst0", dst0);
     // waitKey();
     // Mat kernel = (Mat_<char>(3,3) << 0, -1, 0, -1, 5, -1, 0, -1, 0);
     // t = (double)getTickCount();
     // filter2D(src, dst1, src.depth(), kernel);
     // t = ((double)getTickCount() -t)/getTickFrequency();
     // cout <<"Built-in filter 2D time passed in seconds: " << t <<endl;
     // imshow("Outputdst1", dst1);
     // waitKey(1);
     // return EXIT_SUCCESS;

     // Mat A, C;
     // Mat M(2,2, CV_8UC3, Scalar(0,0,255));
     // cout << "M = " << endl << " " << M << endl << endl;
     // C = imread("../a.jpg");
     // cout << C << endl;
     // Mat M(10,10, CV_8UC1, Scalar(255));
     // cout << "M = " << endl << " " << M << endl << endl;
     // Mat D (M, Rect(2, 3, 4, 4) ); // using a rectangle
     // Mat E = M(Range::all(), Range(1,3)); // using row and column boundaries
     // cout << D << endl;
     // cout << E <<endl;

     // Mat F = D.clone();
     // Mat G;
     // D.copyTo(G);
     // cout << F << endl;
     // cout << G << endl;
     // int sz[3] = {2,2,2};
     // // Mat L(3,sz, CV_8UC(1), Scalar::all(0));
     // Mat M;
     // M.create(4,4, CV_8UC(2));
     // cout << "M " << endl <<" " << M <<endl;
     // Mat E = Mat::eye(4, 4, CV_64F);
     // cout << "E = " << endl << " " << E << endl << endl;
     // Mat O = Mat::ones(2, 2, CV_32F);
     // cout << "O = " << endl << " " << O << endl << endl;
     // Mat Z = Mat::zeros(3,3, CV_8UC1);
     // cout << "Z = " << endl << " " << Z << endl << endl;

     //     Mat C = (Mat_<double>(3, 3) << 0, -1, 0, -1, 5, -1, 0, -1, 0);
     //     cout << "C = " << endl
     //          << " " << C << endl
     //          << endl;
     //     C = (Mat_<double>({0, -1, 0, -1, 5, -1, 0, -1, 0})).reshape(3);
     //     cout << "C = " << endl
     //          << " " << C << endl
     //          << endl;

     //     Mat RowClone = C.row(1).clone();
     //     cout << "RowClone = " << endl
     //          << " " << RowClone << endl
     //          << endl;
     //     Mat R = Mat(3, 2, CV_8UC3);
     //     randu(R, Scalar::all(0), Scalar::all(255));
     //     cout << "R = " << endl
     //          << " " << R << endl
     //          << endl;
     //     cout << "R (python)  = " << endl
     //          << format(R, Formatter::FMT_PYTHON) << endl
     //          << endl;
     //     Point2f P(5, 1);
     //     cout << "Point (2D) = " << P << endl
     //          << endl;
     //     Point3f P3f(2, 6, 7);
     //     cout << "Point (3D) = " << P3f << endl
     //          << endl;

     //     vector<float> v;
     //     v.push_back((float)CV_PI);
     //     v.push_back(2);
     //     v.push_back(3.01f);
     //     cout << "Vector of floats via Mat = " << Mat(v) << endl
     //          << endl;

     //     vector<Point2f> vPoints(20);
     //     for (size_t i = 0; i < vPoints.size(); ++i)
     //     {
     //         vPoints[i] = Point2f(float(i * 5), (float)(i % 7));
     //     }
     //     // cout << "A VECTOR OF 2D Points = " << vPoints << endl
     //     //      << endl;

     //     int divideWith = 0;
     //     stringstream s;
     //     s << 2;
     //     s >> divideWith;
     //     if (!s || !divideWith)
     //     {
     //         cout << "invalid number entered for dividing" << endl;
     //         return -1;
     //     }
     //     uchar table[256];
     //     for (int i = 0; i < 256; ++i)
     //     {
     //         table[i] = (uchar)(divideWith * (i / divideWith));
     //     }
     //     // for(auto data:table)
     //     //     cout << " " << data;

     //     double t = (double)getTickCount();
     //     sleep(1);
     //     sleep(1.5);

     //     usleep(100000);
     //     t = ((double)getTickCount() -t)/getTickFrequency();
     //     cout << "Times passed in seconds: " << t <<endl;

     //     std::this_thread::sleep_for(std::chrono::milliseconds(100));

     return 0;
}