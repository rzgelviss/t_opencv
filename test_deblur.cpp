// #include <iostream>
// #include "opencv2/imgproc.hpp"
// #include "opencv2/imgcodecs.hpp"
// #include <opencv2/core.hpp>
// #include <opencv2/highgui.hpp>

// using namespace cv;
// using namespace std;
// void help();
// void calcPSF(Mat& outputImg, Size filterSize, int R);
// void fftshift(const Mat& inputImg, Mat& outputImg);
// void filter2DFreq(const Mat& inputImg, Mat& outputImg, const Mat& H);
// void calcWnrFilter(const Mat& input_h_PSF, Mat& output_G, double nsr);
// const String keys =
// "{help h usage ? |             | print this message   }"
// "{image          |../b.jpg | input image name     }"
// "{R              |53           | radius               }"
// "{SNR            |5200         | signal to noise ratio}"
// ;
// int main(int argc, char *argv[])
// {
//     help();
//     CommandLineParser parser(argc, argv, keys);
//     if (parser.has("help"))
//     {
//         parser.printMessage();
//         return 0;
//     }
//     int R = parser.get<int>("R");
//     int snr = parser.get<int>("SNR");
//     string strInFileName = parser.get<String>("image");
//     if (!parser.check())
//     {
//         parser.printErrors();
//         return 0;
//     }
//     Mat imgIn;
//     imgIn = imread(strInFileName, IMREAD_GRAYSCALE);
//     if (imgIn.empty()) //check whether the image is loaded or not
//     {
//         cout << "ERROR : Image cannot be loaded..!!" << endl;
//         return -1;
//     }
//     Mat imgOut;
//     // it needs to process even image only
//     Rect roi = Rect(0, 0, imgIn.cols & -2, imgIn.rows & -2);
//     //Hw calculation (start)
//     Mat Hw, h;
//     calcPSF(h, roi.size(), R);
//     calcWnrFilter(h, Hw, 1.0 / double(snr));
//     //Hw calculation (stop)
//     // filtering (start)
//     filter2DFreq(imgIn(roi), imgOut, Hw);
//     // filtering (stop)
//     imgOut.convertTo(imgOut, CV_8U);
//     normalize(imgOut, imgOut, 0, 255, NORM_MINMAX);
//     imwrite("result.jpg", imgOut);
//     imshow("result", imgOut);
//     waitKey(0);
//     return 0;
// }
// void help()
// {
//     cout << "2018-07-12" << endl;
//     cout << "DeBlur_v8" << endl;
//     cout << "You will learn how to recover an out-of-focus image by Wiener filter" << endl;
// }
// void calcPSF(Mat& outputImg, Size filterSize, int R)
// {
//     Mat h(filterSize, CV_32F, Scalar(0));
//     Point point(filterSize.width / 2, filterSize.height / 2);
//     circle(h, point, R, 255, -1, 8);
//     Scalar summa = sum(h);
//     outputImg = h / summa[0];
// }
// void fftshift(const Mat& inputImg, Mat& outputImg)
// {
//     outputImg = inputImg.clone();
//     int cx = outputImg.cols / 2;
//     int cy = outputImg.rows / 2;
//     Mat q0(outputImg, Rect(0, 0, cx, cy));
//     Mat q1(outputImg, Rect(cx, 0, cx, cy));
//     Mat q2(outputImg, Rect(0, cy, cx, cy));
//     Mat q3(outputImg, Rect(cx, cy, cx, cy));
//     Mat tmp;
//     q0.copyTo(tmp);
//     q3.copyTo(q0);
//     tmp.copyTo(q3);
//     q1.copyTo(tmp);
//     q2.copyTo(q1);
//     tmp.copyTo(q2);
// }
// void filter2DFreq(const Mat& inputImg, Mat& outputImg, const Mat& H)
// {
//     Mat planes[2] = { Mat_<float>(inputImg.clone()), Mat::zeros(inputImg.size(), CV_32F) };
//     Mat complexI;
//     merge(planes, 2, complexI);
//     dft(complexI, complexI, DFT_SCALE);
//     Mat planesH[2] = { Mat_<float>(H.clone()), Mat::zeros(H.size(), CV_32F) };
//     Mat complexH;
//     merge(planesH, 2, complexH);
//     Mat complexIH;
//     mulSpectrums(complexI, complexH, complexIH, 0);
//     idft(complexIH, complexIH);
//     split(complexIH, planes);
//     outputImg = planes[0];
// }
// void calcWnrFilter(const Mat& input_h_PSF, Mat& output_G, double nsr)
// {
//     Mat h_PSF_shifted;
//     fftshift(input_h_PSF, h_PSF_shifted);
//     Mat planes[2] = { Mat_<float>(h_PSF_shifted.clone()), Mat::zeros(h_PSF_shifted.size(), CV_32F) };
//     Mat complexI;
//     merge(planes, 2, complexI);
//     dft(complexI, complexI);
//     split(complexI, planes);
//     Mat denom;
//     pow(abs(planes[0]), 2, denom);
//     denom += nsr;
//     divide(planes[0], denom, output_G);
// }


// #include <iostream>
// #include "opencv2/imgproc.hpp"
// #include "opencv2/imgcodecs.hpp"
// #include "opencv2/highgui.hpp"
// using namespace cv;
// using namespace std;
// void help();
// void calcPSF(Mat& outputImg, Size filterSize, int len, double theta);
// void fftshift(const Mat& inputImg, Mat& outputImg);
// void filter2DFreq(const Mat& inputImg, Mat& outputImg, const Mat& H);
// void calcWnrFilter(const Mat& input_h_PSF, Mat& output_G, double nsr);
// void edgetaper(const Mat& inputImg, Mat& outputImg, double gamma = 5.0, double beta = 0.2);
// const String keys =
// "{help h usage ? |             | print this message             }"
// "{image          |../b.jpg    | input image name               }"
// "{LEN            |125          | length of a motion             }"
// "{THETA          |0            | angle of a motion in degrees   }"
// "{SNR            |700          | signal to noise ratio          }"
// ;
// int main(int argc, char *argv[])
// {
//     help();
//     CommandLineParser parser(argc, argv, keys);
//     if (parser.has("help"))
//     {
//         parser.printMessage();
//         return 0;
//     }
//     int LEN = parser.get<int>("LEN");
//     double THETA = parser.get<double>("THETA");
//     int snr = parser.get<int>("SNR");
//     string strInFileName = parser.get<String>("image");
//     if (!parser.check())
//     {
//         parser.printErrors();
//         return 0;
//     }
//     Mat imgIn;
//     imgIn = imread(strInFileName, IMREAD_GRAYSCALE);
//     if (imgIn.empty()) //check whether the image is loaded or not
//     {
//         cout << "ERROR : Image cannot be loaded..!!" << endl;
//         return -1;
//     }
//     Mat imgOut;
//     // it needs to process even image only
//     Rect roi = Rect(0, 0, imgIn.cols & -2, imgIn.rows & -2);
//     //Hw calculation (start)
//     Mat Hw, h;
//     calcPSF(h, roi.size(), LEN, THETA);
//     calcWnrFilter(h, Hw, 1.0 / double(snr));
//     //Hw calculation (stop)
//     imgIn.convertTo(imgIn, CV_32F);
//     edgetaper(imgIn, imgIn);
//     // filtering (start)
//     filter2DFreq(imgIn(roi), imgOut, Hw);
//     // filtering (stop)
//     imgOut.convertTo(imgOut, CV_8U);
//     normalize(imgOut, imgOut, 0, 255, NORM_MINMAX);
//     imwrite("result.jpg", imgOut);
//     imshow("result", imgOut);
//     waitKey(0);
//     return 0;
// }
// void help()
// {
//     cout << "2018-08-14" << endl;
//     cout << "Motion_deblur_v2" << endl;
//     cout << "You will learn how to recover an image with motion blur distortion using a Wiener filter" << endl;
// }
// void calcPSF(Mat& outputImg, Size filterSize, int len, double theta)
// {
//     Mat h(filterSize, CV_32F, Scalar(0));
//     Point point(filterSize.width / 2, filterSize.height / 2);
//     ellipse(h, point, Size(0, cvRound(float(len) / 2.0)), 90.0 - theta, 0, 360, Scalar(255), FILLED);
//     Scalar summa = sum(h);
//     outputImg = h / summa[0];
// }
// void fftshift(const Mat& inputImg, Mat& outputImg)
// {
//     outputImg = inputImg.clone();
//     int cx = outputImg.cols / 2;
//     int cy = outputImg.rows / 2;
//     Mat q0(outputImg, Rect(0, 0, cx, cy));
//     Mat q1(outputImg, Rect(cx, 0, cx, cy));
//     Mat q2(outputImg, Rect(0, cy, cx, cy));
//     Mat q3(outputImg, Rect(cx, cy, cx, cy));
//     Mat tmp;
//     q0.copyTo(tmp);
//     q3.copyTo(q0);
//     tmp.copyTo(q3);
//     q1.copyTo(tmp);
//     q2.copyTo(q1);
//     tmp.copyTo(q2);
// }
// void filter2DFreq(const Mat& inputImg, Mat& outputImg, const Mat& H)
// {
//     Mat planes[2] = { Mat_<float>(inputImg.clone()), Mat::zeros(inputImg.size(), CV_32F) };
//     Mat complexI;
//     merge(planes, 2, complexI);
//     dft(complexI, complexI, DFT_SCALE);
//     Mat planesH[2] = { Mat_<float>(H.clone()), Mat::zeros(H.size(), CV_32F) };
//     Mat complexH;
//     merge(planesH, 2, complexH);
//     Mat complexIH;
//     mulSpectrums(complexI, complexH, complexIH, 0);
//     idft(complexIH, complexIH);
//     split(complexIH, planes);
//     outputImg = planes[0];
// }
// void calcWnrFilter(const Mat& input_h_PSF, Mat& output_G, double nsr)
// {
//     Mat h_PSF_shifted;
//     fftshift(input_h_PSF, h_PSF_shifted);
//     Mat planes[2] = { Mat_<float>(h_PSF_shifted.clone()), Mat::zeros(h_PSF_shifted.size(), CV_32F) };
//     Mat complexI;
//     merge(planes, 2, complexI);
//     dft(complexI, complexI);
//     split(complexI, planes);
//     Mat denom;
//     pow(abs(planes[0]), 2, denom);
//     denom += nsr;
//     divide(planes[0], denom, output_G);
// }
// void edgetaper(const Mat& inputImg, Mat& outputImg, double gamma, double beta)
// {
//     int Nx = inputImg.cols;
//     int Ny = inputImg.rows;
//     Mat w1(1, Nx, CV_32F, Scalar(0));
//     Mat w2(Ny, 1, CV_32F, Scalar(0));
//     float* p1 = w1.ptr<float>(0);
//     float* p2 = w2.ptr<float>(0);
//     float dx = float(2.0 * CV_PI / Nx);
//     float x = float(-CV_PI);
//     for (int i = 0; i < Nx; i++)
//     {
//         p1[i] = float(0.5 * (tanh((x + gamma / 2) / beta) - tanh((x - gamma / 2) / beta)));
//         x += dx;
//     }
//     float dy = float(2.0 * CV_PI / Ny);
//     float y = float(-CV_PI);
//     for (int i = 0; i < Ny; i++)
//     {
//         p2[i] = float(0.5 * (tanh((y + gamma / 2) / beta) - tanh((y - gamma / 2) / beta)));
//         y += dy;
//     }
//     Mat w = w2 * w1;
//     multiply(inputImg, w, outputImg);
// }


// #include <iostream>
// #include "opencv2/imgproc.hpp"
// #include "opencv2/imgcodecs.hpp"
// #include "opencv2/highgui.hpp"
// using namespace cv;
// using namespace std;
// void calcGST(const Mat& inputImg, Mat& imgCoherencyOut, Mat& imgOrientationOut, int w);
// int main()
// {
//     int W = 52;             // window size is WxW
//     double C_Thr = 0.43;    // threshold for coherency
//     int LowThr = 35;        // threshold1 for orientation, it ranges from 0 to 180
//     int HighThr = 57;       // threshold2 for orientation, it ranges from 0 to 180
//     Mat imgIn = imread("../b.jpg", IMREAD_GRAYSCALE);
//     if (imgIn.empty()) //check whether the image is loaded or not
//     {
//         cout << "ERROR : Image cannot be loaded..!!" << endl;
//         return -1;
//     }
//     Mat imgCoherency, imgOrientation;
//     calcGST(imgIn, imgCoherency, imgOrientation, W);
//     Mat imgCoherencyBin;
//     imgCoherencyBin = imgCoherency > C_Thr;
//     Mat imgOrientationBin;
//     inRange(imgOrientation, Scalar(LowThr), Scalar(HighThr), imgOrientationBin);
//     Mat imgBin;
//     imgBin = imgCoherencyBin & imgOrientationBin;
//     normalize(imgCoherency, imgCoherency, 0, 255, NORM_MINMAX);
//     normalize(imgOrientation, imgOrientation, 0, 255, NORM_MINMAX);
//     imwrite("result.jpg", 0.5*(imgIn + imgBin));
//     imwrite("Coherency.jpg", imgCoherency);
//     imwrite("Orientation.jpg", imgOrientation);
//     imshow ("result",0.5*(imgIn + imgBin));
//     imshow("Coherency",imgCoherency);
//     imshow("Orientation",imgOrientation);
//     waitKey(0);
//     return 0;
// }
// void calcGST(const Mat& inputImg, Mat& imgCoherencyOut, Mat& imgOrientationOut, int w)
// {
//     Mat img;
//     inputImg.convertTo(img, CV_32F);
//     // GST components calculation (start)
//     // J =  (J11 J12; J12 J22) - GST
//     Mat imgDiffX, imgDiffY, imgDiffXY;
//     Sobel(img, imgDiffX, CV_32F, 1, 0, 3);
//     Sobel(img, imgDiffY, CV_32F, 0, 1, 3);
//     multiply(imgDiffX, imgDiffY, imgDiffXY);
//     Mat imgDiffXX, imgDiffYY;
//     multiply(imgDiffX, imgDiffX, imgDiffXX);
//     multiply(imgDiffY, imgDiffY, imgDiffYY);
//     Mat J11, J22, J12;      // J11, J22 and J12 are GST components
//     boxFilter(imgDiffXX, J11, CV_32F, Size(w, w));
//     boxFilter(imgDiffYY, J22, CV_32F, Size(w, w));
//     boxFilter(imgDiffXY, J12, CV_32F, Size(w, w));
//     // GST components calculation (stop)
//     // eigenvalue calculation (start)
//     // lambda1 = 0.5*(J11 + J22 + sqrt((J11-J22)^2 + 4*J12^2))
//     // lambda2 = 0.5*(J11 + J22 - sqrt((J11-J22)^2 + 4*J12^2))
//     Mat tmp1, tmp2, tmp3, tmp4;
//     tmp1 = J11 + J22;
//     tmp2 = J11 - J22;
//     multiply(tmp2, tmp2, tmp2);
//     multiply(J12, J12, tmp3);
//     sqrt(tmp2 + 4.0 * tmp3, tmp4);
//     Mat lambda1, lambda2;
//     lambda1 = tmp1 + tmp4;
//     lambda1 = 0.5*lambda1;      // biggest eigenvalue
//     lambda2 = tmp1 - tmp4;
//     lambda2 = 0.5*lambda2;      // smallest eigenvalue
//     // eigenvalue calculation (stop)
//     // Coherency calculation (start)
//     // Coherency = (lambda1 - lambda2)/(lambda1 + lambda2)) - measure of anisotropism
//     // Coherency is anisotropy degree (consistency of local orientation)
//     divide(lambda1 - lambda2, lambda1 + lambda2, imgCoherencyOut);
//     // Coherency calculation (stop)
//     // orientation angle calculation (start)
//     // tan(2*Alpha) = 2*J12/(J22 - J11)
//     // Alpha = 0.5 atan2(2*J12/(J22 - J11))
//     phase(J22 - J11, 2.0*J12, imgOrientationOut, true);
//     imgOrientationOut = 0.5*imgOrientationOut;
//     // orientation angle calculation (stop)
// }

#include <iostream>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
using namespace cv;
using namespace std;
void fftshift(const Mat& inputImg, Mat& outputImg);
void filter2DFreq(const Mat& inputImg, Mat& outputImg, const Mat& H);
void synthesizeFilterH(Mat& inputOutput_H, Point center, int radius);
void calcPSD(const Mat& inputImg, Mat& outputImg, int flag = 0);
int main()
{
    Mat imgIn = imread("../b.jpg", IMREAD_GRAYSCALE);
    if (imgIn.empty()) //check whether the image is loaded or not
    {
        cout << "ERROR : Image cannot be loaded..!!" << endl;
        return -1;
    }
    imgIn.convertTo(imgIn, CV_32F);
    // it needs to process even image only
    Rect roi = Rect(0, 0, imgIn.cols & -2, imgIn.rows & -2);
    imgIn = imgIn(roi);
    // PSD calculation (start)
    Mat imgPSD;
    calcPSD(imgIn, imgPSD);
    fftshift(imgPSD, imgPSD);
    normalize(imgPSD, imgPSD, 0, 255, NORM_MINMAX);
    // PSD calculation (stop)
    //H calculation (start)
    Mat H = Mat(roi.size(), CV_32F, Scalar(1));
    const int r = 21;
    synthesizeFilterH(H, Point(705, 458), r);
    synthesizeFilterH(H, Point(850, 391), r);
    synthesizeFilterH(H, Point(993, 325), r);
    //H calculation (stop)
    // filtering (start)
    Mat imgOut;
    fftshift(H, H);
    filter2DFreq(imgIn, imgOut, H);
    // filtering (stop)
    imgOut.convertTo(imgOut, CV_8U);
    normalize(imgOut, imgOut, 0, 255, NORM_MINMAX);
    imwrite("result.jpg", imgOut);
    imwrite("PSD.jpg", imgPSD);
    fftshift(H, H);
    normalize(H, H, 0, 255, NORM_MINMAX);
    imwrite("filter.jpg", H);
    imshow("result", imgOut);
    imshow("PSD", imgPSD);
    imshow("filter", H);
    waitKey(0);
    return 0;
}
void fftshift(const Mat& inputImg, Mat& outputImg)
{
    outputImg = inputImg.clone();
    int cx = outputImg.cols / 2;
    int cy = outputImg.rows / 2;
    Mat q0(outputImg, Rect(0, 0, cx, cy));
    Mat q1(outputImg, Rect(cx, 0, cx, cy));
    Mat q2(outputImg, Rect(0, cy, cx, cy));
    Mat q3(outputImg, Rect(cx, cy, cx, cy));
    Mat tmp;
    q0.copyTo(tmp);
    q3.copyTo(q0);
    tmp.copyTo(q3);
    q1.copyTo(tmp);
    q2.copyTo(q1);
    tmp.copyTo(q2);
}
void filter2DFreq(const Mat& inputImg, Mat& outputImg, const Mat& H)
{
    Mat planes[2] = { Mat_<float>(inputImg.clone()), Mat::zeros(inputImg.size(), CV_32F) };
    Mat complexI;
    merge(planes, 2, complexI);
    dft(complexI, complexI, DFT_SCALE);
    Mat planesH[2] = { Mat_<float>(H.clone()), Mat::zeros(H.size(), CV_32F) };
    Mat complexH;
    merge(planesH, 2, complexH);
    Mat complexIH;
    mulSpectrums(complexI, complexH, complexIH, 0);
    idft(complexIH, complexIH);
    split(complexIH, planes);
    outputImg = planes[0];
}
void synthesizeFilterH(Mat& inputOutput_H, Point center, int radius)
{
    Point c2 = center, c3 = center, c4 = center;
    c2.y = inputOutput_H.rows - center.y;
    c3.x = inputOutput_H.cols - center.x;
    c4 = Point(c3.x,c2.y);
    circle(inputOutput_H, center, radius, 0, -1, 8);
    circle(inputOutput_H, c2, radius, 0, -1, 8);
    circle(inputOutput_H, c3, radius, 0, -1, 8);
    circle(inputOutput_H, c4, radius, 0, -1, 8);
}
// Function calculates PSD(Power spectrum density) by fft with two flags
// flag = 0 means to return PSD
// flag = 1 means to return log(PSD)
void calcPSD(const Mat& inputImg, Mat& outputImg, int flag)
{
    Mat planes[2] = { Mat_<float>(inputImg.clone()), Mat::zeros(inputImg.size(), CV_32F) };
    Mat complexI;
    merge(planes, 2, complexI);
    dft(complexI, complexI);
    split(complexI, planes);            // planes[0] = Re(DFT(I)), planes[1] = Im(DFT(I))
    planes[0].at<float>(0) = 0;
    planes[1].at<float>(0) = 0;
    // compute the PSD = sqrt(Re(DFT(I))^2 + Im(DFT(I))^2)^2
    Mat imgPSD;
    magnitude(planes[0], planes[1], imgPSD);        //imgPSD = sqrt(Power spectrum density)
    pow(imgPSD, 2, imgPSD);                         //it needs ^2 in order to get PSD
    outputImg = imgPSD;
    // logPSD = log(1 + PSD)
    if (flag)
    {
        Mat imglogPSD;
        imglogPSD = imgPSD + Scalar::all(1);
        log(imglogPSD, imglogPSD);
        outputImg = imglogPSD;
    }
}