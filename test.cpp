// #include<boost/filesystem.hpp>
#include<iostream>
#include <string>
#include<fstream>
// using namespace boost;
#include <opencv2/calib3d.hpp>
#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>  
#include <opencv2/core/core.hpp>  
#include <opencv2/highgui/highgui.hpp>  
#include <opencv2/imgproc/imgproc.hpp>  

using namespace std;
using namespace cv;
cv::Point2f GetPointAfterRotate(cv::Point2f inputpoint, cv::Point2f center, double angle)
{
	angle = angle /180.0 * CV_PI;
    cv::Point2d preturn;
    preturn.x = (inputpoint.x - center.x) * cos(-angle) - (inputpoint.y - center.y) * sin(-angle) + center.x;
    preturn.y = (inputpoint.x - center.x) * sin(-angle) + (inputpoint.y - center.y) * cos(-angle) + center.y;
    return preturn;
}



// 图像旋转
///@ angle 要旋转的角度
void Rotate(const Mat& srcImage, Mat& destImage, double angle)
{
	Point2f center(srcImage.cols / 2, srcImage.rows / 2);//中心
	Mat M = getRotationMatrix2D(center, angle, 1);//计算旋转的仿射变换矩阵 
	warpAffine(srcImage, destImage, M, Size(srcImage.cols, srcImage.rows));//仿射变换  
	circle(destImage, center, 2, Scalar(255, 0, 0));
}

int main()
{
	std::string str = "a.jpg";
	//初始化输入图像
	cv::Mat srcImage = imread(str);
	//初始化输入缩放图像
	cv::Mat srcImage_zoom;
	Size srcImageSize = Size(960, 600);  //填入任意指定尺寸
	resize(srcImage, srcImage_zoom, srcImageSize, 0, 0, INTER_LINEAR);
	//显示输入缩放图像
	if (!srcImage_zoom.data)
		return -1;
	circle(srcImage_zoom,Point2d(300, 200), 2, (0,255,255), 5);
	
	Point2d preturn = GetPointAfterRotate(Point2f(300, 200), Point2f(960/2, 300), -30);

	imshow("原始图像-缩放", srcImage_zoom);
	//初始化输出图像
	Mat destImage0;
	Mat destImage1;
	//初始化输出缩放图像
	Mat destImage_zoom;
	//镜像图片操作
	// flip(srcImage, destImage0, 1);
	//旋转图片操作
	double angle = -30;//角度
	Rotate(srcImage_zoom, destImage1, angle);
	circle(destImage1,preturn, 2, (0,255,0), 5);

	//显示输出缩放图像
	// resize(destImage1, destImage_zoom, srcImageSize, 0, 0, INTER_LINEAR);
	imshow("输出图像-缩放", destImage1);
	//调整输出图片的格式及质量
	std::vector<int> compression_params;
	compression_params.push_back(IMWRITE_JPEG_QUALITY);  //选择jpeg
	compression_params.push_back(100); //在这个填入你要的图片质量
	imwrite("output.jpg", destImage1, compression_params);

	waitKey(0);

	return 0;
}












// int main() {



	
// 	std::string in_param_file = "/home/twc/catkin_ws/src/image_recognition/config/yaml11/intrinsic_parameters_mr813.yaml";
//  cv::FileStorage fs(in_param_file, cv::FileStorage::READ);
// 	std::cout << fs.isOpened() << std::endl;
    //  unsigned char buffer_map[100000];
    // using namespace std;
    // fstream file1("/home/twc/default_map.bin", ios::in | ios::binary);
    // if (!file1)
    // {
    //     printf("%s", "open file1 error");
    //     return 0;
    // }
    // file1.seekg(0, std::ios::end);
    // size_t read_len1 = file1.tellg();
	// cout <<read_len1<<endl;
    // file1.seekg(0);
    // file1.read(reinterpret_cast<char *>(buffer_map), read_len1);
    // file1.close();
    // // tuya_debug_hex_dump("default map  compress: ", 0, buffer_map, read_len1);
    // cout << endl;
	// for(int i; i<sizeof(buffer_map); i++)
	// {
	// 	// cout <<buffer_map[i] <<" " <<endl;
	// 	printf("%02x ", buffer_map[i]);
	// }
    // int i=4;
    // std::string s2 =(std::string)("房间");
    // cout <<"s2: "<< s2 <<endl;
	// filesystem::path path("/home/twc/module");
	// // path /= "module";
    // cout <<path<<endl;
    // boost::filesystem::path::iterator pathI = path.begin();
    // // while (pathI != path.end())
    // // {
    // //     // undefined
    // //     std::cout << *pathI << std::endl;
    // //     ++pathI;
    // // }
	// cout << path.filename() << endl;
	// cout << path.relative_path() << endl;
	// filesystem::directory_iterator end;
	// for (filesystem::directory_iterator it(path); it != end; it++) {
	// 	cout << *it << endl;
	// }
	// // path /= "Debug";
	// cout << path.string() << endl;
	// cout << filesystem::current_path() << endl;;
	
	// for (filesystem::directory_iterator it(path); it != end; it++) 
	// {
	// 	if(filesystem::is_regular(it->path()))
	// 		cout << it->path() <<"="<< filesystem::file_size((it->path()))/1024.0/1024.0<< endl;
	// 	else {
	// 		cout << "directory:" << it->path().string() << endl;
	// 	}
		
	// }
	// path.append("\\cheji1a\\jiajia1");
	// cout << "path1" << path << endl;
	// if(!filesystem::exists(path))
	// 	filesystem::create_directories(path);
	// else{
	// 	cout << "path is exists...." << endl;
	// }
//     return 0;
// }