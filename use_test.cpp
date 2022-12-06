#include <stdio.h>
#include <opencv2/opencv.hpp>
#include "include/test.h"
#include <string>

#include <iostream>
#include <cassert>
#include <vector>
#include <string>
#include "boost/regex.hpp"

using namespace cv;
using namespace std;

// std::vector<std::string> split(std::string str,std::string s)
// {
//     boost::regex reg(s.c_str());
//     std::vector<std::string> vec;
//     boost::sregex_token_iterator it(str.begin(),str.end(),reg,-1);
//     boost::sregex_token_iterator end;
//     while(it!=end)
//     {
//         vec.push_back(*it++);
//     }
//     return vec;
// }

std::vector<std::string> stringsplit(const std::string &str, const char *delim)
{
    std::vector<std::string> strlist;
    int size = str.size();
    char *input = new char[size + 1];
    strcpy(input, str.c_str());
    char *token = std::strtok(input, delim);
    while (token != NULL)
    {
        strlist.push_back(token);
        token = std::strtok(NULL, delim);
    }
    delete[] input;
    return strlist;
}

int main(int argc, char** argv )
{
    if ( argc != 2 )
    {
        printf("usage: DisplayImage.out <Image_Path>\n");
        return -1;
    }
    // test t1;
    // t1.show_img(argv[1]);
    
    // char license[100] = "[uuid:adf][authkey:qwe123]";
    char buffer[] = {0x05, 0x5a, 0x01,0x01,0x00,0x3f,0x00,0x0e,0x5b,0x75,0x75,0x69,0x64,0x3a,0x74,0x75,0x79,0x61,0x33,0x33,0x65,0x61,0x31,0x34,0x37,0x33,
    0x66,0x38,0x37,0x39,0x39,0x61,0x32,0x36,0x5d,0x5b,0x6b,0x65,0x79,0x3a,0x78,0x76,0x67,0x62,0x51,0x4f,0x46,0x78,0x4d,0x4d,0x6d,0x7a,0x43,0x50,0x6b,0x7a,
    0x41,0x55,0x43,0x48,0x65,0x54,0x61,0x75,0x6b,0x4e,0x71,0x47,0x65,0x35,0x70,0x61,0x5d};
    // 73 27 38
    char license[buffer[5]] = {0};
    memcpy(license, &buffer[8], buffer[5]);
    const char *d1 = "[]";
    char *p1;

    std::vector<std::string> list = stringsplit(std::string(license), "[]:");
    vector<std::string>::iterator it; 
    for(it = list.begin(); it!=list.end();it++)
    {
        std::cout<<*it<<std::endl;
    }
    
    // p1 = strtok(license, d1);
    // char *filename;
    // while (p1)
    // {
    //     printf("%s\n", p1);
    //     if (string(p1).find("uuid:") != string::npos)
    //     {
    //         printf("golden=%s\n", p1);
    //         filename = &p1[5];
    //         cout << filename << endl;
    //     }
    //     else if(string(p1).find("key:") != string::npos)
    //     {
    //         filename = &p1[5];
    //         cout <<filename <<endl;
    //     }
    //     p1 = strtok(NULL, d1);
    // }



    // std::string str,s;
    // str="sss/ddd/ggg/hh";
    // s="/";
    // std::vector<std::string> vec=split(str,s);
    // for(int i=0,size=vec.size();i<size;i++)
    // {
    //     std::cout<<vec[i]<<std::endl;
    // }
    // std::cin.get();
    // std::cin.get();

    return 0;
}