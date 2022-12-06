#include <fstream>
#include <iostream>
#include "include/md5.hpp"
#include <string>

using namespace std;

int main()
{
    // char buffer1[] = "ABCE";

    // int read_len = strlen(buffer1);
    // std::cout << read_len<< std::endl;
    // fstream file2("../test.bin", ios::out | ios::binary);

    // if (!file2)
    // {
    //     printf("Error opening file");
    //     return 0;
    // }
    // file2.write(reinterpret_cast<char *>(buffer1), read_len);
    // file2.close();

    char buffer_map[100];
    fstream file1("../test.bin", ios::in | ios::binary);
    if (!file1)
    {
        printf("%s", "open file1 error");
        return 0;
    }
    file1.seekg(0, std::ios::end);
    int read_len1 = file1.tellg();
    file1.seekg(0);
    file1.read(reinterpret_cast<char *>(buffer_map), read_len1);
    file1.close();
    buffer_map[read_len1] = 0;
    std::cout << read_len1 <<std::endl;
    std::cout <<buffer_map<<std::endl;
    cout << strlen(buffer_map) << endl;

    char *a = new char[100];
    memcpy(a, &buffer_map[1], 3);
    cout << a <<endl;
    delete [] a;

    // // char *buffer_map1 = "ABCE";
    // // std::cout <<buffer_map1 << std::endl;
    // std::string hash = websocketpp::md5::md5_hash_hex(std::string(buffer_map));
    // std::cout << hash <<std::endl;


// writing on a text file

ofstream out("out.txt");
if (out.is_open()) 
{
    out << "This is a line.\n";
    out << "This is another line.\n";
    out.close();
}



// reading a text file
#include <stdlib.h>

char buffer[256];
ifstream in("out.txt");
if (! in.is_open())
{ cout << "Error opening file"; exit (1); }
while (!in.eof() )
{
    in.getline (buffer,100);
    cout << buffer << endl;
}


  std::ofstream outfile;
  outfile.open ("test.txt");

  outfile.write ("This is an apple1aa\nabcdefgh",28);
  long pos = outfile.tellp();
  outfile.seekp (pos-7);
  outfile.write (" sam",4);

  outfile.close();


//假设test.txt中的内容是HelloWorld
ifstream fin("test.txt",ios::in);
cout << fin.tellg() << endl;//输出0,流置针指向文本中的第一个字符，类似于数组的下标0

char c;
fin >> c;
fin.tellg();//输出为1,因为上面把fin的第一个字符赋值给了c,同时指针就会向后 移动一个字节（注意是以一个字节为单位移动）指向第二个字符
cout << fin.tellg() << endl;
fin.seekg(0,ios::end);//输出10,注意最后一个字符d的下标是9，而ios::end指向的是最后一个字符的下一个位置

fin.seekg(10,ios::beg);//和上面一样，也到达了尾后的位置

//我们发现利用这个可以算出文件的大小

int m,n;
// m = fin.seekg(0,ios::beg);
// n =  fin.seekg(0,ios::end);
//那么n-m就是文件的所占的字节数

// 我们也可以从文件末尾出发，反向移动流指针，
fin.seekg(-10,ios::end);//回到了第一个字符



//     std::ifstream ifs;

//   ifs.open ("test.txt", std::ifstream::in);

//   char c = ifs.get();

//   while (ifs.good()) {
//     std::cout << c << endl;;
//     c = ifs.get();
//   }

//   ifs.close();




    return 0;
}
