/*********************************************************************
Author:Li Junshi
File Decription: Info Area Extract funtion of each class of invoice
File Create Time: 2017-06-01
Please send email to lijunshi2015@163.com if you any question.
*********************************************************************/


#include <iostream>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <set>

using namespace std;
using namespace cv;

extern int g_info_area_count;

void func(Mat& img,Mat&src)
{
	int thresh = 0.6 * src.rows;  //归一化阈值
	cout << "yuzhi:" << thresh << endl;
	int* h2 = new int[img.cols * 4];
	set<int> x_pos;
	memset(h2, 0, img.cols * 4);
	for (int i = 0; i < img.cols; i+=2) //行
	{
		for (int j = 0; j < img.rows; j++)      //列
		{
			if (img.at<uchar>(j, i) == 0)      //统计的是黑色像素的数量
				h2[i]++;
		}
		cout << "h2: " << h2[i] << "  pos: " << i << endl;
		if (h2[i] > thresh)
		{
			circle(img, Point(i, 10), 5, Scalar(0, 0, 255), -1);
			x_pos.insert(i);
		}
	}

	//imshow("tras", img);

	if (!x_pos.size())  //没有竖线，直接返回原图
	{
		char file[20];
		memset(file, 0, sizeof(file));
		sprintf_s(file, "info%d.jpg", ++g_info_area_count);
		imwrite(file, src);
		return;
	}


#if 1
	int begin = 0;
	int count = 0;
	for (set<int>::iterator it = x_pos.begin(); it != x_pos.end(); it++)
	{
		
/*
		if (begin == 0)
		{
			begin = *it;
			continue;
		}
*/

		//if (count == 2)
		if(1)
		{
			int len = *it - begin;
			if (len < 10)
			{
				continue;
			}
			count++;

			int height = img.rows;
			char file[20];
			memset(file, 0, sizeof(file));
			sprintf_s(file, "info%d.jpg", ++g_info_area_count);
			imwrite(file,src(Rect(begin+2, 0, len-2, height)));
			//imwrite("test.jpg", tmp);
		}

		if (count == x_pos.size())
		{
			int len = src.cols - *it;
			int height = img.rows;
			char file[20];
			memset(file, 0, sizeof(file));
			sprintf_s(file, "info%d.jpg", ++g_info_area_count);
			imwrite(file, src(Rect(*it, 0, len, height)));
		}
		
		begin = *it;
	}
#endif
	delete [] h2;
}



/*广东省通用机打发票*/
void Class7InfoAreaExtract(Mat& src)
{
	resize(src, src, Size(1160, 817));
	imwrite("7p.bmp", src);
	Mat src_gray, src_binary;

	//转化为灰度图像
	cvtColor(src, src_gray, CV_RGB2GRAY);
	//二值化图像
	adaptiveThreshold(src_gray, src_binary, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY, 25, 10);
	int* v = new int[src.cols * 4];
	int* h = new int[src.rows * 4];
	cout << "src.cols = " << src.cols << endl;
	cout << "src.rows = " << src.rows << endl;
	memset(v, 0, src.cols * 4);
	memset(h, 0, src.rows * 4);
	int i, j;
	//方法一遍历
	//垂直方向进行累加（积分）
	int max_x = 0;
	int pos_x = 0;
	int max2_x = 0;
	int pos2_x = 0;
	//注意i的步长，设置这样的步长是为了不把一些临近的像素高的点包括进来，否则会造成点重叠
	//注意i的起点和终点，都+或-了50，这是为了避免发票边缘影响像素检测
	for (i = 50; i<src_binary.cols - 50; i += 3) //列
	{
		for (j = 0; j<src_binary.rows; j++)      //行
		{
			if (src_binary.at<uchar>(j, i) == 0)      //统计的是黑色像素的数量
				v[i]++;
		}
		//cout << "v: " << v[i] << "  pos: " << i << endl;
		//找出top2的像素位置
		if (max2_x < v[i])
		{
			max2_x = v[i];
			pos2_x = i;
			if (max_x < max2_x)
			{
				swap(max_x, max2_x);
				swap(pos_x, pos2_x);
			}
		}
	}

	//水平方向进行累加（积分）
	int max_y = 0;
	int pos_y = 0;
	int max2_y = 0;
	int pos2_y = 0;
	for (i = 50; i<src_binary.rows - 50; i += 3) //行
	{
		for (j = 0; j<src_binary.cols; j++)      //列
		{
			if (src_binary.at<uchar>(i, j) == 0)   //统计黑色像素的数量
				h[i]++;
		}

		//cout << "h :" << h[i] << "  pos: " << i << endl;
		if (max2_y < h[i])
		{
			max2_y = h[i];
			pos2_y = i;
			if (max_y < max2_y)
			{
				swap(max_y, max2_y);
				swap(pos_y, pos2_y);
			}
		}
	}


	cout << "max x = " << max_x << endl;
	cout << "max2 x = " << max2_x << endl;
	cout << "max y = " << max_y << endl;
	cout << "max2 y = " << max2_y << endl;
	cout << "pos x = " << pos_x << endl;
	cout << "pos2 x = " << pos2_x << endl;
	cout << "pos y = " << pos_y << endl;
	cout << "pos2 y = " << pos2_y << endl;


	int bottom_y = MAX(pos_y, pos2_y);
	int left_x = MIN(pos_x, pos2_x);
	int head_y = MIN(pos_y, pos2_y);
	int right_x = MAX(pos_x, pos2_x);

	Point StartPoint(left_x, head_y); //这是发票的原点
									  //标出发票的四个关键点
	circle(src, StartPoint, 5, Scalar(0, 0, 255), -1);
	circle(src, Point(pos2_x, pos2_y), 5, Scalar(0, 0, 255), -1);
	circle(src, Point(pos_x, pos2_y), 5, Scalar(0, 0, 255), -1);
	circle(src, Point(pos2_x, pos_y), 5, Scalar(0, 0, 255), -1);
	//circle(src, Point(5, src.rows / 3), 5, Scalar(0, 255, 0), -1);
	//circle(src, Point(src.cols / 3, 5), 5, Scalar(0, 255, 0), -1);

	/*这些参数都要根据每一类的发票位置的信息区域来调整*/
	/*区域一*/
	int offset_x = 3;
	int offset_y = 8;
	int InfoLength = 910;
	int InfoHeigh = 230;

	Rect InfoArea1(left_x + offset_x, head_y + offset_y, InfoLength, InfoHeigh);
	rectangle(src, InfoArea1, Scalar(255, 255, 255), 2);
	Mat InfoText1 = src(InfoArea1);
	imwrite("info1.jpg", InfoText1);

	/*区域二*/
	int offset2_x = 3;
	int offset2_y = 8;
	int InfoLength2 = 380;
	int InfoHeigh2 = 127;

	Rect InfoArea2(left_x + offset2_x, bottom_y - offset2_y - InfoHeigh2, InfoLength2, InfoHeigh2);
	rectangle(src, InfoArea2, Scalar(255, 255, 255), 2);
	Mat InfoText2 = src(InfoArea2);
	imwrite("info2.jpg", InfoText2);

	/*区域三*/
	int offset3_x = 5;
	int offset3_y = 8;
	int InfoLength3 = 365;
	int InfoHeigh3 = 170;

	Rect InfoArea3(right_x - offset3_x - InfoLength3, bottom_y - offset3_y - InfoHeigh3, InfoLength3, InfoHeigh3);
	rectangle(src, InfoArea3, Scalar(255, 255, 255), 2);
	Mat InfoText3 = src(InfoArea3);
	imwrite("info3.jpg", InfoText3);

	/*区域四*/
	int offset4_x = 15;
	int offset4_y = 25;
	int InfoLength4 = 320;
	int InfoHeigh4 = 87;

	Rect InfoArea4(right_x - InfoLength4, head_y - offset4_y - InfoHeigh4, InfoLength4 + offset4_x, InfoHeigh4);
	rectangle(src, InfoArea4, Scalar(255, 255, 255), 2);
	Mat InfoText4 = src(InfoArea4);
	imwrite("info4.jpg", InfoText4);

	g_info_area_count = 4;
	//显示图像
	imshow("src2", src);
	//namedWindow(wnd_binary, WINDOW_NORMAL);
	//imshow(wnd_binary, src_binary);
	waitKey(0);

	delete [] v;
	delete [] h;
}



/*增值税通用发票*/
void Class10InfoAreaExtract(Mat& src)
{
	resize(src, src, Size(1160, 817));
	imwrite("7p.bmp", src);
	Mat src_gray, src_binary;

	//转化为灰度图像
	cvtColor(src, src_gray, CV_RGB2GRAY);
	//二值化图像
	adaptiveThreshold(src_gray, src_binary, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY, 25, 10);
	int* v = new int[src.cols * 4];
	int* h = new int[src.rows * 4];
	cout << "src.cols = " << src.cols << endl;
	cout << "src.rows = " << src.rows << endl;
	memset(v, 0, src.cols * 4);
	memset(h, 0, src.rows * 4);
	int i, j;
	set<int> x;
	set<int> y;
	//方法一遍历
	//垂直方向进行累加（积分）
	int max_x = 0;
	int pos_x = 0;
	int max2_x = 0;
	int pos2_x = 0;
	for (i = 10; i < 160; i += 2) //列
	{
		for (j = src_binary.rows / 4; j < src_binary.rows /2; j++)      //行
		{
			if (src_binary.at<uchar>(j, i) == 0)      //统计的是黑色像素的数量
				v[i]++;
		}
		//cout << "v: " << v[i] << "  pos: " << i << endl;
		
		//找出top2的像素位置
		if (max2_x < v[i])
		{
			max2_x = v[i];
			pos2_x = i;
			if (max_x < max2_x)
			{
				swap(max_x, max2_x);
				swap(pos_x, pos2_x);
			}
		}
	}

	for (i = src_binary.cols - 10; i > src_binary.cols - 10 - 150; i -= 2) //列
	{
		for (j = src_binary.rows / 4; j < src_binary.rows / 2; j++)      //行
		{
			if (src_binary.at<uchar>(j, i) == 0)      //统计的是黑色像素的数量
				v[i]++;
		}
		//cout << "v: " << v[i] << "  pos: " << i << endl;
		//找出top2的像素位置
		if (max2_x < v[i])
		{
			max2_x = v[i];
			pos2_x = i;
			if (max_x < max2_x)
			{
				swap(max_x, max2_x);
				swap(pos_x, pos2_x);
			}
			}
		}




#if 0
	//注意i的步长，设置这样的步长是为了不把一些临近的像素高的点包括进来，否则会造成点重叠
	//注意i的起点和终点，都+或-了50，这是为了避免发票边缘影响像素检测
	for (i = 20; i<src_binary.cols - 20; i += 2) //列
	{
		for (j = src_binary.rows / 4; j<src_binary.rows*3/4; j++)      //行
		{
			if (src_binary.at<uchar>(j, i) == 0)      //统计的是黑色像素的数量
				v[i]++;
		}
		cout << "v: " << v[i] << "  pos: " << i << endl;


		//找出top2的像素位置
		if (max2_x < v[i])
		{
			max2_x = v[i];
			pos2_x = i;
			if (max_x < max2_x)
			{
				swap(max_x, max2_x);
				swap(pos_x, pos2_x);
			}
		}
	}
#endif
	//水平方向进行累加（积分）
	int max_y = 0;
	int pos_y = 0;
	int max2_y = 0;
	int pos2_y = 0;
	for (i = 50; i<src_binary.rows - 50; i += 2) //行
	{
		for (j = 0; j<src_binary.cols/4; j++)      //列
		{
			if (src_binary.at<uchar>(i, j) == 0)   //统计黑色像素的数量
				h[i]++;
		}

		//cout << "h :" << h[i] << "  pos: " << i << endl;
		if (max2_y < h[i])
		{
			max2_y = h[i];
			pos2_y = i;
			if (max_y < max2_y)
			{
				swap(max_y, max2_y);
				swap(pos_y, pos2_y);
			}
		}
	}


	cout << "max x = " << max_x << endl;
	cout << "max2 x = " << max2_x << endl;
	cout << "max y = " << max_y << endl;
	cout << "max2 y = " << max2_y << endl;
	cout << "pos x = " << pos_x << endl;
	cout << "pos2 x = " << pos2_x << endl;
	cout << "pos y = " << pos_y << endl;
	cout << "pos2 y = " << pos2_y << endl;





	int bottom_y = MAX(pos_y, pos2_y);
	int left_x = MIN(pos_x, pos2_x);
	int head_y = MIN(pos_y, pos2_y);
	int right_x = MAX(pos_x, pos2_x);
	int length = right_x - left_x;

	Point StartPoint(left_x, head_y); //这是发票的原点
									  //标出发票的四个关键点
	//circle(src, StartPoint, 5, Scalar(0, 0, 255), -1);
	//circle(src, Point(pos2_x, pos2_y), 5, Scalar(0, 0, 255), -1);
	//circle(src, Point(pos_x, pos2_y), 5, Scalar(0, 0, 255), -1);
	//circle(src, Point(pos2_x, pos_y), 5, Scalar(0, 0, 255), -1);
	//circle(src, Point(5, src.rows / 3), 5, Scalar(0, 255, 0), -1);
	//circle(src, Point(src.cols / 3, 5), 5, Scalar(0, 255, 0), -1);

	
	for (i = 0; i < src.rows; i++)
	{
		if (h[i] > 120)
		{

			circle(src, Point(pos_x, i), 5, Scalar(0, 0, 255), -1);
			circle(src, Point(pos2_x, i), 5, Scalar(0, 0, 255), -1);
			y.insert(i);
		}
	}
#if 1
	set<int>::iterator iter;
	int begin = 0;
	g_info_area_count = 0;
	for (iter = y.begin(); iter != y.end(); iter++)
	{
		if (begin == 0)
		{
			begin = *iter;
			continue;
		}

		int height = *iter - begin;
		if (height < 30)
		{
			continue;
		}
		
		cout << "y pos: " << *iter << endl;
		cout << "height: " << height << endl;

		int offset_x = 4;
		int offset_y = 4;
		int InfoLength = length;
		int InfoHeigh = height;
		cout << "length: " << InfoLength << endl;
		cout << "point x" << left_x + offset_x << endl;
		cout << "point y:" << *iter - offset_y << endl;
		Rect InfoArea1(left_x + offset_x, begin + offset_y, InfoLength - offset_x, InfoHeigh - offset_y);
		rectangle(src, InfoArea1, Scalar(255, 255, 255), 2);
		Mat InfoText1 = src(InfoArea1);
		/*
		char file[20];
		memset(file, 0, sizeof(file));
		sprintf_s(file, "info%d.jpg", ++g_info_area_count);
		*/
		if (1)
		{
			func(src_binary(InfoArea1),src(InfoArea1));
			//InfoText1 = InfoText1(r);
		}

		//imwrite(file, InfoText1);
		begin = *iter;

	}
#endif
#if 0
	/*这些参数都要根据每一类的发票位置的信息区域来调整*/
	/*区域一*/
	int offset_x = 3;
	int offset_y = 3;
	int InfoLength = 590;
	int InfoHeigh = 230;

	Rect InfoArea1(left_x + offset_x, head_y + offset_y, InfoLength, InfoHeigh);
	rectangle(src, InfoArea1, Scalar(255, 255, 255), 2);
	Mat InfoText1 = src(InfoArea1);
	imwrite("info1.jpg", InfoText1);


	/*区域二*/
	int offset2_x = 3;
	int offset2_y = 8;
	int InfoLength2 = 417;
	int InfoHeigh2 = 160;

	Rect InfoArea2(left_x + offset2_x, bottom_y - offset2_y - InfoHeigh2, InfoLength2, InfoHeigh2);
	rectangle(src, InfoArea2, Scalar(255, 255, 255), 2);
	Mat InfoText2 = src(InfoArea2);
	imwrite("info2.jpg", InfoText2);


	/*区域三*/
	int offset3_x = 5;
	int offset3_y = 20;
	int InfoLength3 = 274;
	int InfoHeigh3 = 146;

	Rect InfoArea3(right_x - offset3_x - InfoLength3, bottom_y - offset3_y - InfoHeigh3, InfoLength3, InfoHeigh3);
	rectangle(src, InfoArea3, Scalar(255, 255, 255), 2);
	Mat InfoText3 = src(InfoArea3);
	imwrite("info3.jpg", InfoText3);

	/*区域四*/
	int offset4_x = 15;
	int offset4_y = 25;
	int InfoLength4 = 320;
	int InfoHeigh4 = 87;

	Rect InfoArea4(right_x - InfoLength4, head_y - offset4_y - InfoHeigh4, InfoLength4 + offset4_x, InfoHeigh4);
	rectangle(src, InfoArea4, Scalar(255, 255, 255), 2);
	Mat InfoText4 = src(InfoArea4);
	imwrite("info4.jpg", InfoText4);
#endif
	namedWindow("src2", 0);
	imshow("src2", src);
	delete[] v;
	delete[] h;
}

