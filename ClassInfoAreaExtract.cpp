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

//extern int g_info_area_count;
extern int g_final_area_count;
extern Mat g_src;
extern Mat g_src_gray;

int g_height;

void Class1InfoAreaExtract(Mat& img);


bool IsFirstKeyArea(Mat& img)
{
	//cout << "width:" << img.cols << " height:" << img.rows << endl;
	if (1)
	{
		if ((img.cols < 600 && img.cols > 500) && (img.rows < 150 && img.rows > 100))
		{
			return true;
		}
	}

	return false;
}

bool IsSecondArea(Mat& img)
{
	cout << "width:" << img.cols << " height:" << img.rows << endl;
	if (1)
	{
		if ((img.cols < 850 && img.cols > 720) && (img.rows < 60 && img.rows > 30))
		{
			return true;
		}
	}

	return false;
}

//列分割
void ColsAreaDivide(Mat& img,Mat&src,int y_pos)
{
	int thresh = 0.6 * src.rows;  //归一化阈值
	cout << "列分割阈值:" << thresh << endl;
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
		//cout << "h2: " << h2[i] << "  pos: " << i << endl;
		if (h2[i] > thresh)
		{
			circle(img, Point(i, 10), 5, Scalar(0, 0, 255), -1);
			x_pos.insert(i);
		}
	}

	//imshow("tras", img);

	if (!x_pos.size())  //没有竖线，直接返回原图
	{
		char file[100];
		memset(file, 0, sizeof(file));
		sprintf_s(file, "E:\\coding\\vs 2015 test\\SmartSystem\\SmartSystem\\final_info_area\\tmp%d.jpg", ++g_final_area_count);
		imwrite(file, src);
		return;
	}

	/*分割每个小区域*/
#if 1
	int begin = 0;
	int count = 0;
	for (set<int>::iterator it = x_pos.begin(); it != x_pos.end(); it++)
	{
		if(1)
		{
			int len = *it - begin;
			if (len < 10)
			{
				continue;
			}
			count++;
		

			int height = img.rows;
			cout << "y pos:" << y_pos <<"g_src:"<< g_height<< endl;
			if ((IsFirstKeyArea(src(Rect(0, 0, len, height)))  && y_pos < g_height) || (IsSecondArea(src(Rect(0, 0, len, height))) && y_pos > g_height))
			{
				char file[100];
				memset(file, 0, sizeof(file));
				sprintf_s(file, "E:\\coding\\vs 2015 test\\SmartSystem\\SmartSystem\\final_info_area\\tmp%d.jpg", ++g_final_area_count);
				imwrite(file, src(Rect(begin + 2, 0, len - 2, height)));
				//imwrite("test.jpg", tmp);
			}


		}
#if 1
		if (count == x_pos.size())
		{
			int len = src.cols - *it;
			int height = img.rows;
			if ((IsFirstKeyArea(src(Rect(0, 0, len, height))) && y_pos < g_height) || (IsSecondArea(src(Rect(0, 0, len, height))) && y_pos > g_height))
			{
				char file[100];
				memset(file, 0, sizeof(file));
				sprintf_s(file, "E:\\coding\\vs 2015 test\\SmartSystem\\SmartSystem\\final_info_area\\tmp%d.jpg", ++g_final_area_count);
				imwrite(file, src(Rect(*it, 0, len, height)));
			}
		}
#endif		
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

	g_final_area_count = 4;
	//显示图像
	imshow("src2", src);
	//namedWindow(wnd_binary, WINDOW_NORMAL);
	//imshow(wnd_binary, src_binary);
	waitKey(0);

	delete [] v;
	delete [] h;
}



/*表格类发票*/
void Class10InfoAreaExtract(Mat& src)
{
	resize(src, src, Size(1160, 817));
	g_height = src.rows / 2;
	Mat src_gray, src_binary;

	//转化为灰度图像
	cvtColor(src, src_gray, CV_RGB2GRAY);
	//二值化图像
	adaptiveThreshold(src_gray, src_binary, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY, 25, 10);
	imshow("二值化", src_binary);
	int* v = new int[src.cols * 4];
	int* h = new int[src.rows * 4];
	cout << "src.cols = " << src.cols << endl;
	cout << "src.rows = " << src.rows << endl;
	memset(v, 0, src.cols * 4);
	memset(h, 0, src.rows * 4);
	int i, j;
	set<int> x;
	set<int> y;

	//垂直方向进行累加（积分）
	int max_x = 0;
	int pos_x = 0;
	int max2_x = 0;
	int pos2_x = 0;

	//从左边数起，在一定范围内的x坐标的最大值
#if 1
	for (i = 10; i < 160; i += 2) //列
	{
		for (j = src_binary.rows*3 / 8; j < src_binary.rows*6/8; j++)      //行
		{
			if (src_binary.at<uchar>(j, i) == 0)      //统计的是黑色像素的数量
				v[i]++;
		}
		//cout << "v: " << v[i] << "  pos: " << i << endl;
		
		//找出top2的像素位置
		if (max2_x < v[i])
		{
			//我们需要保证最大的两个点的距离要大于5
			if (i - pos_x <= 5)
			{
				continue;
			}

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
	//从右边数起，在一定范围内的x坐标的最大值
	for (i = src_binary.cols - 10; i > src_binary.cols - 5 - 150; i -= 1) //列
	{
		for (j = src_binary.rows*3  / 8; j < src_binary.rows*6/8 ; j++)      //行
		{
			if (src_binary.at<uchar>(j, i) == 0)      //统计的是黑色像素的数量
				v[i]++;
		}
		//cout << "v: " << v[i] << "  pos: " << i << endl;
		//找出top2的像素位置
		if (max2_x < v[i])
		{
			//我们需要保证最大的两个点的距离要大于5
			if (i - pos_x <= 5)
			{
				continue;
			}

			max2_x = v[i];
			pos2_x = i;
			if (max_x < max2_x)
			{
				swap(max_x, max2_x);
				swap(pos_x, pos2_x);
			
			}
			}
		}

	//水平方向进行累加（积分），求超过阈值的每个y坐标
	int max_y = 0;
	int pos_y = 0;
	int max2_y = 0;
	int pos2_y = 0;
	for (i = 30; i<src_binary.rows - 30; i += 1) //行
	{
		for (j = src_binary.cols / 4; j<src_binary.cols*3/4; j++)      //列
		{
			if (src_binary.at<uchar>(i, j) == 0)   //统计黑色像素的数量
				h[i]++;
		}

		//cout << "h :" << h[i] << "  pos: " << i << endl;
		if (max2_y < h[i])
		{
			//我们需要保证最大的两个点的距离要大于5
			if (i - pos_y <= 5)
			{
				continue;
			}
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



	//求出最外围矩形框的四个角点坐标

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

	int pre = 0;
	for (i = 0; i < src.rows; i++)
	{
		if (h[i] > 450)
		{
			if (i - pre < 8)
			{
				continue;
			}

			cout << "符合要求的点有：" << i <<"大小是："<<h[i]<< endl;


			circle(src, Point(pos_x, i), 5, Scalar(0, 0, 255), -1);
			circle(src, Point(pos2_x, i), 5, Scalar(0, 0, 255), -1);
			y.insert(i);
			pre = i;
		}
	}

	int YPointCount = y.size();  //符合条件的y点数目
	cout << "y poin num: " << YPointCount << endl;

	if (YPointCount  == 2)  //只有两个点，执行膨胀信息区域提取算法,否则继续往下执行，表格类分割
	{
		cout << "这是一个单一矩形发票！" << endl;
		/*
		//执行膨胀信息区域提取算法
		int h = bottom_y - head_y;
		Rect r(left_x +4, head_y+4, length-8,  h-8); //矩形框适当修正，该矩形区域不包括矩形的线边缘
		Mat ROI = src(r);
		imshow("ROI", ROI);
		Class1InfoAreaExtract(ROI); 
		*/

		//切割两个关键区域，左上角和右下角

		//左上角“付款方”
		int h1 = 120;
		int l1 = 400;
		Rect r1(left_x + 4, head_y + 4, l1 - 8, h1 - 8); //矩形框适当修正，该矩形区域不包括矩形的线边缘
		Mat ROI1 = src(r1);
		imshow("ROI1", ROI1);
		char file[100];
		memset(file, 0, sizeof(file));
		sprintf_s(file, "E:\\coding\\vs 2015 test\\SmartSystem\\SmartSystem\\final_info_area\\tmp%d.jpg", ++g_final_area_count);
		imwrite(file, ROI1);

		//右下角“金额”
		int h2 = (bottom_y - head_y)/2;
		int l2 = (right_x - left_x)/2;
		Rect r2(right_x - l2 + 4, bottom_y - h2+4, l2 - 8, h2 - 8); //矩形框适当修正，该矩形区域不包括矩形的线边缘
		Mat ROI2 = src(r2);
		imshow("ROI2", ROI2);
		memset(file, 0, sizeof(file));
		sprintf_s(file, "E:\\coding\\vs 2015 test\\SmartSystem\\SmartSystem\\final_info_area\\tmp%d.jpg", ++g_final_area_count);
		imwrite(file, ROI2);

		return;
	}

	/*下面开始将抠出来的每一行进行列分割*/
#if 1
	set<int>::iterator iter;
	int begin = 0;
	g_final_area_count = 0;
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
			//列分割
			ColsAreaDivide(src_binary(InfoArea1),src(InfoArea1), begin);
		}

		//imwrite(file, InfoText1);
		begin = *iter;

	}
#endif
	namedWindow("src2", 0);
	imshow("src2", src);
	delete[] v;
	delete[] h;
}

extern void SaveDetailedArea();
//采取腐蚀膨胀的思路抠出矩形区域的信息区
void Class1InfoAreaExtract(Mat& src)
{

	cvtColor(src, g_src, COLOR_RGB2GRAY);
	imshow("灰度化", g_src);

	Mat out;
	boxFilter(g_src, out, -1, Size(5, 5));//-1指原图深度
	imshow("方框滤波", out);



	// 局部二值化
	int blockSize = 25;
	int constValue = 10;
	Mat local;
	adaptiveThreshold(out, local, 255, CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY_INV, blockSize, constValue);
	imshow("二值化", local);
	imwrite("binary.bmp", local);


	Mat out3;
	//获取自定义核
	Mat element2 = getStructuringElement(MORPH_RECT, Size(4, 4)); //第一个参数MORPH_RECT表示矩形的卷积核，当然还可以选择椭圆形的、交叉型的
																  //腐蚀操作
	erode(local, out3, element2,Point(-1,-1),1);
	namedWindow("腐蚀操作", WINDOW_NORMAL);
	imshow("腐蚀操作", out3);
	imwrite("erode.bmp", out3);

	Mat out2;
	//获取自定义核

	Mat element = getStructuringElement(MORPH_RECT, Size(30, 1)); //第一个参数MORPH_RECT表示矩形的卷积核，当然还可以选择椭圆形的、交叉型的
																  //膨胀操作
	dilate(out3, out2, element, Point(-1, -1), 10); //迭代10次
	namedWindow("膨胀操作", WINDOW_NORMAL);
	imshow("膨胀操作", out2);
	imwrite("dd.jpg", out2);

	Mat img;
	/// 载入原图像, 返回3通道图像
	img = imread("dd.jpg", 1);

	/// 转化成灰度图像并进行平滑
	cvtColor(img, g_src_gray, CV_BGR2GRAY);
	blur(g_src_gray, g_src_gray, Size(3, 3));
	imshow("gray blur2", g_src_gray);

	/// 创建窗口
	char* source_window = "Source";
	namedWindow(source_window, CV_WINDOW_AUTOSIZE);
	imshow(source_window, g_src);

	SaveDetailedArea();
}



int matchtemplate1(Mat &img)
{

	Mat templ, result;
	templ = imread("s.png");  //使用的是飞机票模板
	int result_cols = img.cols - templ.cols + 1;
	int result_rows = img.rows - templ.rows + 1;
	result.create(result_cols, result_rows, CV_32FC1);
	matchTemplate(img, templ, result, CV_TM_SQDIFF_NORMED);//CV_TM_SQDIFF_NORMED CV_TM_SQDIFF  

	double minVal;
	double maxVal;
	Point minLoc;
	Point maxLoc;
	Point matchLoc;
	//cout << "匹配度：" << minVal << endl;
	minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc, Mat());

	cout << "匹配度：" << minVal << endl;

	matchLoc = minLoc;

	//rectangle(img, matchLoc, Point(matchLoc.x + templ.cols, matchLoc.y + templ.rows), Scalar(0, 255, 0), 2, 8, 0);


	//阈值判别，小于0.1才认为匹配成功

	if (minVal < 0.03)
	{
		return 1;
	}

	return 2;

}

Point matchtemplate4(Mat &img)
{

	Mat templ, result;
	templ = imread("r.png");  //使用的是飞机票模板
	int result_cols = img.cols - templ.cols + 1;
	int result_rows = img.rows - templ.rows + 1;
	result.create(result_cols, result_rows, CV_32FC1);
	matchTemplate(img, templ, result, CV_TM_SQDIFF_NORMED);//CV_TM_SQDIFF_NORMED CV_TM_SQDIFF  

	double minVal;
	double maxVal;
	Point minLoc;
	Point maxLoc;
	Point matchLoc;
	//cout << "匹配度：" << minVal << endl;
	minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc, Mat());

	cout << "匹配度：" << minVal << endl;

	matchLoc = minLoc;

	rectangle(img, matchLoc, Point(matchLoc.x + templ.cols, matchLoc.y + templ.rows), Scalar(0, 255, 0), 2, 8, 0);

	imshow("ha", img);
	//阈值判别，小于0.1才认为匹配成功

	return matchLoc;



}

/*****************************************每类发票对应的关键信息区域的提取函数********************************************************************/

//火车票类
void Class2InfoExtract(Mat& img)
{
	/*以下参数根据每一类的发票特点进行定制*/
	int x = 10;
	int y = 250;
	int height = 210;
	int length = 450;

	Rect r = Rect(x, y, length, height);

	Mat info = img(r);
	imwrite("E:\\coding\\vs 2015 test\\SmartSystem\\SmartSystem\\final_info_area\\1.jpg",info);

	g_final_area_count++;

	rectangle(img, r, Scalar(0, 255, 0), 2, 8, 0);
	imshow("区域定位", img);

}


/*飞机类发票*/
void Class1InfoExtract(Mat& src)
{
	resize(src, src, Size(1160, 817));
	g_height = src.rows / 2;
	Mat src_gray, src_binary;

	//转化为灰度图像
	cvtColor(src, src_gray, CV_RGB2GRAY);
	//二值化图像
	adaptiveThreshold(src_gray, src_binary, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY, 25, 10);
	imshow("二值化", src_binary);
	int* v = new int[src.cols * 4];
	int* h = new int[src.rows * 4];
	cout << "src.cols = " << src.cols << endl;
	cout << "src.rows = " << src.rows << endl;
	memset(v, 0, src.cols * 4);
	memset(h, 0, src.rows * 4);
	int i, j;
	set<int> x;
	set<int> y;

	//垂直方向进行累加（积分）
	int max_x = 0;
	int pos_x = 0;
	int max2_x = 0;
	int pos2_x = 0;

	//从左边数起，在一定范围内的x坐标的最大值
#if 1
	for (i = 10; i < 160; i += 2) //列
	{
		for (j = src_binary.rows * 3 / 8; j < src_binary.rows * 6 / 8; j++)      //行
		{
			if (src_binary.at<uchar>(j, i) == 0)      //统计的是黑色像素的数量
				v[i]++;
		}
		//cout << "v: " << v[i] << "  pos: " << i << endl;

		//找出top2的像素位置
		if (max2_x < v[i])
		{
			//我们需要保证最大的两个点的距离要大于5
			if (i - pos_x <= 5)
			{
				continue;
			}

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
	//从右边数起，在一定范围内的x坐标的最大值
	for (i = src_binary.cols - 10; i > src_binary.cols - 5 - 150; i -= 1) //列
	{
		for (j = src_binary.rows * 3 / 8; j < src_binary.rows * 6 / 8; j++)      //行
		{
			if (src_binary.at<uchar>(j, i) == 0)      //统计的是黑色像素的数量
				v[i]++;
		}
		//cout << "v: " << v[i] << "  pos: " << i << endl;
		//找出top2的像素位置
		if (max2_x < v[i])
		{
			//我们需要保证最大的两个点的距离要大于5
			if (i - pos_x <= 5)
			{
				continue;
			}

			max2_x = v[i];
			pos2_x = i;
			if (max_x < max2_x)
			{
				swap(max_x, max2_x);
				swap(pos_x, pos2_x);

			}
		}
	}

	//水平方向进行累加（积分），求超过阈值的每个y坐标
	int max_y = 0;
	int pos_y = 0;
	int max2_y = 0;
	int pos2_y = 0;
	for (i = 30; i < src_binary.rows - 30; i += 1) //行
	{
		for (j = src_binary.cols / 4; j < src_binary.cols * 3 / 4; j++)      //列
		{
			if (src_binary.at<uchar>(i, j) == 0)   //统计黑色像素的数量
				h[i]++;
		}

		//cout << "h :" << h[i] << "  pos: " << i << endl;
		if (max2_y < h[i])
		{
			//我们需要保证最大的两个点的距离要大于5
			if (i - pos_y <= 5)
			{
				continue;
			}
			max2_y = h[i];
			pos2_y = i;
			if (max_y < max2_y)
			{
				swap(max_y, max2_y);
				swap(pos_y, pos2_y);

			}
		}
	}

	int pre = 0;
	for (i = 0; i < src.rows; i++)
	{
		if (h[i] > 450)
		{
			if (i - pre < 8)
			{
				continue;
			}

			cout << "符合要求的点有：" << i << "大小是：" << h[i] << endl;


			//circle(src, Point(pos_x, i), 5, Scalar(0, 0, 255), -1);
			//circle(src, Point(pos2_x, i), 5, Scalar(0, 0, 255), -1);
			y.insert(i);
			pre = i;
		}
	}


	set<int>::iterator it = y.begin();
	int begin = *it;
	set<int>::reverse_iterator  it2 = y.rbegin();
	int end = *it2;

	//求出最外围矩形框的四个角点坐标

	int bottom_y = end;
	int left_x = MIN(pos_x, pos2_x);
	int head_y = begin;
	int right_x = MAX(pos_x, pos2_x);


	//circle(src, Point(left_x, bottom_y), 5, Scalar(0, 0, 255), -1);
	//circle(src, Point(left_x, head_y), 5, Scalar(0, 0, 255), -1);
	//circle(src, Point(right_x, bottom_y), 5, Scalar(0, 0, 255), -1);
	//circle(src, Point(right_x, head_y), 5, Scalar(0, 255, 0), -1);

	//namedWindow("hi", 0);
	//imshow("hi", src);


	//细分航空类发票，使用模板匹配
	int match = matchtemplate1(src);

	cout << "match result" << match << endl;

		int y_pos;
		int x_pos;
		int InfoLength;
		int InfoHeigh;


		if (match == 1)//飞机票
		{

			//关键区域1切割，金额
			x_pos = right_x - 230;
			y_pos = bottom_y - 204;
			InfoLength = 230;
			InfoHeigh = 66;

			Rect InfoArea1(x_pos, y_pos, InfoLength, InfoHeigh);
			//rectangle(src, InfoArea1, Scalar(255, 255, 255), 2);
			Mat InfoText1 = src(InfoArea1);
			imwrite("E:\\coding\\vs 2015 test\\SmartSystem\\SmartSystem\\final_info_area\\1.jpg", InfoText1);
			g_final_area_count++;

			//关键区域2切割,姓名
			x_pos = left_x;
			y_pos = head_y;
			InfoLength = 251;
			InfoHeigh = 86;

			Rect InfoArea2(x_pos, y_pos, InfoLength, InfoHeigh);
			//rectangle(src, InfoArea1, Scalar(255, 255, 255), 2);
			Mat InfoText2 = src(InfoArea2);
			imwrite("E:\\coding\\vs 2015 test\\SmartSystem\\SmartSystem\\final_info_area\\2.jpg", InfoText2);
			g_final_area_count++;
			
		}
		else  //保险类
		{
			//关键区域1切割，金额
			x_pos = right_x - 500;
			y_pos = head_y + 210;
			InfoLength = 500;
			InfoHeigh = 60;

			cout << "x:" << x_pos << "y:" << y_pos << endl;

			Rect InfoArea1(x_pos, y_pos, InfoLength, InfoHeigh);
			//rectangle(src, InfoArea1, Scalar(255, 255, 255), 2);
			Mat InfoText1 = src(InfoArea1);
			imwrite("E:\\coding\\vs 2015 test\\SmartSystem\\SmartSystem\\final_info_area\\1.jpg", InfoText1);
			g_final_area_count++;

			//关键区域2切割,姓名
			x_pos = left_x;
			y_pos = head_y;
			InfoLength = 377;
			InfoHeigh = 66;

			Rect InfoArea2(x_pos, y_pos, InfoLength, InfoHeigh);
			//rectangle(src, InfoArea1, Scalar(255, 255, 255), 2);
			Mat InfoText2 = src(InfoArea2);
			imwrite("E:\\coding\\vs 2015 test\\SmartSystem\\SmartSystem\\final_info_area\\2.jpg", InfoText2);
			g_final_area_count++;
		}

	delete[] v;
	delete[] h;
}

//当当京东类发票
void Class4InfoExtract(Mat& img)
{
	resize(img, img, Size(605, 1431));
	Point p = matchtemplate4(img);

	//客户名称信息区域
	int x, y;
	x = 5;
	y = p.y + 113;
	int h = 40;
	int l = 260;

	Rect InfoArea1(x, y, l, h);
	//rectangle(src, InfoArea1, Scalar(255, 255, 255), 2);
	Mat InfoText1 = img(InfoArea1);
	imwrite("E:\\coding\\vs 2015 test\\SmartSystem\\SmartSystem\\final_info_area\\1.jpg", InfoText1);
	g_final_area_count++;

	//金额信息区域
	x = 5;
	y = img.rows - 143;
	h = 46;
	l = 260;

	Rect InfoArea2(x, y, l, h);
	//rectangle(src, InfoArea1, Scalar(255, 255, 255), 2);
	Mat InfoText2 = img(InfoArea2);
	imwrite("E:\\coding\\vs 2015 test\\SmartSystem\\SmartSystem\\final_info_area\\2.jpg", InfoText2);
	g_final_area_count++;


}