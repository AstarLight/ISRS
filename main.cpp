/*************************************************************************
Author:Li Junshi
File Decription: main funtion and some basic image pre process funcions
File Create Time: 2017-06-01
Please send email to lijunshi2015@163.com if you any question.
*************************************************************************/

#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include <iostream>
#include "ClassInfoAreaExtract.h"
#include "Classify.h"
#include <set>

using namespace cv;
using namespace std;

#define RESULT_IMG_PATH  "result.jpg"
#define ERROR -1
#define UNKNOWN 0
#define CLASS_THRESHOLD  0.06    //分类时设定的阈值


//int g_info_area_count = 0;
int g_final_area_count = 0;

Mat g_src;
Mat g_src_gray;
#define DETAILED_INFO_AREA_THRESH  100

void EdgeLocation(Mat& src2);

//度数转换
double DegreeTrans(double theta)
{
	double res = theta / CV_PI * 180;
	return res;
}


//逆时针旋转图像degree角度（原尺寸）    
void rotateImage(Mat src, Mat& img_rotate, double degree)
{
	//旋转中心为图像中心    
	Point2f center;
	center.x = float(src.cols / 2.0);
	center.y = float(src.rows / 2.0);
	int length = 0;
	length = sqrt(src.cols*src.cols + src.rows*src.rows);
	//计算二维旋转的仿射变换矩阵  
	Mat M = getRotationMatrix2D(center, degree, 0.98); //稍微缩小，以便提取边缘
	warpAffine(src, img_rotate, M, Size(length, length));//仿射变换  
}

//通过霍夫变换计算角度
double CalcDegree(Mat srcImage)
{
	Mat midImage, dstImage;

	Canny(srcImage, midImage, 50, 200, 3);
	cvtColor(midImage, dstImage, CV_GRAY2BGR);

	vector<Vec2f> lines;

	HoughLines(midImage, lines, 1, CV_PI / 360, 500, 0, 0);
	cout << lines.size() << endl;

	if (!lines.size())
	{
		HoughLines(midImage, lines, 1, CV_PI / 180, 200, 0, 0);
	}
	cout << lines.size() << endl;

	if (!lines.size())
	{
		HoughLines(midImage, lines, 1, CV_PI / 180, 150, 0, 0);
	}
	cout << lines.size() << endl;

	float sum = 0;
	int count = 0;
	//依次画出每条线段
	for (size_t i = 0; i < lines.size(); i++)
	{
		float rho = lines[i][0];
		float theta = lines[i][1];
		Point pt1, pt2;
		cout <<"theta:" << theta << endl;
		double a = cos(theta), b = sin(theta);
		double x0 = a*rho, y0 = b*rho;
		pt1.x = cvRound(x0 + 1000 * (-b));
		pt1.y = cvRound(y0 + 1000 * (a));
		pt2.x = cvRound(x0 - 1000 * (-b));
		pt2.y = cvRound(y0 - 1000 * (a));

		if (theta < 1.2 || theta > 1.8)  //角度太小或者太大，都不要
		{
			continue;
		}

		//只选角度最小的作为旋转角度
		sum += theta;
		count++;
		line(dstImage, pt1, pt2, Scalar(55, 100, 195), 1, LINE_AA); //Scalar函数用于调节线段颜色

		imshow("直线探测效果图", dstImage);
	}

	float average = sum / count; //对所有角度求平均，这样做旋转效果会更好
	cout << "average theta:" << average << endl;

	double angle = DegreeTrans(average) - 90;

	if (angle < -45)
	{
		angle = 90 + angle;
	}
	//cout << "angle: " << angle << endl;
	rotateImage(dstImage, srcImage, angle);
	//imshow("直线探测效果图2", dstImage);
	imwrite("test2.jpg", srcImage);
	return angle;
}


void ImageRecify(Mat& src, Mat& dst)
{
	double degree;

	//倾斜角度矫正
	degree = CalcDegree(src);
	rotateImage(src, dst, degree);
	cout << "angle:" << degree << endl;

	imshow("旋转调整后", dst);
}


void SystemInit()
{
	for (int i = 1; i <= g_final_area_count; i++)
	{
		char file[100];
		memset(file, 0, sizeof(file));
		sprintf_s(file, "E:\\coding\\vs 2015 test\\SmartSystem\\SmartSystem\\final_info_area\\tmp%d.jpg", i);
		if (remove(file) == 0)
			printf("Removed %s.\n", file);
	}


	//g_info_area_count = 0;
	g_final_area_count = 0;

	//system("md E:\\coding\\vs 2015 test\\SmartSystem\\SmartSystem\\final_info_area");

}

void SaveDetailedArea()
{
	Mat threshold_output;
	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;

	/// 使用Threshold检测边缘
	threshold(g_src_gray, threshold_output, DETAILED_INFO_AREA_THRESH, 255, THRESH_BINARY);

	/// 找到轮廓
	findContours(threshold_output, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));

	/// 多边形逼近轮廓 + 获取矩形和圆形边界框
	vector<vector<Point> > contours_poly(contours.size());
	vector<Rect> boundRect(contours.size());
	vector<Point2f>center(contours.size());
	vector<float>radius(contours.size());

	for (int i = 0; i < contours.size(); i++)
	{
		approxPolyDP(Mat(contours[i]), contours_poly[i], 3, true);
		boundRect[i] = boundingRect(Mat(contours_poly[i]));
		minEnclosingCircle(contours_poly[i], center[i], radius[i]);
	}


	/// 画多边形轮廓 + 包围的矩形框 + 圆形框
	Mat drawing = Mat::zeros(threshold_output.size(), CV_8UC3);
	g_src.copyTo(drawing);
	g_final_area_count = 0; //初始化
	for (int i = 0; i< contours.size(); i++)
	{
		//Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
		//drawContours(drawing, contours_poly, i, color, 1, 8, vector<Vec4i>(), 0, Point());
		int rate = boundRect[i].width / boundRect[i].height;
#if 1
		if (boundRect[i].area() < 300 )  //筛选一些矩形，要面积到达一定大小才选用
		{
			continue;
		}
#endif


		//rectangle(drawing, boundRect[i].tl(), boundRect[i].br(), Scalar(0, 0, 255), 2, 8, 0);
		int x = 0;
		int y = 0;
		if (boundRect[i].x > 3)
		{
			x = boundRect[i].x - 3;
		}

		if (boundRect[i].y > 3)
		{
			y = boundRect[i].y -3;
		}

		Rect r(x , y, boundRect[i].width, boundRect[i].height+4);
		rectangle(g_src, r,Scalar(0, 0, 255), 2, 8, 0);
		
		//Rect r(boundRect[i].tl(), boundRect[i].br());
		Mat tmp = drawing(r);
		char file[100];

		sprintf(file, "E:\\coding\\vs 2015 test\\SmartSystem\\SmartSystem\\final_info_area\\tmp%d.jpg", ++g_final_area_count);
		imwrite(file, tmp);
		//circle(drawing, center[i], (int)radius[i], color, 2, 8, 0);
	}

	/// 显示在一个窗口
	namedWindow("Contours", CV_WINDOW_AUTOSIZE);
	imshow("Contours", g_src);
}

//对第一次提取出来的区域在进行进一步的提取的
void DetailedInfoAreaExtract(Mat& src2)
{
	cvtColor(src2, g_src, COLOR_RGB2GRAY);
	imshow("灰度化", g_src);
/*
	Mat out;
	boxFilter(src, out, -1, Size(5, 5));//-1指原图深度
	imshow("方框滤波", out);
*/
	// 局部二值化
	int blockSize = 25;
	int constValue = 10;
	Mat local;
	adaptiveThreshold(g_src, local, 255, CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY_INV, blockSize, constValue);
	imshow("二值化", local);

	Mat out2;
	//获取自定义核
	Mat element = getStructuringElement(MORPH_RECT, Size(8, 8)); //第一个参数MORPH_RECT表示矩形的卷积核，当然还可以选择椭圆形的、交叉型的
																 //膨胀操作
	dilate(local, out2, element);
	namedWindow("膨胀操作", WINDOW_NORMAL);
	imshow("膨胀操作", out2);
	imwrite("dd.jpg", out2);

	Mat img;
	/// 载入原图像, 返回3通道图像
	img = imread("dd.jpg", 1);

	/// 转化成灰度图像并进行平滑
	cvtColor(img, g_src_gray, CV_BGR2GRAY);
	blur(g_src_gray, g_src_gray, Size(3, 3));

	/// 创建窗口
	char* source_window = "Source";
	namedWindow(source_window, CV_WINDOW_AUTOSIZE);
	imshow(source_window, g_src);

	//SaveDetailedArea();
}

void FinalInfoGenerator()
{
	for (int i = 1; i <= g_final_area_count; i++)
	{
		char file[20];
		sprintf_s(file, "info%d.jpg", i);
		Mat tmpImg = imread(file);

		DetailedInfoAreaExtract(tmpImg);
	}
}



void Class2InfoAreaExtract(Mat& src2)
{
	Mat img2 = src2(Rect(20, 10, src2.cols - 40, src2.rows - 20)); //修正边缘
																   //Mat img2 = src2;
	imshow("img2", img2);
	cvtColor(img2, g_src, COLOR_RGB2GRAY);
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
	Mat element2 = getStructuringElement(MORPH_RECT, Size(5, 5)); //第一个参数MORPH_RECT表示矩形的卷积核，当然还可以选择椭圆形的、交叉型的
																  //腐蚀操作
	erode(local, out3, element2);
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

	/// 创建窗口
	char* source_window = "Source";
	namedWindow(source_window, CV_WINDOW_AUTOSIZE);
	imshow(source_window, g_src);

	SaveDetailedArea();
}



void Class3InfoAreaExtract(Mat& src2)
{
	resize(src2, src2, Size(1160, 817));
	Mat local;

	//转化为灰度图像
	cvtColor(src2, g_src, CV_RGB2GRAY);
	
	//二值化图像
	adaptiveThreshold(g_src, local, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY_INV, 25, 10);

	Mat out;
	//获取自定义核
	Mat element = getStructuringElement(MORPH_RECT, Size(3, 3)); //第一个参数MORPH_RECT表示矩形的卷积核，当然还可以选择椭圆形的、交叉型的
																 //腐蚀操作
	erode(local, out, element, Point(-1, -1), 2);

	out = out(Rect(20, 20, out.cols - 40, out.rows - 40));
	g_src = g_src(Rect(20, 20, g_src.cols - 40, g_src.rows - 40));
	//src_gray = out;
	namedWindow("腐蚀操作", WINDOW_NORMAL);
	imshow("腐蚀操作", out);

	Mat out2;
	//获取自定义核
	Mat element2 = getStructuringElement(MORPH_RECT, Size(40, 2)); //第一个参数MORPH_RECT表示矩形的卷积核，当然还可以选择椭圆形的、交叉型的
																   //膨胀操作
	dilate(out, out2, element2, Point(-1, -1), 5);

	imshow("out2", out2);

	imwrite("dd.jpg", out2);

	Mat img;
	/// 载入原图像, 返回3通道图像
	img = imread("dd.jpg", 1);

	/// 转化成灰度图像并进行平滑
	cvtColor(img, g_src_gray, CV_BGR2GRAY);
	blur(g_src_gray, g_src_gray, Size(3, 3));

	SaveDetailedArea();
}

void EdgeLocation(Mat& src2)
{
	//resize(src, src, Size(1160, 817));
	imwrite("7p.bmp", src2);
	Mat src_gray, src_binary;

	//转化为灰度图像
	cvtColor(src2, src_gray, CV_RGB2GRAY);
	//二值化图像
	adaptiveThreshold(src_gray, src_binary, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY, 25, 10);
	int* v = new int[src2.cols * 4];

	memset(v, 0, src2.cols * 4);

	int i, j;
	//方法一遍历
	//垂直方向进行累加（积分）
	int max_x = 0;
	int pos_x = 0;
	int max2_x = 0;
	int pos2_x = 0;
	//注意i的步长，设置这样的步长是为了不把一些临近的像素高的点包括进来，否则会造成点重叠
	//注意i的起点和终点，都+或-了50，这是为了避免发票边缘影响像素检测
	for (i = 50; i<src_binary.cols - 50; i += 5) //列
	{
		for (j = 0; j<src_binary.rows; j++)      //行
		{
			if (src_binary.at<uchar>(j, i) == 0)      //统计的是白色像素的数量
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


	cout << "max x = " << max_x << endl;
	cout << "max2 x = " << max2_x << endl;

	cout << "pos x = " << pos_x << endl;
	cout << "pos2 x = " << pos2_x << endl;

	int left_x = MIN(pos_x, pos2_x);

	int right_x = MAX(pos_x, pos2_x);

	src2 = src2(Rect(left_x + 10, 10, right_x - left_x - 26, src2.rows - 20));

	imshow("src2", src2);
}



void GetContoursPic(const char* pSrcFileName, const char* pDstFileName)
{
	IplImage* pSrcImg = NULL;
	IplImage* pFirstFindImg = NULL;
	IplImage* pRoiSrcImg = NULL;
	IplImage* pRatationedImg = NULL;
	IplImage* pSecondFindImg = NULL;
	IplImage* pDstImg = NULL;

	CvSeq* pFirstSeq = NULL;
	CvSeq* pSecondSeq = NULL;

	CvMemStorage* storage = cvCreateMemStorage(0);

	pSrcImg = cvLoadImage(pSrcFileName, 1);
	pFirstFindImg = cvCreateImage(cvGetSize(pSrcImg), IPL_DEPTH_8U, 1);

	//检索外围轮廓  
	cvCvtColor(pSrcImg, pFirstFindImg, CV_BGR2GRAY);  //灰度化  
	cvThreshold(pFirstFindImg, pFirstFindImg, 100, 200, CV_THRESH_BINARY);  //设置阈值，二值化  
																			//注意第5个参数为CV_RETR_EXTERNAL，只检索外框  
	int nCount = cvFindContours(pFirstFindImg, storage, &pFirstSeq, sizeof(CvContour), CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);

	//显示看一下  
	//cvNamedWindow("pSrcImg", 1);
	//cvShowImage("pSrcImg", pSrcImg);

	for (; pFirstSeq != NULL; pFirstSeq = pFirstSeq->h_next)
	{
		if (pFirstSeq->total < 600) //太小的不考虑,这个要考虑图片分辨率大小  
		{
			continue;
		}

		//需要获取的坐标  
		CvPoint2D32f rectpoint[4];
		CvBox2D End_Rage2D = cvMinAreaRect2(pFirstSeq); //寻找包围矩形，获取角度  

		cvBoxPoints(End_Rage2D, rectpoint); //获取4个顶点坐标  
											//与水平线的角度  
		float angle = End_Rage2D.angle;


		//如果角度超过1度，就需要做旋转，否则不需要  
		if (angle > 1 || angle < -1)
		{
			//计算两条边的长度  
			int line1 = sqrt((rectpoint[1].y - rectpoint[0].y)*(rectpoint[1].y - rectpoint[0].y) + (rectpoint[1].x - rectpoint[0].x)*(rectpoint[1].x - rectpoint[0].x));
			int line2 = sqrt((rectpoint[3].y - rectpoint[0].y)*(rectpoint[3].y - rectpoint[0].y) + (rectpoint[3].x - rectpoint[0].x)*(rectpoint[3].x - rectpoint[0].x));

			//为了让正方形横着放，所以旋转角度是不一样的  
			if (line1 > line2) //  
			{
				angle = 90 + angle;
			}

			//新建一个感兴趣的区域图，大小跟原图一样大  
			pRoiSrcImg = cvCreateImage(cvGetSize(pSrcImg), pSrcImg->depth, pSrcImg->nChannels);
			cvSet(pRoiSrcImg, CV_RGB(0, 0, 0));  //颜色都设置为黑色  
												 //对得到的轮廓填充一下  
			cvDrawContours(pFirstFindImg, pFirstSeq, CV_RGB(255, 255, 255), CV_RGB(255, 255, 255), -1, CV_FILLED, 8);
			//把pFirstFindImg这个填充的区域从pSrcImg中抠出来放到pRoiSrcImg上  
			cvCopy(pSrcImg, pRoiSrcImg, pFirstFindImg);

			//再显示一下看看，除了感兴趣的区域，其他部分都是黑色的了  
			cvNamedWindow("pRoiSrcImg", 1);
			cvShowImage("pRoiSrcImg", pRoiSrcImg);

			//创建一个旋转后的图像  
			pRatationedImg = cvCreateImage(cvGetSize(pRoiSrcImg), pRoiSrcImg->depth, pRoiSrcImg->nChannels);

			//对pRoiSrcImg进行旋转  
			CvPoint2D32f center = End_Rage2D.center;  //中心点  
			double map[6];
			CvMat map_matrix = cvMat(2, 3, CV_64FC1, map);
			cv2DRotationMatrix(center, angle, 1.0, &map_matrix);
			cvWarpAffine(pRoiSrcImg, pRatationedImg, &map_matrix, CV_INTER_LINEAR | CV_WARP_FILL_OUTLIERS, cvScalarAll(0));

			//显示一下旋转后的图像  
			//cvNamedWindow("pRatationedImg", 1);
			//cvShowImage("pRatationedImg", pRatationedImg);

			//对旋转后的图片进行轮廓提取  
			pSecondFindImg = cvCreateImage(cvGetSize(pRatationedImg), IPL_DEPTH_8U, 1);
			cvCvtColor(pRatationedImg, pSecondFindImg, CV_BGR2GRAY);  //灰度化  
			cvThreshold(pSecondFindImg, pSecondFindImg, 80, 200, CV_THRESH_BINARY);
			nCount = cvFindContours(pSecondFindImg, storage, &pSecondSeq, sizeof(CvContour), CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
			for (; pSecondSeq != NULL; pSecondSeq = pSecondSeq->h_next)
			{
				if (pSecondSeq->total < 600) //太小的不考虑  
				{
					continue;
				}
				//这时候其实就是一个长方形了，所以获取rect  
				CvRect rect = cvBoundingRect(pSecondSeq);
				cvSetImageROI(pRatationedImg, rect);

				CvSize dstSize;
				dstSize.width = rect.width;
				dstSize.height = rect.height;
				pDstImg = cvCreateImage(dstSize, pRatationedImg->depth, pRatationedImg->nChannels);

				cvCopy(pRatationedImg, pDstImg, 0);
				cvResetImageROI(pRatationedImg);
				//保存成图片  
				cvSaveImage(pDstFileName, pDstImg);
			}
		}
		else
		{
			//角度比较小，本来就是正放着的，所以获取矩形  
			CvRect rect = cvBoundingRect(pFirstSeq);
			//把这个矩形区域设置为感兴趣的区域  
			cvSetImageROI(pSrcImg, rect);

			CvSize dstSize;
			dstSize.width = rect.width;
			dstSize.height = rect.height;
			pDstImg = cvCreateImage(dstSize, pSrcImg->depth, pSrcImg->nChannels);
			//拷贝过来  
			cvCopy(pSrcImg, pDstImg, 0);
			cvResetImageROI(pSrcImg);
			//保存  
			cvSaveImage(pDstFileName, pDstImg);
		}
	}
	//显示一下最后的结果  
	//cvNamedWindow("Contour", 1);
	//cvShowImage("Contour", pDstImg);

	//cvWaitKey(0);

	//释放所有  
	cvReleaseMemStorage(&storage);
	if (pRoiSrcImg)
	{
		cvReleaseImage(&pRoiSrcImg);
	}
	if (pRatationedImg)
	{
		cvReleaseImage(&pRatationedImg);
	}
	if (pSecondFindImg)
	{
		cvReleaseImage(&pSecondFindImg);
	}
	cvReleaseImage(&pDstImg);
	cvReleaseImage(&pFirstFindImg);
	cvReleaseImage(&pSrcImg);
}


void GetContoursPic2(const char* pSrcFileName, const char* pDstFileName)
{
	IplImage* pSrcImg = NULL;
	IplImage* pFirstFindImg = NULL;
	IplImage* pRoiSrcImg = NULL;
	IplImage* pRatationedImg = NULL;
	IplImage* pSecondFindImg = NULL;
	IplImage* pDstImg = NULL;

	CvSeq* pFirstSeq = NULL;
	CvSeq* pSecondSeq = NULL;

	CvMemStorage* storage = cvCreateMemStorage(0);

	pSrcImg = cvLoadImage(pSrcFileName, 1);
	pFirstFindImg = cvCreateImage(cvGetSize(pSrcImg), IPL_DEPTH_8U, 1);

	//检索外围轮廓  
	cvCvtColor(pSrcImg, pFirstFindImg, CV_BGR2GRAY);  //灰度化  
	cvThreshold(pFirstFindImg, pFirstFindImg, 100, 200, CV_THRESH_BINARY);  //设置阈值，二值化  
																			//注意第5个参数为CV_RETR_EXTERNAL，只检索外框  
	int nCount = cvFindContours(pFirstFindImg, storage, &pFirstSeq, sizeof(CvContour), CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);

	//显示看一下  
	//cvNamedWindow("pSrcImg", 1);
	//cvShowImage("pSrcImg", pSrcImg);

	for (; pFirstSeq != NULL; pFirstSeq = pFirstSeq->h_next)
	{
		if (pFirstSeq->total < 600) //太小的不考虑,这个要考虑图片分辨率大小  
		{
			continue;
		}

		//需要获取的坐标  
		CvPoint2D32f rectpoint[4];
		CvBox2D End_Rage2D = cvMinAreaRect2(pFirstSeq); //寻找包围矩形，获取角度  

		cvBoxPoints(End_Rage2D, rectpoint); //获取4个顶点坐标  
											//与水平线的角度  
		float angle = End_Rage2D.angle;

		{
			//角度比较小，本来就是正放着的，所以获取矩形  
			CvRect rect = cvBoundingRect(pFirstSeq);
			//把这个矩形区域设置为感兴趣的区域  
			cvSetImageROI(pSrcImg, rect);

			CvSize dstSize;
			dstSize.width = rect.width;
			dstSize.height = rect.height;
			pDstImg = cvCreateImage(dstSize, pSrcImg->depth, pSrcImg->nChannels);
			//拷贝过来  
			cvCopy(pSrcImg, pDstImg, 0);
			cvResetImageROI(pSrcImg);
			//保存  
			cvSaveImage(pDstFileName, pDstImg);
		}
	}
	//显示一下最后的结果  
	cvNamedWindow("Contour2", 1);
	cvShowImage("Contour2", pDstImg);

	//cvWaitKey(0);

	//释放所有  
	cvReleaseMemStorage(&storage);
	if (pRoiSrcImg)
	{
		cvReleaseImage(&pRoiSrcImg);
	}
	if (pRatationedImg)
	{
		cvReleaseImage(&pRatationedImg);
	}
	if (pSecondFindImg)
	{
		cvReleaseImage(&pSecondFindImg);
	}
	cvReleaseImage(&pDstImg);
	cvReleaseImage(&pFirstFindImg);
	cvReleaseImage(&pSrcImg);
}

void HelpText()
{
	cout << "操作说明" << endl;
	cout << "1:发票识别" << endl;
	cout << "2:退出系统" << endl;
}

void ResultOutput(int res, Mat& img)
{
	switch (res)
	{
	case 1:
		cout << "飞机票\n" << endl;
		//Class10InfoAreaExtract(img);
		Class1InfoExtract(img);
		break;
	case 2:
		cout << "火车票\n" << endl;
		//Class2InfoAreaExtract(img);
		Class2InfoExtract(img);
		break;
	case 3:
		cout << "增值发票\n" << endl;
		Class10InfoAreaExtract(img);
		break;
	case 4:
		cout << "当当京东购物票\n" << endl;
		//Class10InfoAreaExtract(img);
		Class4InfoExtract(img);
		break;
	default:
		cout << "没有找到对应的发票种类！\n" << endl;
		//Class10InfoAreaExtract(img);
		break;
	}
}

//分类判定函数，使用模板匹配方法
int ImageClassify(Mat& img)
{
	Mat templ, result;
	//cout << "debug1" << endl;
	//一共有三类发票
	for (int i = 1; i <= 5; i++)
	{
		//cout << "debug2" << endl;
		//每类发票有1个模板
			//cout << "debug3" << endl;
			char file[100];
			memset(file, 0, sizeof(file));
			sprintf_s(file, "E:\\coding\\vs 2015 test\\SmartSystem\\SmartSystem\\template\\%d.png", i);
			//cout << file << endl;
			templ = imread(file);
			if (!templ.data)
			{
				cout << "读取模板图片失败！" << endl;
				return ERROR;
			}

			int result_cols = img.cols - templ.cols + 1;
			int result_rows = img.rows - templ.rows + 1;
			result.create(result_cols, result_rows, CV_32FC1);

			matchTemplate(img, templ, result, CV_TM_SQDIFF_NORMED);//CV_TM_SQDIFF_NORMED CV_TM_SQDIFF  
			//normalize(result, result, 0, 1, NORM_MINMAX, -1, Mat());

			double minVal;
			double maxVal;
			Point minLoc;
			Point maxLoc;
			Point matchLoc;
			//cout << "匹配度：" << minVal << endl;
			minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc, Mat());
			cout <<i<<"类发票"<< "匹配度：" << minVal << endl;

			matchLoc = minLoc;

			//char WinName[10];
			//memset(WinName, 0, sizeof(WinName));
			//sprintf_s(WinName, "img %d%d", i, j);
			//imshow(WinName, img);

			//阈值判别，小于阈值才认为匹配成功
			if (minVal < CLASS_THRESHOLD)
			{
				//rectangle(img, matchLoc, Point(matchLoc.x + templ.cols, matchLoc.y + templ.rows), Scalar(0, 255, 0), 2, 8, 0);
				return i;
			}

		}


	return UNKNOWN;

}



//主函数，控制流程
int main()
{
	int choice;
	SVM_Init();
	while (1)
	{
		//每次循环都初始化一次系统
		SystemInit();
		//显示
		HelpText();
		cout << "\n请输入你要进行的操作：" << endl;
		cin >> choice;

		if (choice == 2) //退出
		{
			cout << "Goodbye SYSU!" << endl;
			return 0;
		}
		else if (choice == 1) //发票识别
		{
			int num;
			char file[50];
			cout << "请输入要识别的图片编号:" << endl;
			cin >> num;
			sprintf(file, "%d.bmp", num);
			Mat tmpImage = imread(file);
			if (!tmpImage.data)
			{
				cout << "读取原始图片失败！" << endl;
				continue;
			}
			//imshow("原始图", tmpImage);

			GetContoursPic(file, RESULT_IMG_PATH);

			Mat PreProcImage = imread(RESULT_IMG_PATH);

			if (!PreProcImage.data)
			{
				cout << "读取预处理后的图片失败！" << endl;
				return -1;
			}


			resize(PreProcImage, PreProcImage, Size(1160, 817), 0, 0, CV_INTER_LINEAR); //统一图片规格，1160*817
			 //cvtColor(PreProcImage, PreProcImage, CV_BGR2GRAY);
			imwrite("resize.jpg", PreProcImage);
			//imshow("预处理后图片", PreProcImage);

			Mat dst;
			//基于直线探测的角度矫正
			ImageRecify(PreProcImage, dst);
			imwrite("dst.jpg", dst);
		
			GetContoursPic2("dst.jpg", "dst2.jpg");
			//需要设计一个发票类型判定函数
			Mat PreProcImage2 = imread("dst2.jpg");
			//imshow("微调前", PreProcImage2);
			PreProcImage2 = PreProcImage2(Rect(5, 5, PreProcImage2.cols - 10, PreProcImage2.rows - 10)); //进行区域微调
			imshow("微调后", PreProcImage2);
			//imshow("再一次轮廓提取", PreProcImage2);
	
			//Mat tmp;
			//cvtColor(PreProcImage2, tmp, CV_RGB2GRAY);
			//imwrite("resize2.jpg", tmp);

			//Mat tmp2 = imread("resize2.jpg");
			//发票分类
			//int result = ImageClassify(tmp2);

			//cout << file << "的类型为：";
			//ResultOutput(result, tmp2);

			//发票分类
			int result = InvoiceClassify(tmpImage);
			if (result == 4) //如果是京东当当发票
			{
				ResultOutput(result, tmpImage);
			}
			else
			{
				ResultOutput(result, PreProcImage2);
			}


			


#if 0
			if (1)  //表格类发票
			{
				cout << "表格类发票\n" << endl;
				Class10InfoAreaExtract(PreProcImage2);
			}
			else  //非表格类发票
			{
				cout << "非表格类发票\n" << endl;
				Class2InfoAreaExtract(PreProcImage2);
			}
#endif
			
			waitKey();
		}
		else
		{
			continue;
		}

	}
	SystemInit();
	return 0;
}

