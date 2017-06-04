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
#include <set>

using namespace cv;
using namespace std;

#define RESULT_IMG_PATH  "result.jpg"
#define ERROR -1
#define UNKNOWN 0
#define CLASS_THRESHOLD  0.02    //分类时设定的阈值

int g_info_area_count = 0;
int g_final_area_count = 0;

Mat src;
Mat src_gray;
#define DETAILED_INFO_AREA_THRESH  100



void SaveDetailedArea()
{
	Mat threshold_output;
	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;

	/// 使用Threshold检测边缘
	threshold(src_gray, threshold_output, DETAILED_INFO_AREA_THRESH, 255, THRESH_BINARY);
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
	src.copyTo(drawing);
	g_final_area_count = 0; //初始化
	for (int i = 0; i< contours.size(); i++)
	{
		//Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
		//drawContours(drawing, contours_poly, i, color, 1, 8, vector<Vec4i>(), 0, Point());
		int rate = boundRect[i].width / boundRect[i].height;
#if 1
		if (boundRect[i].area() < 300)
		{
			continue;
		}
#endif
		rectangle(drawing, boundRect[i].tl(), boundRect[i].br(), Scalar(0, 0, 255), 2, 8, 0);
		Rect r(boundRect[i].tl(), boundRect[i].br());
		Mat tmp = drawing(r);
		char file[100];

		sprintf(file, "E:\\coding\\vs 2015 test\\SmartSystem\\SmartSystem\\final_info_area\\tmp%d.jpg", ++g_final_area_count);
		imwrite(file, tmp);
		//circle(drawing, center[i], (int)radius[i], color, 2, 8, 0);
	}

	/// 显示在一个窗口
	namedWindow("Contours", CV_WINDOW_AUTOSIZE);
	imshow("Contours", drawing);
}

//对第一次提取出来的区域在进行进一步的提取的
void DetailedInfoAreaExtract(Mat& src2)
{
	cvtColor(src2, src, COLOR_RGB2GRAY);
	imshow("灰度化", src);

	Mat out;
	boxFilter(src, out, -1, Size(5, 5));//-1指原图深度
	imshow("方框滤波", out);

	// 局部二值化
	int blockSize = 25;
	int constValue = 10;
	Mat local;
	adaptiveThreshold(out, local, 255, CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY_INV, blockSize, constValue);
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
	cvtColor(img, src_gray, CV_BGR2GRAY);
	blur(src_gray, src_gray, Size(3, 3));

	/// 创建窗口
	char* source_window = "Source";
	namedWindow(source_window, CV_WINDOW_AUTOSIZE);
	imshow(source_window, src);

	SaveDetailedArea();
}

void FinalInfoGenerator()
{
	for (int i = 1; i <= g_info_area_count; i++)
	{
		char file[20];
		sprintf_s(file, "info%d.jpg", i);
		Mat tmpImg = imread(file);

		DetailedInfoAreaExtract(tmpImg);
	}
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
			//cvNamedWindow("pRoiSrcImg", 1);
			//cvShowImage("pRoiSrcImg", pRoiSrcImg);

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
		cout << "1类发票\n" << endl;
		Class10InfoAreaExtract(img);
		break;
	case 2:
		cout << "2类发票\n" << endl;
		break;
	case 3:
		cout << "3类发票\n" << endl;
		Class10InfoAreaExtract(img);
		break;
	case 4:
		cout << "4类发票\n" << endl;
		Class10InfoAreaExtract(img);
		break;
	case 5:
		cout << "广东省通用机打发票\n" << endl;
		Class7InfoAreaExtract(img);
		break;
	case 6:
		cout << "6类发票\n" << endl;
		Class10InfoAreaExtract(img);
		break;
	case 7:
		cout << "7类发票\n" << endl;
		Class7InfoAreaExtract(img);
		break;
	case 8:
		cout << "8类发票\n" << endl;
		break;
	case 9:
		cout << "9类发票\n" << endl;
		Class7InfoAreaExtract(img);
		break;
	case 10:
		cout << "10类发票\n" << endl;
		Class10InfoAreaExtract(img);
		FinalInfoGenerator();
		break;
	default:
		cout << "没有找到对应的发票种类！\n" << endl;
		Class10InfoAreaExtract(img);
		break;
	}
}

//分类判定函数，使用模板匹配方法
int ImageClassify(Mat& img)
{
	Mat templ, result;
	//cout << "debug1" << endl;
	//一共有三类发票
	for (int i = 1; i <= 10; i++)
	{
		//cout << "debug2" << endl;
		//每类发票有两个模板
		for (int j = 1; j <= 2; j++)
		{
			//cout << "debug3" << endl;
			char file[100];
			memset(file, 0, sizeof(file));
			sprintf_s(file, "E:\\coding\\vs 2015 test\\SmartSystem\\SmartSystem\\template\\%d%d.png", i, j);
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

	}

	return UNKNOWN;

}


int main()
{
	int choice;
	while (1)
	{
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

			resize(PreProcImage, PreProcImage, Size(1160, 817), 0, 0, CV_INTER_LINEAR); //统一图片规格，600*450
			 //cvtColor(PreProcImage, PreProcImage, CV_BGR2GRAY);
			imwrite("resize.jpg", PreProcImage);
			imshow("预处理后图片", PreProcImage);
			//发票分类
			int result = ImageClassify(PreProcImage);

			cout << file << "的类型为：";
			ResultOutput(result, tmpImage);

			
			waitKey();
		}
		else
		{
			continue;
		}

	}

	return 0;
}