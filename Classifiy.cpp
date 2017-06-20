
#include <opencv2/opencv.hpp>  
#include "opencv2/highgui/highgui.hpp"  
#include "opencv2/imgproc/imgproc.hpp"  
#include <opencv2\imgproc\imgproc.hpp>
#include <string>

using namespace cv;
using namespace std;
using namespace cv::ml;

int LineDetector(Mat& src);
float ColorRecongize(Mat& img);
bool templateMatch(Mat& img);
int InvoiceClassify(Mat& img);

#define LOG(file) cout<<file<<endl

Ptr<SVM> svm;

void SVM_Init()
{
	//训练需要用到的数据  
	int label[13] = { 1, 1,2,2,2 ,3,3,3,3,3,3,4,4 };
	float trainData[13][4] = { { 114,13674,17 ,100 },{ 115,17922,31,100 },{ 87,16,17 ,0 } ,{ 80,2,12,0 },{ 83,2,11,0 },{ 75,24,9,100 },{ 77,18,9,100 },{ 71,22,7,100 },{ 77,8,7,100 },{ 99,18,14,100 },{ 80,24,11 ,100 } ,{ 21,4,8 ,100 },{ 21,6,10 ,100 } };
	//转为Mat以调用  
	Mat trainMat(13, 4, CV_32FC1, trainData);
	Mat trainlabel(13, 1, CV_32SC1, label);
	//训练的初始化  
	svm = SVM::create();
	svm->setType(SVM::C_SVC);
	svm->setKernel(SVM::LINEAR);
	svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 100, 1e-6));
	//开始训练  
	LOG("SVM training...");
	svm->train(trainMat, ROW_SAMPLE, trainlabel);
	LOG("SVM train done");
}

int InvoiceClassify(Mat& img)
{
	LOG("compute parameters...");
	//分类参数计算
	Mat result, tmp;
	float WHrate = (float)img.cols / img.rows;  //计算宽高的比值
	float para3 = ColorRecongize(img);
	float para1 = WHrate / 2 * 100;
	resize(img, tmp, Size(1160, 817));
	float para2 = (float)LineDetector(tmp) / 50 * 100;

	cout << "WHrate:" << para1 << endl;
	cout << "line cout:" << para2 << endl;
	cout << "clolor cout:" << para3 << endl;

	float para4 = templateMatch(tmp) * 100;
	cout << "is match:" << para4 << endl;

	float teatData[1][4] = { para1,para2,para3 ,para4 };
	Mat query(1, 4, CV_32FC1, teatData);

	int res = (int)svm->predict(query);


	cout << "分类结果:" << res << endl;

	return res;

}

int LineDetector(Mat& src)
{
	int line_count = 0;
	Mat midImage, dstImage;
	//边缘检测
	Canny(src, midImage, 50, 200, 3);
	//灰度化
	cvtColor(midImage, dstImage, CV_GRAY2BGR);
	// 定义矢量结构存放检测出来的直线
	vector<Vec2f> lines;
	//通过这个函数，我们就可以得到检测出来的直线集合了
	HoughLines(midImage, lines, 1, CV_PI / 180, 400, 0, 0);
	//这里注意第五个参数，表示阈值，阈值越大，表明检测的越精准，速度越快，得到的直线越少（得到的直线都是很有把握的直线）
	//这里得到的lines是包含rho和theta的，而不包括直线上的点，所以下面需要根据得到的rho和theta来建立一条直线

	//依次画出每条线段
	int pre_y = 0;
	for (size_t i = 0; i < lines.size(); i++)
	{
		float rho = lines[i][0]; //就是圆的半径r
		float theta = lines[i][1]; //就是直线的角度
		Point pt1, pt2;
		double a = cos(theta), b = sin(theta);
		double x0 = a*rho, y0 = b*rho;
		pt1.x = cvRound(x0 + 1000 * (-b));
		pt1.y = cvRound(y0 + 1000 * (a));
		pt2.x = cvRound(x0 - 1000 * (-b));
		pt2.y = cvRound(y0 - 1000 * (a));

		//if()


		line(dstImage, pt1, pt2, Scalar(55, 100, 195), 1, LINE_AA); //Scalar函数用于调节线段颜色，就是你想检测到的线段显示的是什么颜色
		line_count++;
		//imshow("边缘检测后的图", midImage);
		//namedWindow("最终效果图", 0);
		//imshow("最终效果图", dstImage);
	}

	return line_count;
}

float ColorRecongize(Mat& src)
{
	float Color = 0.0;
	Mat src_gray, src_binary;


	//二值化
	//转化为灰度图像
	cvtColor(src, src_gray, CV_RGB2GRAY);
	//二值化图像
	adaptiveThreshold(src_gray, src_binary, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY, 25, 10);
	//imshow("binary", src_binary);

	int sum = 0;
	//遍历全图，求图片黑色像素点的个数
	for (int i = 0; i<src_binary.rows; i++) //列
	{
		for (int j = 0; j<src_binary.cols; j++)      //行
		{
			//cout << src_binary.at<uchar>(i, j) << endl;
			if (src_binary.at<uchar>(i, j) == 0)   //统计的是白色像素的数量
			{
				sum++;
			}

		}

	}

	//求黑色像素点占全图的比例
	Color = 100 * sum / (src_binary.rows*src_binary.cols);
	cout << "color :" << Color << endl;

	return Color;
}

bool templateMatch(Mat& img)
{
	Mat templ, result;
	templ = imread("t.png");
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

	//char WinName[10];
	//memset(WinName, 0, sizeof(WinName));
	//sprintf_s(WinName, "img %d%d", i, j);
	//imshow("show", img);
	//waitKey();

	//阈值判别，小于0.1才认为匹配成功
	cout << "template match:" << minVal << endl;
	if (minVal < 0.1)
	{
		return true;
	}

	return false;
}