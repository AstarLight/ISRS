
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
	//ѵ����Ҫ�õ�������  
	int label[13] = { 1, 1,2,2,2 ,3,3,3,3,3,3,4,4 };
	float trainData[13][4] = { { 114,13674,17 ,100 },{ 115,17922,31,100 },{ 87,16,17 ,0 } ,{ 80,2,12,0 },{ 83,2,11,0 },{ 75,24,9,100 },{ 77,18,9,100 },{ 71,22,7,100 },{ 77,8,7,100 },{ 99,18,14,100 },{ 80,24,11 ,100 } ,{ 21,4,8 ,100 },{ 21,6,10 ,100 } };
	//תΪMat�Ե���  
	Mat trainMat(13, 4, CV_32FC1, trainData);
	Mat trainlabel(13, 1, CV_32SC1, label);
	//ѵ���ĳ�ʼ��  
	svm = SVM::create();
	svm->setType(SVM::C_SVC);
	svm->setKernel(SVM::LINEAR);
	svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 100, 1e-6));
	//��ʼѵ��  
	LOG("SVM training...");
	svm->train(trainMat, ROW_SAMPLE, trainlabel);
	LOG("SVM train done");
}

int InvoiceClassify(Mat& img)
{
	LOG("compute parameters...");
	//�����������
	Mat result, tmp;
	float WHrate = (float)img.cols / img.rows;  //�����ߵı�ֵ
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


	cout << "������:" << res << endl;

	return res;

}

int LineDetector(Mat& src)
{
	int line_count = 0;
	Mat midImage, dstImage;
	//��Ե���
	Canny(src, midImage, 50, 200, 3);
	//�ҶȻ�
	cvtColor(midImage, dstImage, CV_GRAY2BGR);
	// ����ʸ���ṹ��ż�������ֱ��
	vector<Vec2f> lines;
	//ͨ��������������ǾͿ��Եõ���������ֱ�߼�����
	HoughLines(midImage, lines, 1, CV_PI / 180, 400, 0, 0);
	//����ע��������������ʾ��ֵ����ֵԽ�󣬱�������Խ��׼���ٶ�Խ�죬�õ���ֱ��Խ�٣��õ���ֱ�߶��Ǻ��а��յ�ֱ�ߣ�
	//����õ���lines�ǰ���rho��theta�ģ���������ֱ���ϵĵ㣬����������Ҫ���ݵõ���rho��theta������һ��ֱ��

	//���λ���ÿ���߶�
	int pre_y = 0;
	for (size_t i = 0; i < lines.size(); i++)
	{
		float rho = lines[i][0]; //����Բ�İ뾶r
		float theta = lines[i][1]; //����ֱ�ߵĽǶ�
		Point pt1, pt2;
		double a = cos(theta), b = sin(theta);
		double x0 = a*rho, y0 = b*rho;
		pt1.x = cvRound(x0 + 1000 * (-b));
		pt1.y = cvRound(y0 + 1000 * (a));
		pt2.x = cvRound(x0 - 1000 * (-b));
		pt2.y = cvRound(y0 - 1000 * (a));

		//if()


		line(dstImage, pt1, pt2, Scalar(55, 100, 195), 1, LINE_AA); //Scalar�������ڵ����߶���ɫ�����������⵽���߶���ʾ����ʲô��ɫ
		line_count++;
		//imshow("��Ե�����ͼ", midImage);
		//namedWindow("����Ч��ͼ", 0);
		//imshow("����Ч��ͼ", dstImage);
	}

	return line_count;
}

float ColorRecongize(Mat& src)
{
	float Color = 0.0;
	Mat src_gray, src_binary;


	//��ֵ��
	//ת��Ϊ�Ҷ�ͼ��
	cvtColor(src, src_gray, CV_RGB2GRAY);
	//��ֵ��ͼ��
	adaptiveThreshold(src_gray, src_binary, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY, 25, 10);
	//imshow("binary", src_binary);

	int sum = 0;
	//����ȫͼ����ͼƬ��ɫ���ص�ĸ���
	for (int i = 0; i<src_binary.rows; i++) //��
	{
		for (int j = 0; j<src_binary.cols; j++)      //��
		{
			//cout << src_binary.at<uchar>(i, j) << endl;
			if (src_binary.at<uchar>(i, j) == 0)   //ͳ�Ƶ��ǰ�ɫ���ص�����
			{
				sum++;
			}

		}

	}

	//���ɫ���ص�ռȫͼ�ı���
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
	//cout << "ƥ��ȣ�" << minVal << endl;
	minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc, Mat());

	cout << "ƥ��ȣ�" << minVal << endl;

	matchLoc = minLoc;

	//rectangle(img, matchLoc, Point(matchLoc.x + templ.cols, matchLoc.y + templ.rows), Scalar(0, 255, 0), 2, 8, 0);

	//char WinName[10];
	//memset(WinName, 0, sizeof(WinName));
	//sprintf_s(WinName, "img %d%d", i, j);
	//imshow("show", img);
	//waitKey();

	//��ֵ�б�С��0.1����Ϊƥ��ɹ�
	cout << "template match:" << minVal << endl;
	if (minVal < 0.1)
	{
		return true;
	}

	return false;
}