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
#define CLASS_THRESHOLD  0.02    //����ʱ�趨����ֵ

int g_info_area_count = 0;
int g_final_area_count = 0;

Mat src;
Mat src_gray;
#define DETAILED_INFO_AREA_THRESH  100

void EdgeLocation(Mat& src2);

void SaveDetailedArea()
{
	Mat threshold_output;
	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;

	/// ʹ��Threshold����Ե
	threshold(src_gray, threshold_output, DETAILED_INFO_AREA_THRESH, 255, THRESH_BINARY);

	/// �ҵ�����
	findContours(threshold_output, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));

	/// ����αƽ����� + ��ȡ���κ�Բ�α߽��
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


	/// ����������� + ��Χ�ľ��ο� + Բ�ο�
	Mat drawing = Mat::zeros(threshold_output.size(), CV_8UC3);
	src.copyTo(drawing);
	g_final_area_count = 0; //��ʼ��
	for (int i = 0; i< contours.size(); i++)
	{
		//Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
		//drawContours(drawing, contours_poly, i, color, 1, 8, vector<Vec4i>(), 0, Point());
		int rate = boundRect[i].width / boundRect[i].height;
#if 1
		if (boundRect[i].area() < 300 )
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
		rectangle(src, r,Scalar(0, 0, 255), 2, 8, 0);
		
		//Rect r(boundRect[i].tl(), boundRect[i].br());
		Mat tmp = drawing(r);
		char file[100];

		sprintf(file, "E:\\coding\\vs 2015 test\\SmartSystem\\SmartSystem\\final_info_area\\tmp%d.jpg", ++g_final_area_count);
		imwrite(file, tmp);
		//circle(drawing, center[i], (int)radius[i], color, 2, 8, 0);
	}

	/// ��ʾ��һ������
	namedWindow("Contours", CV_WINDOW_AUTOSIZE);
	imshow("Contours", src);
}

//�Ե�һ����ȡ�����������ڽ��н�һ������ȡ��
void DetailedInfoAreaExtract(Mat& src2)
{
	cvtColor(src2, src, COLOR_RGB2GRAY);
	imshow("�ҶȻ�", src);
/*
	Mat out;
	boxFilter(src, out, -1, Size(5, 5));//-1ָԭͼ���
	imshow("�����˲�", out);
*/
	// �ֲ���ֵ��
	int blockSize = 25;
	int constValue = 10;
	Mat local;
	adaptiveThreshold(src, local, 255, CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY_INV, blockSize, constValue);
	imshow("��ֵ��", local);

	Mat out2;
	//��ȡ�Զ����
	Mat element = getStructuringElement(MORPH_RECT, Size(8, 8)); //��һ������MORPH_RECT��ʾ���εľ���ˣ���Ȼ������ѡ����Բ�εġ������͵�
																 //���Ͳ���
	dilate(local, out2, element);
	namedWindow("���Ͳ���", WINDOW_NORMAL);
	imshow("���Ͳ���", out2);
	imwrite("dd.jpg", out2);

	Mat img;
	/// ����ԭͼ��, ����3ͨ��ͼ��
	img = imread("dd.jpg", 1);

	/// ת���ɻҶ�ͼ�񲢽���ƽ��
	cvtColor(img, src_gray, CV_BGR2GRAY);
	blur(src_gray, src_gray, Size(3, 3));

	/// ��������
	char* source_window = "Source";
	namedWindow(source_window, CV_WINDOW_AUTOSIZE);
	imshow(source_window, src);

	//SaveDetailedArea();
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

void Class2InfoAreaExtract(Mat& src2)
{
	Mat img2 = src2(Rect(20, 10, src2.cols - 40, src2.rows - 20)); //������Ե
																   //Mat img2 = src2;
	imshow("img2", img2);
	cvtColor(img2, src, COLOR_RGB2GRAY);
	imshow("�ҶȻ�", src);

	Mat out;
	boxFilter(src, out, -1, Size(5, 5));//-1ָԭͼ���
	imshow("�����˲�", out);

	// �ֲ���ֵ��
	int blockSize = 25;
	int constValue = 10;
	Mat local;
	adaptiveThreshold(out, local, 255, CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY_INV, blockSize, constValue);
	imshow("��ֵ��", local);
	imwrite("binary.bmp", local);

	Mat out3;
	//��ȡ�Զ����
	Mat element2 = getStructuringElement(MORPH_RECT, Size(5, 5)); //��һ������MORPH_RECT��ʾ���εľ���ˣ���Ȼ������ѡ����Բ�εġ������͵�
																  //��ʴ����
	erode(local, out3, element2);
	namedWindow("��ʴ����", WINDOW_NORMAL);
	imshow("��ʴ����", out3);
	imwrite("erode.bmp", out3);

	Mat out2;
	//��ȡ�Զ����

	Mat element = getStructuringElement(MORPH_RECT, Size(30, 1)); //��һ������MORPH_RECT��ʾ���εľ���ˣ���Ȼ������ѡ����Բ�εġ������͵�
																  //���Ͳ���
	dilate(out3, out2, element, Point(-1, -1), 10); //����10��
	namedWindow("���Ͳ���", WINDOW_NORMAL);
	imshow("���Ͳ���", out2);
	imwrite("dd.jpg", out2);

	Mat img;
	/// ����ԭͼ��, ����3ͨ��ͼ��
	img = imread("dd.jpg", 1);

	/// ת���ɻҶ�ͼ�񲢽���ƽ��
	cvtColor(img, src_gray, CV_BGR2GRAY);
	blur(src_gray, src_gray, Size(3, 3));

	/// ��������
	char* source_window = "Source";
	namedWindow(source_window, CV_WINDOW_AUTOSIZE);
	imshow(source_window, src);

	SaveDetailedArea();
}



void Class3InfoAreaExtract(Mat& src2)
{
	resize(src2, src2, Size(1160, 817));
	Mat local;

	//ת��Ϊ�Ҷ�ͼ��
	cvtColor(src2, src, CV_RGB2GRAY);
	
	//��ֵ��ͼ��
	adaptiveThreshold(src, local, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY_INV, 25, 10);

	Mat out;
	//��ȡ�Զ����
	Mat element = getStructuringElement(MORPH_RECT, Size(3, 3)); //��һ������MORPH_RECT��ʾ���εľ���ˣ���Ȼ������ѡ����Բ�εġ������͵�
																 //��ʴ����
	erode(local, out, element, Point(-1, -1), 2);

	out = out(Rect(20, 20, out.cols - 40, out.rows - 40));
	src = src(Rect(20, 20, src.cols - 40, src.rows - 40));
	//src_gray = out;
	namedWindow("��ʴ����", WINDOW_NORMAL);
	imshow("��ʴ����", out);

	Mat out2;
	//��ȡ�Զ����
	Mat element2 = getStructuringElement(MORPH_RECT, Size(40, 2)); //��һ������MORPH_RECT��ʾ���εľ���ˣ���Ȼ������ѡ����Բ�εġ������͵�
																   //���Ͳ���
	dilate(out, out2, element2, Point(-1, -1), 5);

	imshow("out2", out2);

	imwrite("dd.jpg", out2);

	Mat img;
	/// ����ԭͼ��, ����3ͨ��ͼ��
	img = imread("dd.jpg", 1);

	/// ת���ɻҶ�ͼ�񲢽���ƽ��
	cvtColor(img, src_gray, CV_BGR2GRAY);
	blur(src_gray, src_gray, Size(3, 3));

	SaveDetailedArea();
}

void EdgeLocation(Mat& src2)
{
	//resize(src, src, Size(1160, 817));
	imwrite("7p.bmp", src2);
	Mat src_gray, src_binary;

	//ת��Ϊ�Ҷ�ͼ��
	cvtColor(src2, src_gray, CV_RGB2GRAY);
	//��ֵ��ͼ��
	adaptiveThreshold(src_gray, src_binary, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY, 25, 10);
	int* v = new int[src2.cols * 4];

	memset(v, 0, src2.cols * 4);

	int i, j;
	//����һ����
	//��ֱ��������ۼӣ����֣�
	int max_x = 0;
	int pos_x = 0;
	int max2_x = 0;
	int pos2_x = 0;
	//ע��i�Ĳ��������������Ĳ�����Ϊ�˲���һЩ�ٽ������ظߵĵ�����������������ɵ��ص�
	//ע��i�������յ㣬��+��-��50������Ϊ�˱��ⷢƱ��ԵӰ�����ؼ��
	for (i = 50; i<src_binary.cols - 50; i += 5) //��
	{
		for (j = 0; j<src_binary.rows; j++)      //��
		{
			if (src_binary.at<uchar>(j, i) == 0)      //ͳ�Ƶ��ǰ�ɫ���ص�����
				v[i]++;
		}
		cout << "v: " << v[i] << "  pos: " << i << endl;
		//�ҳ�top2������λ��
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

	//������Χ����  
	cvCvtColor(pSrcImg, pFirstFindImg, CV_BGR2GRAY);  //�ҶȻ�  
	cvThreshold(pFirstFindImg, pFirstFindImg, 100, 200, CV_THRESH_BINARY);  //������ֵ����ֵ��  
																			//ע���5������ΪCV_RETR_EXTERNAL��ֻ�������  
	int nCount = cvFindContours(pFirstFindImg, storage, &pFirstSeq, sizeof(CvContour), CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);

	//��ʾ��һ��  
	//cvNamedWindow("pSrcImg", 1);
	//cvShowImage("pSrcImg", pSrcImg);

	for (; pFirstSeq != NULL; pFirstSeq = pFirstSeq->h_next)
	{
		if (pFirstSeq->total < 600) //̫С�Ĳ�����,���Ҫ����ͼƬ�ֱ��ʴ�С  
		{
			continue;
		}

		//��Ҫ��ȡ������  
		CvPoint2D32f rectpoint[4];
		CvBox2D End_Rage2D = cvMinAreaRect2(pFirstSeq); //Ѱ�Ұ�Χ���Σ���ȡ�Ƕ�  

		cvBoxPoints(End_Rage2D, rectpoint); //��ȡ4����������  
											//��ˮƽ�ߵĽǶ�  
		float angle = End_Rage2D.angle;


		//����Ƕȳ���1�ȣ�����Ҫ����ת��������Ҫ  
		if (angle > 1 || angle < -1)
		{
			//���������ߵĳ���  
			int line1 = sqrt((rectpoint[1].y - rectpoint[0].y)*(rectpoint[1].y - rectpoint[0].y) + (rectpoint[1].x - rectpoint[0].x)*(rectpoint[1].x - rectpoint[0].x));
			int line2 = sqrt((rectpoint[3].y - rectpoint[0].y)*(rectpoint[3].y - rectpoint[0].y) + (rectpoint[3].x - rectpoint[0].x)*(rectpoint[3].x - rectpoint[0].x));

			//Ϊ���������κ��ŷţ�������ת�Ƕ��ǲ�һ����  
			if (line1 > line2) //  
			{
				angle = 90 + angle;
			}

			//�½�һ������Ȥ������ͼ����С��ԭͼһ����  
			pRoiSrcImg = cvCreateImage(cvGetSize(pSrcImg), pSrcImg->depth, pSrcImg->nChannels);
			cvSet(pRoiSrcImg, CV_RGB(0, 0, 0));  //��ɫ������Ϊ��ɫ  
												 //�Եõ����������һ��  
			cvDrawContours(pFirstFindImg, pFirstSeq, CV_RGB(255, 255, 255), CV_RGB(255, 255, 255), -1, CV_FILLED, 8);
			//��pFirstFindImg������������pSrcImg�пٳ����ŵ�pRoiSrcImg��  
			cvCopy(pSrcImg, pRoiSrcImg, pFirstFindImg);

			//����ʾһ�¿��������˸���Ȥ�������������ֶ��Ǻ�ɫ����  
			cvNamedWindow("pRoiSrcImg", 1);
			cvShowImage("pRoiSrcImg", pRoiSrcImg);

			//����һ����ת���ͼ��  
			pRatationedImg = cvCreateImage(cvGetSize(pRoiSrcImg), pRoiSrcImg->depth, pRoiSrcImg->nChannels);

			//��pRoiSrcImg������ת  
			CvPoint2D32f center = End_Rage2D.center;  //���ĵ�  
			double map[6];
			CvMat map_matrix = cvMat(2, 3, CV_64FC1, map);
			cv2DRotationMatrix(center, angle, 1.0, &map_matrix);
			cvWarpAffine(pRoiSrcImg, pRatationedImg, &map_matrix, CV_INTER_LINEAR | CV_WARP_FILL_OUTLIERS, cvScalarAll(0));

			//��ʾһ����ת���ͼ��  
			//cvNamedWindow("pRatationedImg", 1);
			//cvShowImage("pRatationedImg", pRatationedImg);

			//����ת���ͼƬ����������ȡ  
			pSecondFindImg = cvCreateImage(cvGetSize(pRatationedImg), IPL_DEPTH_8U, 1);
			cvCvtColor(pRatationedImg, pSecondFindImg, CV_BGR2GRAY);  //�ҶȻ�  
			cvThreshold(pSecondFindImg, pSecondFindImg, 80, 200, CV_THRESH_BINARY);
			nCount = cvFindContours(pSecondFindImg, storage, &pSecondSeq, sizeof(CvContour), CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
			for (; pSecondSeq != NULL; pSecondSeq = pSecondSeq->h_next)
			{
				if (pSecondSeq->total < 600) //̫С�Ĳ�����  
				{
					continue;
				}
				//��ʱ����ʵ����һ���������ˣ����Ի�ȡrect  
				CvRect rect = cvBoundingRect(pSecondSeq);
				cvSetImageROI(pRatationedImg, rect);

				CvSize dstSize;
				dstSize.width = rect.width;
				dstSize.height = rect.height;
				pDstImg = cvCreateImage(dstSize, pRatationedImg->depth, pRatationedImg->nChannels);

				cvCopy(pRatationedImg, pDstImg, 0);
				cvResetImageROI(pRatationedImg);
				//�����ͼƬ  
				cvSaveImage(pDstFileName, pDstImg);
			}
		}
		else
		{
			//�ǶȱȽ�С���������������ŵģ����Ի�ȡ����  
			CvRect rect = cvBoundingRect(pFirstSeq);
			//�����������������Ϊ����Ȥ������  
			cvSetImageROI(pSrcImg, rect);

			CvSize dstSize;
			dstSize.width = rect.width;
			dstSize.height = rect.height;
			pDstImg = cvCreateImage(dstSize, pSrcImg->depth, pSrcImg->nChannels);
			//��������  
			cvCopy(pSrcImg, pDstImg, 0);
			cvResetImageROI(pSrcImg);
			//����  
			cvSaveImage(pDstFileName, pDstImg);
		}
	}
	//��ʾһ�����Ľ��  
	//cvNamedWindow("Contour", 1);
	//cvShowImage("Contour", pDstImg);

	//cvWaitKey(0);

	//�ͷ�����  
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
	cout << "����˵��" << endl;
	cout << "1:��Ʊʶ��" << endl;
	cout << "2:�˳�ϵͳ" << endl;
}

void ResultOutput(int res, Mat& img)
{
	switch (res)
	{
	case 1:
		cout << "1�෢Ʊ\n" << endl;
		Class10InfoAreaExtract(img);
		break;
	case 2:
		cout << "2�෢Ʊ\n" << endl;
		Class2InfoAreaExtract(img);
		break;
	case 3:
		cout << "3�෢Ʊ\n" << endl;
		Class10InfoAreaExtract(img);
		break;
	case 4:
		cout << "4�෢Ʊ\n" << endl;
		Class10InfoAreaExtract(img);
		break;
	case 5:
		cout << "�㶫ʡͨ�û���Ʊ\n" << endl;
		//Class7InfoAreaExtract(img);
		Class10InfoAreaExtract(img);
		break;
	case 6:
		cout << "6�෢Ʊ\n" << endl;
		Class10InfoAreaExtract(img);
		break;
	case 7:
		cout << "7�෢Ʊ\n" << endl;
		Class7InfoAreaExtract(img);
		break;
	case 8:
		cout << "8�෢Ʊ\n" << endl;
		
		Class3InfoAreaExtract(img);
		break;
	case 9:
		cout << "9�෢Ʊ\n" << endl;
		Class7InfoAreaExtract(img);
		break;
	case 10:
		cout << "10�෢Ʊ\n" << endl;
		Class10InfoAreaExtract(img);
		//FinalInfoGenerator();
		break;
	default:
		cout << "û���ҵ���Ӧ�ķ�Ʊ���࣡\n" << endl;
		Class10InfoAreaExtract(img);
		break;
	}
}

//�����ж�������ʹ��ģ��ƥ�䷽��
int ImageClassify(Mat& img)
{
	Mat templ, result;
	//cout << "debug1" << endl;
	//һ�������෢Ʊ
	for (int i = 1; i <= 10; i++)
	{
		//cout << "debug2" << endl;
		//ÿ�෢Ʊ������ģ��
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
				cout << "��ȡģ��ͼƬʧ�ܣ�" << endl;
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
			//cout << "ƥ��ȣ�" << minVal << endl;
			minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc, Mat());
			cout <<i<<"�෢Ʊ"<< "ƥ��ȣ�" << minVal << endl;

			matchLoc = minLoc;


			//char WinName[10];
			//memset(WinName, 0, sizeof(WinName));
			//sprintf_s(WinName, "img %d%d", i, j);
			//imshow(WinName, img);

			//��ֵ�б�С����ֵ����Ϊƥ��ɹ�
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
		cout << "\n��������Ҫ���еĲ�����" << endl;
		cin >> choice;

		if (choice == 2) //�˳�
		{
			cout << "Goodbye SYSU!" << endl;
			return 0;
		}
		else if (choice == 1) //��Ʊʶ��
		{
			int num;
			char file[50];
			cout << "������Ҫʶ���ͼƬ���:" << endl;
			cin >> num;
			sprintf(file, "%d.bmp", num);
			Mat tmpImage = imread(file);
			if (!tmpImage.data)
			{
				cout << "��ȡԭʼͼƬʧ�ܣ�" << endl;
				continue;
			}
			//imshow("ԭʼͼ", tmpImage);

			GetContoursPic(file, RESULT_IMG_PATH);

			Mat PreProcImage = imread(RESULT_IMG_PATH);

			if (!PreProcImage.data)
			{
				cout << "��ȡԤ������ͼƬʧ�ܣ�" << endl;
				return -1;
			}

			resize(PreProcImage, PreProcImage, Size(1160, 817), 0, 0, CV_INTER_LINEAR); //ͳһͼƬ���600*450
			 //cvtColor(PreProcImage, PreProcImage, CV_BGR2GRAY);
			imwrite("resize.jpg", PreProcImage);
			imshow("Ԥ�����ͼƬ", PreProcImage);
			//��Ʊ����
			int result = ImageClassify(PreProcImage);

			cout << file << "������Ϊ��";
			ResultOutput(result, PreProcImage);

			
			waitKey();
		}
		else
		{
			continue;
		}

	}

	return 0;
}