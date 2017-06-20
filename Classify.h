#ifndef __CLASSIFT_H__
#define __CLASSIFT_H__
using namespace std;
using namespace cv;

#define LOG(file) cout<<file<<endl


extern int InvoiceClassify(Mat& img);
extern void SVM_Init();

#endif

