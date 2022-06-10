#include <opencv2/opencv.hpp>
#include <iostream>
#include <math.h>

using namespace cv;
using namespace std;

const char input[] = "cornkernels";

int main() {
	
	Mat src = imread("D:/OpenCV/picture zone/cornkernels.png");
	//Mat src = imread("D:/OpenCV/picture zone/YB.jpg");
	if (!src.data)
	{
		printf("ERROR");
		return -1;
	}

	namedWindow(input,CV_WINDOW_AUTOSIZE);
	imshow(input,src);

	Mat src_gray;
	cvtColor(src,src_gray,COLOR_BGR2GRAY);

	Mat src_blur;
	blur(src_gray,src_blur,Size(3,3));

	//��ֵ��
		//���ڵ����ͼ����THRESH_TRIANGLE���Ǳ任����͹��
	Mat src_binary;
	threshold(src_blur,src_binary,0,255,THRESH_BINARY|THRESH_TRIANGLE);
	imshow("binary",src_binary);

	//��̬ѧ����
	Mat src_dilate;
	Mat kernel = getStructuringElement(MORPH_RECT,Size(5,5),Point(-1,-1));
	//morphologyEx(src_binary,src_open,CV_MOP_OPEN,kernel,Point(-1,-1));
	dilate(src_binary,src_dilate,kernel,Point(-1,-1),5);
	//erode(src_binary, src_open, kernel, Point(-1, -1), 8);
	imshow("open",src_dilate);

	//����任
		//����任�Ǿ��ڸ������ֵģ������ڶ�ԭͼ����ж�ֵ�任��ʱ����Ҫ�õ�����ֵTHRESH_BINARY_INV
		//����ʹ�ö�ֵ��תbitwise_not
	bitwise_not(src_dilate,src_dilate);
	Mat src_distance;
	distanceTransform(src_dilate,src_distance, DIST_L1,3);
	normalize(src_distance,src_distance,0,1.0,NORM_MINMAX);
	imshow("distance",src_distance);

	//��ֵ����ֵ�ָ�
	Mat src_distance_8u;
	src_distance.convertTo(src_distance_8u,CV_8U);
	adaptiveThreshold(src_distance_8u, src_distance_8u, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY,71, 0.0);
	kernel = getStructuringElement(MORPH_RECT, Size(3, 3), Point(-1, -1));
	dilate(src_distance_8u,src_distance_8u,kernel,Point(-1,-1),2);
	//erode(src_distance_8u, src_distance_8u, kernel, Point(-1, -1));
	imshow("8u",src_distance_8u);



	//����
	vector<vector<Point>> contours;
	findContours(src_distance_8u, contours, CV_RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

	// draw result
	Mat markers = Mat::zeros(src.size(), CV_8UC3);
	RNG rng(12345);
	for (size_t t = 0; t < contours.size(); t++) {
		drawContours(markers, contours, static_cast<int>(t), Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255)),
			-1, 8, Mat());
	}
	printf("number of corns : %d", contours.size());
	imshow("Final result", markers);

	waitKey(0);
	return 0;

}