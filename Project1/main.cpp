#include <iostream>
#include <fstream>
#include<vector>
#include<fstream>
#include<io.h>
#include<string>
#include<windows.h>
#include "opencv2/opencv.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <opencv2/core/hal/hal.hpp>
#include <stdio.h>
#include "opencv2/ml.hpp"

using namespace cv;
using namespace std;
double cos(Point center, Point a, Point b) {
	double ab = sqrt((a.x-b.x)*(a.x-b.x) + (a.y - b.y)*(a.y-b.y));
	double ac = sqrt((a.x - center.x)*(a.x-center.x) + (a.y - center.y)*(a.y-center.y));
	double bc = sqrt((b.x - center.x)*(b.x-center.x) + (b.y - center.y)*(b.y-center.y));
	return (ac * ac + bc * bc - ab * ab) / (2 *ac*bc);
}
vector<Point> find_rect_around(Mat canny_edge) {
	Point left(100000, 0), right(0, 0), top(0, 100000), bottom(0, 0);
	for (int i = 0; i < canny_edge.size().width; i++) {
		for (int j = 0; j < canny_edge.size().height; j++) {
			if (canny_edge.at<uchar>(j, i) == 255) {
				if (i < left.x)left = Point(i, j);
				if (i > right.x)right = Point(i, j);
				if (j > bottom.y)bottom = Point(i, j);
				if (j < top.y)top = Point(i, j);
			}
		}
	}
	Point left_top = Point(left.x, top.y);
	Point right_bottom = Point(right.x, bottom.y);
	vector<Point> Rect;
	Rect.push_back(left_top);
	Rect.push_back(right_bottom);
	return Rect;
}

Mat extract_main_hand(Mat origin) {
	vector<vector<Point>> contours;
	Mat canny_contours;

	Mat struct_kernel = getStructuringElement(MORPH_RECT, Size(3,3));
	Mat struct_kernel1 = getStructuringElement(MORPH_RECT, Size(2, 2));

	//Mat Prewitt_h = (Mat_<int>(3, 3) << -2, -2, -2, 0, 0, 0, 2, 2, 2);
	//Mat Prewitt_v = (Mat_<int>(3, 3) << -2, 0, 2, -2, 0, 2, -2, 0, 2);
	//Mat origin_h, origin_v;
	//filter2D(origin, origin_h, CV_8U,Prewitt_h);
	//filter2D(origin, origin_v, CV_8U, Prewitt_v);
	//add(origin_h, origin_v, origin_v);
	//add(origin, origin_v, origin);

	Canny(origin, canny_contours, 0, 100);

	//adaptiveThreshold(origin, origin, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY_INV, 9, 2);
	//medianBlur(canny_contours, origin, 3);
	dilate(canny_contours, origin, struct_kernel);
	erode(origin, origin, struct_kernel1);
	vector<Point> Rectangle = find_rect_around(origin);
	origin = origin(Rect(Rectangle[0], Rectangle[1]));
	copyMakeBorder(origin, origin, 4, 0, 4, 4, BORDER_CONSTANT, Scalar(0));
	copyMakeBorder(origin, origin, 1, 1, 1, 1, BORDER_CONSTANT, Scalar(255));
	findContours(origin, contours, RETR_TREE, CHAIN_APPROX_SIMPLE);

	Mat contours_image = Mat::zeros(origin.size(), CV_8U);
	int arg_max = 0;
	double max = 0;

	for (size_t i = 0; i < contours.size(); i++) {
		if (arcLength(contours[i],true) > max) {
			max = arcLength(contours[i],true);
			arg_max = i;
		}
	}

	drawContours(contours_image, contours, arg_max, 255);
	drawContours(contours_image, contours, arg_max, 255, cv::FILLED);
	contours_image = contours_image(Rect(Point(1, 1), Point(contours_image.size().width - 1, contours_image.size().height - 1)));
	for (int i = 0; i < contours_image.cols; i++)
		for (int k = 0; k < contours_image.rows; k++) {
			if (contours_image.at<uchar>(k, i) == 0)
				contours_image.at<uchar>(k, i) = 255;
			else
				contours_image.at<uchar>(k, i) = 0;
		}
	//imshow("v", contours_image);
	//waitKey();
	//imwrite("binary_img.jpg", contours_image);

	return contours_image;
}

vector<vector<Point>> prepare(Mat origin) //return the contours of hand
{
	double contrast = 3;

	if (origin.type() != CV_8U) {
		//GaussianBlur(origin, origin, Size(3,3), 3);
		/*Mat Prewitt_h = (Mat_<int>(3, 3) << -1, -1, -1, 0, 0, 0, 1, 1, 1);
		Mat Prewitt_v = (Mat_<int>(3, 3) << -1, 0, 1, -1, 0, 1, -1, 0, 1);
		Mat origin_h, origin_v;
		filter2D(origin, origin_h, CV_8UC3,Prewitt_h);
		filter2D(origin, origin_v, CV_8UC3, Prewitt_v);
		add(origin_h, origin_v, origin);*/
		cvtColor(origin, origin, COLOR_RGB2GRAY);
		vector<vector<Point>> color_contours;
		
	}
	Mat g = getGaussianKernel(3, 0.75);
	Mat gaussian = g * g.t();
	filter2D(origin, origin, CV_32F, gaussian);
	origin.convertTo(origin, CV_8U);

	origin = extract_main_hand(origin);
	vector<vector<Point>> Contours;
	findContours(origin, Contours, RETR_TREE, CHAIN_APPROX_SIMPLE);
	int max = 0, argmax = 0;
	for (int k = 0; k < Contours.size(); k++) {
		if (Contours[k].size() > max) {
			max = Contours[k].size();
			argmax = k;
		}
	}

	vector<vector<Point>> r;
	vector<Point> size_point;
	size_point.push_back(Point(origin.size().width, origin.size().height));
	r.push_back(Contours[argmax]);
	r.push_back(size_point);
	return r;
}

vector<double> extract_feature(vector<vector< Point >>Contours, Size origin_size) {
	vector<double> features;
	Size image_size(Contours[1][0].x, Contours[1][0].y);
	Mat Contours_image(origin_size, CV_8U);
	threshold(Contours_image, Contours_image, 255, 255, THRESH_BINARY);
	drawContours(Contours_image, Contours, 0, Scalar(255), cv::FILLED);
	vector<vector<Point>> hulls(1);
	vector<int> hullsi;
	vector<Vec4i> defects;
	convexHull(Contours[0], hulls[0]);
	convexHull(Contours[0], hullsi);
	try {
		convexityDefects(Contours[0], hullsi, defects);
	}
	catch(cv::Exception) {
		Mat struct_kernel1 = getStructuringElement(MORPH_RECT, Size(2, 2));
		erode(Contours_image, Contours_image, struct_kernel1);
		Contours_image = extract_main_hand(Contours_image);
		image_size = Contours_image.size();
		findContours(Contours_image, Contours, RETR_TREE, CHAIN_APPROX_SIMPLE);
		convexHull(Contours[0], hulls[0]);
		convexHull(Contours[0], hullsi);
		convexityDefects(Contours[0], hullsi, defects);
	}
	drawContours(Contours_image, hulls, 0, Scalar(255), 1);



	int acute_num=0;
	int lie_angle = 0;
	for (int k = 0; k < defects.size(); k++) {
		Point start = Contours[0][defects[k][0]];
		Point end = Contours[0][defects[k][1]];
		Point Far = Contours[0][defects[k][2]];
		
		vector<Point> defect_triangle{ start,end,Far };
		double defect_area = contourArea(defect_triangle);
		if (defect_area < 25)continue;

		circle(Contours_image, start, 2, Scalar(50+20*k));
		circle(Contours_image, end, 2, Scalar(51+20*k));
		circle(Contours_image, Far, 2, Scalar(52+20*k));

		double cosine = cos(Far, start, end);
		if (cosine > 0) {
			acute_num++;

			Point mid((start.x + end.x) / 2, (start.y + end.y) / 2);
			Vec<double, 2> angle(mid.x - Far.x, mid.y - Far.y);
			Vec<double, 2> horizon(1, 0);
			double lie = angle.dot(horizon) / sqrt((angle.val[0]*angle.val[0]+angle.val[1]*angle.val[1]) * (horizon.val[0]*horizon.val[0]+horizon.val[1]*horizon.val[1]));
			if (lie > 0.5 || lie < -0.5)lie_angle ++ ;
		}


	}
	features.push_back(acute_num);
	features.push_back(lie_angle);

	double contour_area = contourArea(Contours[0]);
	double hull_area = contourArea(hulls[0]);
	double contour_versus_hull = contour_area / hull_area;
	features.push_back(contour_versus_hull);

	features.push_back(double(defects.size()) /20);
	features.push_back(double(hullsi.size()) / 20);

	double circumference = arcLength(Contours[0], true);
	double hull_circumference = arcLength(hulls[0], true);
	double circumference_versus_area = circumference / contour_area;
	double circumference_versus_hull_circumference = circumference / hull_circumference;
	features.push_back(circumference_versus_area);
	features.push_back(circumference_versus_hull_circumference);

	//imshow("v", Contours_image);
	//imwrite("convex.jpg", Contours_image);
	//waitKey();
	return features;
}
vector<double> tag;
vector<vector<double>> train_data;

void read_data(string path,int t) {
	for (int k = 1; k < 64; k++) {
		string seq = to_string(k);
		Mat m;
		if (k <= 50)m = imread(path + seq + ".png");
		else if (k > 50) {
			continue;
			m = imread(path + seq + ".jpg");
		}
		m = m(Rect(Point(2, 2), Point(m.size().width - 2, m.size().height)));
		vector<vector<Point>> Contours = prepare(m);
		vector<double> features = extract_feature(Contours, Size(m.size().width, m.size().height));
		train_data.push_back(features);
		tag.push_back(t);
		//imshow("v", m);
		//waitKey();
	}
}

std::vector<std::string> getFilesInFolder(const char* folderPath) {
	std::vector<std::string> fileList;

	WIN32_FIND_DATAA findData;
	HANDLE hFind = FindFirstFileA(folderPath, &findData);

	if (hFind == INVALID_HANDLE_VALUE) {
		std::cerr << "Error finding directory:" << folderPath << std::endl;
		return fileList;
	}

	do {
		if (findData.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) {
			continue; 
		}

		fileList.push_back(findData.cFileName);
	} while (FindNextFileA(hFind, &findData));

	FindClose(hFind);

	return fileList;
}

int main()
{
	string path = "../Hand_Posture_Easy_Stu/Five/Five";
	read_data(path, 0);
	path = "../Hand_Posture_Easy_Stu/A/A";
	read_data(path, 1);
	path = "../Hand_Posture_Easy_Stu/V/V";
	read_data(path, 2);
	path = "../Hand_Posture_Easy_Stu/C/C";
	read_data(path, 3);

	Ptr<ml::SVM> svm_classifier = ml::SVM::create();
	svm_classifier->setType(ml::SVM::C_SVC);
	svm_classifier->setKernel(ml::SVM::RBF);
	svm_classifier->setGamma(0.001);
	svm_classifier->setC(10000);

	Mat train_mat(train_data.size(), 7, CV_32F);
	Mat train_tag(tag.size(), 1, CV_32S);
	for (int i = 0; i < train_data.size(); i++) {
		for (int j = 0; j < 7; j++) {
			train_mat.at<float>(i, j) = train_data[i][j];
		}
		train_tag.at<int>(i, 0) = tag[i];
	}

	Ptr<ml::TrainData> train = ml::TrainData::create(train_mat, ml::ROW_SAMPLE, train_tag);
	svm_classifier->train(train);

	vector<string> filename;
	filename=getFilesInFolder("..\\Hand_Posture_Easy_Stu\\validate\\*.*");
	for (int k = 0; k < filename.size(); k++) {
		Mat test = imread("../Hand_Posture_Easy_Stu/validate/"+filename[k]);
		vector<vector<Point>> Contours = prepare(test);
		vector<double> features = extract_feature(Contours, Size(test.size().width, test.size().height));
		Mat features_mat(1, 7, CV_32F);
		for (int k = 0; k < 7; k++) { features_mat.at<float>(0, k) = features[k]; }
		int res = svm_classifier->predict(features_mat);
		cout <<filename[k]<<' ' << res << endl;
	}

}