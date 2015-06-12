/*
 * ESR.hpp
 *
 *  Created on: 2014年10月15日
 *      Author: netbeen
 */

#ifndef ESR_HPP_
#define ESR_HPP_

#include <iostream>
#include <opencv2/opencv.hpp>
#include <fstream>
#include <opencv2/gpu/gpu.hpp>

using namespace std;
using namespace cv;

class BoundingBox {
public:
	double startX;
	double startY;
	double width;
	double height;
	double centerX;
	double centerY;
	BoundingBox();
	Point_<double> getStartPoint();
	Point_<double> getEndPoint();
	bool isInBoudingBox(Point_<double> pt);
	Rect returnRect();
};

int train();
int test();
bool detectFace(Mat& grayImg, CascadeClassifier& cascade, double scale, BoundingBox& boundingBox);

class Fern {
private:
	int fern_pixel_num_;
	int landmark_num_;
	Mat_<int> selected_nearest_landmark_index_;
	Mat_<double> threshold_;
	Mat_<int> selected_pixel_index_;
	Mat_<double> selected_pixel_locations_;
	vector<Mat_<double> > bin_output_;
public:
	vector<Mat_<double> > Train(const vector<vector<double> >& candidate_pixel_intensity, const Mat_<double>& covariance, const Mat_<double>& candidate_pixel_locations, const Mat_<int>& nearest_landmark_index, const vector<Mat_<double> >& regression_targets, int fern_pixel_num);
	Mat_<double> predict(const Mat_<uchar>& image, const Mat_<double>& shape, const Mat_<double>& rotation, const BoundingBox& bounding_box, double scale);
	void read(ifstream& fin);
	void Write(ofstream& fout);
};

class FernCascade {
public:
	vector<Mat_<double> > train(const vector<Mat_<uchar> >& images, const vector<Mat_<double> >& current_shapes, const vector<Mat_<double> >& ground_truth_shapes, const vector<BoundingBox> & bounding_box, const Mat_<double>& mean_shape, int second_level_num, int candidate_pixel_num, int fern_pixel_num);
	Mat_<double> predict(const Mat_<uchar>& image, const BoundingBox& bounding_box, const Mat_<double>& mean_shape, const Mat_<double>& shape);
	void read(ifstream& fin);
	void Write(ofstream& fout);
private:
	vector<Fern> ferns_;
	int second_level_num_;
};

class ShapeRegressor {
public:
	ShapeRegressor();
	void train(const vector<Mat_<uchar> >& images, const vector<Mat_<double> >& ground_truth_shapes, const vector<BoundingBox>& bounding_box, int first_level_num, int second_level_num, int candidate_pixel_num, int fern_pixel_num, int initial_num);
	Mat_<double> predict(const Mat_<uchar>& image, const BoundingBox& bounding_box, int initial_num);
	void read(ifstream& fin);
	void Write(ofstream& fout);
	void load(string path);
	void save(string path);
private:
	int first_level_num_;
	int landmark_num_;
	vector<FernCascade> fernCascades;
	Mat_<double> meanShape;
	vector<Mat_<double> > training_shapes_;
	vector<BoundingBox> bounding_box_;
};

Mat_<double> getMeanShape(const vector<Mat_<double> >& shapes, const vector<BoundingBox>& bounding_box);
Mat_<double> projectShape(const Mat_<double>& shape, const BoundingBox& bounding_box);
Mat_<double> reProjectShape(const Mat_<double>& shape, const BoundingBox& bounding_box);
void SimilarityTransform(const Mat_<double>& shape1, const Mat_<double>& shape2, Mat_<double>& rotation, double& scale);
double calculate_covariance(const vector<double>& v_1, const vector<double>& v_2);

void YYTest();

void SVMtest();

void opticalFlow();

#endif /* ESR_HPP_ */
