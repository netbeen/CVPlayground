/*
 * SVM.cpp
 *
 *  Created on: 2015年6月1日
 *      Author: netbeen
 */

#include <opencv2/opencv.hpp>
#include <vector>

using namespace cv;
using namespace std;

class TrainPoint {
public:
	float x;
	float y;
	float label;
	TrainPoint(float x, float y, float label) :
			x(x), y(y), label(label) {
	}
};

void SVMtest() {
	// Data for visual representation
	int width = 512, height = 512;
	Mat image = Mat::zeros(height, width, CV_8UC3);

	vector<TrainPoint> trainSet;
	 trainSet.push_back(TrainPoint(501,10,1));
	 trainSet.push_back(TrainPoint(100,100,1));
	 trainSet.push_back(TrainPoint(310,290,1));
	 trainSet.push_back(TrainPoint(300,200,1));

	 trainSet.push_back(TrainPoint(0,200,-1));
	 trainSet.push_back(TrainPoint(502,503,-1));
	 trainSet.push_back(TrainPoint(11,504,-1));
	 trainSet.push_back(TrainPoint(250,250,-1));

	 Mat labelsMat(trainSet.size(), 1, CV_32FC1);
	 Mat trainingDataMat(trainSet.size(), 2, CV_32FC1);

	 for(size_t i = 0; i < trainSet.size(); i++){
		 labelsMat.at<float>(i,0) = trainSet.at(i).label;
		 trainingDataMat.at<float>(i,0) = trainSet.at(i).x;
		 trainingDataMat.at<float>(i,1) = trainSet.at(i).y;
	 }

	// Set up training data
	/*float labels[4] = { 1.0, -1.0, -1.0, -1.0 };
	Mat labelsMat(4, 1, CV_32FC1, labels);
	float trainingData[4][2] = { { 501, 10 }, { 0, 200 }, { 502, 503 }, { 11, 504 } };
	Mat trainingDataMat(4, 2, CV_32FC1, trainingData);

	 cout << labelsMat << endl;
	 cout << trainingDataMat << endl;*/

	// Set up SVM's parameters
	CvSVMParams params;
	CvSVM SVM;
	params.svm_type = CvSVM::C_SVC;
	params.kernel_type = CvSVM::LINEAR;
	params.term_crit = cvTermCriteria(CV_TERMCRIT_ITER, 100, 1e-6);

	SVM.train(trainingDataMat, labelsMat, Mat(), Mat(), params);	//LINEAR

	Vec3b green(0, 255, 0), blue(255, 0, 0), red(0, 0, 255);
	// Show the decision regions given by the SVM
	for (int i = 0; i < image.rows; ++i)
		for (int j = 0; j < image.cols; ++j) {
			Mat sampleMat = (Mat_<float>(1, 2) << i, j);
			float response = SVM.predict(sampleMat);

			if (response == 1)
				image.at<Vec3b>(j, i) = red;
			else if (response == -1)
				image.at<Vec3b>(j, i) = blue;
		}

	// Show the training data
	int thickness = -1;
	int lineType = 8;
	for(TrainPoint elem : trainSet){
		if(elem.label == 1){
			circle(image, Point(elem.x, elem.y), 5, Scalar(0, 0, 0), thickness, lineType);
		}else{
			circle(image, Point(elem.x, elem.y), 5, Scalar(255, 255, 255), thickness, lineType);
		}
	}

	/*circle(image, Point(501, 10), 5, Scalar(0, 0, 0), thickness, lineType);
	circle(image, Point(0, 200), 5, Scalar(255, 255, 255), thickness, lineType);
	circle(image, Point(501, 501), 5, Scalar(255, 255, 255), thickness, lineType);
	circle(image, Point(10, 501), 5, Scalar(255, 255, 255), thickness, lineType);*/

	// Show support vectors
	thickness = 2;
	lineType = 8;
	int c = SVM.get_support_vector_count();
	cout << "get_support_vector_count=" << c << endl;

	for (int i = 0; i < c; ++i) {
		const float* v = SVM.get_support_vector(i);
		cout << v[0] << " " << v[1] << endl;
		circle(image, Point((int) (v[0]), (int) (v[1])), 6, Scalar(128, 128, 128), thickness, lineType);
	}

	imwrite("result.png", image);        // save the image

	imshow("SVM Simple Example", image); // show it to the user
	waitKey(0);
}
