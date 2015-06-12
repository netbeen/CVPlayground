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

void SVMtest(){
	vector<float> trainLabels{-1,-1,-1,-1,-1,2,2,2,2,2};
	Mat labelMat(trainLabels);
	//Mat labelMat(10,1,CV_32FC1,trainLabels.data());

	//vector<vector<float> > trainCoordinates{{0.4,0.09},{0.3,0.19},{0.2,0.29},{0.1,0.39},{0.05,0.44},{0.5,0.01},{0.4,0.11},{0.3,0.21},{0.2,0.31},{0.1,0.41}};
	float trainCoordinates[10][2] = {{0.4,0.09},{0.3,0.19},{0.2,0.29},{0.1,0.39},{0.05,0.44},{0.5,0.01},{0.4,0.11},{0.3,0.21},{0.2,0.31},{0.1,0.41}};
	Mat coordMat(10,2,CV_32FC1,trainCoordinates);
	//Mat coordMat(10,2,CV_32FC1,trainCoordinates.data());

	CvSVMParams svmPara;
	svmPara.svm_type = SVM::C_SVC;
	svmPara.kernel_type = SVM::LINEAR;
	svmPara.term_crit = cvTermCriteria(CV_TERMCRIT_ITER,100,1e-6);

	CvSVM svm;
	svm.train(coordMat, labelMat,  Mat(), Mat(), svmPara);

	vector<float> testCoordinates{0.4,0.09};
	Mat test(2,1,CV_32FC1, testCoordinates.data());
	cout << svm.predict(test);
}
