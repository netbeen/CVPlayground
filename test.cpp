/*
 * test.cpp
 *
 *  Created on: 2014年10月16日
 *      Author: netbeen
 */

#include "ESR.hpp"

int test() {
	vector<Mat_<uchar> > test_images;
	vector<BoundingBox> test_bounding_box;
	int initial_number = 20;
	int landmarkNum = 194;
	//ifstream fin;
	VideoCapture cap;
		//switch (atoi(argv[1])) {
		//case 0:
			if (!cap.open(0)) {
				cout << "Webcam doesn't ready." << endl;
				exit(1);
			}
		//break;
		//case 1:
			/*if (!cap.open("data/5.avi")) {
				cout << "AVI file doesn't ready." << endl;
				exit(1);
			}*/
		//	break;
		/*default:
			cout << "Error!" << endl;
			exit(2);
			break;*/
		//}

	Mat rawImg, grayImg;

	CascadeClassifier cascadeFrontalface;
	cascadeFrontalface.load("data/haarcascade_frontalface_alt2.xml");
	//fin.close();

	BoundingBox boundingBox;
	ShapeRegressor regressor;
	regressor.load("data/model-Helen-HAAR.txt");

	cap >> rawImg;
	//rawImg = imread("data/multipeople3.jpg");
	//int frameNO = 0;		//video
	bool stop = false;		//video
	while (!stop) {				//video
		cap >> rawImg;
		cvtColor(rawImg, grayImg, COLOR_RGB2GRAY);
		if (detectFace(grayImg, cascadeFrontalface, 1, boundingBox)) {
			Mat_<double> current_shape = regressor.predict(grayImg, boundingBox, initial_number);
			cvtColor(grayImg, grayImg, COLOR_GRAY2RGB);
			for (int i = 0; i < landmarkNum; i++) {
				circle(rawImg, Point2d(current_shape(i, 0), current_shape(i, 1)), 3, Scalar(0, 255, 0), -1, 8, 0);
			}
			rectangle(rawImg, boundingBox.returnRect(), Scalar(0, 255, 255), 3, 8, 0);
		}else {
			cout << "-NO FACE-" << endl;
		}
		/*Mat smallImg = Mat(cvRound(grayImg.rows / 2), cvRound(grayImg.cols / 2), CV_8UC1);///////////////////////////////////
		resize(grayImg, grayImg, smallImg.size(), 0, 0, INTER_LINEAR);*/
		imshow("result", rawImg);
		//waitKey(0);
		/*stringstream outputFilename;					////////////////////////////////////////////// video
		outputFilename<<  "data/result/" << ++frameNO << ".jpg";
		imwrite(outputFilename.str(), rawImg);
		cout << frameNO << endl;*/
		if (waitKey(30) >= 0)
			stop = true;
	}		//video
	return 0;
}
