/*
 * YYTest.cpp
 *
 *  Created on: 2014年12月1日
 *      Author: netbeen
 */

#include "ESR.hpp"

void YYTest() {
	int imgNum = 30; //2330;
	int badImgNum = 0;
	int candidatePixelNum = 400;
	int fernPixelNum = 5;
	int firstLevelNum = 10;
	int secondLevelNum = 500;
	int landmarkNum = 194;
	int initialNumber = 20;
	vector<Mat_<uchar> > images;
	vector<BoundingBox> bounding_box;
	vector<Mat_<double> > groundTruthShapes;

	ifstream fin;
	string HelenPath = "/home/netbeen/Helen-all/";
	string HelenAnnotationPath = HelenPath + "annotation/";
	string str;
	string annotationFileName;
	string imgFileName;

	CascadeClassifier cascadeFrontalface;
	cascadeFrontalface.load("data/haarcascade_frontalface_alt2.xml");

	for (int i = 0; i < imgNum; i++) {
		stringstream ss;
		ss << i + 1;
		ss >> str;
		annotationFileName = HelenAnnotationPath + str + ".txt";
		fin.open(annotationFileName.c_str());

		imgFileName.clear();
		fin >> imgFileName;
		imgFileName = HelenPath + imgFileName + ".jpg";
		Mat_<uchar> tempImg = imread(imgFileName, 0);

		BoundingBox tempBoundingBox;
		if (detectFace(tempImg, cascadeFrontalface, 5, tempBoundingBox)) {

			Mat_<double> tempLandmarkMat(landmarkNum, 2);
			for (int j = 0; j < landmarkNum; j++) {
				char c;
				fin >> tempLandmarkMat(j, 0) >> c >> tempLandmarkMat(j, 1);
			}

			if (!tempBoundingBox.isInBoudingBox(Point2d(tempLandmarkMat(67, 0), tempLandmarkMat(67, 1)))) {
				cout << imgFileName + "'s face doesn't in the boundingbox!!! CONTINUE!!!" << endl;
				badImgNum++;
				fin.close();
				continue;
			}
			images.push_back(tempImg);
			cout << "push_backed images.size() = " << images.size() << endl;
			groundTruthShapes.push_back(tempLandmarkMat);
			bounding_box.push_back(tempBoundingBox);

			Mat tempColorImg;
			cvtColor(tempImg, tempColorImg, COLOR_GRAY2RGB);
			for (int i = 0; i < landmarkNum; i++) {
				circle(tempColorImg, Point2d(tempLandmarkMat(i, 0), tempLandmarkMat(i, 1)), 3, Scalar(0, 255, 0), -1, 8, 0);
			}
		for(int j = 0; j < landmarkNum; j++) {
			circle(tempColorImg, Point2d(tempLandmarkMat(j, 0), tempLandmarkMat(j, 1)), 3, Scalar(255, 0, 0), -1, 8, 0);
			rectangle(tempColorImg, tempBoundingBox.getStartPoint(), tempBoundingBox.getEndPoint(), Scalar(255, 0, 0));
			imshow("temp", tempColorImg);
			cout << j <<endl;
			waitKey(0);
		}
	} else {
		cout << imgFileName + " doesn't detect any face!!! CONTINUE!!!" << endl;
		badImgNum++;
		fin.close();
		continue;
	}
	fin.close();
}
imgNum -= badImgNum;
ShapeRegressor regressor;
regressor.train(images, groundTruthShapes, bounding_box, firstLevelNum, secondLevelNum, candidatePixelNum, fernPixelNum, initialNumber);
regressor.save("data/model.txt");
cout << "Done." << endl;

return;
}
