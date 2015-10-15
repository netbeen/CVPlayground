/*
 * FourierTrain.cpp
 *
 *  Created on: 2015年8月4日
 *      Author: netbeen
 */

#include "ESR.hpp"

void fourierTrain() {
	cv::Mat imageF = cv::imread("/home/netbeen/NUAA-raw/ImposterRaw/0001/0001_00_00_01_0.jpg");
	cv::Mat imageT = cv::imread("/home/netbeen/NUAA-raw/ClientRaw/0001/0001_00_00_01_16.jpg");
	cv::Mat gray;
	cv::cvtColor(imageT, gray, COLOR_BGR2GRAY);

	CascadeClassifier cascadeFrontalface;
	cascadeFrontalface.load("data/haarcascade_frontalface_alt2.xml");
	BoundingBox boundingBox;
	detectFace(gray, cascadeFrontalface, 1, boundingBox);

	cv::Mat grayFace = gray(boundingBox.returnRect());
	cv::resize(grayFace, grayFace, Size(128, 128));

	cv::Mat gaussian1, gaussian2;
	cv::GaussianBlur(grayFace, gaussian1, Size(3, 3), 0);
	cv::GaussianBlur(gaussian1, gaussian2, Size(3, 3), 0);
	Mat img_DoG = gaussian1 - gaussian2;
	normalize(img_DoG, img_DoG, 255, 0, CV_MINMAX);

	/*cv::Mat foutier;
	 cv::dft(img_DoG,foutier);*/

	/////////////////////////////////////////////////////////////////////////

	Mat padded;                            //expand input image to optimal size
	int m = getOptimalDFTSize(img_DoG.rows);
	int n = getOptimalDFTSize(img_DoG.cols); // on the border add zero values
	copyMakeBorder(img_DoG, padded, 0, m - img_DoG.rows, 0, n - img_DoG.cols, BORDER_CONSTANT, Scalar::all(0));

	Mat planes[] = { Mat_<float>(padded), Mat::zeros(padded.size(), CV_32F) };
	Mat complexI;
	merge(planes, 2, complexI);         // Add to the expanded another plane with zeros

	dft(complexI, complexI);            // this way the result may fit in the source matrix

	// compute the magnitude and switch to logarithmic scale
	// => log(1 + sqrt(Re(DFT(I))^2 + Im(DFT(I))^2))
	split(complexI, planes);                   // planes[0] = Re(DFT(I), planes[1] = Im(DFT(I))
	magnitude(planes[0], planes[1], planes[0]);                   // planes[0] = magnitude
	Mat magI = planes[0];

	magI += Scalar::all(1);                    // switch to logarithmic scale
	log(magI, magI);

	// crop the spectrum, if it has an odd number of rows or columns
	magI = magI(Rect(0, 0, magI.cols & -2, magI.rows & -2));

	// rearrange the quadrants of Fourier image  so that the origin is at the image center
	int cx = magI.cols / 2;
	int cy = magI.rows / 2;

	Mat q0(magI, Rect(0, 0, cx, cy));   // Top-Left - Create a ROI per quadrant
	Mat q1(magI, Rect(cx, 0, cx, cy));  // Top-Right
	Mat q2(magI, Rect(0, cy, cx, cy));  // Bottom-Left
	Mat q3(magI, Rect(cx, cy, cx, cy)); // Bottom-Right

	Mat tmp;                           // swap quadrants (Top-Left with Bottom-Right)
	q0.copyTo(tmp);
	q3.copyTo(q0);
	tmp.copyTo(q3);

	q1.copyTo(tmp);                    // swap quadrant (Top-Right with Bottom-Left)
	q2.copyTo(q1);
	tmp.copyTo(q2);

	normalize(magI, magI, 0, 1, CV_MINMAX); // Transform the matrix with float values into a
	                                        // viewable image form (float between values 0 and 1).

	imshow("spectrum magnitude", magI);
	/////////////////////////////////////////////////////////////////////////

	std::cout << img_DoG.rows << " " << img_DoG.cols << std::endl;
	std::cout << magI.rows << " " << magI.cols << std::endl;

	//for (float angle = 0; angle < 2 * 3.1415926; angle += 0.1) {
		//std::cout << static_cast<int>(r * cos(angle)) << " "<< static_cast<int>(r*sin(angle)) <<std::endl;
	//}

	//std::cout << magI.at<float>(127, 0) << std::endl;

	for (int r = 0; r < 64; r++) {
		float sum = 0;
		for (float angle = 0; angle < 2 * 3.1415926; angle += 0.1) {
			sum += magI.at<float>(64+static_cast<int>(r * cos(angle)), 64+static_cast<int>(r*sin(angle)));
		}
		std::cout << sum/62 << std::endl;
	}

	cv::waitKey();
}
