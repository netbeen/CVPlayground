/*
 Author: Bi Sai
 Date: 2014/06/18
 This program is a reimplementation of algorithms in "Face Alignment by Explicit
 Shape Regression" by Cao et al.
 If you find any bugs, please email me: soundsilencebisai-at-gmail-dot-com

 Copyright (c) 2014 Bi Sai
 The MIT License (MIT)
 Permission is hereby granted, free of charge, to any person obtaining a copy of
 this software and associated documentation files (the "Software"), to deal in
 the Software without restriction, including without limitation the rights to
 use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
 the Software, and to permit persons to whom the Software is furnished to do so,
 subject to the following conditions:

 The above copyright notice and this permission notice shall be included in all
 copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
 FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
 IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
 CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */
#include "ESR.hpp"

ShapeRegressor::ShapeRegressor() {
	first_level_num_ = 0;
	landmark_num_ = 0;
}

/**
 * @param images gray scale images
 * @param ground_truth_shapes a vector of N*2 matrix, where N is the number of landmarks
 * @param bounding_box BoundingBox of faces
 * @param first_level_num number of first level regressors
 * @param second_level_num number of second level regressors
 * @param candidate_pixel_num number of pixels to be selected as features
 * @param fern_pixel_num number of pixel pairs in a fern
 * @param initial_num number of initial shapes for each input image
 */
void ShapeRegressor::train(const vector<Mat_<uchar> >& images, const vector<Mat_<double> >& groundTruthShapes, const vector<BoundingBox>& boundingBox, int firstLevelNum, int second_level_num, int candidate_pixel_num, int fern_pixel_num, int initialNum) {
	cout << "Start training..." << endl;
	bounding_box_ = boundingBox;
	training_shapes_ = groundTruthShapes;
	first_level_num_ = firstLevelNum;
	landmark_num_ = groundTruthShapes[0].rows;
	// data augmentation and multiple initialization
	vector<Mat_<uchar> > augmentedImages;	//augmented=扩增的
	vector<BoundingBox> augmentedBoundingBox;
	vector<Mat_<double> > augmentedGroundTruthShapes;
	vector<Mat_<double> > currentShapes;

	for (int i = 0; i < images.size(); i++) {
		for (int j = 0; j < initialNum; j++) {
			int index = (i + j + 1) % (images.size());
			augmentedImages.push_back(images[i]);
			augmentedGroundTruthShapes.push_back(groundTruthShapes[i]);
			augmentedBoundingBox.push_back(boundingBox[i]);
			// 1. Select ground truth shapes of other images as initial shapes
			// 2. Project current shape to bounding box of ground truth shapes
			Mat_<double> temp = groundTruthShapes[index];
			temp = projectShape(temp, boundingBox[index]);		//
			temp = reProjectShape(temp, boundingBox[i]);			//经过一轮 projectShape 和 reProjectShape 之后，boundingBox[index]被规范化，与 boundingBox[i] 对齐。
			currentShapes.push_back(temp);
		}
	}

	// get mean shape from training shapes
	meanShape = getMeanShape(groundTruthShapes, boundingBox);

	// train fern cascades
	fernCascades.resize(firstLevelNum);		//调整Fern的数量为一级分类器的数量
	vector<Mat_<double> > prediction;
	for (int i = 0; i < firstLevelNum; i++) {
		cout << "Training fern cascades: " << i + 1 << " out of " << firstLevelNum << endl;
		prediction = fernCascades[i].train(augmentedImages, currentShapes, augmentedGroundTruthShapes, augmentedBoundingBox, meanShape, second_level_num, candidate_pixel_num, fern_pixel_num);

		// update current shapes
		for (int j = 0; j < prediction.size(); j++) {
			currentShapes[j] = prediction[j] + projectShape(currentShapes[j], augmentedBoundingBox[j]);
			currentShapes[j] = reProjectShape(currentShapes[j], augmentedBoundingBox[j]);
		}
	}
}

void ShapeRegressor::Write(ofstream& fout) {
	fout << first_level_num_ << endl;
	fout << meanShape.rows << endl;
	for (int i = 0; i < landmark_num_; i++) {
		fout << meanShape(i, 0) << " " << meanShape(i, 1) << " ";
	}
	fout << endl;

	fout << training_shapes_.size() << endl;
	for (int i = 0; i < training_shapes_.size(); i++) {
		fout << bounding_box_[i].startX << " " << bounding_box_[i].startY << " " << bounding_box_[i].width << " " << bounding_box_[i].height << " " << bounding_box_[i].centerX << " " << bounding_box_[i].centerY << endl;
		for (int j = 0; j < training_shapes_[i].rows; j++) {
			fout << training_shapes_[i](j, 0) << " " << training_shapes_[i](j, 1) << " ";
		}
		fout << endl;
	}

	for (int i = 0; i < first_level_num_; i++) {
		fernCascades[i].Write(fout);
	}
}

void ShapeRegressor::read(ifstream& fin) {
	fin >> first_level_num_;					//第1行是第一层级联分类器的数量
	fin >> landmark_num_;					//第2行是landmark的数量

	meanShape = Mat::zeros(landmark_num_, 2, CV_64FC1);			//64位精度,在32位机上是double，64位机上是float
	for (int i = 0; i < landmark_num_; i++) {								//读入29个坐标（共58个数）作为初始的平均形状
		fin >> meanShape(i, 0) >> meanShape(i, 1);
	}

	int trainingNum;
	fin >> trainingNum;							//第4行是训练图的数量
	training_shapes_.resize(trainingNum);
	bounding_box_.resize(trainingNum);

	for (int i = 0; i < trainingNum; i++) {				//两行两行一组读入训练图的参数
		BoundingBox temp;
		fin >> temp.startX >> temp.startY >> temp.width >> temp.height >> temp.centerX >> temp.centerY;		//组合内：第一行是boundingBox的6个参数
		bounding_box_[i] = temp;

		Mat_<double> temp1(landmark_num_, 2);
		for (int j = 0; j < landmark_num_; j++) {				//组合内：第二行是29个坐标（共58个数），作为landmark参数
			fin >> temp1(j, 0) >> temp1(j, 1);
		}
		training_shapes_[i] = temp1;
	}				//一直读到了2694行

	fernCascades.resize(first_level_num_);
	for (int i = 0; i < first_level_num_; i++) {
		fernCascades[i].read(fin);
	}
}

Mat_<double> ShapeRegressor::predict(const Mat_<uchar>& image, const BoundingBox& bounding_box, int initial_num) {
	// generate multiple initializations
	Mat_<double> result = Mat::zeros(landmark_num_, 2, CV_64FC1);
	RNG random_generator(getTickCount());
	for (int i = 0; i < initial_num; i++) {
		random_generator = RNG(i);
		int index = random_generator.uniform(0, training_shapes_.size());		//uniform函数可以指定随机数的范围
		Mat_<double> currentShape = training_shapes_[index];
		BoundingBox currentBoundingBox = bounding_box_[index];
		currentShape = projectShape(currentShape, currentBoundingBox);		//projectShape函数将当前的shape等比例缩放到(0,1)的boundingBox内
		currentShape = reProjectShape(currentShape, bounding_box);				//reProjectShape函数将已经归一化的shape所放到真实的boundingBox内
		for (int j = 0; j < first_level_num_; j++) {
			Mat_<double> prediction = fernCascades[j].predict(image, bounding_box, meanShape, currentShape);
			// update current shape
			currentShape = prediction + projectShape(currentShape, bounding_box);
			currentShape = reProjectShape(currentShape, bounding_box);
		}
		result = result + currentShape;
	}
	return 1.0 / initial_num * result;
}

void ShapeRegressor::load(string path) {
	cout << "Loading model..." << endl;
	ifstream fin;
	fin.open(path.c_str());
	this->read(fin);
	fin.close();
	cout << "Model loaded successfully..." << endl;
}

void ShapeRegressor::save(string path) {
	cout << "Saving model..." << endl;
	ofstream fout;
	fout.open(path.c_str());
	this->Write(fout);
	fout.close();
}

