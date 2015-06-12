/*
 * BoundingBox.cpp
 *
 *  Created on: 2014年10月15日
 *      Author: netbeen
 */


#include "ESR.hpp"

BoundingBox::BoundingBox() {
		startX = 0;
		startY = 0;
		width = 0;
		height = 0;
		centerX = 0;
		centerY = 0;
}

Point_<double> BoundingBox::getStartPoint(){
	Point_<double> point;
	point.x = startX;
	point.y = startY;
	return point;
}

Point_<double> BoundingBox::getEndPoint(){
	Point_<double> point;
	point.x = startX + width;
	point.y = startY + height;
	return point;
}

bool BoundingBox::isInBoudingBox(Point_<double> pt){
	if(pt.x > startX && pt.x < (startX + width)){
		if(pt.y > startY && pt.y < (startY + height)){
			return true;
		}
	}
	return false;
}

Rect BoundingBox::returnRect(){
	Rect objectRect;
	objectRect.x = startX;
	objectRect.y = startY;
	objectRect.width = width;
	objectRect.height = height;
	return objectRect;
}
