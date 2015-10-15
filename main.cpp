/*
 * main.cpp
 *
 *  Created on: 2014年10月15日
 *      Author: netbeen
 */


#include "ESR.hpp"

int main(int argc, char *argv[]){
	int operate;
	cout << "What do you want to do? \t1. train \t2.test \t4. SVM model \t5.opticalFlow \t6.FourierTrain" << endl;
	cin >> operate;
	switch (operate) {
		case 1:
			train();
			break;
		case 2:
			test();
			break;
		case 3:
			YYTest();
			break;
		case 4:
			SVMtest();
			break;
		case 5:
			opticalFlow();
			break;
		case 6:
			fourierTrain();
			break;
		default:
			cout << "Error Operate!!! EXIT!!!" <<endl;
			break;
	}
	return 0;
}
