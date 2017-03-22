/*
 * util.h
 *
 *  Created on: 13.02.2012
 *      Author: ar
 */

#ifndef UTIL_H_
#define UTIL_H_

#include <iostream>
#include <sstream>

using namespace std;

#ifndef M_PI
    #define M_PI                    3.1415926535897932384626433832795
#endif

#include <cv.h>
#include "lut.h"

cv::Mat getCVLut(int lutIdx);
cv::Mat getLutForGrayImage(cv::Mat& grayImage, int idxLut, bool cvtScale=false);
cv::Mat getColoredImage(cv::Mat& image, int idxLut);

string imgInfo(cv::Mat&);

template <class T> cv::Point2d findMinMaxT(cv::Mat&);
cv::Point2d findMinMax(cv::Mat&);


#endif /* UTIL_H_ */
