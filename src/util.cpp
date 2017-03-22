/*
 * util.cpp
 *
 *  Created on: 13.02.2012
 *      Author: ar
 */


#include "util.h"
#include "lut.h"

cv::Mat getCVLut(int lutIdx) {
	cv::Mat palette(1, 256, CV_8UC3);
	uchar*	p 	= palette.data;
	int nch	= palette.channels();
	int cnt	= 0;
	for(int i=0; i<256; i++) {
		for(int k=0; k<nch; k++) {
			p[cnt] = lut::slut[lutIdx].lut[i][2-k];
			cnt++;
		}
	}
	return palette;
}

cv::Mat getLutForGrayImage(cv::Mat& grayImage, int idxLut, bool cvtScale) {
	CV_Assert( grayImage.channels()==1 );
	int nrow = grayImage.rows;
	int ncol = grayImage.cols;
	cv::Mat tmp(nrow, ncol, CV_8UC3, cv::Scalar(0,0,0));
	//
	if(cvtScale) {
		cv::convertScaleAbs(grayImage, grayImage);
	}
	uchar* p;
	uchar* pg;
	uchar val;
	unsigned char (*clut)[3] = lut::slut[idxLut].lut;
	for(int i=0; i<nrow; i++) {
		p	= tmp.ptr<uchar>(i);
		pg	= grayImage.ptr<uchar>(i);
		int cnt=0;
		for(int j=0; j<ncol; j++) {
			val	= pg[j];
			p[cnt++] = clut[val][2];
			p[cnt++] = clut[val][1];
			p[cnt++] = clut[val][0];
		}
	}
	return tmp;
}

cv::Mat getColoredImage(cv::Mat& image, int idxLut) {
	int nch	= image.channels();
	CV_Assert( nch==1 || nch==3 );
	cv::Mat tmp;
	if(nch==1) {
		return getLutForGrayImage(image, idxLut, true);
	} else if(nch==3) {
		cv::Mat palette	= getCVLut(idxLut);
		cv::LUT(image, palette, tmp);
		return tmp;
	}
	return tmp;
}

string imgInfo(cv::Mat& mat) {
	stringstream ss;
	ss<<"opt: " << mat.rows <<"x"<<mat.cols<<":"<<mat.channels() << ", depth=" <<mat.depth();
	return ss.str();
}

template <class T>
cv::Point2d findMinMaxT(cv::Mat& mat) {
	double min, max;
	min	= (double)mat.at<T>(0,0);
	max	= min;
	int nrow 	= mat.rows;
	int ncol	= mat.cols;
	double tmp;
	for(int i=0; i<nrow; i++) {
		for(int j=0; j<ncol; j++) {
			tmp	= (double)mat.at<T>(i,j);
			if(tmp<min) {
				min	=tmp;
			} else if(tmp>max) {
				max	= tmp;
			}
		}
	}
	return cv::Point2d(min, max);
}

cv::Point2d findMinMax(cv::Mat& mat) {
	int depth	= mat.depth();
	CV_Assert( mat.channels() == 1 || depth == CV_8U || depth == CV_32F || depth == CV_64F );
	switch (depth) {
		case CV_8U:
			return findMinMaxT<uchar>(mat);
			break;
		case CV_32F:
			return findMinMaxT<float>(mat);
			break;
		case CV_64F:
			return findMinMaxT<double>(mat);
			break;
		default:
			break;
	}
	return cv::Point2d(-1., -1.); //FIXME: bad solution
}
