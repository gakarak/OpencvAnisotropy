#include <iostream>
#include <vector>
#include <string>
#include <sstream>

#include <cmath>

#include <stdlib.h>

using namespace std;

#include <cv.h>
#include <highgui.h>

#include "util.h"

#define MAX_WIDTH	1600
#define thresh		25.


int t_width, t_height;
int o_width, o_height;
int cx, cy;
int ocx, ocy;
cv::Size t_size, o_size;

cv::Mat img, imgRsz, imgGray, imgGrayRsz;
cv::Mat magRsz, angRsz, gxRsz, gyRsz, rgxRsz, rgyRsz;
cv::Mat fieldMapRsz, fieldMapRszDelta;
cv::Mat buff;
cv::Mat gx, gy, rgx, rgy;

#define WIN 		"map"
#define WIN_HIST	"hist"

#define HIST_SIZ_W		400
#define HIST_SIZ_BASE	300
#define HIST_SIZ_BASED2	HIST_SIZ_BASE / 2

#define SIZE_AREA_MIN	10
#define SIZE_AREA_MAX	60
int size_area_shift		= 10;
int o_size_area			= SIZE_AREA_MIN + size_area_shift;
int size_area			= o_size_area; // real-size : for scaled image

#define SHOW_ORIG	0x1
#define SHOW_GRAY	0x11
#define SHOW_GX		0x2
#define SHOW_GY		0x3
#define SHOW_MAG	0x4
#define SHOW_ANG	0x5
#define SHOW_DIR	0x6 // FIXME: realised?

int currentSHOW		= SHOW_ORIG;
int currentLUT		= 0;
bool isColorMode	= false;
bool isRotatedGxGy	= false;
bool isDeltaField	= false;

long*	dataHist;
long*	bins;
long*	bins2_plain;
long*	bins2_plain_delta;
double*	bins2_weight;
double*	bins2_weight_delta;
int		NBIN;

////////////////////////////////////////////
void track_lut(int, void*);
void track_size_area(int, void*);
void on_mouse_map(int , int , int , int , void*);
void drawLUT();
void drawLegend(cv::Mat& img, int num);
void check_size_win();
void calcDirectionField();
void calcWinBins(bool);
void printBins();
template <class T> void printBinsTmp(T* bins, int siz, const string&);
//void saveColorImage(const string&, int);
int findMaxIdx(long* , int);
cv::Point findMinMaxIdx(long* bins, int siz);
int getIdxCircular(int siz, int idx);
double getSigmaAvg(long* bin, int siz);
double getAvg(long* bin, int siz);
double getSigmaByMax(long* bin, int siz);
//
template <class T> T findMaxTmp(T* arr, int siz);
template <class T> T findMinTmp(T* arr, int siz);

////////////////////////////////////////////
void track_lut(int, void*) {
	cout<<currentLUT << ": {" << lut::slut[currentLUT].data << "}" << endl;
	drawLUT();
}

void track_size_area(int, void*) {
	o_size_area	= SIZE_AREA_MIN + size_area_shift;
	size_area	= (t_width * o_size_area) / o_width;
	check_size_win();
	calcDirectionField();
	drawLUT();
	cout << "area-size: " << size_area << ", real-area-size: " << o_size_area << endl;
}

void on_mouse_map(int event, int x, int y, int flags, void* params) {
	cx	= x;
	cy	= y;
	ocx	= (cx * o_width) / t_width;
	ocy	= (cy * o_height) / t_height;
	//
	check_size_win();
	calcWinBins(false);
	drawLUT();

	if(event==CV_EVENT_LBUTTONDOWN/* || flags==CV_EVENT_FLAG_LBUTTON*/) {
		calcWinBins(true);
//		printBins();
		printBinsTmp<double>(bins2_weight, NBIN, "weight");
		printBinsTmp<long>(bins2_plain, NBIN, "plane");
		cout << "\n" << endl;
	}
//	cout<< "xy = [" << cx << ", " << cy << "], real-xy = [" << ocx << ", " << ocy << "]" << endl;
}

void printBins() {
	cout 	<< "bins(" << NBIN << ") : {avg=" << getAvg(bins, NBIN-1)
			<< ", max=" << findMaxIdx(bins, NBIN-1)
			<< "} = [";
	for(int i=0; i<NBIN-1; i++) {
		cout << bins[i] << ", ";
	}
	cout << " | " << bins[NBIN-1];
	cout << "]" << endl;
}

template <class T>
void printBinsTmp(T* bins, int siz, const string& text) {
	cout 	<< "bins{"<< text <<"}(" << siz << ") : = [";
	for(int i=0; i<siz; i++) {
		cout << bins[i] << ", ";
	}
	cout << "]" << endl;
}

// used global variables
void check_size_win() {
	int shift	= 4;
	if (ocx - o_size_area < shift) {
		ocx = o_size_area + shift;
	};
	if (ocx + o_size_area >= o_width - shift) {
		ocx = o_width - o_size_area - shift;
	};
	if (ocy - o_size_area < shift) {
		ocy = o_size_area + shift;
	};
	if (ocy + o_size_area >= o_height - shift) {
		ocy = o_height - o_size_area - shift;
	};
	//
	cx	= (ocx * t_width) / o_width;
	cy	= (ocy * t_height) / o_height;
}

void drawLegend(cv::Mat& img, int num) {
	cv::putText(img, lut::slut[currentLUT].data, cv::Point(20,40), cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(0,0,0), 4);
	cv::putText(img, lut::slut[currentLUT].data, cv::Point(20,40), cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(255,255,255), 1);
	//
	int h0	= 60;
	int dh	= 20;
	int ddh	= 15;
	int dw	= 40;
	int nn	= num;
	for(int i=0; i<nn; i++) {
		int bin	= (255 * i)/ nn;
		cv::Scalar color	= cv::Scalar(lut::slut[currentLUT].lut[bin][2],lut::slut[currentLUT].lut[bin][1], lut::slut[currentLUT].lut[bin][0]);
		cv::rectangle(img, cv::Rect(20, h0+i*dh, dw, ddh), color, CV_FILLED, CV_AA, 0);
		stringstream num;
		num << " - " << bin;
		cv::putText(img, num.str(), cv::Point(20+dw+20, h0+i*dh+dh/2), cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(0, 0, 0), 4);
		cv::putText(img, num.str(), cv::Point(20+dw+20, h0+i*dh+dh/2), cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(255, 255, 255), 1);
	}
}

template <class T>
T findMaxTmp(T* arr, int siz) {
	T max	= arr[0];
	for(int i=1; i<siz; i++) {
		if(arr[i]>max) {
			max	= arr[i];
		}
	}
	return max;
}

template <class T>
T findMinTmp(T* arr, int siz) {
	T min	= arr[0];
	for(int i=1; i<siz; i++) {
		if(arr[i]<min) {
			min	= arr[i];
		}
	}
	return min;
}

template <class T>
void drawBarHistOnRect(cv::Mat& base, cv::Rect& rect, const string& text, T* bins, int siz, cv::Scalar color) {
	const int padding	= 5; // FIXME: may be parameter of this function?
	int x0	= rect.x;
	int dx	= (rect.width - 2*padding) / (siz+1);
	int y0	= rect.y;
	int h	= rect.height;
	int w	= rect.width;
	T max 	= findMaxTmp(bins, siz-1); // FIXME: last bin -> bad pixels!
	if(max<1) {
		max	= 1;
	}
	cv::line(base, cv::Point(x0+padding, y0), cv::Point(x0+padding, y0+h), cv::Scalar(0,255,0),1);
	cv::line(base, cv::Point(x0, y0+h-padding), cv::Point(x0+w, y0+h-padding), cv::Scalar(0,255,0),1);
	cv::putText(base, text, cv::Point(x0+padding+5,y0 + 0), CV_FONT_HERSHEY_PLAIN, 1.0, cv::Scalar(255,255,255));
	int y1	= y0+h-padding;
	cv::Scalar tColor;
	for(int i=0; i<siz; i++) {
		T db	= bins[i];
		if (i < (NBIN - 1)) {
			tColor = color;
		} else {
			tColor = cv::Scalar(0, 0, 255);
			if(bins[i]>max) {
				db	= max;
			}
		}
		int dy		= ((h-2*padding) * db) / max;
		cv::Rect r	= cv::Rect(x0+padding + dx*(i+1), y1 - dy, dx-1, dy);
		cv::rectangle(base, r, tColor, CV_FILLED, CV_AA, 0);
	}
}

void drawBarHist(cv::Mat& base, int shift_y, cv::Scalar color, const string& text) {
	int y0	= (shift_y + 1) * HIST_SIZ_BASE - HIST_SIZ_BASED2;
	cv::putText(base, text, cv::Point(10,y0 + 50), CV_FONT_HERSHEY_PLAIN, 1.0, cv::Scalar(255,255,255));
	cv::line(base, cv::Point(0, y0), cv::Point(HIST_SIZ_W, y0), cv::Scalar(0,255,0), 1, CV_AA, 0);
	int padding_x	= 10;
	int dx	= (HIST_SIZ_W - 2*padding_x) / (NBIN+1);
	int idxMax	= findMaxIdx(bins, NBIN-1);
	long maxVal	= bins[idxMax]>bins[NBIN-1]?bins[idxMax]:bins[NBIN-1];
	if(maxVal<1) {
		maxVal	= 1;
	}
	cv::Scalar tColor;
	for(int i=0; i<NBIN; i++) {
		int dy	= (0.9 * HIST_SIZ_BASED2 * bins[i]) / maxVal;
		cv::Rect rect(padding_x + dx*(i+1), y0 - dy, dx-1, dy);
		if(i<(NBIN-1)) {
			tColor	= color;
		} else {
			tColor	= cv::Scalar(0, 0, 255);
		}
		cv::rectangle(base, rect, tColor, CV_FILLED, CV_AA, 0);
	}
	int angMaxAngle	= (int) (180. * idxMax / (NBIN-1));
	double sigma	= getSigmaByMax(bins, NBIN-1);
	//
	stringstream ss;
	ss << (180.-angMaxAngle) << "deg, sigma=" << sigma;
	cv::putText(base, ss.str(), cv::Point(10, y0+HIST_SIZ_BASED2 - 40), CV_FONT_HERSHEY_PLAIN, 1.2, cv::Scalar(0,0,255));
}

void drawHist() {
	cv::Mat hist	= cv::Mat::zeros(cv::Size(HIST_SIZ_W, HIST_SIZ_BASE*2 + HIST_SIZ_BASED2 + HIST_SIZ_BASED2), CV_8UC3);
	cv::Mat roiSrc(img,  cv::Rect(ocx-o_size_area, ocy-o_size_area, 2*o_size_area, 2*o_size_area));
	cv::Size zoom = cv::Size(o_size_area * 3, o_size_area * 3);
	cv::Mat crop, cropZoom;
	roiSrc.copyTo(crop);
	cv::resize(crop, cropZoom, zoom);
	cv::Mat roiDst(hist, cv::Rect(HIST_SIZ_W/2-zoom.width/2, HIST_SIZ_BASE + HIST_SIZ_BASED2 -zoom.height/2, zoom.width, zoom.height));
	cropZoom.copyTo(roiDst);
	//
	stringstream ss;
	int ds = o_size_area*2+1;
	ds	= ds*ds;
	ss << "thresh= " << bins[NBIN-1] << ":" << ds << " or ~" << (100 * bins[NBIN-1]) / ds << "%";
	drawBarHist(hist, 0, cv::Scalar(255, 0, 0), ss.str());
	//
	int dp	= 5;
	cv::Rect rect11	= cv::Rect(dp, 					HIST_SIZ_BASE*2+dp, (-2*dp + HIST_SIZ_W/2), (-2*dp + HIST_SIZ_BASED2) );
	cv::Rect rect12	= cv::Rect(dp + HIST_SIZ_W/2,	HIST_SIZ_BASE*2+dp, (-2*dp + HIST_SIZ_W/2), (-2*dp + HIST_SIZ_BASED2) );
	cv::Rect rect21	= cv::Rect(dp, 					HIST_SIZ_BASE*2+HIST_SIZ_BASED2+dp, (-2*dp + HIST_SIZ_W/2), (-2*dp + HIST_SIZ_BASED2) );
	cv::Rect rect22	= cv::Rect(dp + HIST_SIZ_W/2,	HIST_SIZ_BASE*2+HIST_SIZ_BASED2+dp, (-2*dp + HIST_SIZ_W/2), (-2*dp + HIST_SIZ_BASED2) );
	drawBarHistOnRect<double>(hist, rect11, "weight", bins2_weight,	NBIN, cv::Scalar(255,0,0));
	drawBarHistOnRect<long>  (hist, rect12, "plain",  bins2_plain,	NBIN, cv::Scalar(255,0,0));
	//
	drawBarHistOnRect<double>(hist, rect21, "weight-delta", bins2_weight_delta,	NBIN, cv::Scalar(255,0,0));
	drawBarHistOnRect<long>  (hist, rect22, "plain-delta",  bins2_plain_delta,	NBIN, cv::Scalar(255,0,0));
	cv::imshow(WIN_HIST, hist);
}

void drawLUT() {
	cv::Mat buff;
	bool isPostProcess	= true;
	if(true) {
		switch (currentSHOW) {
			case SHOW_ORIG:
				imgRsz.copyTo(buff);
				isPostProcess	= false;
				break;
			case SHOW_GRAY:
				imgGrayRsz.copyTo(buff);
				break;
			case SHOW_GX:
				if(isRotatedGxGy) {
					rgxRsz.copyTo(buff);
				} else {
					gxRsz.copyTo(buff);
				}
				break;
			case SHOW_GY:
				if(isRotatedGxGy) {
					rgyRsz.copyTo(buff);
				} else {
					gyRsz.copyTo(buff);
				}
				break;
			case SHOW_DIR:
				if(isDeltaField) {
					fieldMapRszDelta.copyTo(buff);
				} else {
					fieldMapRsz.copyTo(buff);
				}
				break;
			case SHOW_MAG:
				magRsz.copyTo(buff);
				break;
			case SHOW_ANG:
				angRsz.copyTo(buff);
				break;
			default:
				break;
		}
	}
	drawHist();
	// draw rect
	cv::rectangle(buff, cv::Rect(cx-size_area, cy-size_area, 2*size_area, 2*size_area), cv::Scalar(10, 255, 10), 2);
	if(!isPostProcess) {
		cv::imshow(WIN, buff);
		return;
	}
	//
	if(!isColorMode) {
		cv::imshow(WIN, buff);
	} else {
		cv::Mat tmpBuff;
		cv::Mat palette	= getCVLut(currentLUT);
		tmpBuff = getLutForGrayImage(buff, currentLUT, false);
		drawLegend(tmpBuff, 15);
		cv::imshow(WIN, tmpBuff);
	}
}
////////////////////////////////////////////
int findMaxIdx(long* bins, int siz) {
	long max 	= bins[0];
	int maxIdx	= 0;
	for(int i=1; i<siz; i++) {
		if(bins[i]>max) {
			max	= bins[i];
			maxIdx	= i;
		}
	}
	return maxIdx;
}

// return {minIdx, maxIdx}
cv::Point findMinMaxIdx(long* bins, int siz) {
	long	max 	= bins[0];
	int		maxIdx	= 0;
	long	min 	= bins[0];
	int		minIdx	= 0;
	for(int i=1; i<siz; i++) {
		if(bins[i]>max) {
			max	= bins[i];
			maxIdx	= i;
		} else if(bins[i]<min) {
			min	= bins[i];
			minIdx	= i;
		}
	}
	return cv::Point(minIdx, maxIdx);
}

// circular array idx: arr=[0,1,2,3] -> arr[-1] = 3, arr[-2] = 2, arr[4] = 0, arr[5] = 1
int getIdxCircular(int siz, int idx) {
    if(idx>=0 && idx<(siz-1)) {
        return idx;
    }
    int ridx    = idx - (idx / siz)*siz;
    if(ridx<0) {
        ridx= siz + ridx;
    }
    return ridx;
}

double getSigmaAvg(long* bin, int siz) {
    int nbin=0;
    int sd2    = (siz-0)/2;
    int avg    = 0;
    double ss  = 0.;
    for(int i=0; i<siz; i++) {
        avg += i*bin[i];
        nbin+= bin[i];
    }
    if(nbin<3) {
    	return 5.;
    }
    avg /= nbin;
    int i0=avg-sd2;
    int i1=avg+sd2;
    for(int i=i0; i<=i1; i++) {
        int idx = getIdxCircular(siz, i);
        ss += (i-avg)*(i-avg)*bin[idx];
    }
    ss *= siz;
    ss /= nbin;
    return sqrt(ss/(siz-1));
}

double getAvg(long* bin, int siz) {
	double ret	= 0.;
	int nbin	= 0;
	for(int i=0; i<siz; i++) {
		ret += i * bin[i];
		nbin+= bin[i];
	}
	return (ret/nbin);
}

double getSigmaByMax(long* bin, int siz) {
    int nbin=0;
    int sd2     = (siz-0)/2;
    int avg     = findMaxIdx(bin, siz);
    long max	= bin[avg];
    if(max<2) {
    	for(int i=0; i<siz; i++) {
    		bin[i] = 1;
    	}
    	avg=0;
    }
    double ss   = 0.;
    for(int i=0; i<siz; i++) {
        nbin+= bin[i];
    }
    int i0=avg-sd2;
    int i1=avg+sd2;
    for(int i=i0; i<=i1; i++) {
        int idx = getIdxCircular(siz, i);
        ss += (i-avg)*(i-avg)*bin[idx];
    }
    ss *= siz;
    ss /= nbin;
    return sqrt(ss/(siz-1));
}

template <class T>
double calcSigmaForBigNorm(T* bins, int siz) {
	T max	= findMaxTmp(bins, siz);
	double ret	= 0.;
	if(max<2) {
		for(int i=0; i<siz; i++) {
			ret += (double) i*i;
		}
		ret = ret/(siz*(siz-1));
		return sqrt(ret);
	}
	//
	T sum = 0;
	for(int i=0; i<siz; i++) {
		sum += bins[i];
		ret += (double)bins[i]*i*i;
	}
	ret = ret / (sum * (siz-1));
	return sqrt(ret);
}

void calcDirectionField() {
	int w	= o_width;
	int h	= o_height;
	int d	= NBIN; //the last bin is not used
	long* tmp_bins	= new long[d];
	cv::Mat fieldMapGS		=  cv::Mat::zeros(cv::Size(w,h), CV_8UC1);
	cv::Mat fieldMapGSDelta	=  cv::Mat::zeros(cv::Size(w,h), CV_8UC1);
	cv::Mat fieldMap	=  cv::Mat::zeros(cv::Size(w,h), CV_64F);
	int iims, jjms;
	int scan_win	= o_size_area * 2 + 1;
	int i0	= scan_win + 1;
	int j0	= scan_win + 1;
	cout << "::" << scan_win << ", d=" << d << ", w=" << w << ", h=" << h << endl;
	int ii, jj;
	int wd	= w * d;
	double sigma;
	double maxSigma	= -1.;
	double minSigma	= DBL_MAX;
	for(int i=i0; i<h; i++) {
		ii	= wd*i;
		iims= wd*(i-scan_win);
		for(int j=j0; j<w; j++) {
			jj	= d*j;
			jjms= d*(j-scan_win);
			for(int k=0; k<d-1; k++) {
				tmp_bins[k]	= dataHist[ii + jj + k] + dataHist[iims + jjms + k] - dataHist[ii + jjms + k] - dataHist[iims + jj + k];
			}
//			sigma	= getSigmaAvg(tmp_bins, d-1);
			sigma	= getSigmaByMax(tmp_bins, d-1);
//			sigma	= findMaxIdx(tmp_bins, d-1);
//			sigma	= getAvg(tmp_bins, d-1);
			if(sigma>maxSigma) {
				maxSigma	= sigma;
			}
			if(sigma<minSigma) {
				minSigma	= sigma;
			}
			fieldMap.at<double>(i-o_size_area,j-o_size_area) = sigma;
		}
	}
	//
	cout << "maxSigma = " << maxSigma << endl;
	double val;
	for(int i=0; i<h; i++) {
		for(int j=0; j<w; j++) {
			val	= fieldMap.at<double>(i,j);
			fieldMapGS.at<uchar>(i, j)	= (uchar) (255. * val /maxSigma);
			fieldMapGSDelta.at<uchar>(i, j)	= (uchar) (255. * (val - minSigma) /(maxSigma-minSigma));
		}
	}
	//
	cv::resize(fieldMapGS,		fieldMapRsz,	t_size);
	cv::resize(fieldMapGSDelta,fieldMapRszDelta,t_size);
	delete [] tmp_bins;
}

/*
 * 	i -> rows, j-> columns
 */
void calcWinBins(bool ext) {
	int d	= NBIN;
	int wd	= o_width * d;
	//
	int i0	= ocy - o_size_area;
	int j0	= ocx - o_size_area;
	int i1	= ocy + o_size_area+1;
	int j1	= ocx + o_size_area+1;
	//
	int ii0	= wd*i0;
	int ii1	= wd*i1;
	int jj0	= d*j0;
	int jj1	= d*j1;
	for(int k=0; k<d; k++) {
		bins[k]	= dataHist[ii1 + jj1 + k] + dataHist[ii0 + jj0 + k] - dataHist[ii1 + jj0 + k] - dataHist[ii0 + jj1 + k];
	}
	// calc bins2
	if(ext) {
		memset(bins2_plain,  0, sizeof(long)	* NBIN);
		memset(bins2_weight, 0, sizeof(double)	* NBIN);
		memset(bins2_plain_delta,  0, sizeof(long)	* NBIN);
		memset(bins2_weight_delta, 0, sizeof(double)* NBIN);
		double x1, y1, x2, y2, dx, dy;
		double v1, v2, dv2;
		double cos_a, angle_rad, angle_deg;
		double angle, angle_avg;
		int idx;
		int counter = 0;
		for(int py1=i0; py1<=i1; py1++) {
			for(int px1=j0; px1<=j1; px1++) {
				x1	= rgx.at<double>(py1, px1);
				y1	= rgy.at<double>(py1, px1);
				for(int py2=i0; py2<=i1; py2++) {
					for(int px2=j0; px2<=j1; px2++) {
						if(!(py2==py1 && px2==px1)) {
							x2	= rgx.at<double>(py2, px2);
							y2	= rgy.at<double>(py2, px2);
							//
							dx	= x2 - x1;
							dy	= y2 - y1;
							//
							v1	= sqrt(x1*x1+y1*y1);
							v2	= sqrt(x2*x2+y2*y2);
							dv2	= dx*dx + dy*dy;
							if(v1>thresh && v2>thresh) {
								cos_a		= (v1*v1+v2*v2-dv2)/(2.*v1*v2);
								if(cos_a>1.) {
									cos_a	= 1.;
								} else if(cos_a<-1.) {
									cos_a	= -1.;
								}
								angle_rad	= acos(cos_a);
								angle		= 180. * angle_rad / M_PI;
								//
								if(angle>90.) {
									angle	= 180. - angle;
								}
								idx	= (int)((angle * (NBIN - 1)) / 90.);
								if(idx<0) {
									cout	<< "xy1=[" << x1 << ", " << y1 << "], xy2=[" << x2 << ", " << y2 << "], idx=" << idx
											<< ", angle=" << angle << ", cos_a=" << cos_a << ", angle_rad="<< angle_rad
											<< ", point1={" << px1 << ", " << py1 << "}, point2={" << px2 << ", " << py2 << "}" << endl;
								}
//								cout << idx << "  ";
								bins2_plain[idx]	+=1;
								bins2_weight[idx]	+=sqrt(v1*v2);
								//
								angle_avg	+= angle;
								counter++;
							} else {
								/*for(int ll=0; ll<NBIN-1; ll++) {
									bins2_plain[ll] +=1;
								}*/
								//
								bins2_plain[NBIN-1]	+=1;
								bins2_weight[NBIN-1]+=1.; // FIXME: bad solution
							}
						}
					}
				}
			}
		}
		long minPlain		= findMinTmp<long>(bins2_plain,		d-1);
		double minWeight	= findMinTmp<double>(bins2_weight,	d-1);
		//
		for(int i=0; i<d-1; i++) {
			bins2_plain_delta[i]	= bins2_plain[i] -minPlain;
			bins2_weight_delta[i]	= bins2_weight[i]-minWeight;
		}
//		cout << "avg-angle = " << (angle_avg / counter) << endl;
		double sigma_plain	= calcSigmaForBigNorm<long>(bins2_plain, d-1);
		double sigma_weight	= calcSigmaForBigNorm<double>(bins2_weight, d-1);
		cout << "sigma-plain = " << sigma_plain << ", sigma-weight=" << sigma_weight << endl;
	}
}

void printHelp(char* argv0) {
	cout << "Usage: " << argv0 << " {/path/to/image} {num_bin}" << endl;
	cout << " * app-keys:"<<endl;
	cout	<< "\t <Esc>, q -> exit\n"
			<< "\to -> show original\n"
			<< "\tg -> show grayscale\n"
			<< "\tc -> switch color mode\n"
			<< "\tr -> switch rotate or not-rotate gradient\n"
			<< "\tm -> show magnitude\n"
			<< "\ta -> show angle\n"
			<< "\tx -> show x-component\n"
			<< "\ty -> show y-component\n"
			<< "\tf -> show field\n"
			<< "\td -> switch scale [0,max] or [min,max]\n"
			<< "\tp -> print this help\n"
			<< endl;
}

/////////////////////////////////////////////////////////////
int main(int argc, char* argv[]) {
	if(argc<3) {
		cerr << "Usage: " << argv[0] << " {/path/to/image} {num_bin}" << endl;
		exit(1);
	}
	string filename;
	istringstream iss(argv[1]);
	while(getline(iss, filename, '/')) {}

	NBIN	= atoi(argv[2]);
	if(NBIN==0) {
		cerr << "bad nbin, set to 15" << endl;
		NBIN	= 9;
	}
	NBIN	+= 1;
/////////////////////////////
	printHelp(argv[0]);
	//
	img = cv::imread(argv[1], CV_LOAD_IMAGE_COLOR);
	cv::cvtColor(img, imgGray, CV_RGB2GRAY);
	//
	cv::Mat mag, ang;
	cv::Sobel(imgGray, gx, CV_64F, 1, 0, 3, 1, 0, cv::BORDER_DEFAULT);
	cv::Sobel(imgGray, gy, CV_64F, 0, 1, 3, 1, 0, cv::BORDER_DEFAULT);
	rgx = -gy;
	rgy =  gx;
	cv::cartToPolar(-gy, gx, mag, ang, true);
///////////////////////////
	o_width = img.cols;
	o_height = img.rows;
	t_width = o_width;
	t_height = o_height;
	o_size = cv::Size(o_width, o_height);
	if (img.cols > MAX_WIDTH) {
		t_width = MAX_WIDTH;
		t_height = (img.rows * MAX_WIDTH) / img.cols;
	}
	t_size = cv::Size(t_width, t_height);
///////////////////////////
	int w	= ang.cols;
	int h	= ang.rows;
	int d	= NBIN;
	cout << "w=" << w << ", h="<<h<<", d=" << d <<endl;
	//
	int mem_in_mb	= (sizeof(long) * w * h * d ) / (1024 * 1024);
	cout << "::allocate memory : " <<  mem_in_mb << "Mb" << endl;
	dataHist	= new long[w*h*d];
	bins		= new long[d];
	bins2_plain	= new long[d];
	bins2_weight= new double[d];
	bins2_plain_delta	= new long[d];
	bins2_weight_delta	= new double[d];
	memset(dataHist,	0, sizeof(long)	* w * h * d );
	memset(bins, 		0, sizeof(long) * NBIN );
	memset(bins2_plain,	0, sizeof(long) * NBIN );
	memset(bins2_weight,0, sizeof(double) * NBIN );

	printBinsTmp<long>(bins2_plain, NBIN,	"plane");
	printBinsTmp<double>(bins2_weight, NBIN,"weight");
	//
	double tang, tangr, tmag;
	int tangIdx;
	for(int i=0; i<ang.rows; i++) {
		int ii	= i*w*d;
		for(int j=0; j<ang.cols; j++) {
			tang	= ang.at<double>(i,j);
			tmag	= ang.at<double>(i,j);
			if(tmag>thresh) {
				tangr	= tang;
				if(tang<0) {
					tangr	= 180. + tang;
//					ang.at<double>(i,j)	= 180. + tang;
				} else if(tang > 180.) {
//					ang.at<double>(i,j)	= tang - 180.;
					tangr	= tang - 180.;
				}
//				tangIdx	= (int) (tangr * (d-0) / 180.);
				tangIdx	= (int) (tangr * (d-1) / 180.); // bin (d-1) for thresholded angles
				dataHist[ii + j*d + tangIdx]	+=1;
			} else {
				dataHist[ii + j*d + (d-1)]	+=1;
			}
		}
	}

//////////// BEGIN : create integral-image-representation
	int wd= w*d;
	int ii, iim1, jj, jjm1;
	// fill 1st column
	for(int i=1; i<h; i++) {
		ii=wd*i;
		iim1=wd*(i-1);
		for(int k=0; k<d; k++) {
			dataHist[ii + k] += dataHist[iim1 + k];
		}
	}
	// fill 1st row
	for(int j=1; j<w; j++) {
		jj=d*j;
		jjm1=d*(j-1);
		for(int k=0; k<d; k++) {
			dataHist[jj + k] += dataHist[jjm1 + k];
		}
	}
	// integrate
	for(int i=1; i<h; i++) {
		ii	= wd*i;
		iim1= wd*(i-1);
		for(int j=1; j<w; j++) {
			jj	= d*j;
			jjm1= d*(j-1);
			for(int k=0; k<d; k++) {
				dataHist[ii + jj +k] += dataHist[iim1+jj+k] + dataHist[ii+jjm1+k]-dataHist[iim1+jjm1+k];
			}
		}
	}
//////////// END : create integral-image-representation
	calcDirectionField();
	size_area	= (t_width * o_size_area) / o_width; //FIXME: bad solution

	// save file
	/*
	stringstream ss;
	ss<<filename << "_b" << NBIN << "_s" << (2*scan_win_d2 + 1) << "_corr" << ".png";
	cv::imwrite(ss.str(), fieldMapRsz);
	*/

	if(true) {
		cv::namedWindow(WIN, CV_WINDOW_AUTOSIZE);
		cv::namedWindow(WIN_HIST, CV_WINDOW_AUTOSIZE);
		cv::createTrackbar("color-scheme: ", WIN, &currentLUT, sizeof(lut::slut)/sizeof(lut::ss)-1, track_lut, 0);
		cv::createTrackbar("size: ", WIN, &size_area_shift, SIZE_AREA_MAX - SIZE_AREA_MIN, track_size_area, 0);
		cv::setMouseCallback(WIN, on_mouse_map, NULL);

		// auto-free block
		{
			cv::Mat gxScaled, gyScaled, magScaled, angScaled,  rgxScaled, rgyScaled;
			cv::convertScaleAbs(gx,	gxScaled);
			cv::convertScaleAbs(gy,	gyScaled);
			cv::convertScaleAbs(rgx,rgxScaled);
			cv::convertScaleAbs(rgy,rgyScaled);
			cv::convertScaleAbs(mag,magScaled);
			cv::convertScaleAbs(ang,angScaled);
			//
			cv::resize(gxScaled,  gxRsz,  t_size);
			cv::resize(gyScaled,  gyRsz,  t_size);
			cv::resize(rgxScaled, rgxRsz, t_size);
			cv::resize(rgyScaled, rgyRsz, t_size);
			cv::resize(magScaled, magRsz, t_size);
			cv::resize(angScaled, angRsz, t_size);
			//
			cv::resize(img, imgRsz, t_size);
			cv::resize(imgGray, imgGrayRsz, t_size);
		}
		track_size_area(9, NULL);

		int key;
		while (true) {
			key	= cv::waitKey(0);
			if(key==27 || key=='q' || key=='Q') {
				break;
			}
			switch (key) {
				case 'x':
				case 'X':
					currentSHOW	= SHOW_GX;
					break;
				case 'y':
				case 'Y':
					currentSHOW	= SHOW_GY;
					break;
				case 'o':
				case 'O':
					currentSHOW	= SHOW_ORIG;
					break;
				case 'g':
				case 'G':
					currentSHOW	= SHOW_GRAY;
					break;
				case 'c':
				case 'C':
					if(isColorMode) {
						isColorMode	= false;
					} else {
						isColorMode	= true;
					}
					break;
				case 'f':
				case 'F':
					currentSHOW	= SHOW_DIR;
					break;
				case 'm':
				case 'M':
					currentSHOW	= SHOW_MAG;
					break;
				case 'a':
				case 'A':
					currentSHOW	= SHOW_ANG;
					break;
				case 'd':
				case 'D':
					if(isDeltaField) {
						isDeltaField = false;
					} else {
						isDeltaField = true;
					}
					break;
				case 'r':
				case 'R':
					if(isRotatedGxGy) {
						isRotatedGxGy	= false;
					} else {
						isRotatedGxGy	= true;
					}
					break;
				case 'p':
				case 'P':
					printHelp(argv[0]);
					break;
				default:
					break;
			}
			drawLUT();
		}

		delete [] dataHist;
		delete [] bins;
		delete [] bins2_plain;
		delete [] bins2_weight;
		delete [] bins2_plain_delta;
		delete [] bins2_weight_delta;
		/*
		stringstream buff;
		buff << "/home/ar/workspace-cdt-3.6/OpencvTutiorials/bin/test_gray2palette " << ss.str();
		system(buff.str().c_str());
		*/
	}
	return 0;
}
