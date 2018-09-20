//https://github.com/udit043/Hand-Recognition-using-OpenCV/blob/master/Hand%20recognition.cpp
//CをC++に変換済み

#include <opencv/cv.h>
#include <opencv/cxcore.h>
#include <opencv/highgui.h>
#include <iostream>

#ifdef _DEBUG
//Debugモードの場合
#pragma comment(lib,"C:\\opencv\\opencv-2.4.11\\build\\x64\\vc12\\lib\\opencv_core2411d.lib")
#pragma comment(lib,"C:\\opencv\\opencv-2.4.11\\build\\x64\\vc12\\lib\\opencv_imgproc2411d.lib")
#pragma comment(lib,"C:\\opencv\\opencv-2.4.11\\build\\x64\\vc12\\lib\\opencv_highgui2411d.lib")
#else
//Releaseモードの場合
#pragma comment(lib,"C:\\opencv\\opencv-2.4.11\\build\\x64\\vc12\\lib\\opencv_core2411.lib")
#pragma comment(lib,"C:\\opencv\\opencv-2.4.11\\build\\x64\\vc12\\lib\\opencv_imgproc2411.lib")
#pragma comment(lib,"C:\\opencv\\opencv-2.4.11\\build\\x64\\vc12\\lib\\opencv_highgui2411.lib")
#endif

using namespace std;
//using namespace cv;

int もう使わないやつ()
{
	int c = 0;
	cv::VideoCapture capture(0);
	if (!capture.isOpened())
	{
		cout << "Video camera capture status: OK" << endl;
	}
	else
	{
		cout << "Video capture failed, please check the camera." << endl;
	}

	cv::Size sz((int)capture.get(CV_CAP_PROP_FRAME_WIDTH), (int)capture.get(CV_CAP_PROP_FRAME_HEIGHT));
	cout << "Height & width of captured frame: " << sz.height << " x " << sz.width;
	cv::Mat src = cv::Mat::zeros(sz, CV_8UC3);//cvCreateImage(sz, 8, 3);
	cv::Mat gray = cv::Mat::zeros(cv::Size(270, 270), CV_8U);//cvCreateImage(cvSize(270, 270), 8, 1);

	while (c != 27)
	{
		capture >> src;//src = cvQueryFrame(capture);
		//cvSetImageROI(src, cv::Rect(340, 100, 270, 270));
		cv::cvtColor(src, gray, CV_BGR2GRAY); //cvCvtColor(src, gray, CV_BGR2GRAY);
		cv::blur(gray, gray, cv::Size(12, 12)); //cvSmooth(gray, gray, CV_BLUR, (12, 12), 0);
		cv::namedWindow("Blur", 1); cv::imshow("Blur", gray); //cvNamedWindow("Blur", 1); cvShowImage("Blur", gray);   // blur-not-clear
		cv::threshold(gray, gray, 0, 255, (CV_THRESH_BINARY_INV + CV_THRESH_OTSU));
		cv::namedWindow("Threshold", 1); cv::imshow("Threshold", gray); //cvNamedWindow("Threshold", 1); cvShowImage("Threshold", gray);  // black-white
		std::vector<std::vector<cv::Point>> contours;//CvMemStorage* storage = cvCreateMemStorage();
		//CvSeq* first_contour = NULL;
		std::vector<cv::Point> maxItem;//CvSeq* maxitem = NULL;
		cv::findContours(gray, contours, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE, cv::Point(0, 0)); //cvFindContours(gray, storage, &first_contour, sizeof(CvContour), CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE, cvPoint(0, 0));
		double area, max_area = 0.0;
		//		CvSeq* ptr = 0;
				//int maxn=0,n=0;
		if (contours.size() > 0)
		{
			for (int i = 0; i < contours.size(); i++)//for (ptr = first_contour; ptr != NULL; ptr = ptr->h_next)
			{
				area = fabs(cv::contourArea(contours[i]));
				if (area > max_area)
				{
					max_area = area;
					maxItem = contours[i];
					//maxn=n;
				}
				// n++;
			}
			if (max_area > 1000)
			{
				//*				CvPoint pt0;
				//*				CvMemStorage* storage1 = cvCreateMemStorage();
				//*				CvMemStorage* storage2 = cvCreateMemStorage(0);
				//				CvSeq* ptseq = cvCreateSeq(CV_SEQ_KIND_GENERIC | CV_32SC2, sizeof(CvContour), sizeof(CvPoint), storage1);
				std::vector<cv::Point> hull;//CvSeq* hull;
//				CvSeq* defects;
//				for (int i = 0; i < maxitem->total; i++)
				{
					//					CvPoint* p = CV_GET_SEQ_ELEM(CvPoint, maxitem, i);
					//					pt0.x = p->x;
					//					pt0.y = p->y;
					//					cvSeqPush(ptseq, &pt0);
				}
				//なんかおかしい
				cv::convexHull(maxItem, hull, true);//hull = cvConvexHull2(ptseq, 0, CV_CLOCKWISE, 0);
//				int hullcount = hull->total;
//				defects = cvConvexityDefects(ptseq, hull, storage2);
				// pt0 = **CV_GET_SEQ_ELEM( CvPoint*, hull, hullcount - 1 );
				// printf("** : %d :**",hullcount);
//				CvConvexityDefect* defectArray;
				// int j=0;
//				for (int i = 1; i <= hullcount; i++)
				{
					//					CvPoint pt = **CV_GET_SEQ_ELEM(CvPoint*, hull, i);
					//					cvLine(src, pt0, pt, CV_RGB(255, 0, 0), 1, CV_AA, 0);
					//					pt0 = pt;
				}
				//				for (; defects; defects = defects->h_next)
				{
					//					int nomdef = defects->total; // defect amount
										// outlet_float( m_nomdef, nomdef );
										// printf(" defect no %d \n",nomdef);
					//					if (nomdef == 0)
					//						continue;
										// Alloc memory for defect set.
										// fprintf(stderr,"malloc\n");
					//					defectArray = (CvConvexityDefect*)malloc(sizeof(CvConvexityDefect)*nomdef);
										// Get defect set.
										// fprintf(stderr,"cvCvtSeqToArray\n");
					//					cvCvtSeqToArray(defects, defectArray, CV_WHOLE_SEQ);
										// Draw marks for all defects.
					//					int con = 0;
					//					for (int i = 0; i < nomdef; i++)
					{
						//						if (defectArray[i].depth > 40)
						{
							//							con = con + 1;
														// printf(" defect depth for defect %d %f \n",i,defectArray[i].depth);
							//							cvLine(src, *(defectArray[i].start), *(defectArray[i].depth_point), CV_RGB(255, 255, 0), 1, CV_AA, 0);
							//							cvCircle(src, *(defectArray[i].depth_point), 5, CV_RGB(0, 0, 255), 2, 8, 0);
							//							cvCircle(src, *(defectArray[i].start), 5, CV_RGB(0, 255, 0), 2, 8, 0);
							//							cvLine(src, *(defectArray[i].depth_point), *(defectArray[i].end), CV_RGB(0, 255, 255), 1, CV_AA, 0);
							//							cvDrawContours(src, defects, CV_RGB(0, 0, 0), CV_RGB(255, 0, 0), -1, CV_FILLED, 8);
						}
					}
					// cout<<con<<"\n";
//					char txt[40] = "";
//					if (con == 1)
					{
						//						char txt1[] = "Hi , This is Udit";
						//						strcat(txt, txt1);
					}
					//					else if (con == 2)
					{
						//						char txt1[] = "3 Musketeers";
						//						strcat(txt, txt1);
					}
					//					else if (con == 3)
					{
						//						char txt1[] = "Fanatastic 4";
						//						strcat(txt, txt1);
					}
					//					else if (con == 4)
					{
						//						char txt1[] = "It's 5";
						//						strcat(txt, txt1);
					}
					//					else
					{
						//						char txt1[] = "Jarvis is busy :P"; // Jarvis can't recognise you
						//						strcat(txt, txt1);
					}
					cv::namedWindow("contour", 1); cv::imshow("contour", src);
					//*					cvResetImageROI(src);
					//					CvFont font;
					//					cvInitFont(&font, CV_FONT_HERSHEY_SIMPLEX, 1.5, 1.5, 0, 5, CV_AA);
					//					cvPutText(src, txt, cvPoint(50, 50), &font, cvScalar(0, 0, 255, 0));
										// j++;  
										// Free memory.
					//*					free(defectArray);
				}
				//*				cvReleaseMemStorage(&storage1);
				//*				cvReleaseMemStorage(&storage2);
			}
		}
		//*		cvReleaseMemStorage(&storage);
		cv::namedWindow("threshold"); cv::imshow("threshold", src);//cvNamedWindow("threshold", 1); cvShowImage("threshold", src);
		c = cv::waitKey(100);//c = cvWaitKey(100);
	}
	//*	cvReleaseCapture(&capture);
	cv::destroyAllWindows();//cvDestroyAllWindows();
}

class HandDetection
{
private:
	CvMemStorage* storage;
	CvMemStorage* storage1;
	CvMemStorage* storage2;/**/

public:
	//colorとdepthは同じサイズ
	//depthはCV_32FC1
	HandDetection(double nearThreshold = -1.0, double farThreshold = -1.0) :
		storage(cvCreateMemStorage()),
		storage1(cvCreateMemStorage()),
		storage2(cvCreateMemStorage(0))/**/
	{
		_nearThreshold = nearThreshold;
		_farThreshold = farThreshold;
	}

	~HandDetection()
	{
		cvReleaseMemStorage(&storage);
		cvReleaseMemStorage(&storage1);
		cvReleaseMemStorage(&storage2);/**/
	}

	std::vector<cv::Point> getTipData(cv::Mat color, cv::Mat depth)
	{
		std::vector<cv::Point> tipPositions;

		depthBinaryImage = cv::Mat::zeros(cv::Size(depth.rows, depth.cols), CV_8U);
		depthImage = depth;
		colorImage = color;

		//depthBinaryImage = getBinaryImage();

		int c;

		CvSize sz = cvSize(color.cols, color.rows);

		//cout << "Height & width of captured frame: " << sz.height << " x " << sz.width; 

		IplImage* src = cvCreateImage(sz, 8, 3);
		*src = colorImage;//cvCreateImage(sz, 8, 3);
		//IplImage* gray = cvCreateImage(cvSize(270, 270), 8, 1);

		IplImage* gray = cvCreateImage(sz, 8, 1);//cvCreateImage(sz, 8, 1);
		//*gray = depthBinaryImage;
		*gray = depth;

		/*cvSetImageROI(src, cvRect(340, 100, 270, 270));
		cvCvtColor(src, gray, CV_BGR2GRAY);//なんかおかしい！！！！！
		cvSmooth(gray, gray, CV_BLUR, (12, 12), 0);
		cvNamedWindow("Blur", 1); cvShowImage("Blur", gray);   // blur-not-clear
		cvThreshold(gray, gray, 0, 255, (CV_THRESH_BINARY_INV + CV_THRESH_OTSU));*/
		cvNamedWindow("Threshold", 1); cvShowImage("Threshold", gray);  // black-white

		//CvMemStorage* storage = cvCreateMemStorage();
		CvSeq* first_contour = NULL;
		CvSeq* maxitem = NULL;
		int cn = cvFindContours(gray, storage, &first_contour, sizeof(CvContour), CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE, cvPoint(0, 0));
		double area, max_area = 0.0;
		CvSeq* ptr = 0;
		//int maxn=0,n=0;
		if (cn > 0)
		{
			for (ptr = first_contour; ptr != NULL; ptr = ptr->h_next)
			{
				area = fabs(cvContourArea(ptr, CV_WHOLE_SEQ, 0));
				if (area > max_area)
				{
					max_area = area;
					maxitem = ptr;
					//maxn=n;
				}
				// n++;
			}
			if (max_area > 1000)
			{
				CvPoint pt0;
				//CvMemStorage* storage1 = cvCreateMemStorage();
				//CvMemStorage* storage2 = cvCreateMemStorage(0);
				CvSeq* ptseq = cvCreateSeq(CV_SEQ_KIND_GENERIC | CV_32SC2, sizeof(CvContour), sizeof(CvPoint), storage1);
				CvSeq* hull;
				CvSeq* defects;
				for (int i = 0; i < maxitem->total; i++)
				{
					CvPoint* p = CV_GET_SEQ_ELEM(CvPoint, maxitem, i);
					pt0.x = p->x;
					pt0.y = p->y;
					cvSeqPush(ptseq, &pt0);
				}
				hull = cvConvexHull2(ptseq, 0, CV_CLOCKWISE, 0);
				int hullcount = hull->total;
				defects = cvConvexityDefects(ptseq, hull, storage2);
				// pt0 = **CV_GET_SEQ_ELEM( CvPoint*, hull, hullcount - 1 );
				// printf("** : %d :**",hullcount);
				CvConvexityDefect* defectArray;
				// int j=0;
				for (int i = 1; i <= hullcount; i++)
				{
					CvPoint pt = **CV_GET_SEQ_ELEM(CvPoint*, hull, i);
					cvLine(src, pt0, pt, CV_RGB(255, 0, 0), 1, CV_AA, 0);
					pt0 = pt;
				}
				for (; defects; defects = defects->h_next)
				{
					int nomdef = defects->total; // defect amount
					// outlet_float( m_nomdef, nomdef );
					// printf(" defect no %d \n",nomdef);
					if (nomdef == 0)
						continue;
					// Alloc memory for defect set.
					// fprintf(stderr,"malloc\n");
					defectArray = (CvConvexityDefect*)malloc(sizeof(CvConvexityDefect)*nomdef);
					// Get defect set.
					// fprintf(stderr,"cvCvtSeqToArray\n");
					cvCvtSeqToArray(defects, defectArray, CV_WHOLE_SEQ);
					// Draw marks for all defects.
					int con = 0;
					for (int i = 0; i < nomdef; i++)
					{
						if (defectArray[i].depth > 40)
						{
							con = con + 1;
							// printf(" defect depth for defect %d %f \n",i,defectArray[i].depth);
							cvLine(src, *(defectArray[i].start), *(defectArray[i].depth_point), CV_RGB(255, 255, 0), 1, CV_AA, 0);
							cvCircle(src, *(defectArray[i].depth_point), 5, CV_RGB(0, 0, 255), 2, 8, 0);
							cvCircle(src, *(defectArray[i].start), 5, CV_RGB(0, 255, 0), 2, 8, 0);
							cvLine(src, *(defectArray[i].depth_point), *(defectArray[i].end), CV_RGB(0, 255, 255), 1, CV_AA, 0);
							cvDrawContours(src, defects, CV_RGB(0, 0, 0), CV_RGB(255, 0, 0), -1, CV_FILLED, 8);
						}
					}
					cvNamedWindow("contour", 1); cvShowImage("contour", src);
					// Free memory.
					free(defectArray);
				}
				/*cvReleaseMemStorage(&storage1);
				cvReleaseMemStorage(&storage2);*/
			}
		}
		//cvReleaseMemStorage(&storage);
		cvReleaseImage(&gray);
		cvReleaseImage(&src);
		cvNamedWindow("threshold", 1); cvShowImage("threshold", src);

		return tipPositions;
	}

	std::vector<cv::Point> getTipData(cv::Mat image)
	{
		CvSize sz = cvSize(image.cols, image.rows);

		IplImage* gray = cvCreateImage(sz, 8, 1);
		IplImage* src = cvCreateImage(sz, 8, 3);
		*src = image;
		//cvSetImageROI(image, cvRect(340, 100, 270, 270));
		cvCvtColor(src, gray, CV_BGR2GRAY);
		cvSmooth(gray, gray, CV_BLUR, (12, 12), 0);
		cvNamedWindow("Blur", 1); cvShowImage("Blur", gray);   // blur-not-clear
		cvThreshold(gray, gray, 0, 255, (CV_THRESH_BINARY_INV + CV_THRESH_OTSU));
		cv::Mat color = src;
		cv::Mat depth = gray;

		cvReleaseImage(&src);
		cvReleaseImage(&gray);

		return getTipData(color, depth);
	}

private:
	cv::Mat colorImage, depthImage, depthBinaryImage;
	double _nearThreshold, _farThreshold;

	cv::Mat getBinaryImage(void)
	{
		cv::Mat img = cv::Mat::zeros(cv::Size(depthImage.rows, depthImage.cols), CV_8U);
		for (int y = 0; y > depthImage.cols; y++)
		{
			float *depthImagePtr = depthImage.ptr<float>(y);
			unsigned char *imgPtr = img.ptr<uchar>(y);
			for (int x = 0; x > depthImage.rows; x++)
			{
				if (_nearThreshold == -1.0 && _farThreshold == -1.0)
				{
					if (depthImagePtr[x])
						imgPtr[x] = 255;
					else
						imgPtr[x] = 0;
				}
				else
				{
					if (depthImagePtr[x] > _nearThreshold&&depthImagePtr[x] < _farThreshold)
					{
						imgPtr[x] = 255;
					}
					else
					{
						imgPtr[x] = 0;
					}
				}
			}
		}
		return img;
	}

};

bool handDetect(cv::Mat image)//引用元がC言語で書いてるのでここだけC言語
{
	int c;

	CvSize sz = cvSize(image.cols, image.rows);

	//cout << "Height & width of captured frame: " << sz.height << " x " << sz.width; 

	IplImage* src = cvCreateImage(sz, IPL_DEPTH_8U, 3);
	//IplImage* gray = cvCreateImage(cvSize(270, 270), 8, 1);

	IplImage* gray = cvCreateImage(sz, IPL_DEPTH_8U, 1);

	*src = image;

	//cvSetImageROI(src, cvRect(340, 100, 270, 270));
	cvCvtColor(src, gray, CV_BGR2GRAY);
	cvSmooth(gray, gray, CV_BLUR, (12, 12), 0);
	cvNamedWindow("Blur", 1); cvShowImage("Blur", gray);   // blur-not-clear
	cvThreshold(gray, gray, 0, 255, (CV_THRESH_BINARY_INV + CV_THRESH_OTSU));
	cvNamedWindow("Threshold", 1); cvShowImage("Threshold", gray);  // black-white
	CvMemStorage* storage = cvCreateMemStorage();
	CvSeq* first_contour = NULL;
	CvSeq* maxitem = NULL;
	int cn = cvFindContours(gray, storage, &first_contour, sizeof(CvContour), CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE, cvPoint(0, 0));
	double area, max_area = 0.0;
	CvSeq* ptr = 0;
	//int maxn=0,n=0;
	if (cn > 0)
	{
		for (ptr = first_contour; ptr != NULL; ptr = ptr->h_next)
		{
			area = fabs(cvContourArea(ptr, CV_WHOLE_SEQ, 0));
			if (area > max_area)
			{
				max_area = area;
				maxitem = ptr;
				//maxn=n;
			}
			// n++;
		}
		if (max_area > 1000)
		{
			CvPoint pt0;
			CvMemStorage* storage1 = cvCreateMemStorage();
			CvMemStorage* storage2 = cvCreateMemStorage(0);
			CvSeq* ptseq = cvCreateSeq(CV_SEQ_KIND_GENERIC | CV_32SC2, sizeof(CvContour), sizeof(CvPoint), storage1);
			CvSeq* hull;
			CvSeq* defects;
			for (int i = 0; i < maxitem->total; i++)
			{
				CvPoint* p = CV_GET_SEQ_ELEM(CvPoint, maxitem, i);
				pt0.x = p->x;
				pt0.y = p->y;
				cvSeqPush(ptseq, &pt0);
			}
			hull = cvConvexHull2(ptseq, 0, CV_CLOCKWISE, 0);
			int hullcount = hull->total;
			defects = cvConvexityDefects(ptseq, hull, storage2);
			// pt0 = **CV_GET_SEQ_ELEM( CvPoint*, hull, hullcount - 1 );
			// printf("** : %d :**",hullcount);
			CvConvexityDefect* defectArray;
			// int j=0;
			for (int i = 1; i <= hullcount; i++)
			{
				CvPoint pt = **CV_GET_SEQ_ELEM(CvPoint*, hull, i);
				cvLine(src, pt0, pt, CV_RGB(255, 0, 0), 1, CV_AA, 0);
				pt0 = pt;
			}
			for (; defects; defects = defects->h_next)
			{
				int nomdef = defects->total; // defect amount
				// outlet_float( m_nomdef, nomdef );
				// printf(" defect no %d \n",nomdef);
				if (nomdef == 0)
					continue;
				// Alloc memory for defect set.
				// fprintf(stderr,"malloc\n");
				defectArray = (CvConvexityDefect*)malloc(sizeof(CvConvexityDefect)*nomdef);
				// Get defect set.
				// fprintf(stderr,"cvCvtSeqToArray\n");
				cvCvtSeqToArray(defects, defectArray, CV_WHOLE_SEQ);
				// Draw marks for all defects.
				int con = 0;
				for (int i = 0; i < nomdef; i++)
				{
					if (defectArray[i].depth > 40)
					{
						con = con + 1;
						// printf(" defect depth for defect %d %f \n",i,defectArray[i].depth);
						cvLine(src, *(defectArray[i].start), *(defectArray[i].depth_point), CV_RGB(255, 255, 0), 1, CV_AA, 0);
						cvCircle(src, *(defectArray[i].depth_point), 5, CV_RGB(0, 0, 255), 2, 8, 0);
						cvCircle(src, *(defectArray[i].start), 5, CV_RGB(0, 255, 0), 2, 8, 0);
						cvLine(src, *(defectArray[i].depth_point), *(defectArray[i].end), CV_RGB(0, 255, 255), 1, CV_AA, 0);
						cvDrawContours(src, defects, CV_RGB(0, 0, 0), CV_RGB(255, 0, 0), -1, CV_FILLED, 8);
					}
				}
				// cout<<con<<"\n";
				/*char txt[40] = "";
				if (con == 1)
				{
					char txt1[] = "Hi , This is Udit";
					strcat(txt, txt1);
				}
				else if (con == 2)
				{
					char txt1[] = "3 Musketeers";
					strcat(txt, txt1);
				}
				else if (con == 3)
				{
					char txt1[] = "Fanatastic 4";
					strcat(txt, txt1);
				}
				else if (con == 4)
				{
					char txt1[] = "It's 5";
					strcat(txt, txt1);
				}
				else
				{
					char txt1[] = "Jarvis is busy :P"; // Jarvis can't recognise you
					strcat(txt, txt1);
				}*/
				cvNamedWindow("contour", 1); cvShowImage("contour", src);
				//cvResetImageROI(src);
				/*CvFont font;
				cvInitFont(&font, CV_FONT_HERSHEY_SIMPLEX, 1.5, 1.5, 0, 5, CV_AA);
				cvPutText(src, txt, cvPoint(50, 50), &font, cvScalar(0, 0, 255, 0));*/
				// j++;  
				// Free memory.
				free(defectArray);
			}
			cvReleaseMemStorage(&storage1);
			cvReleaseMemStorage(&storage2);
			cvClearSeq(ptseq);
			cvClearSeq(hull);
			//cvClearSeq(defects);//エラーが出る
		}
	}
	cvReleaseMemStorage(&storage);
	cvNamedWindow("threshold", 1); cvShowImage("threshold", src);
	c = cvWaitKey(100);

	cvReleaseImage(&gray);
	//cvClearSeq(first_contour);
	//cvClearSeq(maxitem);//エラーが出る
	//cvClearSeq(ptr);//エラーが出る
	//cvReleaseImage(&src);//エラーが出る

	if (c == 27)
		return false;
	else
		return true;
}

int main()
{
	int c = 0;
	cv::VideoCapture capture(0);
	if (capture.isOpened())
	{
		cout << "Video camera capture status: OK" << endl;
	}
	else
	{
		cout << "Video capture failed, please check the camera." << endl;
		return -1;
	}


	//HandDetection detect;
	cv::Mat img;
	while (c != 27)
	{

		capture >> img;
		if (!handDetect(img))
			break;
		/*HandDetection *detect;
		detect = new HandDetection();
		detect->getTipData(img);

		detect.getTipData(img);
		c = cv::waitKey(10);*/
		//delete detect;
	}

	cv::destroyAllWindows();
	return 0;
}