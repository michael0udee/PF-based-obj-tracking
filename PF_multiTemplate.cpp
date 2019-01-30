
#include <iostream>
#include <opencv2/opencv.hpp>
//#include <opencv2/video/video.hpp>
//#include <opencv2/imgproc/imgproc.hpp>
//#include <opencv2/highgui/highgui.hpp>
//#include <opencv2/core/core.hpp>
#include "color_detection.h" // MultiBackProjectionColorDetector
#include "mask_analysis.h" // BinaryMaskAnalyser
#include "motion_tracking.h" // ParticleFilter

using namespace cv; 
using namespace std; 

bool outOfFrame = false;

string intToString(int number)
{
	std::stringstream ss;
	ss << number;
	return ss.str();
}

void drawObject(int x, int y, int w, int h, Mat &frame)
{

	//some openCV drawing functions to draw crosshairs
	//on tracked image


	//'if' and 'else' statements to prevent
	//memory errors from writing off the screen (ie. (-25,-25) is not within the window)
	//char k = (char)waitKey(5);
	//if( k == 'd' ) detected = true;
	Rect out(Point(frame.cols/10, frame.rows/10), Point(9*frame.cols/10, 9*frame.rows/10));
	if(out.contains(Point(x,y)))
	{
		outOfFrame = false;
		circle(frame, Point(x, y), 15, Scalar(0, 255, 0), 2);
		if (y - 25 > 0)
			line(frame, Point(x, y), Point(x, y - 25), Scalar(0, 255, 0), 2);
		else line(frame, Point(x, y), Point(x, 0), Scalar(0, 255, 0), 2);
		if (y + 25 < h)
			line(frame, Point(x, y), Point(x, y + 25), Scalar(0, 255, 0), 2);
		else line(frame, Point(x, y), Point(x, h), Scalar(0, 255, 0), 2);
		if (x - 25 > 0)
			line(frame, Point(x, y), Point(x - 25, y), Scalar(0, 255, 0), 2);
		else line(frame, Point(x, y), Point(0, y), Scalar(0, 255, 0), 2);
		if (x + 25 < w)
			line(frame, Point(x, y), Point(x + 25, y), Scalar(0, 255, 0), 2);
		else line(frame, Point(x, y), Point(w, y), Scalar(0, 255, 0), 2);
	}
	else outOfFrame = true;


	//putText(frame, intToString(x) + "," + intToString(y), Point(x, y + 30), 1, 1, Scalar(0, 255, 0), 2);
	//Mat centralRows = frame.rowRange(frame.rows/2 - 1, frame.rows/2 + 1);
	//Mat centralColumns = frame.colRange(frame.cols/2 - 1, frame.cols/2 + 1);
	//centralRows = Scalar(200,0,0);
	//centralColumns = Scalar(200,0,0);


		Rect r(Point(frame.cols/5, frame.rows/5), Point(4*frame.cols/5, 4*frame.rows/5));
		rectangle(frame, r, CV_RGB(255, 255, 255));
		if(r.contains(Point(x, y)))
		{
			rectangle(frame, r, CV_RGB(255, 255, 0), 3);
			putText(frame, "DETECTED", Point(frame.cols/5, frame.rows/8), 1, 1, CV_RGB(0, 255, 0), 2);
			std::cout << "DETECTED!" << std::endl;
		}
		else putText(frame, "DETECTED", Point(frame.cols/5, frame.rows/8), 1, 1, CV_RGB(155, 155, 155), 2);

		if(x < frame.cols/5) line(frame, r.tl(), Point(frame.cols/5, 4*frame.rows/5), CV_RGB(255, 255, 0), 3);
		if(x > 4*frame.cols/5) line(frame, Point(4*frame.cols/5, frame.rows/5), r.br(), CV_RGB(255, 255, 0), 3);
		if(y < frame.rows/5) line(frame, r.tl(), Point(4*frame.cols/5, frame.rows/5), CV_RGB(255, 255, 0), 3);
		if(y > 4*frame.rows/5) line(frame, Point(frame.cols/5, 4*frame.rows/5), r.br(), CV_RGB(255, 255, 0), 3);
		//line(frame, Point(frame.cols/2, (frame.rows/2)+150), Point(frame.cols/2, (frame.rows/2)+170), Scalar(255, 255, 255), 2);
		//line(frame, Point((frame.cols/5), (frame.rows/2)+150), Point(frame.cols/5, (frame.rows/2)+170), Scalar(255, 255, 255), 2);
		//line(frame, Point((4*frame.cols/5), (frame.rows/2)+150), Point(4*frame.cols/5, (frame.rows/2)+170), Scalar(255, 255, 255), 2);
		//drawMarker(frame, Point(x, frame.rows/2+160),  Scalar(255, 0, 255), MARKER_TILTED_CROSS, 20, 2);
	
}

int main(int argc, char** argv)
{
	// Load template
	vector<String> filenames; 
    String folder = "./templates";
	glob(folder, filenames);

	Mat templateImage;
	list<Mat> frameList;
    for(int i=0 ; i < filenames.size() ; i++) 
	{
    	cout << i << ": " << filenames[i] << endl;
		templateImage = imread(filenames[i]);
		// Check for invalid input
		if(templateImage.empty())
		{
		    cerr <<  "Could not open or find the template image" << endl;
		    return -1;
		}
		frameList.push_back(templateImage);
	}


	VideoCapture cap;
	cap.open(argv[1]); // Open the video

    if(!cap.isOpened())  // Check if succeeded
	{
		cerr <<  "Could not open webcam or video file" << endl;
        return -1;
	}

	// Create VideoWriter object
   	double fps =  cap.get(CV_CAP_PROP_FPS);
   	int frameWidth = cap.get(CV_CAP_PROP_FRAME_WIDTH);
   	int frameHeight = cap.get(CV_CAP_PROP_FRAME_HEIGHT);
   	VideoWriter video("out.avi", CV_FOURCC('M','J','P','G'), fps, Size(frameWidth, frameHeight), true); // or XVID, X264
	
	// Declaring the binary mask analyser object
	BinaryMaskAnalyser myMaskAnalyser;

	// Defining the color detector object
	MultiBackProjectionColorDetector myBackDetector;
	myBackDetector.setTemplateList(frameList); // Set the list of templates

	// Filter parameters
	int totalParticles = 1000; //3000
	// Standard deviation which represent how to spread the particles
	// in the prediction phase.
	int std = 15;  //25
	ParticleFilter myParticle(frameWidth, frameHeight, totalParticles);
	Point frameCenterPoint = Point(frameWidth/2, frameHeight/2);
	Point estimatedPoint;
	Point center;
	short dir=1;

	while(true)
	{
		
		Mat frame, frameMask;
	   	cap >> frame;
		// Return the binary mask from the backprojection algorithm
    	frameMask = myBackDetector.frameMask(frame, true, true, 5, 2);
		//cout << "Number of contours: " << myMaskAnalyser.numberOfContours(frame) << endl;

		if(myMaskAnalyser.numberOfContours(frameMask) > 0)	
		{
			// Use the binary mask to find the contour with largest area
        	// and the center of this contour which is the point we
        	// want to track with the particle filter
        	// x_rect,y_rect,w_rect,h_rect
			Rect maxAreaRect = myMaskAnalyser.maxAreaRectangle(frameMask);
        	center = myMaskAnalyser.maxAreaCenter(frameMask);
			// x_center, y_center
			rectangle(frame, maxAreaRect, Scalar(255,0,0)); // BLUE rect
		}	

		//----------PARTICLE FILTER------------
		// Predict the position of the target
		myParticle.predict(std);

		// Drawing the particles.
		if(!outOfFrame) myParticle.drawParticles(frame);

		// Estimate the next position using the internal model
		estimatedPoint = myParticle.estimate();
		//cv::circle(frame, estimatedPoint, 3, cv::Scalar(0,255,0), 5); // GREEN dot
		drawObject(estimatedPoint.x, estimatedPoint.y, frameWidth, frameHeight, frame);
		// Update the filter with the last measurements
		myParticle.update(center);

		// Resample the particles
		myParticle.resample();
		//-------------------------------------
		//----------CONTROL--------------------
		
		//if(estimatedPoint.x < frameWidth/2) dir = -1; else dir = 1;
		//line(frame, frameCenterPoint, estimatedPoint, Scalar(255,50,50), 2, LINE_AA, 0);
		//cv::arrowedLine(frame, Point(frameWidth/2, frameHeight/2), 
		//				Point(frameWidth/2 + dir*norm(frameCenterPoint - estimatedPoint), frameHeight/2), 
		//				Scalar(255,255,255), 3, LINE_AA);
		
		// (...)
		//-------------------------------------
		// write video to file
	   	video.write(frame);
		// show on window
		//cout << "Channels: " << frameMask.channels() << endl;
    	cv::imshow("Original", frame);
    	cv::imshow("Mask", frameMask);
		// Exit when Esc is pressed
	   	char c = (char)waitKey(20);
	   	if( c == 27 ) break;
	}

	destroyAllWindows();
	cout << "Bye..." << endl;
	return 0;
}
