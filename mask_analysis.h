#ifndef MASK_ANALYSIS_H
#define MASK_ANALYSIS_H 

#include <opencv2/opencv.hpp>
#include <iostream>
#include <iterator>
#include <vector>
/*!
* This class analyses binary masks, like the ones returned by
* the color detection classes.
*
* The class implements function for finding the contour with the
* largest area and its properties (centre, sorrounding rectangle).
*
*/
class BinaryMaskAnalyser
{
public:
	/*!
	* it gets list of contours from the mask
 	* 
    * @param mask the binary image to use in the function
	*/
	void getContoursFromMask(cv::Mat &mask);
	/*!
	* it gets contour of element with max area
 	* 
    * @return get the desired contour
	*/
	std::vector<cv::Point> getMaxAreaElementContour();
	/*!
	* it returns the total number of contours present on the mask
 	* 
    * this method must be used during video analysis to check if the frame contains
    * at least one contour before calling the other function below.
    * @param mask the binary image to use in the function
	* @return get the number of contours 
	*/
	int numberOfContours(cv::Mat &mask);
	/*!
	* it returns the rectangle sorrounding the contour with the largest area.
	*
    * This method could be useful to find a face when a skin detector filter is used.
    * @param mask the binary image to use in the function
    * @return get the coords of the upper corner of the rectangle (x, y) and the rectangle size (widht, hight)
    * In case of error it returns a tuple (None, None, None, None) 
	*/
	cv::Rect maxAreaRectangle(cv::Mat &mask);
	/*!
	* it returns the centre of the contour with largest area.
 	*
    * This method could be useful to find the center of a face when a skin detector filter is used.
    * @param mask the binary image to use in the function
    * @return get the x and y center coords of the contour whit the largest area.
    * In case of error it returns a tuple (None, None)
	*/
	cv::Point maxAreaCenter(cv::Mat &mask);

private:
	std::vector<std::vector<cv::Point> > contours;
	std::vector<double> areaArray; // contains the area of the contours

};

#endif
