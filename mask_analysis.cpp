#include "mask_analysis.h"

void BinaryMaskAnalyser::getContoursFromMask(cv::Mat &mask)
{
	// check if mask is empty
	if( mask.empty() )
    	CV_Error( cv::Error::StsBadArg, "mask is empty!" );
	// convert to gray image if coloured
	if( mask.type() != 0)
		cv::cvtColor( mask, mask, cv::COLOR_BGR2GRAY );
	// get contours
	cv::findContours(mask, contours, CV_RETR_LIST,
                         CV_CHAIN_APPROX_SIMPLE);
};

std::vector<cv::Point> BinaryMaskAnalyser::getMaxAreaElementContour()
{
	for (auto cnt : contours) 
        areaArray.push_back(cv::contourArea(cnt));    

	if(areaArray.size()==0)
		std::cout << "No contours!" << std::endl; // the array is empty
	// return the index of the max element
	int maxElementIndex = std::distance(areaArray.begin(), std::max_element(areaArray.begin(), areaArray.end()));
    auto cnt = contours[maxElementIndex];
	return cnt;
};

int BinaryMaskAnalyser::numberOfContours(cv::Mat &mask)
{
	getContoursFromMask(mask);
	return contours.size();
};

cv::Rect BinaryMaskAnalyser::maxAreaRectangle(cv::Mat &mask)
{
	getContoursFromMask(mask);

    cv::Rect outputRect = cv::boundingRect(getMaxAreaElementContour());
    return outputRect;
};

cv::Point BinaryMaskAnalyser::maxAreaCenter(cv::Mat &mask)
{
	getContoursFromMask(mask);

	// calculate the moments
    cv::Moments M = cv::moments(getMaxAreaElementContour());
    if(M.m00 == 0)
		std::cout << "No moment!" << std::endl;
	// get the center from the moments
	cv::Point c;
	c.x = (int)( M.m10 / M.m00 );
	c.y = (int)( M.m01 / M.m00 );
	// return the center coords
    return c;
};
