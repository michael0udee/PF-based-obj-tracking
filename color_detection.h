#ifndef COLOR_DETECTION_H
#define COLOR_DETECTION_H 

#include <opencv2/opencv.hpp>
#include <vector>
/*!
* Implementation of the Histogram Backprojection algorithm with multi-template.
*
* This class is the reimplementation of the BackProjectionColorDetector class for
* multi-template color detection. Instead of specifing a single template it is 
* possible to pass a list of templates, which can be multiple subframe taken from
* different part of an object. Multiple version of the Backprojection algorithm
* are then run at the same time and the filtered output added togheter. The result
* of this process is much robust (but slower) than the standard class.
*
*/
class MultiBackProjectionColorDetector
{
public:
	/*!
	* Set the BGR image list used as container for the templates
 	*
    * The template can be a specific region of interest of the main
    * frame or a representative color scheme to identify. the template
    * is internally stored as an HSV image.
    * @param frameList the list of templates to use in the algorithm 
	*/
	void setTemplateList(std::list<cv::Mat> frameList);
	/*!
	* Get the BGR image list used as container for the templates
 	*
    * The template can be a specific region of interest of the main
    * frame or a representative color scheme to identify.
	*/
	std::list<cv::Mat> getTemplateList();
	/*!
	* Given an input frame in BGR return the filtered version.
	*
    * @param frame the original frame (color)
    * @param morph_opening it is a erosion followed by dilatation to remove noise
    * @param blur to smooth the image it is possible to apply Gaussian Blur
    * @param kernel_size is the kernel dimension used for morph and blur
	*/
	cv::Mat frameFiltered(cv::Mat frame, bool morph_opening=true, bool blur=true, int kernel_size=5, int iterations=1);
	/*!
	* Given an input frame in BGR return the black/white mask.
 	*
    * @param frame the original frame (color)
    * @param morph_opening it is a erosion followed by dilatation to remove noise
    * @param blur to smooth the image it is possible to apply Gaussian Blur
    * @param kernel_size is the kernel dimension used for morph and blur
	*/
	cv::Mat frameMask(cv::Mat frame, bool morph_opening=true, bool blur=true, int kernel_size=5, int iterations=1);

private:
	std::list<cv::Mat> hsvTemplates;

};

#endif
