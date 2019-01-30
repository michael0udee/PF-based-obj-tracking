#include "color_detection.h"
//#include <iostream>
void MultiBackProjectionColorDetector::setTemplateList(std::list<cv::Mat> frameList)
{
    for (auto frame : frameList)
	{
		cv::cvtColor(frame, frame, cv::COLOR_BGR2HSV); 
        hsvTemplates.push_back(frame);
	}
};

std::list<cv::Mat> MultiBackProjectionColorDetector::getTemplateList()
{
	std::list<cv::Mat> outputList;
	for (auto frame : hsvTemplates)
	{
		cv::cvtColor(frame, frame, cv::COLOR_HSV2BGR);
		outputList.push_back(frame);
	}
	return outputList;
};

cv::Mat MultiBackProjectionColorDetector::frameFiltered(cv::Mat frame, bool morph_opening, bool blur, int kernel_size, int iterations)
{
	if(hsvTemplates.size() == 0) CV_Error( cv::Error::StsBadArg, "templates are empty!" );
	// Get the mask from the internal function
    cv::Mat frameThreshold = frameMask(frame, morph_opening, blur, kernel_size, iterations);
    // Return the AND image
	cv::Mat andImage;
	cv::bitwise_and(frame, frameThreshold, andImage);
    return andImage; 
};

cv::Mat MultiBackProjectionColorDetector::frameMask(cv::Mat frame, bool morph_opening, bool blur, int kernel_size, int iterations)
{
	if(hsvTemplates.size() == 0) CV_Error( cv::Error::StsBadArg, "templates are empty!" );
	cv::Mat frameHSV;
	cv::cvtColor(frame, frameHSV, cv::COLOR_BGR2HSV);
	cv::Mat mask = cv::Mat::zeros(frame.rows, frame.cols, CV_8UC1);

	// Histogram parameters:
	int hbins = 180, sbins = 256; // 30, 32
    int histSize[] = {hbins, sbins};
    // hue varies from 0 to 179
    float hranges[] = { 0, 180 };
    // saturation varies from 0 (black-gray-white) to
    // 255 (pure spectrum color)
    float sranges[] = { 0, 256 };
    const float* ranges[] = { hranges, sranges };
    cv::MatND templateHist;
    // we compute the histogram from the 0-th and 1-st channels
    int channels[] = {0, 1};

	for (auto hsvTemplate : hsvTemplates)
	{
		// Set the template histogram
		cv::Mat templateHist;
		cv::calcHist(&hsvTemplate, 1, channels, cv::Mat(), // do not use mask
             templateHist, 2, histSize, ranges,
             true, // the histogram is uniform
             false );
	    // Normalize the template histogram and apply backprojection
		cv::Mat frameHSVbackProj;
        cv::normalize(templateHist, templateHist, 0, 255, cv::NORM_MINMAX);
		cv::calcBackProject(&frameHSV, 1, channels, templateHist, frameHSVbackProj, ranges, 1, true);
		// Get the kernel and apply a convolution
		cv::Mat frameHSVclean;
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(kernel_size, kernel_size));
        cv::filter2D(frameHSVbackProj, frameHSVclean, -1, kernel);
        // Applying the morph open operation (erosion followed by dilation)
        if(morph_opening==true)
		{
        	cv::Mat kernel = cv::Mat::ones(kernel_size, kernel_size, CV_8UC1);
        	cv::morphologyEx(frameHSVclean, frameHSVclean, cv::MORPH_OPEN, kernel, cv::Point(-1,-1), iterations);
		}
        // Applying Gaussian Blur
        if(blur==true)
            cv::GaussianBlur(frameHSVclean, frameHSVclean, cv::Size(kernel_size, kernel_size), 0, 0);
        // Get the threshold
		cv::Mat frameHSVthreshold;
		cv::threshold(frameHSVclean, frameHSVthreshold, 50, 255, cv::THRESH_BINARY);
		//std::cout << "Mask channels: " << mask.channels() << std::endl;
		//std::cout << "Threshold channels: " << frameHSVthreshold.channels() << std::endl;
		//std::cout << "Mask size: " << mask.rows <<"x"<< mask.cols << std::endl;
		//std::cout << "Threshold size: " << frameHSVthreshold.rows <<"x"<< frameHSVthreshold.cols << std::endl;
		
        mask += frameHSVthreshold; // Add the threshold to the mask
	}
    // Normalize the mask because it contains
  	// values added during the previous loop
    // Attention: here it is not necessary to normalize because the astype(np.uint8) method
    // will resize to 255 each value which is higher that that...
    // cv::normalize(mask, mask, 0, 255, cv::NORM_MINMAX) // Not necessary

	// mask.convertTo(mask, CV_8UC1);
    cv::threshold(mask, mask, 50, 255, cv::THRESH_BINARY);
	std::vector<cv::Mat> outputMask(3);
	cv::Mat outputMask2;
	for(int i=0; i<outputMask.size(); ++i) outputMask[i] = mask;
	cv::merge(outputMask, outputMask2); // !!!
	//std::cout << "Mask3 channels: " << outputMask2.channels() << std::endl;
    return outputMask2;
};
