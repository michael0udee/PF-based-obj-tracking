#ifndef MOTION_TRACKING_H
#define MOTION_TRACKING_H 

#include <opencv2/opencv.hpp>
#include <unistd.h>
#include <cstdlib>
#include <vector>
#include <cmath>
#include <ctime>
/*!
* Particle filter motion tracking.
*
* This class estimates the position of a single point
* in a image. It can be used to predict the position of a
* landmark for example when tracking the corner of a bounding box.
*/
class ParticleFilter
{
public:
	/*!
    * @param width the width of the frame
    * @param height the height of the frame
    * @param N the number of particles
	*/	
	ParticleFilter(int width, int height, int N);
	/*!
    * Predict the position of the point in the next frame.
	*
    * Move the particles based on how the real system is predicted to behave.
    * @param std the standard deviation of the gaussian distribution used to add noise
	*/	
	void predict(int std);
	/*!
	* Draw the particles on a frame
	*
	* @param frame the image to draw
	* @param radius is the radius of the particles
	*/
	void drawParticles(cv::Mat frame, int radius=2);
	/*!
	* Estimate the position of the point given the particle weights.
	*
    * Using the weighted average of the particles
    * gives an estimation of the position of the point
	* @return estimated point
	*/
	cv::Point estimate();
	/*!
	* Update the weights associated which each particle based on the (x,y) coords measured.
    * Particles that closely match the measurements give an higher contribution.
	*
	* The position of the point at the next time step is predicted using the 
    * estimated speed along X and Y axis and adding Gaussian noise sampled 
    * from a distribution with MEAN=0.0 and STD=std. It is a linear model.
	* @param center position of the center point
	*/
	void update(cv::Point center);
	/*!
	* The resample function removes useless particles and keep the
    * useful ones. It is not necessary to resample at every loop.
    * If there are not new measurements then there is not any information 
    * from which the resample can benefit.
	*/
	void resample();

private:
	cv::Mat particles;
	std::vector<double> weights, tempWeights, residual, cumulativeSum;
	std::vector<int> indices, numCopies;
	cv::Point estimatePoint;
	double sum, acc, sum_fract, sum_weights, diff_fract;
	int N;

};

#endif
