#include "motion_tracking.h"
//#include <iostream>
ParticleFilter::ParticleFilter(int width, int height, int N)
{
	this->N = N;
	if(N <= 0 || N > (width*height))
		CV_Error( cv::Error::StsBadArg, "the ParticleFilter class does not accept a value of N which is <= 0 or > (widht*height)" );
	particles  = cv::Mat::zeros(2, N, CV_64FC1);
    cv::randn(particles.row(0), width/2, width/8); // init the X coord
    cv::randn(particles.row(1), height/2, height/8); // init the Y coord
	// Init the weights vector as a uniform distribution
    // at the begining each particle has the same probability
    // to represent the point we are following
	for(int i=0; i < N; ++i)
		weights.push_back(1.0/N);
};

void ParticleFilter::predict(int std)
{
	cv::Mat tempMatrix = cv::Mat::zeros(1, particles.cols, CV_64FC1); 
	//Adding some noises to State Matrix 
   	cv::randn(tempMatrix, 0, 1);
	//std::cout << "TempMatrix: " << tempMatrix << std::endl;
	particles.row(0) = particles.row(0) + tempMatrix*std;
	cv::randn(tempMatrix, 0, 1);
	particles.row(1) = particles.row(1) + tempMatrix*std;
};

void ParticleFilter::drawParticles(cv::Mat frame, int radius)
{
	cv::Point_<double> cur_point;
	for(int j = 0 ; j < particles.cols ; j++)
	{
		cur_point.x = particles.at<double>(0,j); 
		cur_point.y = particles.at<double>(1,j);  
	// cout << cur_point << endl ;
		cv::circle(frame, cur_point, radius, CV_RGB(255,0,0), -1) ; // RED Particles
	}

};

cv::Point ParticleFilter::estimate()
{
	estimatePoint.x = std::round(cv::sum(particles.row(0).mul(weights))[0]);
 	estimatePoint.y = std::round(cv::sum(particles.row(1).mul(weights))[0]);
	
    return estimatePoint;
};

void ParticleFilter::update(cv::Point center)
{
    // Generating a temporary array for the input position
	cv::Mat posMatrix = cv::Mat::ones(2, particles.cols, CV_64FC1); 
	cv::Mat xMat(1, particles.cols, CV_64FC1, cv::Scalar::all(center.x));
	cv::Mat yMat(1, particles.cols, CV_64FC1, cv::Scalar::all(center.y));

	posMatrix.row(0) = (posMatrix.row(0)).mul(xMat);
	posMatrix.row(1) = (posMatrix.row(1)).mul(yMat);

	// 1- We can take the difference between each particle new
    // position and the measurement. In this case is the Euclidean Distance.
	std::vector <double> distance;
	cv::Mat subMatrix;
	cv::subtract(particles, posMatrix, subMatrix); 
	//std::cout << " subMatrix: " << subMatrix << std::endl;
	double x,y;
	for(int i=0; i < N; ++i)
	{
		x = subMatrix.at<double>(0,i); 
		y = subMatrix.at<double>(1,i); 
		distance.push_back(sqrt(x*x+y*y));
	}
    // 2- Particles which are closer to the real position have smaller
    // Euclidean Distance, here we subtract the maximum distance in order
    // to get the opposite (particles close to the real position have
    // an higher wieght)
	double maxDistance = *std::max_element(distance.begin(), distance.end());
	
	for(int i=0; i < distance.size(); ++i)
		distance[i] = abs(maxDistance-distance[i]);
	// 3-Particles that best predict the measurement 
    // end up with the highest weight.
	sum = 0;
	for(int i=0; i < weights.size(); ++i)
	{
		weights[i] = 1.0;
		weights[i] *= distance[i];
		weights[i] += 1.e-300; // avoid zeros
		sum += weights[i];
	}
	// 4- after the multiplication the sum of the weights won't be 1. 
    // Renormalize by dividing all the weights by the sum of all the weights.
	for(int i=0; i < weights.size(); ++i)
		weights[i] /= sum;
};

void ParticleFilter::resample()
{
	// take int(N*weights) copies of each weight
	for(int i=0; i<N; ++i)
	{
		indices.push_back(0);
		cumulativeSum.push_back(0);
		tempWeights.push_back(0);
		numCopies.push_back(int(N*weights[i]));
	}
	int k = 0;
	for(int i=0; i<N; ++i)
		for (int j=0; j<numCopies[i]; ++i)
		{
			indices[k] = i;
			k++;
		}
	//std::cout << " Indices: ";
	//for(auto i : indices) std::cout << i << " ";	
	sum_fract = 0;
	diff_fract = 0;
	// multinomial resample
	for(int i=0; i<N; ++i)
	{
		// get fractional part
		diff_fract = weights[i] - numCopies[i];
		residual.push_back(diff_fract);
		sum_fract += diff_fract;
	}
	// normalize
	for(int i=0; i < residual.size(); ++i)
		residual[i] /= sum_fract;
	// cumulative sum
	acc = 0;
 	for(int i=0; i < residual.size(); i++)
	{
 		acc += residual[i];
 		cumulativeSum[i] = acc;
	}
	// ensures sum is exactly one
	cumulativeSum.back() = 1.0;
	// use current time as seed for random generator
	std::srand(std::time(nullptr));
    double random_variable;
	for(int i=k; i<N; ++i)
	{
		random_variable = (double) std::rand()/(RAND_MAX + 1.0);
		for(int j=0; j<N; ++j) 
			if(random_variable <= cumulativeSum[j])
			{
				indices[i] = j;
				break;
			}
	} 	
	//std::cout << " cumulativeSum: ";
	//for(auto i : cumulativeSum) std::cout << " " << i ;
	//std::cout << " Particles: " << particles << std::endl;

	// resample according to indices
	cv::Mat tempParticles = cv::Mat::zeros(2, N, CV_64FC1);
	for(int ind = 0 ; ind < N ; ++ind)
	{
		tempParticles.at<double>(0,ind) = particles.at<double>(0,indices[ind]); 
		tempParticles.at<double>(1,ind) = particles.at<double>(1,indices[ind]); 
		tempWeights[ind] = weights[indices[ind]];
	}
    // Create a new set of particles by randomly choosing particles 
    // from the current set according to their weights.
	particles = tempParticles;
	weights = tempWeights;
	// normalize the new set of particles
	sum_weights = 0;
	for(int i=0; i < N; ++i)
		sum_weights += weights[i];
	for(int i=0; i < N; ++i)
		weights[i] /= sum_weights;
	//std::cout << "Weights: " << weights.size() << std::endl;
	//std::cout << "Particles: " << particles.size() << std::endl;
	//std::cout << "WORK!!!!" << std::endl;
	residual.clear();
	numCopies.clear();
	cumulativeSum.clear();
	tempWeights.clear();
	indices.clear();
};
