/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
	default_random_engine gen;
	normal_distribution<double> dist_x(x, std[0]);
	normal_distribution<double> dist_y(y, std[1]);
	normal_distribution<double> dist_theta(theta, std[2]);

	num_particles = 25;

	for(int i = 0; i < num_particles; i++) {
		Particle particle;
		particle.id = i;
		particle.x = dist_x(gen);
		particle.y = dist_y(gen);
		particle.theta = dist_theta(gen);
		particle.weight = 1.0;
		particles.push_back(particle);
	}
	is_initialized = true;

}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
	default_random_engine gen;;
	normal_distribution<double> dist_x(0, std_pos[0]);
  normal_distribution<double> dist_y(0, std_pos[1]);
  normal_distribution<double> dist_theta(0, std_pos[2]);

  for (int i = 0; i < num_particles; i++) {

    if (abs(yaw_rate) < 0.0001) {  
      particles[i].x += velocity * delta_t * cos(particles[i].theta);
      particles[i].y += velocity * delta_t * sin(particles[i].theta);
    } else {
      particles[i].x += velocity / yaw_rate * (sin(particles[i].theta + yaw_rate*delta_t) - sin(particles[i].theta));
      particles[i].y += velocity / yaw_rate * (cos(particles[i].theta) - cos(particles[i].theta + yaw_rate*delta_t));
      particles[i].theta += yaw_rate * delta_t;
    }

    particles[i].x += dist_x(gen);
    particles[i].y += dist_y(gen);
    particles[i].theta += dist_theta(gen);
  }

}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.
	for(int i = 0; i < observations.size(); i++) {
		double distance = dist(predicted[0].x, predicted[0].y, observations[i].x, observations[i].y);
		double tmp_dist;
		observations[i].id = predicted[0].id;

		for(int j = 0; j < predicted.size(); j++) {
			tmp_dist = dist(predicted[j].x, predicted[j].y, observations[i].x, observations[i].y);
			if(tmp_dist < distance) {
				observations[i].id = predicted[j].id;
				distance = tmp_dist;
			}
		}
	}

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html
	for(int i = 0; i < num_particles; i++) {
		double x_p = particles[i].x;
		double y_p = particles[i].y;
		double theta = particles[i].theta;

		std::vector<LandmarkObs> selected_landmarks;
		for(int k = 0; k < map_landmarks.landmark_list.size(); k++) {
			if((abs(map_landmarks.landmark_list[k].x_f - x_p) <= sensor_range) && (abs(map_landmarks.landmark_list[k].y_f - y_p) <= sensor_range)){
				LandmarkObs landmark_selected;
				landmark_selected.id = map_landmarks.landmark_list[k].id_i;
				landmark_selected.x = map_landmarks.landmark_list[k].x_f;
				landmark_selected.y = map_landmarks.landmark_list[k].y_f;
				selected_landmarks.push_back(landmark_selected);
			}
		}

		std::vector<LandmarkObs> transformed_obs;
		for(int j = 0; j < observations.size(); j++) {
			LandmarkObs tran_obs;
			double x_m = x_p + (cos(theta) * observations[j].x) - (sin(theta) * observations[j].y);
			double y_m = y_p + (sin(theta) * observations[j].x) + (cos(theta) * observations[j].y);
			tran_obs.x = x_m;
			tran_obs.y = y_m;
			transformed_obs.push_back(tran_obs);
		}

		dataAssociation(selected_landmarks, transformed_obs);

		double new_w = 1.0;
		double sig_x = std_landmark[0];
		double sig_y = std_landmark[1];
		for(int l = 0; l < transformed_obs.size(); l++) {
			LandmarkObs tran_obs = transformed_obs[l];
			double x = tran_obs.x;
			double y = tran_obs.y;

			double u_x;
			double u_y;

			for (int m = 0; m < selected_landmarks.size(); m++) {
        if (selected_landmarks[m].id == tran_obs.id) {
          u_x = selected_landmarks[m].x;
          u_y = selected_landmarks[m].y;
        }
      }

			double gauss_norm= (1.0/(2.0 * M_PI * sig_x * sig_y));

			double delta_x = (x - u_x);
			double delta_y = (y - u_y);

			double delta_x_sq = pow(delta_x, 2);
			double delta_y_sq = pow(delta_y, 2);

			double std_x_sq = 2.0 * pow(sig_x, 2);
			double std_y_sq = 2.0 * pow(sig_y, 2);

			double exponent = (delta_x_sq / std_x_sq) + (delta_y_sq / std_y_sq);

			double weight = gauss_norm * exp(-exponent);

			new_w *= weight;
		}
		particles[i].weight = new_w;
	}

}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
	
	default_random_engine gen;
	vector<Particle> new_particles;

  vector<double> weights;
  for (int i = 0; i < num_particles; i++) {
    weights.push_back(particles[i].weight);
  }

  uniform_int_distribution<int> uniintdist(0, num_particles-1);
  auto index = uniintdist(gen);

  double max_weight = *max_element(weights.begin(), weights.end());

  uniform_real_distribution<double> unirealdist(0.0, max_weight);

  double beta = 0.0;

  for (int i = 0; i < num_particles; i++) {
    beta += unirealdist(gen) * 2.0;
    while (beta > weights[index]) {
      beta -= weights[index];
      index = (index + 1) % num_particles;
    }
    new_particles.push_back(particles[index]);
  }

  particles = new_particles;

}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations, 
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
