#include "CohesionRule.h"
#include <glm/glm.hpp>

glm::vec2 CohesionRule::computeForce(const std::vector<BoidView>& neighborhood, const BoidView& boid) {
  glm::vec2 cohesionForce(0.f);

  // glm::length(vec) returns the length of a vector,
  // glm::normalize(vec) returns the normalized vector (length 1) in the same direction as vec.

  // begin solution
  if (neighborhood.empty())
    return glm::vec2(0.f);  // no neighbors: no center of mass, so no cohesion pull

  glm::vec2 centerOfMass = {0, 0};

  for (int i = 0; i < neighborhood.size(); i++) {
    centerOfMass += neighborhood[i].position;
  }

  centerOfMass /= neighborhood.size();

  cohesionForce = centerOfMass - boid.position;

  if (glm::length(cohesionForce) > 0.0F)
    cohesionForce = glm::normalize(cohesionForce);

  // end solution

  return cohesionForce;
}
