#include "VFormationRule.h"
#include "imgui.h"
#include <glm/glm.hpp>
#include <cmath>

bool VFormationRule::hasNeighborAhead(const std::vector<BoidView>& neighborhood, const BoidView& boid, float coneDot) {
  glm::vec2 forward = boid.velocity;
  if (glm::length(forward) < 0.0001f) return false;
  forward = glm::normalize(forward);
  for (const auto& n : neighborhood) {
    if (n.steppingDown) continue;  // an outgoing leader is leaving; it no longer leads anyone
    glm::vec2 toN = n.position - boid.position;
    float dist = glm::length(toN);
    if (dist < 0.0001f) continue;
    if (glm::dot(forward, toN / dist) >= coneDot) return true;
  }
  return false;
}

glm::vec2 VFormationRule::computeForce(const std::vector<BoidView>& neighborhood, const BoidView& boid) {
  // glm::length / glm::normalize / glm::dot are used below for vector math.

  // begin solution
  glm::vec2 forward = boid.velocity;
  if (glm::length(forward) < 0.0001f)
    return glm::vec2(0.f); // no heading yet
  forward = glm::normalize(forward);

  // Followers steer toward the slot the assignment pass picked for them (this includes an
  // outgoing leader, which is given a rear slot). Leaders and unattached boids get no
  // formation force (other rules carry them).
  //
  // Velocity servo, not a bare position spring: we aim for the slot's own velocity plus a
  // correction toward the slot (slotPull * position error), then steer to reach that velocity.
  // At the slot the desired velocity is just the slot's velocity, so the boid cruises WITH the
  // formation (a relative stop) instead of overshooting and orbiting. Weight scales the servo.
  if ((boid.role == FormationRole::Left || boid.role == FormationRole::Right) && boid.hasTarget) {
    glm::vec2 desiredVel = boid.targetVel + (boid.target - boid.position) * slotPull;
    return desiredVel - boid.velocity;
  }

  return glm::vec2(0.f);
  // end solution
}

bool VFormationRule::drawImguiRuleExtra() {
  bool valueHasChanged = false;
  if (ImGui::SliderAngle("Slot Angle", &slotAngle, 0.f, 90.f)) valueHasChanged = true;
  if (ImGui::SliderFloat("Slot Distance", &slotDistance, 5.f, 120.f, "%.f")) valueHasChanged = true;
  if (ImGui::SliderFloat("Slot Pull", &slotPull, 0.5f, 20.f, "%.1f")) valueHasChanged = true;
  if (ImGui::SliderFloat("Catch-up Boost", &catchupBoost, 0.f, 4.f, "%.2f")) valueHasChanged = true;
  if (ImGui::SliderFloat("Stamina Time (s)", &staminaTime, 1.f, 60.f, "%.1f")) valueHasChanged = true;
  return valueHasChanged;
}
