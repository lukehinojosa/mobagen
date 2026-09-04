#include "MouseInfluenceRule.h"
#include "imgui.h"

glm::vec2 MouseInfluenceRule::computeForce(const std::vector<BoidView>& neighborhood, const BoidView& boid) {
  glm::vec2 force(0.f);

  // ImGui::IsMouseDown(ImGuiMouseButton_Left) returns true if the left mouse button is currently pressed.
  // ImGui::GetIO().MousePos returns the current mouse position as an ImVec2.
  // glm::length(vec) returns the length of a vector

  // begin solution
  if (ImGui::IsMouseDown(ImGuiMouseButton_Left)) {
    if (isRepulsive) {
      glm::vec2 distToMouse = boid.position - glm::vec2{ImGui::GetIO().MousePos.x, ImGui::GetIO().MousePos.y};
      float distance = glm::length(distToMouse);

      if (distance > 0.0001f) {
        glm::vec2 repelDirection = glm::normalize(distToMouse);

        force += repelDirection * (1 / distance);
      }
    }
    else {
      glm::vec2 distToMouse = glm::vec2{ImGui::GetIO().MousePos.x, ImGui::GetIO().MousePos.y} - boid.position;
      float distance = glm::length(distToMouse);

      if (distance > 0.0001f) {
        glm::vec2 attractDirecton = glm::normalize(distToMouse);

        force += attractDirecton * (1 / distance);
      }
    }
  }

  // end solution

  return force;
}

bool MouseInfluenceRule::drawImguiRuleExtra() {
  bool valueHasChanged = false;

  if (ImGui::RadioButton("Attractive", !isRepulsive)) {
    isRepulsive = false;
    valueHasChanged = true;
  }
  ImGui::SameLine();
  if (ImGui::RadioButton("Repulsive", isRepulsive)) {
    isRepulsive = true;
    valueHasChanged = true;
  }

  return valueHasChanged;
}
