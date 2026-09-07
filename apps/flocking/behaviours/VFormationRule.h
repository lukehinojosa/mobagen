#ifndef VFORMATIONRULE_H
#define VFORMATIONRULE_H

#include "FlockingRule.h"

// Steering half of the V-formation behaviour. The organizing half (who leads, which arm
// each boid joins, where its slot is) lives in FlockingManager's formation-assignment
// pass, which writes each boid's role and target slot. This rule just turns that target
// into a steering force, and brakes a fatigued leader so it drops back.
class VFormationRule : public FlockingRule {
private:
  float slotAngle;      // angle (radians) of a wingtip slot, measured back from the parent's tail
  float slotDistance;   // how far behind the parent each wingtip slot sits
  float catchupBoost;   // extra speed (x cruise per slotDistance of slot error) a follower may use to close the gap
  float slotPull;       // how strongly slot position error is turned into a closing velocity (1/s)
  float staminaTime;    // seconds of stamina a leader has before it hands off the apex

public:
  // Shared neighbor ahead cone. The formation-assignment and energy passes use the
  // same threshold so leader/follower detection there matches this rule's.
  static constexpr float kAheadConeDot = 0.5f;

  // True if any active (not stepping-down) neighbor sits inside boid's forward cone. A boid with none
  // is at the front of its group (a candidate leader).
  static bool hasNeighborAhead(const std::vector<BoidView>& neighborhood, const BoidView& boid, float coneDot = kAheadConeDot);

  explicit VFormationRule(float weight = 1.f, float slotAngle_ = 0.7853982f /* 45 deg */, float slotDistance_ = 30.f,
                          bool isEnabled = true)
      : FlockingRule(Color::Green, weight, isEnabled),
        slotAngle(slotAngle_),
        slotDistance(slotDistance_),
        catchupBoost(1.5f),
        slotPull(4.f),
        staminaTime(15.f) {}

  VFormationRule(const VFormationRule& toCopy) : FlockingRule(toCopy) {
    slotAngle = toCopy.slotAngle;
    slotDistance = toCopy.slotDistance;
    catchupBoost = toCopy.catchupBoost;
    slotPull = toCopy.slotPull;
    staminaTime = toCopy.staminaTime;
  }

  std::unique_ptr<FlockingRule> clone() override { return std::make_unique<VFormationRule>(*this); }

  const char* getRuleName() override { return "V Formation"; }
  const char* getRuleExplanation() override {
    return "Boids classify themselves as leader, left wing or right wing and draft into a slot behind the "
           "same-wing boid ahead. Formation identity is passed neighbor to neighbor, so a boid only needs to "
           "see the boid ahead of it. Fatigued leaders drop back and let the next boid take the apex.";
  }
  float getBaseWeightMultiplier() override { return 1.f; }

  // Slot geometry, read by the manager's assignment pass so the target it computes matches
  // what this rule expects to steer toward.
  float getSlotAngle() const { return slotAngle; }
  float getSlotDistance() const { return slotDistance; }
  float getCatchupBoost() const { return catchupBoost; }
  float getStaminaTime() const { return staminaTime; }

  glm::vec2 computeForce(const std::vector<BoidView>& neighborhood, const BoidView& boid) override;
  bool drawImguiRuleExtra() override;
};

#endif
