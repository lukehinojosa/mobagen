#ifndef WORLD_H
#define WORLD_H

#include "../behaviours/FlockingRule.h"
#include "Boid.h"
#include "ecs/world.hpp"
#include "jobs/scheduler.hpp"

#include <memory>
#include <vector>

class VFormationRule;

class FlockingManager {
private:
  ecs::World& ecs_;
  jobs::Scheduler& sched_;

  int nbBoids = 100;
  bool hasConstantSpeed = false;
  float desiredSpeed = 120.0f;
  bool hasMaxAcceleration = false;
  float maxAcceleration = 10.0f;
  float detectionRadius = 250.f;

  bool showRadius = false;
  bool showRules = false;
  bool showAcceleration = false;

  // V-formation stamina model (rates are per second, energy is clamped to [0,1]).
  float leaderDrainRate = 0.06f; // a leader (max drag) loses energy this fast
  float draftRecoverRate = 0.04f; // a drafting follower recovers this fast
  float stepDownThreshold = 0.15f; // leader energy this low so hand off and head to a rear slot

  std::vector<std::unique_ptr<FlockingRule>> boidsRules;
  std::vector<float> defaultWeights;
  std::vector<ecs::Entity> boidEntities;
  VFormationRule* vRule_ = nullptr;  // non-owning; points into boidsRules for slot params + enabled state

  void initializeRules();
  void setNumberOfBoids(int number);
  ecs::Entity createBoid();
  void randomizeBoidPosVel(ecs::Entity e);
  void warpIfOutOfBounds(BoidPos& p);
  void updateBoidEnergy(ecs::Entity e, const std::vector<BoidView>& neighborhood, const BoidView& self, float deltaTime);
  void updateBoidFormation(ecs::Entity e, const std::vector<BoidView>& neighborhood, const BoidView& self);

  void drawGeneralUI();
  void drawRulesUI();
  void drawPerformanceUI(float deltaTime);
  void showConfigurationWindow(float deltaTime);

public:
  explicit FlockingManager(ecs::World& world, jobs::Scheduler& sched);

  void Start();
  void Update(float deltaTime);
  void OnGui();
  void OnDraw();
};

#endif
