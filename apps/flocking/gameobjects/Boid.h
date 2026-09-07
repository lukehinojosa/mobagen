#ifndef BOID_H
#define BOID_H

#include <glm/glm.hpp>
#include "math/ColorT.h"
#include "../behaviours/FlockingRule.h"
#include <cstdint>
#include <vector>

struct BoidPos {
  glm::vec2 pos{0.f};
};

struct BoidVel {
  glm::vec2 vel{0.f};
};

struct BoidAcc {
  glm::vec2 acc{0.f};
  glm::vec2 prevAcc{0.f};
};

struct BoidConfig {
  float detectionRadius = 100.f;
  float speed = 120.f;
  bool hasConstantSpeed = false;
  float maxAcceleration = 10.f;
};

struct BoidDebug {
  bool drawDebugRadius = false;
  bool drawDebugRules = false;
  bool drawAcceleration = false;
  Color32 color;
};

struct BoidForceCache {
  std::vector<glm::vec2> forces;
};

// Persistent per-boid V-formation state. Stamina (leaders drain it, drafters recover
// it) plus the boid's communicated role in the formation: which arm it belongs to, how
// deep along that arm it sits, which formation (the leader's entity index), and the slot
// it is currently steering toward.
struct BoidFormation {
  float energy = 1.f;
  bool steppingDown = false;
  FormationRole role = FormationRole::None;
  int depth = 0;
  std::uint32_t formationId = kNoFormation;
  glm::vec2 formationForward{0.f};
  glm::vec2 target{0.f};
  glm::vec2 targetVel{0.f};
  bool hasTarget = false;
};

#endif
