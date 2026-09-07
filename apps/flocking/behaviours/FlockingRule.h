#ifndef FLOCKINGRULE_H
#define FLOCKINGRULE_H

#include <cstdint>
#include <memory>
#include <vector>
#include <glm/glm.hpp>
#include "math/ColorT.h"
#include "imgui.h"

// How a boid classifies itself within a V formation. A Left boid never belongs to
// the Right arm and vice versa; wing membership is inherited down the chain.
enum class FormationRole : std::uint8_t { None, Leader, Left, Right };

constexpr std::uint32_t kNoFormation = 0xFFFFFFFFu;

struct BoidView {
  glm::vec2 position{0.f};
  glm::vec2 velocity{0.f};
  std::uint32_t id{0};                     // stable entity index; used to break leadership ties
  float energy{1.f};                       // stamina in [0,1]; leaders drain it, drafters recover it
  bool steppingDown{false};                // true while an outgoing leader is heading to a rear slot
  FormationRole role{FormationRole::None}; // self-classification within the formation
  int depth{0};                            // 0 = leader, increases down each arm
  std::uint32_t formationId{kNoFormation}; // leader's entity index; propagated to followers
  glm::vec2 formationForward{0.f};   // leader's heading, propagated down the chain; anchors slot offsets
  glm::vec2 target{0.f};             // slot this boid is steering toward (followers only)
  glm::vec2 targetVel{0.f};          // velocity the slot is moving at (the formation's cruise velocity)
  bool hasTarget{false};
};

class FlockingRule {
protected:
  Color32 debugColor;

  explicit FlockingRule(Color32 debugColor_, float weight_, bool isEnabled_ = true)
      : debugColor(debugColor_), weight(weight_), isEnabled(isEnabled_) {}

  virtual glm::vec2 computeForce(const std::vector<BoidView>& neighborhood, const BoidView& boid) = 0;

  virtual float getBaseWeightMultiplier() { return 1.f; }

  virtual const char* getRuleName() = 0;
  virtual const char* getRuleExplanation() = 0;
  virtual bool drawImguiRuleExtra() { return false; }

public:
  float weight;
  bool isEnabled;

  FlockingRule(const FlockingRule& toCopy);
  virtual ~FlockingRule() = default;

  virtual std::unique_ptr<FlockingRule> clone() = 0;

  glm::vec2 computeWeightedForce(const std::vector<BoidView>& neighborhood, const BoidView& boid);

  virtual bool drawImguiRule();

  virtual void draw(const BoidView& boid, ImDrawList* dl, glm::vec2 cachedForce) const;

  virtual void drawWorldOverlay(ImDrawList* dl) const {}
};

#endif
