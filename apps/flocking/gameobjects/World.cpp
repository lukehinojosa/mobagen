#include "World.h"
#include "imgui.h"
#include "../utils/ImGuiExtra.h"
#include "Random.h"

#include "../behaviours/SeparationRule.h"
#include "../behaviours/CohesionRule.h"
#include "../behaviours/AlignmentRule.h"
#include "../behaviours/MouseInfluenceRule.h"
#include "../behaviours/BoundedAreaRule.h"
#include "../behaviours/WindRule.h"
#include "../behaviours/VFormationRule.h"

#include <glm/glm.hpp>
#include <algorithm>
#include <cmath>

#if defined(_WIN32)
#  include "Windows.h"
#  include "Psapi.h"
#endif

FlockingManager::FlockingManager(ecs::World& world, jobs::Scheduler& sched) : ecs_(world), sched_(sched) {}

void FlockingManager::initializeRules() {
  boidsRules.emplace_back(std::make_unique<SeparationRule>(25.f, 90.f));
  boidsRules.emplace_back(std::make_unique<CohesionRule>(300.f));
  boidsRules.emplace_back(std::make_unique<AlignmentRule>(2.9f));
  boidsRules.emplace_back(std::make_unique<MouseInfluenceRule>(1000000.f));
  boidsRules.emplace_back(std::make_unique<BoundedAreaRule>(100, 200.f, false));
  boidsRules.emplace_back(std::make_unique<WindRule>(1.f, 6.f, false));
  boidsRules.emplace_back(std::make_unique<VFormationRule>(40.f, 0.7853982f, 30.f, false));

  defaultWeights.clear();
  for (const auto& rule : boidsRules) defaultWeights.push_back(rule->weight);

  vRule_ = nullptr;
  for (const auto& rule : boidsRules)
    if (auto* v = dynamic_cast<VFormationRule*>(rule.get())) vRule_ = v;

  SetupImGuiStyle();
}

void FlockingManager::randomizeBoidPosVel(ecs::Entity e) {
  ImVec2 displaySize = ImGui::GetIO().DisplaySize;
  float w = displaySize.x > 0.f ? displaySize.x : 1280.f;
  float h = displaySize.y > 0.f ? displaySize.y : 800.f;

  ecs_.get<BoidPos>(e).pos.x = Random::Range(0.f, w);
  ecs_.get<BoidPos>(e).pos.y = Random::Range(0.f, h);
  float angle = Random::Range(0.f, 6.28318530718f);
  ecs_.get<BoidVel>(e).vel = glm::vec2(std::cos(angle), std::sin(angle)) * desiredSpeed;
}

ecs::Entity FlockingManager::createBoid() {
  ecs::Entity e = ecs_.create();
  ecs_.add<BoidPos>(e);
  ecs_.add<BoidVel>(e);
  ecs_.add<BoidAcc>(e);
  ecs_.add<BoidForceCache>(e);
  ecs_.add<BoidFormation>(e);

  BoidConfig& cfg = ecs_.add<BoidConfig>(e);
  cfg.detectionRadius = detectionRadius;
  cfg.speed = desiredSpeed;
  cfg.hasConstantSpeed = hasConstantSpeed;
  cfg.maxAcceleration = hasMaxAcceleration ? maxAcceleration : 10000.f;

  BoidDebug& dbg = ecs_.add<BoidDebug>(e);
  dbg.drawDebugRadius = showRadius;
  dbg.drawDebugRules = showRules;
  dbg.drawAcceleration = showAcceleration;
  dbg.color = Color32::RandomColor(31, 255);

  randomizeBoidPosVel(e);
  return e;
}

void FlockingManager::setNumberOfBoids(int number) {
  int diff = static_cast<int>(boidEntities.size()) - number;
  if (diff == 0) return;

  if (diff < 0) {
    for (int i = 0; i < -diff; i++) boidEntities.push_back(createBoid());
  } else {
    for (int i = 0; i < diff; i++) {
      ecs_.destroy(boidEntities.back());
      boidEntities.pop_back();
    }
  }
}

void FlockingManager::warpIfOutOfBounds(BoidPos& p) {
  ImVec2 displaySize = ImGui::GetIO().DisplaySize;
  float w = displaySize.x > 0.f ? displaySize.x : 1280.f;
  float h = displaySize.y > 0.f ? displaySize.y : 800.f;

  if (p.pos.x < 0.f)
    p.pos.x += w;
  else if (p.pos.x > w)
    p.pos.x -= w;
  if (p.pos.y < 0.f)
    p.pos.y += h;
  else if (p.pos.y > h)
    p.pos.y -= h;
}

void FlockingManager::updateBoidEnergy(ecs::Entity e, const std::vector<BoidView>& /*neighborhood*/, const BoidView& /*self*/, float deltaTime) {
  BoidFormation& fm = ecs_.get<BoidFormation>(e);

  // A boid at the apex drains while it leads; everyone else (followers, and an outgoing leader
  // on its way to a rear slot) recovers. When a leader runs low it flags steppingDown, which
  // triggers the hand-off in the formation pass. Recovery is silent (no rest/brake phase).
  const bool leadingNow = fm.role == FormationRole::Leader && !fm.steppingDown;
  if (leadingNow) {
    // Drain a full tank over the rule's stamina time (energy 1 -> 0 across staminaTime seconds);
    // the leader hands off once it dips below the step-down threshold.
    float staminaTime = vRule_ != nullptr ? vRule_->getStaminaTime() : 15.f;
    float drainRate = staminaTime > 0.01f ? 1.f / staminaTime : leaderDrainRate;
    fm.energy -= drainRate * deltaTime;
    if (fm.energy <= stepDownThreshold) fm.steppingDown = true;
  } else {
    fm.energy += draftRecoverRate * deltaTime;
  }

  if (fm.energy < 0.f) fm.energy = 0.f;
  if (fm.energy > 1.f) fm.energy = 1.f;
}

void FlockingManager::updateBoidFormation(ecs::Entity e, const std::vector<BoidView>& neighborhood, const BoidView& self) {
  BoidFormation& fm = ecs_.get<BoidFormation>(e);

  // No formation work when the rule is off, or when this boid is drifting.
  const float speed = glm::length(self.velocity);
  if (vRule_ == nullptr || !vRule_->isEnabled || speed < 0.0001f) {
    fm.role = FormationRole::None;
    fm.steppingDown = false;
    fm.hasTarget = false;
    return;
  }
  const glm::vec2 forward = self.velocity / speed;
  const float cone = VFormationRule::kAheadConeDot;

  // Attachment parent = nearest ESTABLISHED member (leader or a wing boid) directly ahead. We
  // only attach to an established member: an unattached (None) boid ahead is not a parent, so
  // two lone boids can never end up parenting each other in a cycle with no leader. Outgoing
  // leaders are skipped too (they are leaving). If none is ahead we handle seeding below.
  const BoidView* parent = nullptr;
  float parentDist = 0.f;  // nearest same-wing-or-leader boid ahead (preferred)
  const BoidView* anyParent = nullptr;
  float anyDist = 0.f;  // nearest established boid ahead of any wing (fallback)
  const bool committed = self.role == FormationRole::Left || self.role == FormationRole::Right;
  for (const auto& n : neighborhood) {
    if (n.steppingDown || n.role == FormationRole::None) continue;
    glm::vec2 rel = n.position - self.position;
    float d = glm::length(rel);
    if (d < 0.0001f) continue;
    if (glm::dot(forward, rel / d) < cone) continue;
    if (anyParent == nullptr || d < anyDist) {
      anyParent = &n;
      anyDist = d;
    }
    // Prefer a same-wing boid or the leader so a committed boid keeps following its own arm
    // (stable, no wing flip-fighting) instead of latching onto a nearer opposite-wing boid.
    bool preferred = n.role == FormationRole::Leader || !committed || n.role == self.role;
    if (preferred && (parent == nullptr || d < parentDist)) {
      parent = &n;
      parentDist = d;
    }
  }
  // No same-wing/leader boid ahead: fall back to the opposite arm and swap into it (the wing is
  // taken from the parent below), instead of coasting with no parent.
  if (parent == nullptr) parent = anyParent;

  // Outgoing leader: hand off and go to the back instead of leading or decelerating. While we
  // are still up front (nobody ahead), steer to a specific rear slot on the emptier wing; being
  // steppingDown, we are skipped by everyone else, so a wing head promotes into the apex behind
  // us. Once a boid is ahead of us we have reached the back, so we rejoin as a normal follower.
  if (fm.steppingDown) {
    if (parent == nullptr) {
      glm::vec2 fdir = glm::length(fm.formationForward) > 0.0001f ? glm::normalize(fm.formationForward) : forward;
      int lc = 0, rc = 0;
      const BoidView* deepestLeft = nullptr;
      const BoidView* deepestRight = nullptr;
      const BoidView* rearmost = nullptr;
      float rearProj = -1.0e9f;
      for (const auto& n : neighborhood) {
        if (n.steppingDown) continue;
        if (n.role == FormationRole::Left) {
          lc++;
          if (deepestLeft == nullptr || n.depth > deepestLeft->depth) deepestLeft = &n;
        } else if (n.role == FormationRole::Right) {
          rc++;
          if (deepestRight == nullptr || n.depth > deepestRight->depth) deepestRight = &n;
        }
        float proj = glm::dot(-fdir, n.position - self.position);
        if (proj > rearProj) {
          rearProj = proj;
          rearmost = &n;
        }
      }
      FormationRole wing = lc <= rc ? FormationRole::Left : FormationRole::Right;
      const BoidView* tail = wing == FormationRole::Left ? deepestLeft : deepestRight;
      if (tail == nullptr) tail = rearmost;

      if (tail != nullptr) {
        glm::vec2 back = -fdir;
        glm::vec2 lp = {-fdir.y, fdir.x};
        float sgn = wing == FormationRole::Left ? 1.f : -1.f;
        float a = vRule_->getSlotAngle();
        float d = vRule_->getSlotDistance();
        glm::vec2 slotOff = (back * std::cos(a) + lp * (sgn * std::sin(a))) * d;

        fm.role = wing;
        fm.formationForward = fdir;
        fm.depth = tail->depth + 1;
        fm.formationId = tail->formationId != kNoFormation ? tail->formationId : tail->id;
        fm.target = tail->position + slotOff;
        fm.targetVel = tail->velocity;
        fm.hasTarget = true;
        return;
      }
      // Isolated: nobody left to drop behind (the formation drifted out of range). Abandon the
      // hand-off and re-seed below instead of chasing a phantom point behind ourselves forever.
      fm.steppingDown = false;
    } else {
      fm.steppingDown = false;  // a boid is ahead of us now: we have reached the back, rejoin below
    }
  }

  if (parent == nullptr) {
    // No established member ahead: this is a seeding decision, made by identity, not geometry
    // (geometry is asymmetric, which let two lone boids both lead or both follow). We contest
    // only against candidates so unattached (None) boids and existing Leaders are scanned in all
    // directions so both boids reach the same verdict. Our own Left/Right followers are not
    // candidates, so a leader never yields to its own arm. Priority: a Leader outranks a
    // non-leader; among equals the lower id wins. The single highest-priority candidate leads;
    // everyone else forms up behind it. Two lone boids therefore split into one leader, one
    // follower, and both agree on which is which.
    const bool selfLeader = self.role == FormationRole::Leader;
    const bool selfEstablished = self.role != FormationRole::None;
    auto outranks = [](bool aLeader, std::uint32_t aId, bool bLeader, std::uint32_t bId) {
      if (aLeader != bLeader) return aLeader;
      return aId < bId;
    };

    const BoidView* best = nullptr;
    bool bestLeader = false;
    std::uint32_t bestId = 0;
    bool anyone = false;
    for (const auto& n : neighborhood) {
      if (n.steppingDown) continue;
      anyone = true;
      bool nLeader = n.role == FormationRole::Leader;
      bool nNone = n.role == FormationRole::None;
      if (!nLeader && !nNone) continue; // only None/Leader are candidates
      if (selfEstablished && nNone) continue; // an established boid never anchors to an unattached boid
      if (best == nullptr || outranks(nLeader, n.id, bestLeader, bestId)) {
        best = &n;
        bestLeader = nLeader;
        bestId = n.id;
      }
    }

    if (best != nullptr && outranks(bestLeader, bestId, selfLeader, self.id)) {
      if (bestLeader) {
        // Defer to an actual leader (also merges two formations): attach as a follower below.
        parent = best;
      } else {
        // Deferred to a lower-id unattached seed that has not become a leader yet. Wait a frame
        // with no slot rather than childing to its meaningless heading (which would put the slot
        // in front of us); once it leads, we attach to it properly.
        fm.role = FormationRole::None;
        fm.formationId = kNoFormation;
        fm.formationForward = forward;
        fm.hasTarget = false;
        return;
      }
    } else if (selfEstablished) {
      // Our formation just lost its leader (it stepped down or drifted off). Promote
      // deterministically so the formation never collapses into unattached boids: exactly one
      // boid is the head of a wing, i.e. the minimum (depth, then id) among our visible
      // formation-mates takes the apex. Everyone else keeps their wing this frame and
      // re-anchors to the new leader next frame.
      bool iAmSuccessor = true;
      for (const auto& n : neighborhood) {
        if (n.steppingDown || n.formationId != self.formationId) continue;
        if (n.role != FormationRole::Left && n.role != FormationRole::Right && n.role != FormationRole::Leader) continue;
        if (n.depth < self.depth || (n.depth == self.depth && n.id < self.id)) {
          iAmSuccessor = false;
          break;
        }
      }
      if (iAmSuccessor) {
        fm.role = FormationRole::Leader;
        fm.depth = 0;
        fm.formationId = self.id;
        fm.formationForward = forward;
      } else {
        fm.role = self.role; // keep our wing; re-anchor to the promoted head next frame
      }
      fm.hasTarget = false;
      return;
    } else {
      // Unattached: seed a new formation if anyone is near, else stay unattached.
      fm.role = anyone ? FormationRole::Leader : FormationRole::None;
      fm.depth = 0;
      fm.formationId = anyone ? self.id : kNoFormation;
      fm.formationForward = forward;
      fm.hasTarget = false;
      return;
    }
  }

  // Follower. Inherit the formation identity from the boid ahead (this is how formation
  // membership propagates outward without every boid needing to see the leader).
  std::uint32_t fid = parent->formationId != kNoFormation ? parent->formationId : parent->id;
  fm.formationId = fid;
  fm.depth = parent->depth + 1;

  // Inherit the formation's travel heading from the boid ahead. Every boid measures its slot
  // offset from this single shared direction (not from its own or its parent's instantaneous
  // velocity), so the arm stays a straight diagonal instead of bending with each boid's wobble.
  glm::vec2 fdir = parent->formationForward;
  if (glm::length(fdir) < 0.0001f) fdir = glm::length(parent->velocity) > 0.0001f ? parent->velocity : forward;
  fdir = glm::normalize(fdir);
  fm.formationForward = fdir;

  // Tally this formation's two arms among visible members. Also note, per wing, whether any
  // same-wing boid outranks us as the arm's tail: deeper, or equally deep with a lower id.
  // That (depth, id) ordering makes exactly one boid the tail, so a rebalance moves only it.
  int leftCount = 0, rightCount = 0;
  bool tailBlockedLeft = false, tailBlockedRight = false;
  for (const auto& n : neighborhood) {
    if (n.steppingDown || n.formationId != fid) continue;
    if (n.role == FormationRole::Left) {
      leftCount++;
      if (n.depth > fm.depth || (n.depth == fm.depth && n.id < self.id)) tailBlockedLeft = true;
    } else if (n.role == FormationRole::Right) {
      rightCount++;
      if (n.depth > fm.depth || (n.depth == fm.depth && n.id < self.id)) tailBlockedRight = true;
    }
  }

  // Choose a wing. If we attached to a wing boid, take its wing: down an arm this keeps us on
  // the same side (purity), and if we fell back to the opposite arm it swaps us into it. Behind
  // the leader we keep our committed side, or, with no wing yet, join the arm we are
  // geometrically on so a boid next to the right arm joins the right arm rather than being
  // forced to the empty left side. Actual balancing is done afterward by the tail rebalance.
  FormationRole wing;
  if (parent->role == FormationRole::Left || parent->role == FormationRole::Right) {
    wing = parent->role;
  } else if (self.role == FormationRole::Left || self.role == FormationRole::Right) {
    wing = self.role;
  } else {
    glm::vec2 pLeft = {-fdir.y, fdir.x};
    wing = glm::dot(self.position - parent->position, pLeft) > 0.f ? FormationRole::Left : FormationRole::Right;
  }

  // Rebalance: if the arms differ by 2 or more, only the single tail boid on the heavier arm
  // moves to the lighter one (moving the tail closes the gap by exactly 2). Restricting it to
  // the unique tail stops several boids swapping at once and overshooting into an oscillation.
  int totalLeft = leftCount + (wing == FormationRole::Left ? 1 : 0);
  int totalRight = rightCount + (wing == FormationRole::Right ? 1 : 0);
  if (wing == FormationRole::Left && totalLeft - totalRight >= 2 && !tailBlockedLeft) {
    wing = FormationRole::Right;
  } else if (wing == FormationRole::Right && totalRight - totalLeft >= 2 && !tailBlockedRight) {
    wing = FormationRole::Left;
  }
  fm.role = wing;

  // Steer toward the wingtip slot of the same-wing boid ahead (extends a straight arm);
  // fall back to a leader ahead, then to the parent. Building the slot relative to the boid
  // ahead is what keeps the arm from collapsing into a single-file line.
  const BoidView* ref = nullptr;
  float refDist = 0.f;
  const BoidView* leaderAhead = nullptr;
  float leaderDist = 0.f;
  for (const auto& n : neighborhood) {
    if (n.steppingDown || n.formationId != fid) continue;
    glm::vec2 rel = n.position - self.position;
    float d = glm::length(rel);
    if (d < 0.0001f) continue;
    if (glm::dot(forward, rel / d) < cone) continue;
    if (n.role == wing && (ref == nullptr || d < refDist)) {
      ref = &n;
      refDist = d;
    }
    if (n.role == FormationRole::Leader && (leaderAhead == nullptr || d < leaderDist)) {
      leaderAhead = &n;
      leaderDist = d;
    }
  }
  if (ref == nullptr) ref = leaderAhead;
  if (ref == nullptr) ref = parent;

  // Offset is anchored to the shared formation heading (fdir), not the reference boid's own
  // velocity, so every slot lies on the same straight diagonal from its predecessor.
  glm::vec2 back = -fdir;
  glm::vec2 rLeft = {-fdir.y, fdir.x};
  float sgn = wing == FormationRole::Left ? 1.f : -1.f;
  float a = vRule_->getSlotAngle();
  float dist = vRule_->getSlotDistance();
  fm.target = ref->position + (back * std::cos(a) + rLeft * (sgn * std::sin(a))) * dist;
  fm.targetVel = ref->velocity;  // the slot rides on the reference boid, so it moves at its velocity
  fm.hasTarget = true;
}

void FlockingManager::Start() {
  initializeRules();
  setNumberOfBoids(nbBoids);
}

void FlockingManager::Update(float deltaTime) {
  const int n = static_cast<int>(boidEntities.size());
  if (n == 0) return;

  std::vector<BoidView> snapshot(n);
  for (int i = 0; i < n; i++) {
    snapshot[i].position = ecs_.get<BoidPos>(boidEntities[i]).pos;
    snapshot[i].velocity = ecs_.get<BoidVel>(boidEntities[i]).vel;
    snapshot[i].id = ecs::entity_index(boidEntities[i]);
    const BoidFormation& fm = ecs_.get<BoidFormation>(boidEntities[i]);
    snapshot[i].energy = fm.energy;
    snapshot[i].steppingDown = fm.steppingDown;
    snapshot[i].role = fm.role;
    snapshot[i].depth = fm.depth;
    snapshot[i].formationId = fm.formationId;
    snapshot[i].formationForward = fm.formationForward;
    snapshot[i].target = fm.target;
    snapshot[i].targetVel = fm.targetVel;
    snapshot[i].hasTarget = fm.hasTarget;
  }

  glm::vec2 inputArrow(0.f);
  if (ImGui::IsKeyDown(ImGuiKey_UpArrow)) inputArrow.y -= 1.f;
  if (ImGui::IsKeyDown(ImGuiKey_DownArrow)) inputArrow.y += 1.f;
  if (ImGui::IsKeyDown(ImGuiKey_LeftArrow)) inputArrow.x -= 1.f;
  if (ImGui::IsKeyDown(ImGuiKey_RightArrow)) inputArrow.x += 1.f;
  if (glm::length(inputArrow) > 0.f) {
    ecs_.get<BoidAcc>(boidEntities[0]).acc += inputArrow * 20.f;
    ecs_.get<BoidDebug>(boidEntities[0]).drawDebugRadius = true;
    ecs_.get<BoidDebug>(boidEntities[0]).color = Color::Red;
  }

  const auto& rules = boidsRules;
  const bool vOn = vRule_ != nullptr && vRule_->isEnabled;
  const float vSlotDist = vRule_ != nullptr ? vRule_->getSlotDistance() : 30.f;
  const float vCatchup = vRule_ != nullptr ? vRule_->getCatchupBoost() : 1.5f;
  jobs::WaitGroup wg;
  sched_.parallel_for(
      static_cast<std::size_t>(n), 16,
      [&](std::size_t begin, std::size_t end) {
        for (std::size_t i = begin; i < end; i++) {
          ecs::Entity e = boidEntities[i];
          BoidPos& pos = ecs_.get<BoidPos>(e);
          BoidVel& vel = ecs_.get<BoidVel>(e);
          BoidAcc& acc = ecs_.get<BoidAcc>(e);
          BoidConfig& cfg = ecs_.get<BoidConfig>(e);
          BoidForceCache& fc = ecs_.get<BoidForceCache>(e);

          std::vector<BoidView> neighborhood;
          const float r2 = cfg.detectionRadius * cfg.detectionRadius;
          for (int j = 0; j < n; j++) {
            if (static_cast<std::size_t>(j) == i) continue;
            glm::vec2 d = snapshot[j].position - snapshot[i].position;
            if (glm::dot(d, d) <= r2) neighborhood.push_back(snapshot[j]);
          }

          fc.forces.resize(rules.size());
          for (std::size_t ri = 0; ri < rules.size(); ri++) {
            glm::vec2 f = rules[ri]->computeWeightedForce(neighborhood, snapshot[i]);
            fc.forces[ri] = f;
            acc.acc += f;
          }

          // Advance this boid's stamina, then recompute its formation role/target from the
          // neighborhood snapshot. Both write only this boid's own BoidFormation component,
          // so they are safe alongside the other threads (each owns a distinct entity).
          updateBoidEnergy(e, neighborhood, snapshot[i], deltaTime);
          updateBoidFormation(e, neighborhood, snapshot[i]);

          float mag = glm::length(acc.acc);
          if (mag > cfg.maxAcceleration && mag > 0.0001f) acc.acc = acc.acc * (cfg.maxAcceleration / mag);

          glm::vec2 newVel = vel.vel + acc.acc * deltaTime;
          if (!std::isfinite(newVel.x) || !std::isfinite(newVel.y)) newVel = glm::vec2(0.f);  // never let NaN/Inf reach position/draw
          acc.prevAcc = acc.acc;
          acc.acc = glm::vec2(0.f);

          // Speed cap. A boid seeking a formation slot gets headroom to exceed the formation's
          // cruise speed, scaled by how far it is from its slot; otherwise every boid tops out at
          // the same speed and a lagging follower could never close the gap. The headroom fades to
          // zero as it arrives, so it settles at cruise speed in the slot.
          float cap = cfg.speed;
          if (vOn) {
            const BoidFormation& bf = ecs_.get<BoidFormation>(e);
            if (bf.hasTarget && (bf.role == FormationRole::Left || bf.role == FormationRole::Right)) {
              float slotError = glm::length(bf.target - pos.pos);
              float sd = vSlotDist > 0.0001f ? vSlotDist : 1.f;
              float boost = slotError / sd;
              if (boost > vCatchup) boost = vCatchup;
              cap = cfg.speed * (1.f + boost);
            }
          }

          float speed = glm::length(newVel);
          if (cfg.hasConstantSpeed || speed > cap) {
            if (speed > 0.0001f) newVel = newVel * (cap / speed);
          }

          vel.vel = newVel;
          pos.pos += vel.vel * deltaTime;
        }
      },
      wg);
  sched_.wait(wg);

  for (auto e : boidEntities) warpIfOutOfBounds(ecs_.get<BoidPos>(e));
}

void FlockingManager::OnGui() { showConfigurationWindow(ImGui::GetIO().DeltaTime); }

void FlockingManager::OnDraw() {
  ImDrawList* dl = ImGui::GetBackgroundDrawList();

  bool vEnabled = false;
  for (const auto& rule : boidsRules)
    if (rule->isEnabled && dynamic_cast<VFormationRule*>(rule.get())) {
      vEnabled = true;
      break;
    }

  for (int i = 0; i < static_cast<int>(boidEntities.size()); i++) {
    ecs::Entity e = boidEntities[i];
    BoidPos& pos = ecs_.get<BoidPos>(e);
    BoidVel& vel = ecs_.get<BoidVel>(e);
    BoidAcc& acc = ecs_.get<BoidAcc>(e);
    BoidConfig& cfg = ecs_.get<BoidConfig>(e);
    BoidDebug& dbg = ecs_.get<BoidDebug>(e);

    glm::vec2 p = pos.pos;
    glm::vec2 v = vel.vel;

    float len = glm::length(v);
    glm::vec2 fwd = len > 0.0001f ? v / len : glm::vec2(0.f, -1.f);
    glm::vec2 perp(-fwd.y, fwd.x);
    ImVec2 tip = {p.x + fwd.x * 9.f, p.y + fwd.y * 9.f};
    ImVec2 left = {p.x - perp.x * 4.5f - fwd.x * 4.f, p.y - perp.y * 4.5f - fwd.y * 4.f};
    ImVec2 right = {p.x + perp.x * 4.5f - fwd.x * 4.f, p.y + perp.y * 4.5f - fwd.y * 4.f};
    ImU32 col = IM_COL32(static_cast<int>(dbg.color.r * 255), static_cast<int>(dbg.color.g * 255), static_cast<int>(dbg.color.b * 255),
                         static_cast<int>(dbg.color.a * 255));
    if (vEnabled && (showRules || dbg.drawDebugRules)) {
      switch (ecs_.get<BoidFormation>(e).role) {  // recolor by formation role for debugging
        case FormationRole::Leader: col = IM_COL32(255, 215, 0, 255); break;   // apex: gold
        case FormationRole::Left: col = IM_COL32(80, 190, 255, 255); break;    // left arm: blue
        case FormationRole::Right: col = IM_COL32(255, 120, 200, 255); break;  // right arm: pink
        case FormationRole::None: col = IM_COL32(140, 140, 140, 255); break;   // unattached: gray
      }
    }
    dl->AddTriangleFilled(tip, left, right, col);

    if (showRadius || dbg.drawDebugRadius) {
      dl->AddCircle({p.x, p.y}, cfg.detectionRadius,
                    IM_COL32(static_cast<int>(dbg.color.r * 255), static_cast<int>(dbg.color.g * 255), static_cast<int>(dbg.color.b * 255), 64), 32);
    }

    if (showAcceleration || dbg.drawAcceleration) {
      glm::vec2 end = p + acc.prevAcc * 0.08f;
      dl->AddLine({p.x, p.y}, {end.x, end.y}, IM_COL32(128, 0, 128, 220), 1.5f);
    }

    if (vEnabled && (showRules || dbg.drawDebugRules)) {
      const BoidFormation& fm = ecs_.get<BoidFormation>(e);
      const float bw = 14.f, bh = 2.5f, by = p.y - 12.f;
      ImVec2 a = {p.x - bw * 0.5f, by};
      ImVec2 b = {p.x + bw * 0.5f, by + bh};
      dl->AddRectFilled(a, b, IM_COL32(0, 0, 0, 120));
      ImVec2 fill = {a.x + bw * fm.energy, b.y};
      dl->AddRectFilled(a, fill, IM_COL32(static_cast<int>(255 * (1.f - fm.energy)), static_cast<int>(255 * fm.energy), 60, 220));
      if (fm.steppingDown) dl->AddRect({a.x - 1.f, a.y - 1.f}, {b.x + 1.f, b.y + 1.f}, IM_COL32(80, 160, 255, 255));

      auto drawSlot = [&](glm::vec2 s, ImU32 c, bool thread) {
        if (thread) dl->AddLine({p.x, p.y}, {s.x, s.y}, IM_COL32(255, 255, 255, 60), 1.0f);
        dl->AddCircle({s.x, s.y}, 4.f, c, 12, 1.5f);
        dl->AddLine({s.x - 3.f, s.y}, {s.x + 3.f, s.y}, c, 1.0f);
        dl->AddLine({s.x, s.y - 3.f}, {s.x, s.y + 3.f}, c, 1.0f);
      };

      // Slot marker: where this follower is trying to sit, with a thread from the boid to it.
      if (fm.hasTarget)
        drawSlot(fm.target, fm.role == FormationRole::Left ? IM_COL32(80, 190, 255, 200) : IM_COL32(255, 120, 200, 200), true);

      // Leader shows both open wingtip slots so the V's arms are visible before boids fill them.
      if (fm.role == FormationRole::Leader && vRule_ != nullptr) {
        glm::vec2 fdir = fm.formationForward;
        if (glm::length(fdir) < 0.0001f) fdir = fwd;
        fdir = glm::normalize(fdir);
        glm::vec2 back = -fdir;
        glm::vec2 lp = {-fdir.y, fdir.x};
        float sa = vRule_->getSlotAngle();
        float sd = vRule_->getSlotDistance();
        glm::vec2 baseOff = back * std::cos(sa) * sd;
        glm::vec2 latOff = lp * std::sin(sa) * sd;
        drawSlot(p + baseOff + latOff, IM_COL32(80, 190, 255, 130), false);   // left slot
        drawSlot(p + baseOff - latOff, IM_COL32(255, 120, 200, 130), false);  // right slot
      }
    }

    if (showRules || dbg.drawDebugRules) {
      BoidForceCache& fc = ecs_.get<BoidForceCache>(e);
      BoidView bv{p, v};
      for (std::size_t ri = 0; ri < boidsRules.size() && ri < fc.forces.size(); ri++) {
        if (boidsRules[ri]->isEnabled) boidsRules[ri]->draw(bv, dl, fc.forces[ri]);
      }
    }
  }

  if (showRules) {
    for (const auto& rule : boidsRules) {
      if (rule->isEnabled) rule->drawWorldOverlay(dl);
    }
  }
}

void FlockingManager::drawGeneralUI() {
  ImGui::SetNextItemOpen(true, ImGuiCond_Once);
  if (ImGui::CollapsingHeader("General")) {
    if (ImGui::DragInt("Number of Boids", &nbBoids)) {
      if (nbBoids < 0) nbBoids = 0;
      setNumberOfBoids(nbBoids);
    }
    ImGui::SameLine();
    HelpMarker("Drag to change the weight's value or CTRL+Click to input a new value.");

    if (ImGui::SliderFloat("Neighborhood Radius", &detectionRadius, 0.0f, 250.0f, "%.f"))
      for (auto e : boidEntities) ecs_.get<BoidConfig>(e).detectionRadius = detectionRadius;

    ImGui::SetNextItemOpen(false, ImGuiCond_Once);
    if (ImGui::TreeNode("Movement Settings")) {
      if (ImGui::Checkbox("Has Constant Speed", &hasConstantSpeed))
        for (auto e : boidEntities) ecs_.get<BoidConfig>(e).hasConstantSpeed = hasConstantSpeed;

      const char* speedLabel = hasConstantSpeed ? "Speed" : "Max Speed";
      if (ImGui::SliderFloat(speedLabel, &desiredSpeed, 0.0f, 300.0f, "%.f"))
        for (auto e : boidEntities) ecs_.get<BoidConfig>(e).speed = desiredSpeed;

      if (ImGui::Checkbox("Has Max Acceleration", &hasMaxAcceleration)) {
        for (auto e : boidEntities) ecs_.get<BoidConfig>(e).maxAcceleration = hasMaxAcceleration ? maxAcceleration : 10000.f;
      }
      ImguiTooltip("Boids keeps more momentum when the acceleration is capped.");

      if (hasMaxAcceleration)
        if (ImGui::SliderFloat("Max Acceleration", &maxAcceleration, 0.0f, 35.0f, "%.f"))
          for (auto e : boidEntities) ecs_.get<BoidConfig>(e).maxAcceleration = maxAcceleration;

      ImGui::TreePop();
    }

    ImGui::SetNextItemOpen(true, ImGuiCond_Once);
    if (ImGui::TreeNode("Display Settings")) {
      if (ImGui::Checkbox("Show Acceleration", &showAcceleration))
        for (auto e : boidEntities) ecs_.get<BoidDebug>(e).drawAcceleration = showAcceleration;
      if (ImGui::Checkbox("Show Radius", &showRadius))
        for (auto e : boidEntities) ecs_.get<BoidDebug>(e).drawDebugRadius = showRadius;
      if (ImGui::Checkbox("Show Rules", &showRules))
        for (auto e : boidEntities) ecs_.get<BoidDebug>(e).drawDebugRules = showRules;
      ImGui::TreePop();
    }

    if (ImGui::Button("Randomize Boids position and velocity"))
      for (auto e : boidEntities) randomizeBoidPosVel(e);
  }
}

void FlockingManager::drawRulesUI() {
  if (ImGui::CollapsingHeader("Rules")) {
    for (auto& rule : boidsRules) {
      rule->drawImguiRule();
      ImGui::Separator();
    }
    if (ImGui::Button("Restore Default Weights")) {
      int i = 0;
      for (auto& rule : boidsRules) rule->weight = defaultWeights[i++];
    }
    ImGui::Spacing();
  }
}

void FlockingManager::showConfigurationWindow(float deltaTime) {
  if (ImGui::BeginMainMenuBar()) {
    if (ImGui::BeginMenu("File")) {
      ImGui::EndMenu();
    }
    ImGui::Text("%.1fms %.0fFPS | AVG: %.2fms %.1fFPS", ImGui::GetIO().DeltaTime * 1000, 1.0f / ImGui::GetIO().DeltaTime,
                1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);
    ImGui::EndMainMenuBar();
  }

  ImGui::SetNextWindowPos(ImVec2(20, 20), ImGuiCond_Once);
  ImGui::SetNextWindowSize(ImVec2(320, 550), ImGuiCond_Once);
  if (ImGui::Begin("Settings", nullptr, ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoTitleBar)) {
    ImGui::Text("Control the simulation with those settings.");
    ImGui::Spacing();
    ImGui::PushItemWidth(ImGui::GetWindowWidth() * 0.45f);

    drawGeneralUI();
    ImGui::SetNextItemOpen(true, ImGuiCond_Once);
    drawRulesUI();
    drawPerformanceUI(deltaTime);

    ImGui::End();
  }
}

void FlockingManager::drawPerformanceUI(float deltaTime) {
#if defined(_WIN32)
  if (ImGui::CollapsingHeader("Performance")) {
    ImGui::Text("Frames Per Second (FPS) : %.f", 1.f / deltaTime);
    PlotVar("Frame duration (ms)", deltaTime * 1000);
    ImGui::Separator();

    MEMORYSTATUSEX memInfo;
    memInfo.dwLength = sizeof(MEMORYSTATUSEX);
    GlobalMemoryStatusEx(&memInfo);

    PROCESS_MEMORY_COUNTERS_EX pmc;
    GetProcessMemoryInfo(GetCurrentProcess(), (PROCESS_MEMORY_COUNTERS*)&pmc, sizeof(pmc));

    const int div = 1048576;
    ImGui::Text("Total Virtual Memory : %uMb", (unsigned)(memInfo.ullTotalPageFile / div));
    ImGui::Text("Total RAM : %uMb", (unsigned)(memInfo.ullTotalPhys / div));
    ImGui::Separator();
    ImGui::Text("Virtual Memory used by process : %uMb", (unsigned)(pmc.PrivateUsage / div));
    PlotVar("Virtual Memory Consumption (Mb)", (float)(pmc.PrivateUsage / div));
    ImGui::Text("RAM used by process : %uMb", (unsigned)(pmc.WorkingSetSize / div));
    PlotVar("Ram Consumption (Mb)", (float)(pmc.WorkingSetSize / div));
  }
#else
  (void)deltaTime;
#endif
}
