//
// Created by atolstenko on 2/9/2023.
//

#include "HexagonGameOfLife.h"
#include "../fsm/Action.h"
#include "../fsm/AgentContext.h"
#include "../fsm/Condition.h"

#include <SDL3/SDL_log.h>

#include <cstdlib>
#include <stdexcept>

// Hexagonal variant: each cell has 6 neighbors instead of 8. This one is
// interactive-only (no formal fixtures), so the exact rule is up to you - the
// classic hex grid plays B2/S34: a dead cell is born with exactly 2 live
// neighbors, a live cell survives with 3 or 4.
//
// hint: the app draws odd rows displaced by half a cell, so the neighbors
// above and below shift by one column depending on the row parity.
// Reference: https://arunarjunakani.github.io/HexagonalGameOfLife/
//
// The rules as machine parts (same shape as JohnConway):
//   underpopulation (<3) / overpopulation (>4) -> conditions that leave Alive
//   reproduction (==2)                          -> condition that leaves Dead
//   survival is implicit: no transition firing means the stay actions run.

// begin solution
namespace hexagon {
class Underpopulation : public Condition {
public:
  bool Test(const AgentContext& context) override {
    // B2/S34: a live cell is underpopulated below 3 neighbors
    return context.aliveNeighbors < 3;
  }
};

class Overpopulation : public Condition {
public:
  bool Test(const AgentContext& context) override {
    // B2/S34: a live cell is overpopulated above 4 neighbors
    return context.aliveNeighbors > 4;
  }
};

class Reproduction : public Condition {
public:
  bool Test(const AgentContext& context) override {
    // B2/S34: a dead cell is born with exactly 2 neighbors
    return context.aliveNeighbors == 2;
  }
};

class DieAction : public Action {
public:
  void Execute(const AgentContext& context) override {
    context.world.SetNext(context.position, false);
  }
};

class BornAction : public Action {
public:
  void Execute(const AgentContext& context) override {
    context.world.SetNext(context.position, true);
  }
};

class StayAliveAction : public Action {
public:
  void Execute(const AgentContext& context) override {
    context.world.SetNext(context.position, true);
  }
};

class StayDeadAction : public Action {
public:
  void Execute(const AgentContext& context) override {
    context.world.SetNext(context.position, false);
  }
};
}  // namespace hexagon

// end solution

HexagonGameOfLife::HexagonGameOfLife() {
  using namespace hexagon;

  alive = std::make_shared<State>("Alive");
  dead = std::make_shared<State>("Dead");

  const auto die = std::make_shared<DieAction>();
  const auto born = std::make_shared<BornAction>();

  // todo: add transitions and actions for alive, dead. example:
  //   alive->AddTransition(std::make_shared<Underpopulation>(), dead, {die});
  //   dead->AddAction(std::make_shared<StayDeadAction>());
  // begin solution

  alive->AddTransition(std::make_shared<Underpopulation>(), dead, {die});
  alive->AddTransition(std::make_shared<Overpopulation>(), dead, {die});
  dead->AddTransition(std::make_shared<Reproduction>(), alive, {born});

  alive->AddAction(std::make_shared<StayAliveAction>());
  dead->AddAction(std::make_shared<StayDeadAction>());

  // end solution
}

void HexagonGameOfLife::Step(World& world) {
  // relevant functions:
  //   world.Height() and world.Width() to get the world dimensions,
  //   world.Get() reads the CURRENT generation, world.SetNext() writes the NEXT one
  // Build one context per cell and let the machine decide: conditions read the
  // current generation through the context, actions write the next one.
  //
  // note: the double buffering does NOT happen here. Your actions only write
  // the next buffer via SetNext; the demo app's Manager::step calls
  // world.SwapBuffers() right AFTER this function returns. Never call
  // SwapBuffers from inside a rule.
  // begin solution
  // sparse sweep: only live cells and their neighbors can change this step, so
  // tally alive neighbors from the live set instead of scanning the whole grid.
  neighborCounts.clear();
  const auto& live = world.LiveCells();
  neighborCounts.reserve(live.size() * 7);
  for (const auto& cell : live) {
    // the live cell is itself a candidate, even with 0 live neighbors
    neighborCounts.emplace(cell, 0);
    for (const auto& n : Neighbors(world, cell)) ++neighborCounts[world.Wrap(n)];
  }
  // run the FSM only on the candidate cells
  for (const auto& entry : neighborCounts) {
    const Point2D& pos = entry.first;
    AgentContext context{world, pos, world.Get(pos), entry.second};
    machine.SetCurrent(context.isAlive ? alive : dead);
    machine.Update(context);
  }
  // end solution
}

std::array<Point2D, 6> HexagonGameOfLife::Neighbors(const World& world, Point2D point) const {
  // odd-r offset layout: rows whose distance from the center is odd are drawn
  // shifted half a cell to the right (see Manager::OnDraw), so the two upper and
  // two lower neighbors shift by one column with that same parity.
  const bool shifted = std::abs(point.y - world.Height() / 2) % 2 == 1;
  return {{{point.x - 1, point.y},
           {point.x + 1, point.y},
           {point.x + (shifted ? 0 : -1), point.y - 1},
           {point.x + (shifted ? 1 : 0), point.y - 1},
           {point.x + (shifted ? 0 : -1), point.y + 1},
           {point.x + (shifted ? 1 : 0), point.y + 1}}};
}

int HexagonGameOfLife::CountNeighbors(World& world, Point2D point) {
  // count the ALIVE hex neighbors of the cell at point (toroidal via world.Get)
  // begin solution
  int count = 0;
  for (const auto& n : Neighbors(world, point))
    if (world.Get(n)) ++count;
  return count;
  // end solution
}
