#ifndef MOBAGEN_JOHNCONWAY_H
#define MOBAGEN_JOHNCONWAY_H
#include "../PointHash.h"
#include "../RuleBase.h"
#include "../fsm/State.h"
#include "../fsm/StateMachine.h"
#include <memory>
#include <string>
#include <unordered_map>

class JohnConway : public RuleBase {
public:
  JohnConway();
  ~JohnConway() override = default;
  std::string GetName() override { return "JohnConway"; }
  void Step(World& world) override;
  int CountNeighbors(World& world, Point2D point);
  GameOfLifeTileSetEnum GetTileSet() override { return GameOfLifeTileSetEnum::Square; };

private:
  // the shared state graph: every cell runs this same machine
  std::shared_ptr<State> alive;
  std::shared_ptr<State> dead;
  StateMachine machine;
  // alive-neighbor tally for the candidate cells, reused across steps
  std::unordered_map<Point2D, int, PointHash> neighborCounts;
};

#endif  // MOBAGEN_JOHNCONWAY_H
