#ifndef MOBAGEN_EXAMPLES_LIFE_RULES_HEXAGONGAMEOFLIFE_H_
#define MOBAGEN_EXAMPLES_LIFE_RULES_HEXAGONGAMEOFLIFE_H_

#include "../PointHash.h"
#include "../RuleBase.h"
#include "../fsm/State.h"
#include "../fsm/StateMachine.h"
#include <array>
#include <memory>
#include <string>
#include <unordered_map>

class HexagonGameOfLife : public RuleBase {
public:
  HexagonGameOfLife();
  ~HexagonGameOfLife() override = default;
  std::string GetName() override { return "Hexagon"; }
  void Step(World& world) override;
  int CountNeighbors(World& world, Point2D point);
  GameOfLifeTileSetEnum GetTileSet() override { return GameOfLifeTileSetEnum::Hexagon; };

private:
  // the 6 neighbor offsets for the cell at point, matching the renderer's
  // odd-row shift so on-screen adjacency equals simulated adjacency
  std::array<Point2D, 6> Neighbors(const World& world, Point2D point) const;
  // the shared state graph: every cell runs this same machine
  std::shared_ptr<State> alive;
  std::shared_ptr<State> dead;
  StateMachine machine;
  // alive-neighbor tally for the candidate cells, reused across steps
  std::unordered_map<Point2D, int, PointHash> neighborCounts;
};

#endif  // MOBAGEN_EXAMPLES_LIFE_RULES_HEXAGONGAMEOFLIFE_H_
