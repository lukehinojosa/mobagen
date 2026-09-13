#ifndef LIFE_WORLD_H
#define LIFE_WORLD_H

#include "PointHash.h"
#include "math/Point2D.h"
#include <unordered_set>
#include <utility>
#include <vector>

struct World {
private:
  // dense mirror of the current generation, kept only for rendering and tests
  std::vector<bool> buffer;
  int width;
  int height;
  // sparse source of truth: the live cells of the current and next generations
  std::unordered_set<Point2D, PointHash> liveCells;
  std::unordered_set<Point2D, PointHash> nextLiveCells;
  // writes staged by SetNext, applied to the mirror on SwapBuffers
  std::vector<std::pair<Point2D, bool>> pending;

public:
  inline const int& Width() const { return width; };
  inline const int& Height() const { return height; };
  // live cells of the current generation, read by the sparse sweep
  inline const std::unordered_set<Point2D, PointHash>& LiveCells() const { return liveCells; }
  // wraps a point onto the toroidal grid
  Point2D Wrap(Point2D point) const;
  // square grids (visual app)
  void Resize(int sideSize);
  // rectangular grids (formal tests): C columns x L lines
  void Resize(int columns, int lines);
  // promotes the next generation to current: applies the staged writes to the
  // mirror and swaps the live sets. Called by whoever drives the simulation
  // right AFTER a rule Step returns, never from inside a rule.
  void SwapBuffers();
  bool Get(Point2D point);
  void SetNext(Point2D point, bool value);
  void SetCurrent(Point2D point, bool value);
  void Randomize();
};

#endif  // MOBAGEN_WORLD_H
