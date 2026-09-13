#include "World.h"
#include "Random.h"
Point2D World::Wrap(Point2D point) const {
  // toroidal wrap into [0,width) x [0,height)
  point.x %= width;
  if (point.x < 0) point.x += width;
  point.y %= height;
  if (point.y < 0) point.y += height;
  return point;
}
void World::Resize(int size) { Resize(size, size); }
void World::Resize(int columns, int lines) {
  width = columns;
  height = lines;
  buffer.assign(columns * lines, false);
  liveCells.clear();
  nextLiveCells.clear();
  pending.clear();
}
void World::SwapBuffers() {
  // apply the staged next-generation writes to the dense mirror
  for (const auto& [point, value] : pending) buffer[point.y * width + point.x] = value;
  // next generation becomes current, ready the staging for the following step
  std::swap(liveCells, nextLiveCells);
  pending.clear();
  nextLiveCells.clear();
}
void World::SetNext(Point2D point, bool value) {
  point = Wrap(point);
  pending.emplace_back(point, value);
  if (value)
    nextLiveCells.insert(point);
  else
    nextLiveCells.erase(point);
}
void World::SetCurrent(Point2D point, bool value) {
  point = Wrap(point);
  buffer[point.y * width + point.x] = value;
  if (value)
    liveCells.insert(point);
  else
    liveCells.erase(point);
}
bool World::Get(Point2D point) {
  point = Wrap(point);
  return buffer[point.y * width + point.x];
}
void World::Randomize() {
  liveCells.clear();
  nextLiveCells.clear();
  pending.clear();
  for (int y = 0; y < height; ++y)
    for (int x = 0; x < width; ++x) {
      bool alive = (Random::Range(0, 1) != 0);
      buffer[y * width + x] = alive;
      if (alive) liveCells.insert({x, y});
    }
}
