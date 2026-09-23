#include "PrimExample.h"
#include "../World.h"
#include "Random.h"

// Cell colors double as the visited state: dark gray = untouched,
// frontier = waiting in toBeVisited, visited = already part of the maze.
static const Color32 frontierColor = Color::Orange;
static const Color32 visitedColor = Color::Black;

static bool sameColor(const Color32& a, const Color32& b) { return a.GetPacked() == b.GetPacked(); }

static bool inBounds(World* w, const Point2D& p) { return p.x >= 0 && p.x < w->GetWidth() && p.y >= 0 && p.y < w->GetHeight(); }

bool PrimExample::Step(World* w) {
  if (!initialized) // First step: pick a random start cell and add its neighbors to the frontier.
  {
    Point2D start = {0, 0};
    w->SetNodeColor(start, visitedColor);

    for (const auto& neighbor : getVisitables(w, start))
    {
      toBeVisited.push_back(neighbor);
      w->SetNodeColor(neighbor, frontierColor);
    }

    initialized = true;
    return !toBeVisited.empty();
  }

  if (toBeVisited.empty()) // Maze is done.
    return false;

  // Take a random cell off the frontier.
  int frontierIndex = Random::Range(0, (int)toBeVisited.size() - 1);
  Point2D current = toBeVisited[frontierIndex];
  toBeVisited.erase(toBeVisited.begin() + frontierIndex);

  // Connect it to a random neighbor that is already in the maze.
  std::vector<Point2D> visitedNeighbors = getVisitedNeighbors(w, current);
  Point2D neighbor = visitedNeighbors[Random::Range(0, (int)visitedNeighbors.size() - 1)];

  // Move
  Point2D direction = neighbor - current;
  Point2D up = {0, -1};
  Point2D down = {0, 1};
  Point2D left = {-1, 0};
  Point2D right = {1, 0};

  if (direction == up)
    w->SetNorth(current, false);
  else if (direction == down)
    w->SetSouth(current, false);
  else if (direction == left)
    w->SetWest(current, false);
  else if (direction == right)
    w->SetEast(current, false);

  w->SetNodeColor(current, visitedColor); // Mark as part of the maze.

  // Add its untouched neighbors to the frontier.
  for (const auto& next : getVisitables(w, current))
  {
    toBeVisited.push_back(next);
    w->SetNodeColor(next, frontierColor);
  }

  return !toBeVisited.empty();
}
void PrimExample::Clear(World* world) {
  toBeVisited.clear();
  initialized = false;
}

std::vector<Point2D> PrimExample::getVisitables(World* w, const Point2D& p) {
  std::vector<Point2D> visitables;
  auto clearColor = Color32(169.0f / 255.0f, 169.0f / 255.0f, 169.0f / 255.0f, 1.0f);  // dark gray

  // Neighbors inside the grid that are neither visited nor already on the frontier.
  Point2D candidates[] = {
    {p.x, p.y - 1}, // UP
    {p.x + 1, p.y}, // RIGHT
    {p.x, p.y + 1}, // DOWN
    {p.x - 1, p.y}, // LEFT
  };
  for (const auto& c : candidates)
  {
    if (inBounds(w, c) && sameColor(w->GetNodeColor(c), clearColor))
      visitables.push_back(c);
  }

  return visitables;
}

std::vector<Point2D> PrimExample::getVisitedNeighbors(World* w, const Point2D& p) {
  std::vector<Point2D> deltas = {Point2D(0, -1), Point2D(0, 1), Point2D(-1, 0), Point2D(1, 0)};  // N, S, W, E
  std::vector<Point2D> neighbors;

  // Neighbors inside the grid that are already part of the maze.
  for (const auto& delta : deltas)
  {
    Point2D neighbor = p + delta;
    if (inBounds(w, neighbor) && sameColor(w->GetNodeColor(neighbor), visitedColor))
      neighbors.push_back(neighbor);
  }

  return neighbors;
}
