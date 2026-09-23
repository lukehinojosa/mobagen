#include "HuntAndKillExample.h"
#include "../World.h"
#include "Random.h"
#include <climits>

static const Color32 visitedColor = Color::Black;

static bool inBounds(World* w, const Point2D& p) { return p.x >= 0 && p.x < w->GetWidth() && p.y >= 0 && p.y < w->GetHeight(); }

// Opens the wall between two adjacent cells.
static void openWall(World* w, const Point2D& from, const Point2D& to) {
  Point2D direction = to - from;
  Point2D up = {0, -1};
  Point2D down = {0, 1};
  Point2D left = {-1, 0};
  Point2D right = {1, 0};

  if (direction == up)
    w->SetNorth(from, false);
  else if (direction == down)
    w->SetSouth(from, false);
  else if (direction == left)
    w->SetWest(from, false);
  else if (direction == right)
    w->SetEast(from, false);
}

bool HuntAndKillExample::Step(World* w) {
  if (stack.empty()) // Hunt mode: find an unvisited cell next to the maze and connect it.
  {
    for (int y = 0; y < w->GetHeight(); y++)
    {
      for (int x = 0; x < w->GetWidth(); x++)
      {
        if (visited[y][x])
          continue;

        std::vector<Point2D> visitedNeighbors = getVisitedNeighbors(w, {x, y});
        if (visitedNeighbors.empty())
          continue;

        // Found one: connect it to a random neighbor already in the maze and start walking from it.
        Point2D neighbor = visitedNeighbors[Random::Range(0, (int)visitedNeighbors.size() - 1)];
        openWall(w, {x, y}, neighbor);
        stack.push_back({x, y});
        return true;
      }
    }

    // Nothing is next to the maze. Either the maze hasn't started yet, or every cell is visited.
    Point2D start = randomStartPoint(w);
    if (start.x == INT_MAX) // Maze is done.
      return false;

    stack.push_back(start);
    return true;
  }

  // Kill mode: random walk from the current cell.
  Point2D current = stack.back();
  visited[current.y][current.x] = true; // Mark current as visited.
  w->SetNodeColor(current, visitedColor);

  std::vector<Point2D> visitable = getVisitables(w, current);

  if (visitable.empty()) // Dead end, switch to hunt mode.
  {
    stack.clear();
    return true;
  }

  Point2D next;
  if (visitable.size() == 1) // Only 1 not visited, so don't consume a random number.
    next = visitable[0];
  else
    next = visitable[Random::Range(0, (int)visitable.size() - 1)];

  // Move
  openWall(w, current, next);
  stack.push_back(next);

  return true;
}
void HuntAndKillExample::Clear(World* world) {
  visited.clear();
  stack.clear();

  for (int i = 0; i < world->GetHeight(); i++) {
    for (int j = 0; j < world->GetWidth(); j++) {
      visited[i][j] = false;
    }
  }
}
Point2D HuntAndKillExample::randomStartPoint(World* world) {
  // Todo: improve this if you want
  for (int y = 0; y < world->GetHeight(); y++)
    for (int x = 0; x < world->GetWidth(); x++)
      if (!visited[y][x]) return {x, y};
  return {INT_MAX, INT_MAX};
}

std::vector<Point2D> HuntAndKillExample::getVisitables(World* w, const Point2D& p) {
  std::vector<Point2D> visitables;

  // Neighbors inside the grid that are not visited, in clockwise order.
  Point2D candidates[] = {
    {p.x, p.y - 1}, // UP
    {p.x + 1, p.y}, // RIGHT
    {p.x, p.y + 1}, // DOWN
    {p.x - 1, p.y}, // LEFT
  };
  for (const auto& c : candidates)
  {
    if (inBounds(w, c) && !visited[c.y][c.x])
      visitables.push_back(c);
  }

  return visitables;
}
std::vector<Point2D> HuntAndKillExample::getVisitedNeighbors(World* w, const Point2D& p) {
  std::vector<Point2D> deltas = {{-1, 0}, {0, -1}, {1, 0}, {0, 1}};
  std::vector<Point2D> neighbors;

  // Neighbors inside the grid that are already part of the maze.
  for (const auto& delta : deltas)
  {
    Point2D neighbor = p + delta;
    if (inBounds(w, neighbor) && visited[neighbor.y][neighbor.x])
      neighbors.push_back(neighbor);
  }

  return neighbors;
}
