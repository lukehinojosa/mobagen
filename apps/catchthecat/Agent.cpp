#include "Agent.h"
#include <climits>
#include <queue>
#include <unordered_map>
#include <unordered_set>
#include "World.h"

using namespace std;

// neighbors that are inside the board, not visited, not the cat, not blocked and not already queued
static vector<Point2D> getVisitableNeighbors(CatWorld* w, const Point2D& current, const unordered_map<Point2D, bool>& visited,
                                             const unordered_set<Point2D>& frontierSet) {
  vector<Point2D> result;
  auto catPos = w->getCat();
  for (const auto& n : CatWorld::neighbors(current)) {
    if (!w->isValidPosition(n) || n == catPos || w->getContent(n)) continue;
    if (visited.count(n) || frontierSet.count(n)) continue;
    result.push_back(n);
  }
  return result;
}

std::vector<Point2D> Agent::generatePath(CatWorld* w) {
  unordered_map<Point2D, Point2D> cameFrom;  // to build the flowfield and build the path
  queue<Point2D> frontier;                   // to store next ones to visit
  unordered_set<Point2D> frontierSet;        // OPTIMIZATION to check faster if a point is in the queue
  unordered_map<Point2D, bool> visited;      // use .at() to get data, if the element dont exist [] will give you wrong results

  // bootstrap state
  auto catPos = w->getCat();
  frontier.push(catPos);
  frontierSet.insert(catPos);
  Point2D borderExit = {INT32_MAX, INT32_MAX};  // sentinel: no border found yet

  while (!frontier.empty() && borderExit.x == INT32_MAX) {
    auto current = frontier.front();
    frontier.pop();
    frontierSet.erase(current);
    visited[current] = true;

    for (const auto& next : getVisitableNeighbors(w, current, visited, frontierSet)) {
      cameFrom[next] = current;
      if (w->catWinsOnSpace(next)) {
        borderExit = next;
        break;
      }
      frontier.push(next);
      frontierSet.insert(next);
    }
  }

  // no reachable border
  if (borderExit.x == INT32_MAX) return {};

  // filled from the border to the cat (cat excluded): front is the catcher move, back is the cat move
  vector<Point2D> path;
  for (auto p = borderExit; p != catPos; p = cameFrom.at(p)) path.push_back(p);
  return path;
}
