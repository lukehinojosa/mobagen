#include "../World.h"
#include "../SeededRandom.h"
#include "RecursiveBacktrackerExample.h"
#include <climits>

// Recursive backtracker, in grid units: (0, 0) is the top-left cell, x grows
// right, y grows down — the same units as the World API. The caller seeds
// SeededRandom before the first Step; every decision consumes the seed in
// order, so the maze is deterministic.
//
// Procedure per Step, on the cell at the top of the path stack:
//   1. mark it visited;
//   2. list its visitable (unvisited) neighbors in clockwise order starting
//      from the top: UP, RIGHT, DOWN, LEFT (getVisitables does this);
//   3. none        -> dead end: pop the stack (backtrack). Empty stack = done;
//   4. exactly one -> move to it, do not consume a random number;
//   5. two or more -> consume SeededRandom::next() and pick
//      next() % visitableCount;
//   6. moving opens the wall between the two cells
//      (World::SetNorth/SetEast/SetSouth/SetWest with false).

void RecursiveBacktrackerExample::Clear(World* world) {
  // todo: reset the walk
  // hint:
  //   clear visited and the path stack, then start the walk at the
  //   top-left cell: stack.push_back({0, 0})
  // begin solution
  visited.clear();
  stack.clear();
  stack.push_back({0, 0});
  // end solution
}

bool RecursiveBacktrackerExample::Step(World* w) {
  // todo: implement one iteration of the recursive backtracker
  // hint:
  //   empty stack  -> the maze is done, return false
  //   otherwise, on the cell at the top of the stack:
  //   1. mark it visited;
  //   2. list its visitable neighbors with getVisitables
  //      (already in clockwise order: UP, RIGHT, DOWN, LEFT);
  //   3. none        -> dead end: pop the stack (backtrack);
  //   4. exactly one -> move to it, do not consume a random number;
  //   5. two or more -> consume SeededRandom::next() and pick
  //      next() % visitables.size();
  //   moving = opening the wall between the two cells:
  //     UP    -> w->SetNorth(current, false)
  //     RIGHT -> w->SetEast(current, false)
  //     DOWN  -> w->SetSouth(current, false)
  //     LEFT  -> w->SetWest(current, false)
  //   return true while there is still work (stack not empty after the move)
  // begin solution
  if (stack.empty()) // Maze is done.
    return false;

  visited[stack[stack.size() - 1].x][stack[stack.size() - 1].y] = true; // Mark top of stack as visited.

  std::vector<Point2D> visitable = getVisitables(w, stack.back()); // List top of stack's visitable neighbors.

  if (visitable.empty()) // No neighbors that weren't visited.
  {
    stack.pop_back();
    return true;
  }

  if (visitable.size() == 1) // Only 1 not visited, so don't consume a random number.
    stack.push_back(visitable[0]);
  else // Put the random unvisited neighbor at the top of the stack.
    stack.push_back(visitable[SeededRandom::next() % visitable.size()]);

  // Move
  Point2D direction = stack[stack.size() - 1] - stack[stack.size() - 2];
  Point2D up = {0, -1};
  Point2D down = {0, 1};
  Point2D left = {-1, 0};
  Point2D right = {1, 0};

  if (direction == up)
    w->SetNorth(stack[stack.size() - 2], false);
  else if (direction == down)
    w->SetSouth(stack[stack.size() - 2], false);
  else if (direction == left)
    w->SetWest(stack[stack.size() - 2], false);
  else if (direction == right)
    w->SetEast(stack[stack.size() - 2], false);

  // Stack is not empty after the move.
  return true;
  // end solution
}

std::vector<Point2D> RecursiveBacktrackerExample::getVisitables(World* w, const Point2D& point) {
  // todo: list the unvisited neighbors of point, in clockwise order
  // hint:
  //   candidates in order: UP {x, y-1}, RIGHT {x+1, y}, DOWN {x, y+1}, LEFT {x-1, y}
  //   keep a candidate only if it is inside the grid
  //   (0 <= x < w->GetWidth(), 0 <= y < w->GetHeight()) and not visited
  // begin solution
  std::vector<Point2D> notVisited;
  Point2D candidates[] = {{point.x, point.y - 1},{point.x + 1, point.y},{point.x, point.y + 1},{point.x - 1, point.y}};
  for (const Point2D& neighbor : candidates) {
    if (neighbor.x < 0 || neighbor.x >= w->GetWidth() || neighbor.y < 0 || neighbor.y >= w->GetHeight())
      continue;
    if (visited[neighbor.x][neighbor.y])
      continue;
    notVisited.push_back(neighbor);
  }
  return notVisited;
  // end solution
}
