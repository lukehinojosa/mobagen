#include "CatcherModels.h"
#include "World.h"
#include <algorithm>
#include <array>
#include <chrono>
#include <climits>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <queue>
#include <vector>

namespace {
  using grid::kInf;

  // model search tuning (local tournament against last year's 21 catchers, 3 board seeds)
  constexpr int kModelDepth = 6;         // cat moves searched
  constexpr int kModelBudget = 20000;    // search nodes per move
  constexpr int kEdgeSharePercent = 30;  // general edge models run when this share of blocks is on the border
  constexpr int kPortSignature = 6;      // extra blocks on a student's pattern before trusting their port
  constexpr int kJordanBlocks = 2;       // last blocks the JordanCoolbeth port must explain
  constexpr int kNearestDepth = 8;       // cat moves searched against kNearestExits (6 and 10 won fewer games)
  constexpr int kNearestBudget = 60000;  // search nodes per move against it
  constexpr int kMaxExits = 32;          // nearest exits kNearestExits chooses from
  constexpr int kPortDepth = 8;
  constexpr int kPortBudget = 30000;
  constexpr int kConfirmBlocks = 3;  // last blocks the arena cat's memory needs a model to predict (2 to 5 did as well)
  constexpr int kPortConfirmBlocks = 6;

  // Model search. Each model predicts the catcher's block for a given cat cell; the search looks for a
  // line of cat moves that reaches the border against that prediction.

  enum Model {
    kPlugSecondExit,   // plug an exit next to the cat, else block the exit of the cat's second-best path
    kPlugAltEdge,      // plug, else the nearest cell of an every-other-cell edge wall
    kPlugNearestEdge,  // plug, else the nearest open edge cell
    kAndrew,           // close port of last year's AndrewGenualdo catcher
    kToag,             // close port of last year's TOAG21 catcher
    kJordan,           // exact port of JordanCoolbeth's catcher (this year's entrant)
    kNearestExits,     // block one of the border cells nearest the cat, which one unknown (arena cat's memory only)
    kAylwin,           // exact port of last year's AylwinMorgan catcher (arena cat's memory only)
    kCosmey,           // exact port of last year's Cosmey catcher (arena cat's memory only)
    kModelCount
  };

  struct ModelScratch {
    int side = 0;
    std::vector<int> modelDist, modelQueue;
    std::vector<int> depthDist[std::max({kModelDepth, kNearestDepth, kPortDepth}) + 1];
  };

  ModelScratch& scratchFor(const grid::Board& b) {
    static ModelScratch s;
    if (s.side == b.side) {
      return s;
    }
    s.side = b.side;
    s.modelDist.resize(b.total + 1);
    s.modelQueue.resize(b.total + 1);
    for (auto& d : s.depthDist) {
      d.resize(b.total + 1);
    }
    return s;
  }

  // alternating border cells, as an every-other-cell edge wall would pick them
  bool altEdgeCell(int i, int side) {
    int x = i % side, y = i / side;
    if (y == 0 || y == side - 1) {
      return x % 2 == 1;
    }
    return y % 2 == 1;
  }

  // nearest open border cell to `from` (only alternating ones if asked; `skip` treated as closed)
  int nearestExit(const grid::Board& b, ModelScratch& s, int from, bool altOnly, int skip) {
    auto& d = s.modelDist;
    auto& q = s.modelQueue;
    std::fill(d.begin(), d.end(), kInf);
    int tail = 0, found = -1;
    d[from] = 0;
    q[tail++] = from;
    for (int head = 0; head < tail; head++) {
      int c = q[head];
      if (found >= 0 && d[c] > d[found]) {
        break;
      }
      if (c != from && b.isBorder[c] && (!altOnly || altEdgeCell(c, b.side)) && (found < 0 || c < found)) {
        found = c;
      }
      for (int m : b.neigh[c]) {
        if (m < 0 || !b.open[m] || m == skip || d[m] != kInf) {
          continue;
        }
        d[m] = d[c] + 1;
        q[tail++] = m;
      }
    }
    return found;
  }

  // close ports of last year's two hardest catchers (AndrewGenualdo, TOAG21)
  // They copy the originals' quirks on purpose (last year's SE/SW naming, AndrewGenualdo's overflowed
  // A* heuristic, TOAG21's rounding of negative odd rows), because the quirks decide which cell gets
  // blocked. Off-board counts as blocked.

  struct Pt {
    int x, y;
    bool operator==(const Pt& o) const { return x == o.x && y == o.y; }
    bool operator!=(const Pt& o) const { return !(*this == o); }
  };

  // last year's World directions (SE and SW are swapped relative to this year's CatWorld)
  Pt legNE(Pt p) { return (p.y % 2) ? Pt{p.x + 1, p.y - 1} : Pt{p.x, p.y - 1}; }
  Pt legNW(Pt p) { return (p.y % 2) ? Pt{p.x, p.y - 1} : Pt{p.x - 1, p.y - 1}; }
  Pt legE(Pt p) { return {p.x + 1, p.y}; }
  Pt legW(Pt p) { return {p.x - 1, p.y}; }
  Pt legSE(Pt p) { return (p.y % 2) ? Pt{p.x + 1, p.y + 1} : Pt{p.x, p.y + 1}; }
  Pt legSW(Pt p) { return (p.y % 2) ? Pt{p.x, p.y + 1} : Pt{p.x - 1, p.y + 1}; }
  Pt legDir(int dir, Pt p) {  // Agent::dirToPos order
    switch (dir % 6) {
      case 0:
        return legNE(p);
      case 1:
        return legNW(p);
      case 2:
        return legE(p);
      case 3:
        return legW(p);
      case 4:
        return legSW(p);
      default:
        return legSE(p);
    }
  }

  struct Coords {
    const grid::Board& b;
    int h;
    bool inside(Pt p) const { return p.x >= -h && p.x <= h && p.y >= -h && p.y <= h; }
    int idx(Pt p) const { return (p.y + h) * b.side + p.x + h; }
    Pt pt(int i) const { return {i % b.side - h, i / b.side - h}; }
    bool blocked(Pt p) const { return !inside(p) || !b.open[idx(p)]; }
  };

  // 32-bit wrapping arithmetic, as the original int expressions behave in practice
  int32_t wrap(int64_t v) { return static_cast<int32_t>(static_cast<uint32_t>(v)); }
  int32_t wrapAbs(int32_t v) { return v == INT32_MIN ? v : std::abs(v); }

  // Agent::heuristic(goal, next) with goal still Point2D::INFINITE (INT32_MAX, INT32_MAX)
  double andrewHeuristic(Pt n) {
    const int64_t M = INT32_MAX;
    int32_t t1 = wrapAbs(wrap(M - n.x));
    int32_t t2 = wrapAbs(wrap(int64_t(wrap(M + M)) - n.x - n.y));
    int32_t t3 = wrapAbs(wrap(M - n.y));
    return wrap(int64_t(wrap(int64_t(t1) + t2)) + t3) / 2.0;
  }

  struct AndrewScratch {
    int side = 0;
    std::vector<float> cost;  // < 0 means absent (costSoFar.contains)
    std::vector<Pt> walls;
  };

  // his check(x, y): the cell's index if it's on the board, open and not the cat's cell, else -1
  int andrewLegal(const Coords& bd, Pt cat, Pt p) {
    if (bd.inside(p) && !bd.blocked(p) && p != cat) {
      return bd.idx(p);
    }
    return -1;
  }

  // the edge cells he blocks first when the cat is one step from an edge, in his order
  int andrewPlugCells(Pt cat, int side, std::array<Pt, 10>& cells) {
    int count = 0;
    if (cat.x <= -side + 1) {
      cells[count++] = {-side, cat.y};
      if (cat.y != side - 1) {
        cells[count++] = {-side, cat.y + 1};
      }
      if (cat.y != -side + 1) {
        cells[count++] = {-side, cat.y - 1};
      }
    }
    if (cat.x >= side - 1) {
      cells[count++] = {side, cat.y};
      cells[count++] = {side, cat.y + 1};
      cells[count++] = {side, cat.y - 1};
    }
    if (cat.y <= -side + 1) {
      cells[count++] = {cat.x, -side};
      cells[count++] = {cat.x + 1, -side};
    }
    if (cat.y >= side - 1) {
      cells[count++] = {cat.x, side};
      cells[count++] = {cat.x + 1, side};
    }
    return count;
  }

  // AStar(world, cat, walls): pops until it pops an open wall cell; INT32_MAX when none is reachable
  Pt andrewAStarGoal(const Coords& bd, AndrewScratch& s, Pt cat) {
    struct Node {
      Pt point;
      double priority;
      bool operator>(const Node& o) const { return priority > o.priority; }
    };
    std::priority_queue<Node, std::vector<Node>, std::greater<Node>> frontier;
    std::fill(s.cost.begin(), s.cost.end(), -1.0f);
    frontier.push({cat, 0.0});
    s.cost[bd.idx(cat)] = 0.0f;
    while (!frontier.empty()) {
      Pt current = frontier.top().point;
      frontier.pop();
      size_t wallCounter = 0;
      for (const Pt& w : s.walls) {
        if (bd.blocked(w)) {
          wallCounter++;
          continue;
        }
        if (w == current) {
          return current;
        }
      }
      if (wallCounter == s.walls.size()) {
        break;
      }
      for (int d = 0; d < 6; d++) {
        Pt next = legDir(d, current);
        if (bd.blocked(next)) {
          continue;
        }
        float newCost = s.cost[bd.idx(current)] + 1;
        float& c = s.cost[bd.idx(next)];
        if (c < 0 || newCost < c) {
          c = newCost;
          frontier.push({next, newCost + andrewHeuristic(next)});
        }
      }
    }
    return {INT32_MAX, INT32_MAX};
  }

  int andrewReply(const grid::Board& b, int catIdx) {
    static AndrewScratch s;
    const Coords bd{b, b.side / 2};
    const int side = bd.h;  // their "side" is half the board
    if (s.side != b.side) {
      s.side = b.side;
      s.cost.resize(b.side * b.side);
      s.walls.clear();
      for (int y = -side + 1; y <= side; y += 2) {
        s.walls.push_back({-side, y});
      }
      for (int y = -side; y <= side; y += 2) {
        s.walls.push_back({side, y});
      }
      for (int x = -side + 1; x <= side - 1; x += 2) {
        s.walls.push_back({x, -side});
      }
      for (int x = -side + 1; x <= side - 1; x += 2) {
        s.walls.push_back({x, side});
      }
    }
    const Pt cat = bd.pt(catIdx);

    std::array<Pt, 10> plugs;
    int plugCount = andrewPlugCells(cat, side, plugs);
    for (int i = 0; i < plugCount; i++) {
      int r = andrewLegal(bd, cat, plugs[i]);
      if (r >= 0) {
        return r;
      }
    }

    Pt goal = andrewAStarGoal(bd, s, cat);
    if (goal.x != INT32_MAX && goal != cat) {
      int r = andrewLegal(bd, cat, goal);
      if (r >= 0) {
        return r;
      }
    }
    for (int i = 0; i < 6; i++) {  // the original starts at Random::Range(0, 5); the model uses 0
      int r = andrewLegal(bd, cat, legDir(i, cat));
      if (r >= 0) {
        return r;
      }
    }
    return -1;
  }

  // TOAG21's hex distance: parity = y % 2 truncates, so negative odd rows round the other way
  int toagDistance(Pt a, Pt e) {
    int parity = a.y % 2;
    int sq = a.x - (a.y - parity) / 2, sr = a.y;
    parity = e.y % 2;
    int eq = e.x - (e.y - parity) / 2, er = e.y;
    int vx = sq - eq, vy = sr - er;
    return (std::abs(vx) + std::abs(vx + vy) + std::abs(vy)) / 2;
  }

  int toagFiberDistance(Pt a, Pt e) {
    const int caution = 7;
    int d = toagDistance(a, e);
    if (d == 0) {
      return -50;
    }
    if (d == 1) {
      return 500;
    }
    if (d > caution) {
      d = caution;
    }
    return (caution - d) * 10;
  }

  struct Fiber {
    int netStrength = 0, fromIndex = 0, catDist = 0;
  };

  // the corners of TOAG21's hexagon, and one step of its walk around it
  void toagTurns(int o2, Pt turns[6]) {
    const Pt t[6] = {{o2, 0}, {o2 - o2 / 2, o2}, {-o2 + o2 / 2, o2}, {-o2, 0}, {-o2 + o2 / 2, -o2}, {o2 - o2 / 2, -o2}};
    std::copy(t, t + 6, turns);
  }
  Pt toagStep(int direction, Pt c) {
    switch (direction) {
      case 0:
        return legSW(c);
      case 1:
        return legW(c);
      case 2:
        return legNW(c);
      case 3:
        return legNE(c);
      case 4:
        return legE(c);
      default:
        return legSE(c);
    }
  }

  int toagReply(const grid::Board& b, int catIdx) {
    static std::vector<Fiber> net;
    const Coords bd{b, b.side / 2};
    if (net.size() != b.open.size()) {
      net.assign(b.open.size(), Fiber{});
    }
    const Pt cat = bd.pt(catIdx);
    Pt turns[6];
    toagTurns(bd.h, turns);

    if (!bd.blocked(turns[0])) {
      return bd.idx(turns[0]);
    }
    int direction = 0;
    Pt checking = legSW(turns[0]);
    int fromI = bd.idx(turns[0]);
    Fiber outputF{0, fromI, toagFiberDistance(cat, turns[0])};
    net[fromI] = outputF;

    // walk the hexagon; strength counts consecutive open cells
    while (checking != turns[0]) {
      if (!bd.inside(checking)) {
        return -1;  // the original would read off the board here
      }
      int ci = bd.idx(checking);
      int str = bd.blocked(checking) ? 0 : net[fromI].netStrength + 1;
      net[ci] = Fiber{str, fromI, toagFiberDistance(cat, checking)};
      fromI = ci;
      checking = toagStep(direction, checking);
      if (direction < 5 && checking == turns[direction + 1]) {
        direction++;
      }
    }

    // recursiveStrengthCalc, unrolled: walk back from the last cell to turns[0]
    int base = bd.idx(turns[0]);
    for (int at = fromI;;) {
      Fiber& focus = net[at];
      if (focus.fromIndex == at) {
        break;
      }
      Fiber& from = net[base];
      int strBonus = 1;
      if (from.netStrength > 0) {
        strBonus++;
      }
      if (net[focus.fromIndex].netStrength > 0) {
        strBonus++;
      }
      if (focus.netStrength > from.netStrength + strBonus) {
        focus.netStrength = from.netStrength + strBonus;
      }
      if (focus.netStrength != 0 && (focus.netStrength + focus.catDist > outputF.netStrength + outputF.catDist || outputF.netStrength == 0)) {
        outputF.netStrength = focus.netStrength;
        outputF.catDist = focus.catDist;
        outputF.fromIndex = at;
      }
      base = at;
      at = focus.fromIndex;
    }

    Pt out = bd.pt(outputF.fromIndex);
    for (int guard = 0; bd.blocked(out) && guard < 12; guard++) {
      out = toagStep(direction, cat);
      direction = (direction + 1) % 6;
    }
    return bd.blocked(out) || out == cat ? -1 : bd.idx(out);
  }

  // libc++'s std::priority_queue with a greater-than comparator (top: the smallest key), which last year's
  // AylwinMorgan and Cosmey catchers search with: their ties fall to this heap's order, so the ports copy it.
  // Node needs a `key`.
  namespace libcxxHeap {
    template <class Node> bool below(const Node& a, const Node& b) { return a.key > b.key; }

    // push_heap (__sift_up) for the last of the first len elements
    template <class Node> void siftUp(std::vector<Node>& heap, int len) {
      if (len <= 1) {
        return;
      }
      int last = len - 1, parent = (len - 2) / 2;
      if (!below(heap[parent], heap[last])) {
        return;
      }
      const Node t = heap[last];
      do {
        heap[last] = heap[parent];
        last = parent;
        if (parent == 0) {
          break;
        }
        parent = (parent - 1) / 2;
      } while (below(heap[parent], t));
      heap[last] = t;
    }

    template <class Node> void push(std::vector<Node>& heap, const Node& n) {
      heap.push_back(n);
      siftUp(heap, static_cast<int>(heap.size()));
    }

    // pop_heap (__floyd_sift_down, then __sift_up) and pop_back
    template <class Node> Node pop(std::vector<Node>& heap) {
      const int len = static_cast<int>(heap.size());
      const Node top = heap[0];
      if (len > 1) {
        int hole = 0, child = 0;
        while (true) {
          child = 2 * child + 1;
          if (child + 1 < len && below(heap[child], heap[child + 1])) {
            child++;
          }
          heap[hole] = heap[child];
          hole = child;
          if (child > (len - 2) / 2) {
            break;
          }
        }
        if (hole != len - 1) {
          heap[hole] = heap[len - 1];
          siftUp(heap, hole + 1);
        }
      }
      heap.pop_back();
      return top;
    }
  }  // namespace libcxxHeap

  // AylwinMorgan's catcher (last year, AylwinMorgan/mobagen b3caadc, built by the arena from his own World): the
  // same block on every turn the cat can reach the border (1,009 of 1,009 against his arena bot), with flat
  // arrays. His A* (step cost 1, heuristic: rings to the border) stops at the first border cell it takes off the
  // queue, ties in his priority_queue's (libc++'s) heap order (libcxxHeap): first queued or last queued made
  // only 90% and 80% of his blocks. Then he blocks the exit next to the cat; else the path's cell before the exit
  // when it is next to a corner, or on a two step path to a left or right edge exit with another open border
  // cell next to it; else, walking his border list outward from the exit, the first open cell with no blocked
  // list neighbor. -1 once no border is reachable (he then blocks a random cell next to the cat).
  namespace aylwin {
    struct Node {
      int cell, acc, key;
    };

    // his border list (clockwise from the top left, without (-h, -h) and (-h, h)) and his index into it, which is
    // off by one or two from the list's own order; both decide which cell he blocks
    const std::vector<Pt>& borders(int side) {
      static std::vector<Pt> list;
      static int cached = 0;
      if (cached != side) {
        cached = side;
        list.clear();
        const int h = side / 2;
        for (int x = -h + 1; x < h; x++) {
          list.push_back({x, -h});
        }
        for (int y = -h; y < h; y++) {
          list.push_back({h, y});
        }
        for (int x = h; x > -h; x--) {
          list.push_back({x, h});
        }
        for (int y = h - 1; y > -h; y--) {
          list.push_back({-h, y});
        }
      }
      return list;
    }
    int borderIndex(Pt p, int side) {
      const int h = side / 2;
      if (p.y == -h) {
        return p.x + h;
      }
      if (p.x == h) {
        return side + p.y + h;
      }
      if (p.y == h) {
        return 3 * side - p.x - h - 1;
      }
      return 4 * side - p.y - h - 2;
    }

    int reply(const grid::Board& b, ModelScratch& s, int catIdx) {
      const Coords bd{b, b.side / 2};
      const int h = bd.h;
      auto& cameFrom = s.modelDist;
      auto& state = s.modelQueue;  // 0 unseen, 1 queued, 2 done
      std::fill(cameFrom.begin(), cameFrom.end(), -1);
      std::fill(state.begin(), state.end(), 0);
      auto rings = [&](Pt p) { return h - std::max(std::abs(p.x), std::abs(p.y)); };
      static std::vector<Node> q;
      q.clear();
      q.push_back({catIdx, 0, rings(bd.pt(catIdx))});
      state[catIdx] = 1;
      int exit = -1;
      while (!q.empty() && exit < 0) {
        const Node cur = libcxxHeap::pop(q);
        state[cur.cell] = 2;
        const Pt p = bd.pt(cur.cell);
        const Pt around[6] = {legNE(p), legNW(p), legE(p), legW(p), legSE(p), legSW(p)};  // his neighbors order
        for (const Pt& n : around) {
          if (!bd.inside(n)) {
            exit = cur.cell;  // a border cell: he stops once its neighbors are queued
            continue;
          }
          const int ni = bd.idx(n);
          if (ni == catIdx || !b.open[ni] || state[ni] != 0) {
            continue;
          }
          cameFrom[ni] = cur.cell;
          libcxxHeap::push(q, Node{ni, cur.acc + 1, cur.acc + 1 + rings(n)});
          state[ni] = 1;
        }
      }
      if (exit < 0 || exit == catIdx) {
        return -1;  // no border reachable: he blocks a random cell next to the cat
      }
      if (cameFrom[exit] == catIdx) {
        return exit;  // the cat is next to its exit
      }
      const int prev = cameFrom[exit];  // his catPath[1]
      const bool twoSteps = cameFrom[prev] == catIdx;
      const Pt target = bd.pt(exit), before = bd.pt(prev);
      if (std::abs(before.x) == h - 1 && std::abs(before.y) == h - 1) {
        return prev;  // the path turns at a corner
      }
      const auto& list = borders(b.side);
      const int count = static_cast<int>(list.size());
      const int t = borderIndex(target, b.side);
      auto at = [&](int k) { return list[k % count]; };
      auto open = [&](Pt c) { return b.open[bd.idx(c)] != 0; };
      if (twoSteps && std::abs(target.x) == h) {
        for (int i = -2; i <= 2; i++) {
          const Pt c = at(t + count + i);
          if (c != target && open(c)) {
            const int ci = bd.idx(c);
            for (int nb : b.neigh[prev]) {
              if (nb == ci) {
                return prev;  // a second exit next to the cell before the exit
              }
            }
          }
        }
      }
      // outward from the exit along his list, alternating sides: the first open cell with no blocked list neighbor
      for (int i = 0; i < count; i++) {
        const int k = t + count + (i % 2 == 0 ? i / 2 : -(i / 2));
        if (!open(at(k))) {
          continue;
        }
        const int walls = !open(at(k - 1)) + !open(at(k + 1));
        if (walls == 0 || (walls < 2 && at(k) == target && twoSteps)) {
          if (walls > 0) {
            if (open(at(k - 1))) {
              return bd.idx(at(k - 1));
            }
            if (open(at(k + 1))) {
              return bd.idx(at(k + 1));
            }
          }
          return bd.idx(at(k));
        }
      }
      return exit;
    }
  }  // namespace aylwin

  // Cosmey's catcher (last year, Cosmey/mobagenReeceEnthoven 3443e4a, built by the arena from his own World): his
  // generatePath is a best-first search from the cat, priority 0.99 * steps - rings from the center (in float,
  // as his code computes it), that stops at the first cell it takes off the queue with a neighbor off the board;
  // ties in libc++'s heap order (libcxxHeap). He blocks that exit when it is next to the cat; else the exit of a
  // second search with the first exit closed off, or the first exit when the second finds none. -1 once no
  // border is reachable (he then blocks a random cell next to the cat).
  namespace cosmey {
    struct Node {
      int cell;
      float key;
      int steps;
    };

    // his calculateHeuristic
    float priority(Pt p, int steps) {
      float scaled = steps * 0.99f;
      if (steps - scaled >= 1) {
        scaled = static_cast<float>(steps);
      }
      return -static_cast<float>(std::max(std::abs(p.x), std::abs(p.y))) + scaled;
    }

    // his generatePath's exit (-1 if none), never through `closed`; leaves each cell's parent in s.modelDist
    int exitOf(const grid::Board& b, ModelScratch& s, int catIdx, int closed) {
      const Coords bd{b, b.side / 2};
      auto& cameFrom = s.modelDist;
      auto& state = s.modelQueue;  // 0 unseen, 1 queued, 2 done (or closed)
      std::fill(cameFrom.begin(), cameFrom.end(), -1);
      std::fill(state.begin(), state.end(), 0);
      if (closed >= 0) {
        state[closed] = 2;
      }
      static std::vector<Node> q;
      q.clear();
      q.push_back({catIdx, 0.0f, 0});
      state[catIdx] = 1;
      while (!q.empty()) {
        const Node cur = libcxxHeap::pop(q);
        state[cur.cell] = 2;
        const Pt p = bd.pt(cur.cell);
        const Pt around[6] = {legNE(p), legNW(p), legE(p), legW(p), legSW(p), legSE(p)};  // his neighbors order
        for (const Pt& n : around) {
          if (!bd.inside(n)) {
            return cur.cell;  // a border cell
          }
          const int ni = bd.idx(n);
          if (!b.open[ni] || state[ni] != 0) {
            continue;
          }
          cameFrom[ni] = cur.cell;
          libcxxHeap::push(q, Node{ni, priority(n, cur.steps + 1), cur.steps + 1});
          state[ni] = 1;
        }
      }
      return -1;
    }

    int reply(const grid::Board& b, ModelScratch& s, int catIdx) {
      const int exit = exitOf(b, s, catIdx, -1);
      if (exit < 0 || exit == catIdx || s.modelDist[exit] == catIdx) {
        return exit == catIdx ? -1 : exit;
      }
      const int second = exitOf(b, s, catIdx, exit);
      return second >= 0 ? second : exit;
    }
  }  // namespace cosmey

  // JordanCoolbeth's catcher (dewdrop-ripple/GPR-340-mobagen e0c007a) while the cat can reach the border:
  // his generatePath finds the first border cell a BFS from the cat reaches (neighbors in his order NE, NW,
  // SE, SW, E, W). Next to the cat, he blocks it; otherwise he walks the border from it, a step clockwise
  // then a step counterclockwise, and blocks the first open cell. -1 once no border is reachable (he then
  // boxes the cat in).
  int jordanReply(const grid::Board& b, ModelScratch& s, int cat) {
    static constexpr int kOrder[6] = {0, 1, 5, 4, 2, 3};  // his order, as CatWorld::neighbors indices
    auto& parent = s.modelDist;
    auto& q = s.modelQueue;
    std::fill(parent.begin(), parent.end(), -1);
    parent[cat] = cat;
    int tail = 0, exit = -1;
    q[tail++] = cat;
    for (int head = 0; head < tail && exit < 0; head++) {
      const int c = q[head];
      for (int k : kOrder) {
        const int m = b.neigh[c][k];
        if (m < 0 || !b.open[m] || parent[m] >= 0) {
          continue;
        }
        parent[m] = c;
        if (b.isBorder[m]) {
          exit = m;
          break;
        }
        q[tail++] = m;
      }
    }
    if (exit < 0) {
      return -1;
    }
    if (parent[exit] == cat) {
      return exit;  // the cat is next to it
    }
    const int h = b.side / 2;
    Pt cw{exit % b.side - h, exit / b.side - h}, ccw = cw;
    auto openAt = [&](Pt p) {
      const int i = (p.y + h) * b.side + p.x + h;
      return i != cat && b.open[i] ? i : -1;
    };
    for (int guard = 0; guard < 8 * b.side; guard++) {
      if (cw.x == -h && cw.y != h) {
        cw.y++;
      } else if (cw.y == h && cw.x != h) {
        cw.x++;
      } else if (cw.x == h && cw.y != -h) {
        cw.y--;
      } else if (cw.y == -h && cw.x != -h) {
        cw.x--;
      }
      if (openAt(cw) >= 0) {
        return openAt(cw);
      }
      if (ccw.x == -h && ccw.y != -h) {
        ccw.y--;
      } else if (ccw.y == -h && ccw.x != h) {
        ccw.x++;
      } else if (ccw.x == h && ccw.y != h) {
        ccw.y++;
      } else if (ccw.y == h && ccw.x != -h) {
        ccw.x--;
      }
      if (openAt(ccw) >= 0) {
        return openAt(ccw);
      }
    }
    return -1;
  }

  // kNearestExits: the open border cells nearest the cat (fewest steps through open cells) into `out`; returns
  // how many. Last year's edge catchers mostly block one of these, each breaking ties its own way (A* and
  // priority queues whose order no simple rule reproduces).
  int nearestExitSet(const grid::Board& b, ModelScratch& s, int cat, int (&out)[kMaxExits]) {
    auto& d = s.modelDist;
    auto& q = s.modelQueue;
    std::fill(d.begin(), d.end(), -1);
    d[cat] = 0;
    int tail = 0, found = 0, reach = -1;
    q[tail++] = cat;
    for (int head = 0; head < tail; head++) {
      const int c = q[head];
      if (reach >= 0 && d[c] >= reach) {
        break;  // every border cell at that distance is found
      }
      for (int m : b.neigh[c]) {
        if (m < 0 || !b.open[m] || d[m] >= 0) {
          continue;
        }
        d[m] = d[c] + 1;
        if (b.isBorder[m]) {
          reach = d[m];
          if (found < kMaxExits) {
            out[found++] = m;
          }
          continue;
        }
        q[tail++] = m;
      }
    }
    return found;
  }

  // the model's block with the cat standing on `cat`, or -1
  int modelReply(const grid::Board& b, ModelScratch& s, int model, int cat) {
    if (model == kJordan) {
      return jordanReply(b, s, cat);
    }
    if (model == kAylwin) {
      return aylwin::reply(b, s, cat);
    }
    if (model == kCosmey) {
      return cosmey::reply(b, s, cat);
    }
    if (model == kAndrew) {
      return andrewReply(b, cat);
    }
    if (model == kToag) {
      return toagReply(b, cat);
    }
    for (int n : b.neigh[cat]) {
      if (n >= 0 && b.open[n] && b.isBorder[n]) {
        return n;  // plug an exit next to the cat
      }
    }
    if (model == kPlugSecondExit) {
      int e1 = nearestExit(b, s, cat, false, -1);
      if (e1 < 0) {
        return -1;
      }
      int e2 = nearestExit(b, s, cat, false, e1);
      return e2 >= 0 ? e2 : e1;
    }
    if (model == kPlugAltEdge) {
      int e = nearestExit(b, s, cat, true, -1);
      return e >= 0 ? e : nearestExit(b, s, cat, false, -1);
    }
    return nearestExit(b, s, cat, false, -1);
  }

  bool trappedAt(const grid::Board& b, int cat) {
    for (int n : b.neigh[cat]) {
      if (n >= 0 && b.open[n]) {
        return false;
      }
    }
    return true;
  }

  // cat on `cat` to move, `depth` cat moves left: is there a line that beats this model?
  // only cells within depth - 1 of the border matter, so the distance pass stops there
  bool beatsModel(grid::Board& b, ModelScratch& s, int cat, int model, int depth, int& budget) {
    if (depth <= 0 || --budget < 0) {
      return false;
    }
    auto& d = s.depthDist[depth];
    grid::bfsDistanceWithin(b, d, depth - 1);
    for (int n : b.neigh[cat]) {
      if (n < 0 || !b.open[n] || d[n] > depth - 1) {
        continue;  // too far to arrive in time
      }
      if (b.isBorder[n]) {
        return true;
      }
      int blk = modelReply(b, s, model, n);
      if (blk >= 0 && blk != n && b.open[blk]) {
        grid::setOpen(b, blk, false);
      } else {
        blk = -1;
      }
      bool win = !trappedAt(b, n) && beatsModel(b, s, n, model, depth - 1, budget);
      if (blk >= 0) {
        grid::setOpen(b, blk, true);
      }
      if (win) {
        return true;
      }
    }
    return false;
  }

  // Search against kNearestExits. Demanding a line that beats every choice of exit (as beatsModel would)
  // finds too few: the real catchers break ties one fixed way, rarely the worst for the cat. So each choice
  // is taken as equally likely and the cat goes for the best chance.

  double nearestExitOdds(grid::Board& b, ModelScratch& s, int cat, int depth, int& budget);

  // the cat just moved to `cat`: its chance to win within `depth` - 1 more moves if the catcher blocks one of
  // its nearest exits at random
  double oddsAfterMove(grid::Board& b, ModelScratch& s, int cat, int depth, int& budget) {
    int exits[kMaxExits];
    const int count = nearestExitSet(b, s, cat, exits);
    if (count == 0) {
      return 0.0;  // sealed in
    }
    double sum = 0.0;
    for (int k = 0; k < count; k++) {
      grid::setOpen(b, exits[k], false);
      if (!trappedAt(b, cat)) {
        sum += nearestExitOdds(b, s, cat, depth - 1, budget);
      }
      grid::setOpen(b, exits[k], true);
    }
    return sum / count;
  }

  // the cat on `cat` to move: its best chance to reach the border within `depth` moves (only cells within
  // depth - 1 of the border matter, as in beatsModel)
  double nearestExitOdds(grid::Board& b, ModelScratch& s, int cat, int depth, int& budget) {
    if (depth <= 0 || --budget < 0) {
      return 0.0;
    }
    auto& d = s.depthDist[depth];
    grid::bfsDistanceWithin(b, d, depth - 1);
    double best = 0.0;
    for (int n : b.neigh[cat]) {
      if (n < 0 || !b.open[n] || d[n] > depth - 1) {
        continue;  // too far to arrive in time
      }
      if (b.isBorder[n]) {
        return 1.0;
      }
      best = std::max(best, oddsAfterMove(b, s, n, depth, budget));
      if (best >= 1.0) {
        break;
      }
    }
    return best;
  }

  // Reading the catcher's style from the board
  // On the leaderboard the cat has no memory, but the catcher's past blocks stay on the board.

  // share of blocked cells that sit on the border, in percent
  int borderBlockPercent(const grid::Board& b, const ModelScratch& s) {
    int onBorder = 0, all = 0;
    for (int i = 0; i < b.total; i++) {
      if (!b.open[i]) {
        all++;
        onBorder += b.isBorder[i];
      }
    }
    return all ? onBorder * 100 / all : 0;
  }

  // TOAG21's wall: replay its walk around the hexagon (board independent)
  const std::vector<uint8_t>& toagWallCells(int side) {
    static std::vector<uint8_t> cells;
    static int cachedSide = 0;
    if (cachedSide == side) {
      return cells;
    }
    cachedSide = side;
    int h = side / 2;
    cells.assign(side * side, 0);
    auto mark = [&](Pt p) {
      if (p.x >= -h && p.x <= h && p.y >= -h && p.y <= h) {
        cells[(p.y + h) * side + p.x + h] = 1;
      }
    };
    Pt turns[6];
    toagTurns(h, turns);
    mark(turns[0]);
    int direction = 0;
    Pt c = legSW(turns[0]);
    for (int guard = 0; c != turns[0] && guard < 4 * side * side; guard++) {
      mark(c);
      c = toagStep(direction, c);
      if (direction < 5 && c == turns[direction + 1]) {
        direction++;
      }
    }
    return cells;
  }

  // blocks on a footprint beyond what random obstacles would explain
  int footprintExcess(const grid::Board& b, const std::vector<uint8_t>& footprint) {
    int total = b.total, size = 0, blocked = 0, blockedOn = 0;
    for (int i = 0; i < total; i++) {
      size += footprint[i];
      if (!b.open[i]) {
        blocked++;
        blockedOn += footprint[i];
      }
    }
    return blockedOn - blocked * size / total;
  }

  // AndrewGenualdo's wall pattern: blocks on his every-other edge cells minus blocks on the other edge
  // cells (random obstacles hit both about equally; his walls only hit the first)
  int andrewPatternExcess(const grid::Board& b) {
    int h = b.side / 2, onWalls = 0, onOther = 0;
    for (int i : b.borders) {
      if (b.open[i]) {
        continue;
      }
      int x = i % b.side - h, y = i / b.side - h;
      bool wall = (x == -h && y != -h && (y + h) % 2 == 1) || (x == h && (y + h) % 2 == 0)
                  || ((y == -h || y == h) && x != -h && x != h && (x + h) % 2 == 1);
      (wall ? onWalls : onOther)++;
    }
    return onWalls - onOther;
  }

  // could JordanCoolbeth's catcher have made the last `blocks` blocks? The last came with the cat on the cell
  // it is on now: some blocked border cell must be his reply there with that cell open again. The one before
  // came with the cat on a neighbor, with both cells open, and so on. Last year's edge catchers often fit too
  // (their blocks near the cat's exit look alike), which is why modelChoice only asks on the leaderboard.
  bool jordanFits(grid::Board& b, ModelScratch& s, int cat, int blocks) {
    for (int i : b.borders) {
      if (b.open[i]) {
        continue;
      }
      grid::setOpen(b, i, true);
      bool fits = jordanReply(b, s, cat) == i;
      if (fits && blocks > 1) {
        fits = false;
        for (int p : b.neigh[cat]) {
          if (p >= 0 && b.open[p] && !b.isBorder[p] && jordanFits(b, s, p, blocks - 1)) {
            fits = true;
            break;
          }
        }
      }
      grid::setOpen(b, i, false);
      if (fits) {
        return true;
      }
    }
    return false;
  }

  // does the board look like this model's catcher is playing?
  bool modelFitsBoard(const grid::Board& b, const ModelScratch& s, int model) {
    if (model == kJordan) {
      return false;  // jordanFits decides
    }
    if (model == kNearestExits || model == kAylwin || model == kCosmey) {
      return false;  // only the arena cat's memory can tell (see predictedBy)
    }
    if (model == kAndrew) {
      return andrewPatternExcess(b) >= kPortSignature;
    }
    if (model == kToag) {
      int h = b.side / 2;
      if (b.open[h * b.side + b.side - 1]) {
        return false;  // TOAG21 always blocks (h, 0) first
      }
      return footprintExcess(b, toagWallCells(b.side)) >= kPortSignature;
    }
    return borderBlockPercent(b, s) >= kEdgeSharePercent;
  }

}  // namespace

// Survival against AaronArchambault's catcher (his 2026-10-07 version, 2e882a2)
//
// aaron::move is his Catcher::Move rewritten to pick exactly his block with less work:
//  - sealed in (no escape route): every open cell is tried, the one whose block leaves the cat's biggest run
//    smallest wins, then the fewest exits (first in board order on ties);
//  - first look: every open cell scored as a block by the cat's best reply (two-distance, BFS distance,
//    shortest-path count, region size when sealed), stable sorted, board order on ties;
//  - closing in (new in 2e882a2): when his best block leaves the cat no finite two-distance, the blocks that do
//    the same, nearest the cat first (steps ignoring blocks; then his order): the first of 5 whose worst case
//    (as in the lookahead) still leaves the cat none;
//  - lookahead: the 5 best blocks, the cat's 2 best replies, the best follow-up among cells within 3 steps of
//    the reply plus the 10 best first-look blocks; the block with the best worst case wins.
// The shortcuts, each giving exactly his result:
//  - a block only changes the score at a position if it lies on a cell its neighbors' values are built from
//    (reachable from them by strictly decreasing value; his passes run through the cat's cell), and only from
//    the neighbors tied for the best two-distance unless the block raises it; and it only reruns that pass;
//  - a block on a neighbor's shortest paths removes (paths from the neighbor to it) * (its paths to the border)
//    of them, so the BFS pass only runs when it removes them all, and then only when that neighbor was the
//    closest (it can only get farther) and the score could still matter to the caller;
//  - only his 10 best first-look blocks are ever used, so the rest never get the BFS pass and only those 10 are
//    kept in order; a follow-up search only needs its best score, so only blocks tied for the best two-distance
//    get it;
//  - a lookahead block stops once it can't beat the best block's worst case, and its second reply's search
//    once it can't lower the first's.
// Over 45,000 turns against every cat in the harness it played his block every time. Timed against his code on
// the same 8,610 positions (same block on every one): about 12 times faster in ordinary games, and 18 times
// faster where the cat is held, where the survival playouts below spend their time.
namespace {
  namespace aaron {
    constexpr int kUnreachable = 1 << 29;
    constexpr int kTopBlocks = 5;
    constexpr int kCatReplies = 2;
    constexpr int kNearSteps = 3;
    constexpr int kTopFollowUps = 10;
    constexpr int kMaxFollowUps = 1 + 3 * kNearSteps * (kNearSteps + 1) + kTopFollowUps;

    struct Score {
      int twoDist;
      int dist;
      double paths;
      int area;

      bool beats(const Score& o) const {
        if (twoDist != o.twoDist) {
          return twoDist > o.twoDist;
        }
        if (dist != o.dist) {
          return dist > o.dist;
        }
        if (paths != o.paths) {
          return paths < o.paths;
        }
        return area < o.area;
      }
    };

    const Score kCaught{kUnreachable + 1, kUnreachable, 0.0, 0};
    const Score kEscaped{-1, -1, 0.0, 0};

    // a candidate block at a position: its best two-distance and the neighbors' two-distances after it
    struct Trial {
      int cell;
      int twoDist;
      int two[6];
    };

    struct Ranked {
      Score score;
      int cell;
    };

    struct Model {
      int side = 0;
      int cells = 0;
      std::vector<int> neigh;  // 6 per cell, -1 off the board, in CatWorld::neighbors order
      std::vector<int> border;
      std::vector<uint8_t> blocked;
      std::vector<int> dist, two;  // the current board's passes
      std::vector<double> paths;
      std::vector<int> trialDist, trialTwo;  // after a trial block
      std::vector<double> trialPaths;
      std::vector<int> count, queue, mark, steps, scoreDist;
      std::vector<double> scorePaths;
      std::vector<uint8_t> inTwo, inDist, inTwoMin, inDistMin, isNb, nearPos, inRegion;
      std::vector<double> via;                   // 6 per cell: shortest paths from each neighbor of the scored position
      uint8_t viaReady[6] = {0, 0, 0, 0, 0, 0};  // which via tables the last markSupport has built so far
      int baseMin = 0;                           // the scored position's best two-distance
      int stamp = 0;
      std::vector<Trial> trials, followTrials;
      std::vector<Ranked> ranked;
      std::vector<int> twos;
    };

    Model& modelFor(int side) {
      static Model m;
      if (m.side == side) {
        return m;
      }
      m.side = side;
      m.cells = side * side;
      const int h = side / 2;
      m.neigh.assign(m.cells * 6, -1);
      m.border.clear();
      for (int i = 0; i < m.cells; i++) {
        const Point2D p = {i % side - h, i / side - h};
        const auto around = CatWorld::neighbors(p);
        for (int k = 0; k < 6; k++) {
          if (std::abs(around[k].x) <= h && std::abs(around[k].y) <= h) {
            m.neigh[i * 6 + k] = (around[k].y + h) * side + around[k].x + h;
          }
        }
        if (std::abs(p.x) == h || std::abs(p.y) == h) {
          m.border.push_back(i);
        }
      }
      for (auto* v : {&m.dist, &m.two, &m.trialDist, &m.trialTwo, &m.count, &m.queue, &m.mark, &m.steps, &m.scoreDist}) {
        v->assign(m.cells, 0);
      }
      for (auto* v : {&m.blocked, &m.inTwo, &m.inDist, &m.inTwoMin, &m.inDistMin, &m.isNb, &m.nearPos, &m.inRegion}) {
        v->assign(m.cells, 0);
      }
      m.paths.assign(m.cells, 0.0);
      m.trialPaths.assign(m.cells, 0.0);
      m.scorePaths.assign(m.cells, 0.0);
      m.via.assign(6 * m.cells, 0.0);
      m.stamp = 0;
      m.trials.resize(m.cells);
      m.followTrials.resize(kMaxFollowUps);
      m.ranked.resize(m.cells);
      m.twos.resize(m.cells);
      return m;
    }

    // his computeEscapeField: BFS distance from the open border cells and the number of shortest paths. With
    // `wanted` > 0 (those neighbors of the scored position marked in isNb), it stops once their distances and
    // counts are final (a count is complete once every cell one step closer has left the queue); returns false
    // if one of them can't reach the border (the pass then ran to the end).
    bool escapeField(Model& m, std::vector<int>& distVec, std::vector<double>& pathsVec, int wanted) {
      int* dist = distVec.data();
      double* paths = pathsVec.data();
      int* q = m.queue.data();
      const uint8_t* isNb = m.isNb.data();
      const uint8_t* blocked = m.blocked.data();
      const int* neigh = m.neigh.data();
      for (int i = 0; i < m.cells; i++) {
        dist[i] = kUnreachable;
        paths[i] = 0.0;
      }
      int tail = 0;
      int left = wanted, far = -1;
      for (int i : m.border) {
        if (!blocked[i]) {
          dist[i] = 0;
          paths[i] = 1.0;
          q[tail++] = i;
          if (wanted > 0 && isNb[i]) {
            left--;
            far = 0;
          }
        }
      }
      for (int head = 0; head < tail; head++) {
        const int c = q[head];
        if (wanted > 0 && left == 0 && dist[c] >= far) {
          break;
        }
        const int* around = neigh + c * 6;
        for (int k = 0; k < 6; k++) {
          const int n = around[k];
          if (n < 0 || blocked[n]) {
            continue;
          }
          if (dist[n] == kUnreachable) {
            dist[n] = dist[c] + 1;
            paths[n] = paths[c];
            q[tail++] = n;
            if (wanted > 0 && isNb[n]) {
              left--;
              far = std::max(far, dist[n]);
            }
          } else if (dist[n] == dist[c] + 1) {
            paths[n] += paths[c];
          }
        }
      }
      return left <= 0;
    }

    // his computeTwoDistance. With pos >= 0 it stops once the best value among pos's open neighbors is final
    // (cells leave the queue in nondecreasing value); neighbors still unset then have a bigger value, which
    // never decides the score.
    void twoDistance(Model& m, std::vector<int>& valueVec, int pos) {
      int* value = valueVec.data();
      int* settled = m.count.data();
      int* q = m.queue.data();
      uint8_t* isNb = m.isNb.data();
      const uint8_t* blocked = m.blocked.data();
      const int* neigh = m.neigh.data();
      for (int i = 0; i < m.cells; i++) {
        value[i] = kUnreachable;
        settled[i] = 0;
      }
      int tail = 0;
      for (int i : m.border) {
        if (!blocked[i]) {
          value[i] = 0;
          q[tail++] = i;
        }
      }
      int best = kUnreachable + 1;  // never reached: run to the end
      const int* around = pos >= 0 ? neigh + pos * 6 : nullptr;
      if (pos >= 0) {
        best = kUnreachable;
        for (int k = 0; k < 6; k++) {
          const int n = around[k];
          if (n >= 0 && !blocked[n]) {
            isNb[n] = 1;
            best = std::min(best, value[n]);
          }
        }
      }
      for (int head = 0; head < tail; head++) {
        const int c = q[head];
        if (value[c] >= best) {
          break;
        }
        const int* next = neigh + c * 6;
        for (int k = 0; k < 6; k++) {
          const int n = next[k];
          if (n < 0 || blocked[n] || value[n] != kUnreachable) {
            continue;
          }
          settled[n]++;
          if (settled[n] == 2) {
            value[n] = value[c] + 1;
            q[tail++] = n;
            if (isNb[n] && value[n] < best) {
              best = value[n];
            }
          }
        }
      }
      if (pos >= 0) {
        for (int k = 0; k < 6; k++) {
          if (around[k] >= 0) {
            isNb[around[k]] = 0;
          }
        }
      }
    }

    int floodFill(Model& m, int start) {
      m.stamp++;
      int* mark = m.mark.data();
      int* q = m.queue.data();
      const uint8_t* blocked = m.blocked.data();
      const int* neigh = m.neigh.data();
      int tail = 0;
      q[tail++] = start;
      mark[start] = m.stamp;
      for (int head = 0; head < tail; head++) {
        const int* around = neigh + q[head] * 6;
        for (int k = 0; k < 6; k++) {
          const int n = around[k];
          if (n < 0 || blocked[n] || mark[n] == m.stamp) {
            continue;
          }
          mark[n] = m.stamp;
          q[tail++] = n;
        }
      }
      return tail;
    }

    // his evaluate's score at pos from the given passes (with the region size when pos is sealed in)
    Score scoreFrom(Model& m, int pos, const int* two, const int* dist, const double* paths, bool sealed) {
      const int area = sealed ? floodFill(m, pos) : 0;
      Score reply = kCaught;
      const int* around = m.neigh.data() + pos * 6;
      for (int k = 0; k < 6; k++) {
        const int n = around[k];
        if (n < 0 || m.blocked[n]) {
          continue;
        }
        const Score s{two[n], dist[n], paths[n], area};
        if (reply.beats(s)) {
          reply = s;
        }
      }
      return reply;
    }

    // the best two-distance among pos's open neighbors (his Score.twoDist), kCaught's if none is open
    int bestTwo(const Model& m, int pos, const int* two) {
      int best = kCaught.twoDist;
      const int* around = m.neigh.data() + pos * 6;
      for (int k = 0; k < 6; k++) {
        const int n = around[k];
        if (n >= 0 && !m.blocked[n]) {
          best = std::min(best, two[n]);
        }
      }
      return best;
    }

    // the cells whose block can change the two-distance / BFS distance and path count at pos's neighbors (from
    // this board's passes): the open cells reachable from them by strictly decreasing value. inTwo / inDist
    // start from every open neighbor, inTwoMin / inDistMin only from those with the best two-distance (the only
    // ones that decide the score while it stays the best, since blocks only raise values). The shortest path
    // counts from each neighbor (via, for the path-count shortcut) are left for ensureVia.
    void markSupport(Model& m, int pos) {
      int* q = m.queue.data();
      const int* neigh = m.neigh.data();
      const int* around = neigh + pos * 6;
      m.baseMin = bestTwo(m, pos, m.two.data());
      for (int pass = 0; pass < 4; pass++) {
        const int* v = pass % 2 == 0 ? m.two.data() : m.dist.data();
        uint8_t* in = pass == 0 ? m.inTwo.data() : pass == 1 ? m.inDist.data() : pass == 2 ? m.inTwoMin.data() : m.inDistMin.data();
        std::memset(in, 0, m.cells);
        int tail = 0;
        for (int k = 0; k < 6; k++) {
          const int n = around[k];
          if (n >= 0 && !m.blocked[n] && !in[n] && (pass < 2 || m.two[n] == m.baseMin)) {
            in[n] = 1;
            q[tail++] = n;
          }
        }
        // every neighbor's two-distance infinite: no block can change it, only blocking a neighbor matters
        if (pass % 2 == 0 && m.baseMin >= kUnreachable) {
          continue;
        }
        for (int head = 0; head < tail; head++) {
          const int c = q[head];
          const int* next = neigh + c * 6;
          for (int k = 0; k < 6; k++) {
            const int x = next[k];
            if (x >= 0 && !m.blocked[x] && !in[x] && v[x] < v[c]) {
              in[x] = 1;
              q[tail++] = x;
            }
          }
        }
      }
      for (int k = 0; k < 6; k++) {
        m.viaReady[k] = 0;  // built on first use (ensureVia)
      }
    }

    // the two-distance part of blocking t.cell with the cat on pos (markSupport ran for pos)
    void trialTwo(Model& m, int pos, Trial& t) {
      const int* two = m.two.data();
      if (m.inTwoMin[t.cell] && m.baseMin < kUnreachable) {
        m.blocked[t.cell] = 1;
        twoDistance(m, m.trialTwo, pos);
        m.blocked[t.cell] = 0;
        two = m.trialTwo.data();
      }
      const int* around = m.neigh.data() + pos * 6;
      for (int k = 0; k < 6; k++) {
        t.two[k] = around[k] >= 0 ? two[around[k]] : 0;
      }
      m.blocked[t.cell] = 1;
      t.twoDist = bestTwo(m, pos, two);
      m.blocked[t.cell] = 0;
    }

    // Most blocks are neither in the support of pos's best neighbors' two-distances nor one of pos's neighbors,
    // so their trial is the board's own values: startTrials makes that one trial (and marks pos's neighbors),
    // trialOne copies it for those blocks, endTrials clears the marks. markSupport must have run for pos.
    Trial startTrials(Model& m, int pos) {
      Trial base;
      base.cell = -1;
      const int* around = m.neigh.data() + pos * 6;
      for (int k = 0; k < 6; k++) {
        base.two[k] = around[k] >= 0 ? m.two[around[k]] : 0;
        if (around[k] >= 0) {
          m.nearPos[around[k]] = 1;
        }
      }
      base.twoDist = m.baseMin;
      return base;
    }

    void trialOne(Model& m, int pos, const Trial& base, int c, Trial& t) {
      if (m.nearPos[c] || (m.inTwoMin[c] && m.baseMin < kUnreachable)) {
        t.cell = c;
        trialTwo(m, pos, t);
      } else {
        t = base;
        t.cell = c;
      }
    }

    void endTrials(Model& m, int pos) {
      const int* around = m.neigh.data() + pos * 6;
      for (int k = 0; k < 6; k++) {
        if (around[k] >= 0) {
          m.nearPos[around[k]] = 0;
        }
      }
    }

    // the shortest paths from pos's k-th neighbor to every cell, for the path-count shortcut: built the first time
    // they are needed after markSupport(pos), on its board
    void ensureVia(Model& m, int pos, int k) {
      if (m.viaReady[k]) {
        return;
      }
      m.viaReady[k] = 1;
      int* q = m.queue.data();
      const int* neigh = m.neigh.data();
      double* via = m.via.data() + k * m.cells;
      std::fill(via, via + m.cells, 0.0);
      const int n = neigh[pos * 6 + k];
      if (n < 0 || m.blocked[n] || m.dist[n] >= kUnreachable) {
        return;
      }
      via[n] = 1.0;
      int tail = 0;
      q[tail++] = n;
      for (int head = 0; head < tail; head++) {
        const int c = q[head];
        const int* next = neigh + c * 6;
        for (int j = 0; j < 6; j++) {
          const int x = next[j];
          if (x < 0 || m.blocked[x] || m.dist[x] != m.dist[c] - 1) {
            continue;
          }
          if (via[x] == 0.0) {
            q[tail++] = x;
          }
          via[x] += via[c];
        }
      }
    }

    // a block that raised the best two-distance brings in other neighbors, so their support counts too
    bool inSupport(const Model& m, const Trial& t) { return t.twoDist != m.baseMin ? m.inDist[t.cell] : m.inDistMin[t.cell]; }

    // the shortest paths from the neighbors of pos whose path counts t.cell's block can change (before blocking it)
    void viaForTrial(Model& m, int pos, const Trial& t) {
      if (!inSupport(m, t)) {
        return;
      }
      const int* around = m.neigh.data() + pos * 6;
      for (int k = 0; k < 6; k++) {
        const int n = around[k];
        if (n >= 0 && n != t.cell && !m.blocked[n] && t.two[k] == t.twoDist) {
          ensureVia(m, pos, k);
        }
      }
    }

    // the score of blocking t.cell with the cat on pos (m.dist / m.paths hold the board without it) when it
    // needs no pass, or false. With `floor`, the caller only uses a score that beats it, so a score that can't
    // may come back as a bound.
    bool quickScore(Model& m, int pos, const Trial& t, const Score* floor, Score& out) {
      const int* dist = m.dist.data();
      const double* paths = m.paths.data();
      const int* around = m.neigh.data() + pos * 6;
      if (dist[pos] == kUnreachable) {
        return false;  // pos is sealed in: the region size counts
      }
      const bool support = inSupport(m, t);
      viaForTrial(m, pos, t);
      // Most blocks need no pass: the tied neighbors (best two-distance) only lose the paths through the block,
      // and a neighbor that loses all of them (a cut) can only get farther, at least one step. So the reply is
      // the best of the others whenever one of them is no farther than every cut neighbor was; only a cut of the
      // closest neighbors needs the pass. A blocked neighbor just drops out (its trial two-distances already
      // leave it out).
      Score reply = kCaught;
      int cutDist = kUnreachable + 1;  // the closest cut neighbor's distance before the block
      for (int k = 0; k < 6; k++) {
        const int n = around[k];
        if (n < 0 || n == t.cell || m.blocked[n]) {
          continue;
        }
        double p = paths[n];
        if (support && t.two[k] == t.twoDist) {
          const double through = m.via[k * m.cells + t.cell] * paths[t.cell];
          p = paths[n] - through;
          if (through > 0.0 && p <= 0.0) {
            cutDist = std::min(cutDist, dist[n]);
            continue;
          }
        }
        const Score s{t.two[k], dist[n], p, 0};
        if (reply.beats(s)) {
          reply = s;
        }
      }
      if (cutDist > kUnreachable || (reply.twoDist == t.twoDist && reply.dist <= cutDist)) {
        out = reply;
        return true;
      }
      // the cut neighbors can only lower the reply below the others' best: if that can't beat the floor,
      // neither can the true score
      if (floor && reply.twoDist == t.twoDist && !reply.beats(*floor)) {
        out = reply;
        return true;
      }
      return false;
    }

    // the score of blocking t.cell with the cat on pos, with the pass where it needs one
    Score trialScore(Model& m, int pos, const Trial& t, const Score* floor) {
      Score quick;
      if (quickScore(m, pos, t, floor, quick)) {
        return quick;
      }
      const int* dist = m.dist.data();
      const double* paths = m.paths.data();
      const int* around = m.neigh.data() + pos * 6;
      bool sealed = dist[pos] == kUnreachable;
      viaForTrial(m, pos, t);
      m.blocked[t.cell] = 1;
      for (int k = 0; k < 6; k++) {
        if (around[k] >= 0) {
          m.steps[around[k]] = t.two[k];
        }
      }
      if (inSupport(m, t)) {
        // only the neighbors with the best two-distance decide the score; their path counts drop by the paths
        // through the block, and only a block through all of one's paths needs the pass
        int wanted = 0;
        bool cut = false;
        for (int k = 0; k < 6; k++) {
          const int n = around[k];
          if (n < 0 || m.blocked[n] || t.two[k] != t.twoDist) {
            continue;
          }
          wanted++;
          const double through = m.via[k * m.cells + t.cell] * paths[t.cell];
          m.scoreDist[n] = dist[n];
          m.scorePaths[n] = paths[n] - through;
          if (through > 0.0 && m.scorePaths[n] <= 0.0) {
            cut = true;
          }
        }
        if (!cut) {
          const Score s = scoreFrom(m, pos, m.steps.data(), m.scoreDist.data(), m.scorePaths.data(), sealed);
          m.blocked[t.cell] = 0;
          return s;
        }
        for (int k = 0; k < 6; k++) {
          const int n = around[k];
          if (n >= 0 && !m.blocked[n] && t.two[k] == t.twoDist) {
            m.isNb[n] = 1;
          }
        }
        // every wanted neighbor reached: some neighbor reaches the border; otherwise pos's distance tells
        const bool reached = escapeField(m, m.trialDist, m.trialPaths, wanted);
        sealed = !reached && m.trialDist[pos] == kUnreachable;
        for (int k = 0; k < 6; k++) {
          if (around[k] >= 0) {
            m.isNb[around[k]] = 0;
          }
        }
        dist = m.trialDist.data();
        paths = m.trialPaths.data();
      }
      const Score s = scoreFrom(m, pos, m.steps.data(), dist, paths, sealed);
      m.blocked[t.cell] = 0;
      return s;
    }

    // his followUpCells: the cells within kNearSteps of pos (blocked ones included), then his best first-look
    // blocks not listed yet; returns how many
    int followUpCells(Model& m, int pos, int rankedCount, int* cells) {
      int steps[kMaxFollowUps];
      int count = 0;
      m.stamp++;
      int* seen = m.mark.data();
      seen[pos] = m.stamp;
      cells[count] = pos;
      steps[count] = 0;
      count++;
      for (int head = 0; head < count; head++) {
        if (steps[head] == kNearSteps) {
          continue;
        }
        const int* around = m.neigh.data() + cells[head] * 6;
        for (int k = 0; k < 6; k++) {
          const int n = around[k];
          if (n < 0 || seen[n] == m.stamp) {
            continue;
          }
          seen[n] = m.stamp;
          cells[count] = n;
          steps[count] = steps[head] + 1;
          count++;
        }
      }
      for (int k = 0; k < kTopFollowUps && k < rankedCount; k++) {
        const int r = m.ranked[k].cell;
        if (seen[r] != m.stamp) {
          seen[r] = m.stamp;
          cells[count++] = r;
        }
      }
      return count;
    }

    // the best follow-up block's score with the cat on pos (m.dist / m.two / m.paths hold this board). With
    // `cap`, stops once the best score beats or ties it (returning one that does): the caller only keeps the
    // smaller of the two.
    Score bestFollowUp(Model& m, int pos, int rankedCount, const Score* cap) {
      markSupport(m, pos);
      const bool sealed = m.dist[pos] == kUnreachable;  // the region size counts, so every block can matter
      int cells[kMaxFollowUps];
      const int count = followUpCells(m, pos, rankedCount, cells);
      int trialCount = 0;
      int top = INT_MIN;
      const Trial baseTrial = startTrials(m, pos);
      for (int i = 0; i < count; i++) {
        const int c = cells[i];
        if (m.blocked[c] || c == pos) {
          continue;
        }
        Trial& t = m.followTrials[trialCount++];
        trialOne(m, pos, baseTrial, c, t);
        if (cap && t.twoDist > cap->twoDist) {
          endTrials(m, pos);
          return Score{t.twoDist, 0, 0.0, 0};
        }
        top = std::max(top, t.twoDist);
      }
      endTrials(m, pos);
      // only blocks tied for the best two-distance can give the best score
      Score best = kEscaped;
      const Score base = scoreFrom(m, pos, m.two.data(), m.dist.data(), m.paths.data(), sealed);
      for (int i = 0; i < trialCount; i++) {
        const Trial& t = m.followTrials[i];
        if (t.twoDist != top) {
          continue;
        }
        const Score s = (sealed || m.inTwoMin[t.cell] || m.inDistMin[t.cell]) ? trialScore(m, pos, t, &best) : base;
        if (s.beats(best)) {
          best = s;
        }
        if (cap && !cap->beats(best)) {
          return best;
        }
      }
      return best;
    }

    // his worstCase, the root block already placed. Returns early (with a worst case no better than the true
    // one) once it can't beat `need` (when haveNeed): the caller then won't pick this block.
    Score worstCase(Model& m, int cat, int rankedCount, bool haveNeed, const Score& need) {
      escapeField(m, m.dist, m.paths, 0);
      twoDistance(m, m.two, -1);
      int replies[6];
      int count = 0;
      const int* around = m.neigh.data() + cat * 6;
      for (int k = 0; k < 6; k++) {
        if (around[k] >= 0 && !m.blocked[around[k]]) {
          replies[count++] = around[k];
        }
      }
      if (count == 0) {
        return kCaught;
      }
      const int* two = m.two.data();
      const int* dist = m.dist.data();
      const double* paths = m.paths.data();
      std::stable_sort(replies, replies + count, [&](int a, int c) {
        if (two[a] != two[c]) {
          return two[a] < two[c];
        }
        if (dist[a] != dist[c]) {
          return dist[a] < dist[c];
        }
        return paths[a] > paths[c];
      });
      if (two[replies[0]] == 0) {
        return kEscaped;
      }
      Score worst = kCaught;
      for (int r = 0; r < kCatReplies && r < count; r++) {
        // the follow-ups only write the trial passes, so m.dist / m.two stay this board's
        const Score s = bestFollowUp(m, replies[r], rankedCount, r == 0 ? nullptr : &worst);
        if (worst.beats(s)) {
          worst = s;
        }
        if (haveNeed && !worst.beats(need)) {
          return worst;
        }
      }
      return worst;
    }

    // his closing in (2e882a2), m.ranked holding his best first-look blocks: among the blocks that leave the cat
    // no finite two-distance, nearest the cat first (steps ignoring blocks, then his first-look order, then board
    // order), the first of kTopBlocks whose worst case still leaves it none; -1 if none does. Only the nearest
    // rings get full scores: enough of them for kTopBlocks blocks.
    int closingIn(Model& m, int cat, int trialCount, int rankedCount) {
      static std::vector<int> ring;
      ring.assign(m.cells, -1);
      ring[cat] = 0;
      int tail = 0;
      m.queue[tail++] = cat;
      for (int head = 0; head < tail; head++) {
        const int c = m.queue[head];
        const int* around = m.neigh.data() + c * 6;
        for (int k = 0; k < 6; k++) {
          if (around[k] >= 0 && ring[around[k]] < 0) {
            ring[around[k]] = ring[c] + 1;
            m.queue[tail++] = around[k];
          }
        }
      }
      // full scores need this board's passes (the first look left them), so all before any worst case
      const Score base = scoreFrom(m, cat, m.two.data(), m.dist.data(), m.paths.data(), m.dist[cat] == kUnreachable);
      static std::vector<Ranked> picks;
      picks.resize(m.cells);
      int pickCount = 0;
      for (int r = 1; pickCount < kTopBlocks && r < 2 * m.side; r++) {
        const int ringStart = pickCount;
        for (int i = 0; i < trialCount; i++) {
          const Trial& t = m.trials[i];
          if (t.twoDist != kUnreachable || ring[t.cell] != r) {
            continue;
          }
          const Score sc = (m.inTwoMin[t.cell] || m.inDistMin[t.cell]) ? trialScore(m, cat, t, nullptr) : base;
          int at = pickCount++;  // in board order, a block only goes ahead of those it beats
          while (at > ringStart && sc.beats(picks[at - 1].score)) {
            picks[at] = picks[at - 1];
            at--;
          }
          picks[at] = {sc, t.cell};
        }
      }
      // stop a worst case once a reply leaves the cat a finite two-distance
      const Score stillHeld{kUnreachable, INT_MIN, 0.0, 0};
      for (int k = 0; k < kTopBlocks && k < pickCount; k++) {
        m.blocked[picks[k].cell] = 1;
        const Score worst = worstCase(m, cat, rankedCount, true, stillHeld);
        m.blocked[picks[k].cell] = 0;
        if (worst.twoDist >= kUnreachable) {
          return picks[k].cell;
        }
      }
      return -1;
    }

    // his sealed-in endgame; -2 to fall through as his code does. Blocks outside the cat's region change
    // nothing, so they keep the unblocked run and exits.
    int endgame(Model& m, int cat) {
      m.blocked[cat] = 1;
      int baseRun = 0, baseExits = 0;
      const int* around = m.neigh.data() + cat * 6;
      for (int k = 0; k < 6; k++) {
        if (around[k] >= 0 && !m.blocked[around[k]]) {
          baseExits++;
          baseRun = std::max(baseRun, floodFill(m, around[k]));
        }
      }
      m.blocked[cat] = 0;
      floodFill(m, cat);
      for (int i = 0; i < m.cells; i++) {
        m.inRegion[i] = m.mark[i] == m.stamp;
      }
      int best = -1, bestRun = kUnreachable, bestExits = kUnreachable;
      for (int c = 0; c < m.cells; c++) {
        if (c == cat || m.blocked[c]) {
          continue;
        }
        int run = baseRun, exits = baseExits;
        if (m.inRegion[c]) {
          m.blocked[c] = 1;
          m.blocked[cat] = 1;
          run = 0;
          exits = 0;
          for (int k = 0; k < 6; k++) {
            if (around[k] >= 0 && !m.blocked[around[k]]) {
              exits++;
              run = std::max(run, floodFill(m, around[k]));
            }
          }
          m.blocked[cat] = 0;
          m.blocked[c] = 0;
        }
        if (run < bestRun || (run == bestRun && exits < bestExits)) {
          bestRun = run;
          bestExits = exits;
          best = c;
        }
      }
      return bestRun != kUnreachable ? best : -2;
    }

    // his block on m.blocked with the cat on `cat`
    int move(Model& m, int cat) {
      escapeField(m, m.dist, m.paths, 0);
      if (m.dist[cat] == kUnreachable) {
        const int e = endgame(m, cat);
        if (e != -2) {
          return e;
        }
      }

      // first look: every open cell's two-distance, then full scores only for those that can make the top 10
      twoDistance(m, m.two, -1);
      markSupport(m, cat);
      const Trial baseTrial = startTrials(m, cat);
      int trialCount = 0;
      for (int c = 0; c < m.cells; c++) {
        if (c == cat || m.blocked[c]) {
          continue;
        }
        trialOne(m, cat, baseTrial, c, m.trials[trialCount++]);
      }
      endTrials(m, cat);
      if (trialCount == 0) {
        return cat == 0 ? 1 : 0;  // his fallback: the first cell that isn't the cat's
      }
      for (int i = 0; i < trialCount; i++) {
        m.twos[i] = m.trials[i].twoDist;
      }
      int cut = INT_MIN;
      if (trialCount > kTopFollowUps) {
        std::nth_element(m.twos.begin(), m.twos.begin() + (kTopFollowUps - 1), m.twos.begin() + trialCount, std::greater<int>());
        cut = m.twos[kTopFollowUps - 1];
      }
      // his stable sort, but only its top 10 are ever used: keep them in order as the blocks come (a block
      // only goes ahead of those it beats, so ties stay in board order). A block below the cut can't make it.
      const Score base = scoreFrom(m, cat, m.two.data(), m.dist.data(), m.paths.data(), m.dist[cat] == kUnreachable);
      int rankedCount = 0;
      for (int i = 0; i < trialCount; i++) {
        const Trial& t = m.trials[i];
        if (t.twoDist < cut) {
          continue;
        }
        const Score* floor = rankedCount == kTopFollowUps ? &m.ranked[kTopFollowUps - 1].score : nullptr;
        const Score s = (m.inTwoMin[t.cell] || m.inDistMin[t.cell]) ? trialScore(m, cat, t, floor) : base;
        if (rankedCount == kTopFollowUps && !s.beats(m.ranked[kTopFollowUps - 1].score)) {
          continue;
        }
        int at = rankedCount < kTopFollowUps ? rankedCount++ : kTopFollowUps - 1;
        while (at > 0 && s.beats(m.ranked[at - 1].score)) {
          m.ranked[at] = m.ranked[at - 1];
          at--;
        }
        m.ranked[at] = {s, t.cell};
      }
      if (m.ranked[0].score.twoDist == kCaught.twoDist) {
        return m.ranked[0].cell;
      }
      if (m.ranked[0].score.twoDist == kUnreachable) {
        const int close = closingIn(m, cat, trialCount, rankedCount);
        if (close >= 0) {
          return close;
        }
      }

      // lookahead
      int best = m.ranked[0].cell;
      Score bestWorst = kEscaped;
      bool found = false;
      for (int k = 0; k < kTopBlocks && k < rankedCount; k++) {
        const int x = m.ranked[k].cell;
        m.blocked[x] = 1;
        const Score worst = worstCase(m, cat, rankedCount, found, bestWorst);
        m.blocked[x] = 0;
        if (!found || worst.beats(bestWorst)) {
          best = x;
          bestWorst = worst;
          found = true;
        }
      }
      return best;
    }

    // Playouts: every move of the cat is played out against his catcher (aaron::move), the cat then stepping to
    // its roomiest neighbor. All moves advance together one catcher move per round until each playout has ended
    // or the budget is spent, and the move whose playout lasts longest wins (an escape beats any length; ties go
    // to the usual move). Against his catcher this keeps the cat where his blocks stop closing in on it.
    constexpr int kPlayoutCalls = 240;   // his catcher's moves per cat move, over all playouts
    constexpr int kPlayoutMillis = 700;  // and a hard stop, well inside the runner's 2 s
    constexpr int kEscapedValue = 1 << 20;
    constexpr int kCappedValue = -1;  // the playout could reach the move cap, which scores nothing

    struct Line {
      std::vector<uint8_t> blocked;
      int cat = 0;
      int first = 0;
      int plies = 0;
      int value = 0;
      bool running = true;
    };

    bool isBorder(const Model& m, int c) {
      const int h = m.side / 2;
      const int x = c % m.side - h, y = c / m.side - h;
      return std::abs(x) == h || std::abs(y) == h;
    }

    int openAround(const Model& m, const uint8_t* blocked, int c) {
      int k = 0;
      for (int j = 0; j < 6; j++) {
        const int n = m.neigh[c * 6 + j];
        k += n >= 0 && !blocked[n];
      }
      return k;
    }

    // the open neighbor with the most open neighbors, then the most room behind them, never the border unless
    // nothing else is open; -1 when the cat is trapped
    int roomiest(const Model& m, const uint8_t* blocked, int cat) {
      int out = -1, bestKey = -1;
      for (int j = 0; j < 6; j++) {
        const int n = m.neigh[cat * 6 + j];
        if (n < 0 || blocked[n]) {
          continue;
        }
        if (out < 0) {
          out = n;
        }
        if (isBorder(m, n)) {
          continue;
        }
        int near = 0, room = 0;
        for (int i = 0; i < 6; i++) {
          const int x = m.neigh[n * 6 + i];
          if (x >= 0 && !blocked[x]) {
            near++;
            room += openAround(m, blocked, x);
          }
        }
        const int key = near * 100 + room;
        if (key > bestKey) {
          bestKey = key;
          out = n;
        }
      }
      return out;
    }

    int survivalMove(const std::vector<bool>& state, int side, int cat, int usual) {
      const auto start = std::chrono::steady_clock::now();
      Model& m = modelFor(side);
      // a safe bound on the plies left before the cap: at least 5% of the board started blocked (the runner's
      // obstacles), so the catcher has placed at most the rest, two plies each
      int blockedCount = 0;
      for (int i = 0; i < m.cells; i++) {
        blockedCount += state[i];
      }
      const int placed = std::max(0, blockedCount - m.cells * 5 / 100);
      const int left = m.cells - 2 * placed;
      if (left < 0) {
        return usual;  // the board is nearly full: no playout can end before the cap
      }

      static std::vector<Line> lines(6);
      int lineCount = 0;
      for (int j = 0; j < 6; j++) {
        const int n = m.neigh[cat * 6 + j];
        if (n < 0 || state[n]) {
          continue;
        }
        if (isBorder(m, n)) {
          return n;  // the cat gets out
        }
        Line& l = lines[lineCount++];
        l.blocked.resize(m.cells);
        for (int i = 0; i < m.cells; i++) {
          l.blocked[i] = state[i] ? 1 : 0;
        }
        l.cat = n;
        l.first = n;
        l.plies = 1;
        l.value = 0;
        l.running = true;
      }
      int calls = 0;
      bool any = lineCount > 0;
      while (any && calls < kPlayoutCalls) {
        const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start).count();
        if (elapsed >= kPlayoutMillis) {
          break;
        }
        any = false;
        for (int i = 0; i < lineCount; i++) {
          Line& l = lines[i];
          if (!l.running) {
            continue;
          }
          calls++;
          std::memcpy(m.blocked.data(), l.blocked.data(), m.cells);
          const int b = move(m, l.cat);
          if (b < 0 || b >= m.cells || b == l.cat || l.blocked[b]) {
            l.running = false;
            l.value = kEscapedValue - l.plies;  // his move would be illegal: the cat wins
            continue;
          }
          l.blocked[b] = 1;
          l.plies++;
          if (l.plies >= left) {
            l.running = false;
            l.value = kCappedValue;
            continue;
          }
          const int next = roomiest(m, l.blocked.data(), l.cat);
          if (next < 0) {
            l.running = false;
            l.value = l.plies;  // trapped
            continue;
          }
          l.cat = next;
          l.plies++;
          if (isBorder(m, next)) {
            l.running = false;
            l.value = kEscapedValue - l.plies;
            continue;
          }
          any = true;
        }
      }
      int best = usual, bestValue = kCappedValue - 1;
      for (int i = 0; i < lineCount; i++) {
        const Line& l = lines[i];
        const int v = l.running ? l.plies : l.value;
        if (v > bestValue || (v == bestValue && l.first == usual)) {
          bestValue = v;
          best = l.first;
        }
      }
      // every move could reach the cap: step where the cat has the least room, to be caught before it
      if (bestValue == kCappedValue) {
        std::vector<uint8_t>& now = lines[0].blocked;
        for (int i = 0; i < m.cells; i++) {
          now[i] = state[i] ? 1 : 0;
        }
        int fewest = 7;
        for (int j = 0; j < 6; j++) {
          const int n = m.neigh[cat * 6 + j];
          if (n >= 0 && !now[n] && !isBorder(m, n) && openAround(m, now.data(), n) < fewest) {
            fewest = openAround(m, now.data(), n);
            best = n;
          }
        }
      }
      return best;
    }
  }  // namespace aaron
}  // namespace

namespace models {
  int predictedBy(grid::Board& b, int cat, int block) {
    auto& s = scratchFor(b);
    grid::setOpen(b, block, true);
    int mask = 0;
    for (int m = 0; m < kModelCount; m++) {
      if (m != kNearestExits) {
        mask |= (modelReply(b, s, m, cat) == block) << m;
      }
    }
    int exits[kMaxExits];
    const int count = nearestExitSet(b, s, cat, exits);
    for (int k = 0; k < count; k++) {
      mask |= (exits[k] == block) << kNearestExits;
    }
    grid::setOpen(b, block, false);
    return mask;
  }

  int confirmed(const std::vector<int>& hits) {
    const int n = static_cast<int>(hits.size());
    int general = n >= kConfirmBlocks ? ~0 : 0, ports = n >= kPortConfirmBlocks ? ~0 : 0;
    for (int k = std::max(0, n - kConfirmBlocks); k < n; k++) {
      general &= hits[k];
    }
    for (int k = std::max(0, n - kPortConfirmBlocks); k < n; k++) {
      ports &= hits[k];
    }
    const int portBits = 1 << kAylwin | 1 << kCosmey;
    return (general & ~portBits) | (ports & portBits);
  }

  int modelChoice(grid::Board& b, int cat, const int* cells, int count, bool tryJordan, int playedLike) {
    auto& s = scratchFor(b);
    bool fits[kModelCount];
    for (int m = 0; m < kModelCount; m++) {
      fits[m] = playedLike ? (playedLike >> m & 1) : modelFitsBoard(b, s, m);
    }
    // an exact port that explains the last blocks outranks the general styles
    if (!playedLike && tryJordan && jordanFits(b, s, cat, kJordanBlocks)) {
      for (int m = 0; m < kModelCount; m++) {
        fits[m] = m == kJordan;
      }
    }
    // an exact port (AylwinMorgan's, Cosmey's) explained the last blocks: plan against it alone (some of their
    // blocks are nearest exits too)
    for (int port : {kAylwin, kCosmey}) {
      if (fits[port]) {
        for (int m = 0; m < kModelCount; m++) {
          fits[m] = m == port;
        }
        break;
      }
    }
    // a catcher that keeps blocking nearest exits: plan against that alone, for the best chance (it covers
    // the general styles that pick a nearest exit their own way)
    if (fits[kNearestExits]) {
      int budget = kNearestBudget, best = -1;
      double bestOdds = 0.0;
      for (int i = 0; i < count; i++) {
        const double odds = oddsAfterMove(b, s, cells[i], kNearestDepth, budget);
        if (odds > bestOdds) {
          bestOdds = odds;
          best = cells[i];
        }
      }
      return best;
    }
    const bool portOnly = fits[kAylwin] || fits[kCosmey];
    const int depth = portOnly ? kPortDepth : kModelDepth;
    int budget = portOnly ? kPortBudget : kModelBudget, best = -1, bestCount = 0;
    for (int i = 0; i < count; i++) {
      int n = cells[i], beaten = 0;
      for (int model = 0; model < kModelCount; model++) {
        if (!fits[model]) {
          continue;
        }
        int blk = modelReply(b, s, model, n);
        if (blk >= 0 && blk != n && b.open[blk]) {
          grid::setOpen(b, blk, false);
        } else {
          blk = -1;
        }
        beaten += !trappedAt(b, n) && beatsModel(b, s, n, model, depth - 1, budget);
        if (blk >= 0) {
          grid::setOpen(b, blk, true);
        }
      }
      if (beaten > bestCount) {
        bestCount = beaten;
        best = n;
      }
    }
    return best;
  }

  int survivalMove(const std::vector<bool>& state, int side, int cat, int usual) { return aaron::survivalMove(state, side, cat, usual); }
}  // namespace models
