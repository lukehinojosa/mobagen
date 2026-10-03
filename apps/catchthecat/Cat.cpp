#include "Cat.h"
#include "World.h"
#include <algorithm>
#include <array>
#include <climits>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <queue>
#include <vector>

// Cat strategy (stateless, decided from the current board only):
//  1. Two-distance: an open border cell scores 0; any other open cell scores 1 + the SECOND smallest
//     score among its neighbors, because the catcher will block the best one. Rank moves by it.
//  2. Model search: catchers that defend the edge (instead of blocking next to the cat) beat plain
//     two-distance by plugging each exit as the cat laps the board. When the board shows such a style,
//     search up to 6 cat moves for a line that beats cheap copies of those catchers, and take it.
//  3. Veto: try the catcher replies that can matter to the top move (cells its score depends on). Only if
//     one of them cuts off every escape, switch to the move whose worst reply leaves the cat best off.
//     Always trusting the worst case made the cat too timid against last year's real catchers.
//  4. If every score is infinite but a border is still reachable, follow the plain BFS distance and hope
//     the catcher is weak.
//  5. If no border is reachable, survive: move to the neighbor with the most open neighbors.
// All buffers are flat arrays reused across calls, so nothing is allocated after the first move.

namespace {
  constexpr int kInf = INT_MAX;
  constexpr int kNoEscape = 1023;  // evaluate() caps scores here, so it also means "no finite escape"

  // replies tried per move, nearest first. With the lookahead only used as a veto, 8 scored best on
  // 21x21 once the runner's time penalty was counted (12 and 22 won no extra games)
  constexpr int kMaxReplies = 8;

  // model search tuning (local tournament against last year's 21 catchers, 3 board seeds)
  constexpr int kModelDepth = 6;         // cat moves searched
  constexpr int kModelBudget = 20000;    // search nodes per move
  constexpr int kEdgeSharePercent = 30;  // general edge models run when this share of blocks is on the border
  constexpr int kPortSignature = 6;      // extra blocks on a student's pattern before trusting their port

  struct Buffers {
    int side = 0;
    std::vector<std::array<int, 6>> neigh;  // neighbor linear indices, -1 when off the board
    std::vector<int> borders;
    std::vector<uint8_t> open;
    std::vector<int> score;  // two-distance
    std::vector<int> trial;  // two-distance after a hypothetical catcher reply
    std::vector<int> dist;   // plain BFS distance to an open border cell
    std::vector<uint8_t> count;
    std::vector<int> queue;
    std::vector<int> mark;  // stamp per cell for the reply search
    int stamp = 0;
    std::vector<int> replies;
  };

  Buffers& buffersFor(int side) {
    static Buffers b;
    if (b.side == side) {
      return b;
    }

    b.side = side;
    int half = side / 2;
    int total = side * side;
    b.neigh.resize(total);
    b.borders.clear();
    b.open.resize(total);
    b.score.resize(total);
    b.trial.resize(total);
    b.dist.resize(total);
    b.count.resize(total);
    b.queue.resize(total);
    b.mark.assign(total, 0);
    b.stamp = 0;
    b.replies.reserve(total);
    for (int i = 0; i < total; i++) {
      Point2D p = {i % side - half, i / side - half};
      if (std::abs(p.x) == half || std::abs(p.y) == half) {
        b.borders.push_back(i);
      }
      Point2D ns[6] = {CatWorld::NE(p), CatWorld::NW(p), CatWorld::E(p), CatWorld::W(p), CatWorld::SW(p), CatWorld::SE(p)};
      for (int d = 0; d < 6; d++) {
        bool inside = ns[d].x >= -half && ns[d].x <= half && ns[d].y >= -half && ns[d].y <= half;
        b.neigh[i][d] = inside ? (ns[d].y + half) * side + ns[d].x + half : -1;
      }
    }
    return b;
  }

  // Distance passes
  // Raw pointers instead of vector calls, and the 6 neighbors written out instead of looped. `neigh` is 6 ints per cell, -1 off the board.
  // cells leave the queue in nondecreasing score, so the second neighbor to leave is the second best one
  void rawTwoDistance(const int* neigh, const int* borders, int borderCount, const uint8_t* open, int total, int* score, uint8_t* count, int* queue) {
    for (int i = 0; i < total; i++) {
      score[i] = kInf;
    }
    memset(count, 0, total);
    int tail = 0;
    for (int k = 0; k < borderCount; k++) {
      const int i = borders[k];
      if (open[i]) {
        score[i] = 0;
        queue[tail] = i;
        tail++;
      }
    }
    for (int head = 0; head < tail; head++) {
      const int c = queue[head];
      const int next = score[c] + 1;
      const int* nb = neigh + c * 6;
      const int m0 = nb[0];
      if (m0 >= 0 && open[m0] && score[m0] == kInf) {
        count[m0]++;
        if (count[m0] == 2) {
          score[m0] = next;
          queue[tail] = m0;
          tail++;
        }
      }
      const int m1 = nb[1];
      if (m1 >= 0 && open[m1] && score[m1] == kInf) {
        count[m1]++;
        if (count[m1] == 2) {
          score[m1] = next;
          queue[tail] = m1;
          tail++;
        }
      }
      const int m2 = nb[2];
      if (m2 >= 0 && open[m2] && score[m2] == kInf) {
        count[m2]++;
        if (count[m2] == 2) {
          score[m2] = next;
          queue[tail] = m2;
          tail++;
        }
      }
      const int m3 = nb[3];
      if (m3 >= 0 && open[m3] && score[m3] == kInf) {
        count[m3]++;
        if (count[m3] == 2) {
          score[m3] = next;
          queue[tail] = m3;
          tail++;
        }
      }
      const int m4 = nb[4];
      if (m4 >= 0 && open[m4] && score[m4] == kInf) {
        count[m4]++;
        if (count[m4] == 2) {
          score[m4] = next;
          queue[tail] = m4;
          tail++;
        }
      }
      const int m5 = nb[5];
      if (m5 >= 0 && open[m5] && score[m5] == kInf) {
        count[m5]++;
        if (count[m5] == 2) {
          score[m5] = next;
          queue[tail] = m5;
          tail++;
        }
      }
    }
  }

  // BFS distance from the open border cells; cells farther than `limit` are left at kInf
  void rawBfsDistance(const int* neigh, const int* borders, int borderCount, const uint8_t* open, int total, int* dist, int* queue, int limit) {
    for (int i = 0; i < total; i++) {
      dist[i] = kInf;
    }
    int tail = 0;
    for (int k = 0; k < borderCount; k++) {
      const int i = borders[k];
      if (open[i]) {
        dist[i] = 0;
        queue[tail] = i;
        tail++;
      }
    }
    for (int head = 0; head < tail; head++) {
      const int c = queue[head];
      if (dist[c] >= limit) {
        break;
      }
      const int next = dist[c] + 1;
      const int* nb = neigh + c * 6;
      const int m0 = nb[0];
      if (m0 >= 0 && open[m0] && dist[m0] == kInf) {
        dist[m0] = next;
        queue[tail] = m0;
        tail++;
      }
      const int m1 = nb[1];
      if (m1 >= 0 && open[m1] && dist[m1] == kInf) {
        dist[m1] = next;
        queue[tail] = m1;
        tail++;
      }
      const int m2 = nb[2];
      if (m2 >= 0 && open[m2] && dist[m2] == kInf) {
        dist[m2] = next;
        queue[tail] = m2;
        tail++;
      }
      const int m3 = nb[3];
      if (m3 >= 0 && open[m3] && dist[m3] == kInf) {
        dist[m3] = next;
        queue[tail] = m3;
        tail++;
      }
      const int m4 = nb[4];
      if (m4 >= 0 && open[m4] && dist[m4] == kInf) {
        dist[m4] = next;
        queue[tail] = m4;
        tail++;
      }
      const int m5 = nb[5];
      if (m5 >= 0 && open[m5] && dist[m5] == kInf) {
        dist[m5] = next;
        queue[tail] = m5;
        tail++;
      }
    }
  }

  void twoDistance(Buffers& b, std::vector<int>& score) {
    rawTwoDistance(b.neigh[0].data(), b.borders.data(), static_cast<int>(b.borders.size()), b.open.data(), static_cast<int>(b.open.size()),
                   score.data(), b.count.data(), b.queue.data());
  }

  void bfsDistance(Buffers& b) {
    rawBfsDistance(b.neigh[0].data(), b.borders.data(), static_cast<int>(b.borders.size()), b.open.data(), static_cast<int>(b.open.size()),
                   b.dist.data(), b.queue.data(), kInf);
  }

  // value of the cat standing on n with the catcher having just moved; smaller is better.
  // (best neighbor score, how many neighbors share it, BFS distance of that neighbor)
  int64_t evaluate(const Buffers& b, int n, const std::vector<int>& score) {
    int best = kInf, ties = 0, bestDist = kInf;
    bool anyOpen = false;
    for (int m : b.neigh[n]) {
      if (m < 0 || !b.open[m]) {
        continue;
      }
      anyOpen = true;
      if (score[m] < best) {
        best = score[m];
        ties = 1;
        bestDist = b.dist[m];
      } else if (score[m] == best) {
        ties++;
        bestDist = std::min(bestDist, b.dist[m]);
      }
    }
    if (!anyOpen) {
      return INT64_MAX;  // trapped
    }
    int64_t s = std::min(best, kNoEscape);
    int64_t d = std::min(bestDist, kNoEscape);
    return (s << 20) | (int64_t(6 - ties) << 12) | d;
  }

  // worst case over the catcher replies that can change n's score; stops once it reaches alpha
  int64_t worstReply(Buffers& b, int n, int64_t alpha) {
    int64_t worst = evaluate(b, n, b.score);  // a reply that changes nothing
    if (worst >= alpha) {
      return worst;
    }

    // replies: n's neighbors, then every cell reachable from n by strictly decreasing score, nearest first
    b.stamp++;
    b.replies.clear();
    b.mark[n] = b.stamp;
    int tail = 0;
    b.queue[tail++] = n;
    for (int head = 0; head < tail && static_cast<int>(b.replies.size()) < kMaxReplies; head++) {
      int c = b.queue[head];
      for (int m : b.neigh[c]) {
        if (m < 0 || !b.open[m] || b.mark[m] == b.stamp) {
          continue;
        }
        if (c != n && b.score[m] >= b.score[c]) {
          continue;
        }
        b.mark[m] = b.stamp;
        b.replies.push_back(m);
        b.queue[tail++] = m;
      }
    }

    // twoDistance reuses queue, so the reply list lives in its own buffer
    for (int x : b.replies) {
      b.open[x] = 0;
      twoDistance(b, b.trial);
      int64_t v = evaluate(b, n, b.trial);
      b.open[x] = 1;
      worst = std::max(worst, v);
      if (worst >= alpha) {
        break;
      }
    }
    return worst;
  }

  // Model search. Each model predicts the catcher's block for a given cat cell; the search looks for a
  // line of cat moves that reaches the border against that prediction.

  enum Model {
    kPlugSecondExit,   // plug an exit next to the cat, else block the exit of the cat's second-best path
    kPlugAltEdge,      // plug, else the nearest cell of an every-other-cell edge wall
    kPlugNearestEdge,  // plug, else the nearest open edge cell
    kAndrew,           // close port of last year's AndrewGenualdo catcher
    kToag,             // close port of last year's TOAG21 catcher
    kModelCount
  };

  struct ModelScratch {
    int side = 0;
    std::vector<uint8_t> isBorder;
    std::vector<int> modelDist, modelQueue;
    std::vector<int> depthDist[kModelDepth + 1];
  };

  ModelScratch& scratchFor(const Buffers& b) {
    static ModelScratch s;
    if (s.side == b.side) {
      return s;
    }
    int total = b.side * b.side;
    s.side = b.side;
    s.isBorder.assign(total, 0);
    for (int i : b.borders) {
      s.isBorder[i] = 1;
    }
    s.modelDist.resize(total);
    s.modelQueue.resize(total);
    for (auto& d : s.depthDist) {
      d.resize(total);
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
  int nearestExit(const Buffers& b, ModelScratch& s, int from, bool altOnly, int skip) {
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
      if (c != from && s.isBorder[c] && (!altOnly || altEdgeCell(c, b.side)) && (found < 0 || c < found)) {
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

  struct Board {
    const Buffers& b;
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
  int andrewLegal(const Board& bd, Pt cat, Pt p) {
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
  Pt andrewAStarGoal(const Board& bd, AndrewScratch& s, Pt cat) {
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

  int andrewReply(const Buffers& b, int catIdx) {
    static AndrewScratch s;
    const Board bd{b, b.side / 2};
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

  int toagReply(const Buffers& b, int catIdx) {
    static std::vector<Fiber> net;
    const Board bd{b, b.side / 2};
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

  // the model's block with the cat standing on `cat`, or -1
  int modelReply(const Buffers& b, ModelScratch& s, int model, int cat) {
    if (model == kAndrew) {
      return andrewReply(b, cat);
    }
    if (model == kToag) {
      return toagReply(b, cat);
    }
    for (int n : b.neigh[cat]) {
      if (n >= 0 && b.open[n] && s.isBorder[n]) {
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

  void borderDistInto(const Buffers& b, std::vector<int>& d, std::vector<int>& q, int limit) {
    rawBfsDistance(b.neigh[0].data(), b.borders.data(), static_cast<int>(b.borders.size()), b.open.data(), static_cast<int>(b.open.size()), d.data(),
                   q.data(), limit);
  }

  bool trappedAt(const Buffers& b, int cat) {
    for (int n : b.neigh[cat]) {
      if (n >= 0 && b.open[n]) {
        return false;
      }
    }
    return true;
  }

  // cat on `cat` to move, `depth` cat moves left: is there a line that beats this model?
  // only cells within depth - 1 of the border matter, so the distance pass stops there
  bool beatsModel(Buffers& b, ModelScratch& s, int cat, int model, int depth, int& budget) {
    if (depth <= 0 || --budget < 0) {
      return false;
    }
    auto& d = s.depthDist[depth];
    borderDistInto(b, d, s.modelQueue, depth - 1);
    for (int n : b.neigh[cat]) {
      if (n < 0 || !b.open[n] || d[n] > depth - 1) {
        continue;  // too far to arrive in time
      }
      if (s.isBorder[n]) {
        return true;
      }
      int blk = modelReply(b, s, model, n);
      if (blk >= 0 && blk != n && b.open[blk]) {
        b.open[blk] = 0;
      } else {
        blk = -1;
      }
      bool win = !trappedAt(b, n) && beatsModel(b, s, n, model, depth - 1, budget);
      if (blk >= 0) {
        b.open[blk] = 1;
      }
      if (win) {
        return true;
      }
    }
    return false;
  }

  // Reading the catcher's style from the board
  // The cat has no memory, but the catcher's past blocks stay on the board.

  // share of blocked cells that sit on the border, in percent
  int borderBlockPercent(const Buffers& b, const ModelScratch& s) {
    int onBorder = 0, all = 0;
    for (size_t i = 0; i < b.open.size(); i++) {
      if (!b.open[i]) {
        all++;
        onBorder += s.isBorder[i];
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
  int footprintExcess(const Buffers& b, const std::vector<uint8_t>& footprint) {
    int total = static_cast<int>(b.open.size()), size = 0, blocked = 0, blockedOn = 0;
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
  int andrewPatternExcess(const Buffers& b) {
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

  // does the board look like this model's catcher is playing?
  bool modelFitsBoard(const Buffers& b, const ModelScratch& s, int model) {
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

  // the stage 1 move (by rank) that beats the most fitting models, or -1 when none beats any
  template <typename Moves> int modelChoice(Buffers& b, const Moves& moves, int moveCount) {
    auto& s = scratchFor(b);
    bool fits[kModelCount];
    for (int m = 0; m < kModelCount; m++) {
      fits[m] = modelFitsBoard(b, s, m);
    }
    int budget = kModelBudget, best = -1, bestCount = 0;
    for (int i = 0; i < moveCount && moves[i].first[0] < 2; i++) {
      int n = moves[i].second, count = 0;
      for (int model = 0; model < kModelCount; model++) {
        if (!fits[model]) {
          continue;
        }
        int blk = modelReply(b, s, model, n);
        if (blk >= 0 && blk != n && b.open[blk]) {
          b.open[blk] = 0;
        } else {
          blk = -1;
        }
        count += !trappedAt(b, n) && beatsModel(b, s, n, model, kModelDepth - 1, budget);
        if (blk >= 0) {
          b.open[blk] = 1;
        }
      }
      if (count > bestCount) {
        bestCount = count;
        best = n;
      }
    }
    return best;
  }
}  // namespace

Point2D Cat::Move(CatWorld* world) {
  const int side = world->getWorldSideSize();
  const int half = side / 2;
  const int total = side * side;
  const auto& state = world->worldState();
  auto& b = buffersFor(side);

  for (int i = 0; i < total; i++) {
    b.open[i] = !state[i];
  }
  twoDistance(b, b.score);
  bfsDistance(b);

  // stage 1 ranking of the open neighbors; smaller keys are better
  const Point2D cat = world->getCat();
  const int catIdx = (cat.y + half) * side + cat.x + half;
  std::array<std::pair<std::array<int, 5>, int>, 6> moves;
  int moveCount = 0;
  for (int n : b.neigh[catIdx]) {
    if (n < 0 || !b.open[n]) {
      continue;
    }

    int tier, primary;
    if (b.score[n] != kInf) {
      tier = 0;
      primary = b.score[n];
    } else if (b.dist[n] != kInf) {
      tier = 1;
      primary = b.dist[n];
    } else {
      tier = 2;
      primary = 0;
    }

    // forward: neighbors that bring the cat closer (more ways to continue); liberties: open neighbors
    int forward = 0, liberties = 0;
    for (int m : b.neigh[n]) {
      if (m < 0 || !b.open[m]) {
        continue;
      }
      liberties++;
      if (tier == 0 && b.score[m] < b.score[n]) {
        forward++;
      }
      if (tier == 1 && b.dist[m] < b.dist[n]) {
        forward++;
      }
    }
    moves[moveCount++] = {{tier, primary, b.dist[n], -forward, -liberties}, n};
  }

  // no open neighbor means the catcher already won; return anything
  if (moveCount == 0) {
    return CatWorld::NE(cat);
  }
  std::sort(moves.begin(), moves.begin() + moveCount);

  // model search, unless the win is already forced
  if (!(moves[0].first[0] == 0 && moves[0].first[1] <= 1)) {
    int mv = modelChoice(b, moves, moveCount);
    if (mv >= 0) {
      return {mv % side - half, mv / side - half};
    }
  }

  // stage 2 veto: keep the top move unless some catcher reply leaves it no finite escape (and the win
  // isn't already forced); then pick the finite-score move with the best worst case
  int best = moves[0].second;
  const auto& top = moves[0].first;
  if (top[0] == 0 && top[1] > 1) {
    int64_t alpha = worstReply(b, best, INT64_MAX);
    const bool doomed = (alpha >> 20) >= kNoEscape;
    for (int i = 1; doomed && i < moveCount && moves[i].first[0] == 0; i++) {
      int64_t v = worstReply(b, moves[i].second, alpha);
      if (v < alpha) {
        alpha = v;
        best = moves[i].second;
      }
    }
  }

  return {best % side - half, best / side - half};
}
