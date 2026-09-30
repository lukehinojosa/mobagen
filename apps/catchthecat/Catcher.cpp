#include "Catcher.h"
#include "World.h"
#include <algorithm>
#include <array>
#include <climits>
#include <cstdint>
#include <cstdlib>
#include <vector>

// Catcher strategy (stateless, decided from the current board only):
//  1. Score a position by the cat's two-distance (an open border cell scores 0; any other open cell
//     scores 1 + the second smallest score among its neighbors), then its BFS distance, then how many of
//     its neighbors share the best score and how many are open. Bigger is better for the catcher.
//  2. Try every block that can change that score: the cat's neighbors and the cells their scores are
//     built from. Every other cell leaves the score as it is.
//  3. Look ahead on the 5 best blocks: assume the cat's best reply, then the catcher's best follow-up,
//     and keep the block whose worst case is best.
//  4. Once the cat can no longer reach the border, stop defending and shrink its region: block the cell
//     that leaves the cat the least room.
// Blocking next to the cat (instead of walling the edge) caught last year's cats about 3 times faster.

namespace {
  constexpr int kInf = INT_MAX;
  constexpr int kSearchBlocks = 5;     // best first blocks looked ahead on
  constexpr int kFollowUpBlocks = 16;  // follow-up blocks tried after each cat reply

  struct Buffers {
    int side = 0;
    std::vector<std::array<int, 6>> neigh;  // neighbor linear indices, -1 when off the board
    std::vector<int> borders;
    std::vector<uint8_t> isBorder, open, count, cand;
    std::vector<int> score, dist;            // the current board
    std::vector<int> trialScore, trialDist;  // after a hypothetical block
    std::vector<int> nodeScore, nodeDist;    // inside the lookahead
    std::vector<int> queue, list;
    std::vector<std::pair<int64_t, int>> rootValues;
  };

  Buffers& buffersFor(int side) {
    static Buffers b;
    if (b.side == side) {
      return b;
    }
    b.side = side;
    int h = side / 2, total = side * side;
    b.neigh.resize(total);
    b.borders.clear();
    for (auto* v : {&b.open, &b.count, &b.cand}) {
      v->resize(total);
    }
    for (auto* v : {&b.score, &b.dist, &b.trialScore, &b.trialDist, &b.nodeScore, &b.nodeDist, &b.queue}) {
      v->resize(total);
    }
    b.isBorder.assign(total, 0);
    for (int i = 0; i < total; i++) {
      Point2D p = {i % side - h, i / side - h};
      if (std::abs(p.x) == h || std::abs(p.y) == h) {
        b.borders.push_back(i);
        b.isBorder[i] = 1;
      }
      auto ns = CatWorld::neighbors(p);
      for (int k = 0; k < 6; k++) {
        bool inside = std::abs(ns[k].x) <= h && std::abs(ns[k].y) <= h;
        b.neigh[i][k] = inside ? (ns[k].y + h) * side + ns[k].x + h : -1;
      }
    }
    return b;
  }

  // cells leave the queue in nondecreasing score, so the second neighbor to leave is the second best one
  void twoDistance(Buffers& b, std::vector<int>& score) {
    std::fill(score.begin(), score.end(), kInf);
    std::fill(b.count.begin(), b.count.end(), 0);
    int tail = 0;
    for (int i : b.borders) {
      if (b.open[i]) {
        score[i] = 0;
        b.queue[tail++] = i;
      }
    }
    for (int head = 0; head < tail; head++) {
      int c = b.queue[head];
      for (int m : b.neigh[c]) {
        if (m >= 0 && b.open[m] && score[m] == kInf && ++b.count[m] == 2) {
          score[m] = score[c] + 1;
          b.queue[tail++] = m;
        }
      }
    }
  }

  void bfsDistance(Buffers& b, std::vector<int>& dist) {
    std::fill(dist.begin(), dist.end(), kInf);
    int tail = 0;
    for (int i : b.borders) {
      if (b.open[i]) {
        dist[i] = 0;
        b.queue[tail++] = i;
      }
    }
    for (int head = 0; head < tail; head++) {
      int c = b.queue[head];
      for (int m : b.neigh[c]) {
        if (m >= 0 && b.open[m] && dist[m] == kInf) {
          dist[m] = dist[c] + 1;
          b.queue[tail++] = m;
        }
      }
    }
  }

  // how good the position is for the catcher, cat on `cat` and about to move; bigger is better
  int64_t value(const Buffers& b, int cat, const std::vector<int>& score, const std::vector<int>& dist) {
    int bestScore = kInf, bestDist = kInf, open = 0, ties = 0;
    for (int n : b.neigh[cat]) {
      if (n < 0 || !b.open[n]) {
        continue;
      }
      open++;
      if (score[n] < bestScore) {
        bestScore = score[n], ties = 1;
      } else if (score[n] == bestScore) {
        ties++;
      }
      bestDist = std::min(bestDist, dist[n]);
    }
    if (open == 0) {
      return INT64_MAX;  // trapped
    }
    int64_t v = int64_t(bestScore == kInf ? 1000 : bestScore) * 1000000 + int64_t(bestDist == kInf ? 1000 : bestDist) * 1000 - open;
    return v * 8 - ties;
  }

  // cells reachable from the cat (the cat's own cell included), with `skip` treated as blocked
  int regionSize(Buffers& b, int cat, int skip) {
    std::fill(b.count.begin(), b.count.end(), 0);  // reused as a visited mark
    int tail = 0;
    b.count[cat] = 1;
    b.queue[tail++] = cat;
    for (int head = 0; head < tail; head++) {
      for (int m : b.neigh[b.queue[head]]) {
        if (m >= 0 && b.open[m] && m != skip && !b.count[m]) {
          b.count[m] = 1;
          b.queue[tail++] = m;
        }
      }
    }
    return tail;
  }

  int openAround(const Buffers& b, int c, int skip) {
    int k = 0;
    for (int m : b.neigh[c]) {
      k += m >= 0 && b.open[m] && m != skip;
    }
    return k;
  }

  // sealed in: the block that leaves the cat the least room (region size, then its roomiest next cell)
  int trapMove(Buffers& b, int cat) {
    regionSize(b, cat, -1);
    std::vector<int> region;
    for (size_t i = 0; i < b.count.size(); i++) {
      if (b.count[i] && static_cast<int>(i) != cat) {
        region.push_back(static_cast<int>(i));
      }
    }
    int best = -1;
    int64_t bestKey = INT64_MAX;
    for (int x : region) {
      int64_t size = regionSize(b, cat, x), roomiest = 0, catOpen = openAround(b, cat, x);
      if (catOpen == 0) {
        return x;  // this block finishes the trap
      }
      for (int m : b.neigh[cat]) {
        if (m >= 0 && b.open[m] && m != x) {
          roomiest = std::max<int64_t>(roomiest, openAround(b, m, x));
        }
      }
      int64_t key = size * 64 + roomiest * 8 + catOpen;
      if (key < bestKey) {
        bestKey = key;
        best = x;
      }
    }
    return best;
  }

  // cells whose blocking can change value(): the cat's neighbors, and every cell reachable from them by
  // strictly decreasing two-distance or BFS distance (what their scores are built from)
  void markCandidates(Buffers& b, int cat) {
    std::fill(b.cand.begin(), b.cand.end(), 0);
    int tail = 0;
    for (int n : b.neigh[cat]) {
      if (n >= 0 && b.open[n]) {
        b.cand[n] = 1;
        b.queue[tail++] = n;
      }
    }
    for (int head = 0; head < tail; head++) {
      int c = b.queue[head];
      for (int m : b.neigh[c]) {
        if (m < 0 || !b.open[m] || b.cand[m] || m == cat) {
          continue;
        }
        if (b.score[m] < b.score[c] || b.dist[m] < b.dist[c]) {
          b.cand[m] = 1;
          b.queue[tail++] = m;
        }
      }
    }
  }

  // the catcher's best value with the cat on `cat` (cat to move after the block), over the cat's
  // neighbors and the nearest cells its scores depend on (at most `limit` blocks tried)
  int64_t bestFollowUp(Buffers& b, int cat, int limit) {
    twoDistance(b, b.nodeScore);
    bfsDistance(b, b.nodeDist);
    std::fill(b.cand.begin(), b.cand.end(), 0);
    b.list.clear();
    for (int n : b.neigh[cat]) {
      if (n >= 0 && b.open[n]) {
        b.cand[n] = 1;
        b.list.push_back(n);
      }
    }
    for (size_t head = 0; head < b.list.size() && static_cast<int>(b.list.size()) < limit; head++) {
      int c = b.list[head];
      for (int m : b.neigh[c]) {
        if (m < 0 || !b.open[m] || b.cand[m] || m == cat) {
          continue;
        }
        if (b.nodeScore[m] < b.nodeScore[c] || b.nodeDist[m] < b.nodeDist[c]) {
          b.cand[m] = 1;
          b.list.push_back(m);
        }
      }
    }
    int64_t best = value(b, cat, b.nodeScore, b.nodeDist);
    int tried = 0;
    const std::vector<int> list = b.list;
    for (int y : list) {
      if (tried++ >= limit) {
        break;
      }
      b.open[y] = 0;
      twoDistance(b, b.trialScore);
      bfsDistance(b, b.trialDist);
      best = std::max(best, value(b, cat, b.trialScore, b.trialDist));
      b.open[y] = 1;
    }
    return best;
  }
}  // namespace

Point2D Catcher::Move(CatWorld* world) {
  const int side = world->getWorldSideSize(), h = side / 2, total = side * side;
  Buffers& b = buffersFor(side);
  const auto& state = world->worldState();
  for (int i = 0; i < total; i++) {
    b.open[i] = !state[i];
  }
  const Point2D catPos = world->getCat();
  const int cat = (catPos.y + h) * side + catPos.x + h;
  auto toPoint = [&](int i) { return Point2D{i % side - h, i / side - h}; };

  twoDistance(b, b.score);
  bfsDistance(b, b.dist);
  const int64_t base = value(b, cat, b.score, b.dist);

  // sealed in: no neighbor of the cat reaches the border anymore
  bool sealed = true;
  for (int n : b.neigh[cat]) {
    if (n >= 0 && b.open[n] && b.dist[n] != kInf) {
      sealed = false;
    }
  }
  if (sealed) {
    int t = trapMove(b, cat);
    if (t >= 0) {
      return toPoint(t);
    }
  }

  // greedy pass over every open cell; cells that can't change the score keep the base value
  markCandidates(b, cat);
  int best = -1;
  int64_t bestValue = INT64_MIN;
  b.rootValues.clear();
  for (int i = 0; i < total; i++) {
    if (!b.open[i] || i == cat) {
      continue;
    }
    int64_t v = base;
    if (b.cand[i]) {
      b.open[i] = 0;
      twoDistance(b, b.trialScore);
      bfsDistance(b, b.trialDist);
      v = value(b, cat, b.trialScore, b.trialDist);
      b.rootValues.push_back({v, i});
      b.open[i] = 1;
    }
    if (v > bestValue) {
      bestValue = v;
      best = i;
    }
  }

  // lookahead on the best few blocks, unless one of them already traps the cat
  if (bestValue != INT64_MAX) {
    std::stable_sort(b.rootValues.begin(), b.rootValues.end(), [](const auto& a, const auto& c) { return a.first > c.first; });
    const auto roots = b.rootValues;
    int64_t bestWorst = INT64_MIN;
    for (int k = 0; k < static_cast<int>(roots.size()) && k < kSearchBlocks; k++) {
      int x = roots[k].second;
      b.open[x] = 0;
      int64_t worst = INT64_MAX;
      for (int m : b.neigh[cat]) {
        if (m < 0 || !b.open[m]) {
          continue;
        }
        if (b.isBorder[m]) {
          worst = INT64_MIN;  // the cat escapes
          break;
        }
        worst = std::min(worst, bestFollowUp(b, m, kFollowUpBlocks));
      }
      b.open[x] = 1;
      if (worst > bestWorst) {
        bestWorst = worst;
        best = x;
      }
    }
  }

  // the runner rejects a block on a blocked cell, the cat's cell or off the board
  if (best < 0 || !b.open[best] || best == cat) {
    for (int i = 0; i < total; i++) {
      if (b.open[i] && i != cat) {
        return toPoint(i);
      }
    }
  }
  if (best < 0) {
    return {catPos.x == h ? catPos.x - 1 : catPos.x + 1, catPos.y};  // nothing legal is left
  }
  return toPoint(best);
}
