#include "Catcher.h"
#include "World.h"
#include <algorithm>
#include <array>
#include <climits>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <vector>

// Catcher strategy:
//  1. Score a position by the cat's two-distance (an open border cell scores 0; any other open cell
//     scores 1 + the second smallest score among its neighbors), then its BFS distance, then how many of
//     its neighbors share the best score and how many are open. Bigger is better for the catcher.
//  2. Try every block that can change that score: the cat's neighbors and the cells their scores are
//     built from. Every other cell leaves the score as it is.
//  3. Look ahead on the 5 best blocks: assume the cat's best reply, then the catcher's best follow-up,
//     looking one more cat reply and follow-up deeper for the 2 best follow-ups, and keep the block
//     whose worst case is best. Blocks with the same worst case are split by how they do against a cat
//     that follows its shortest path (the generatePath cat).
//  4. Once the cat can no longer reach the border, stop defending and shrink its region: block the cell
//     that leaves the cat the least room.
//  5. Once the chosen block leaves the cat no finite two-distance (it can't force an escape), switch to the
//     block within 2 steps of the cat, among those that also leave it none, that traps a plain two-distance
//     cat soonest in a playout where the catcher blocks the cat's best next cell.
//  6. Before that, take the block within 2 steps whose playout traps that cat sooner, if the lookahead of
//     step 3 rates its worst case as good as the chosen block's.
// Blocking next to the cat (instead of walling the edge) caught last year's cats about 3 times faster.
//
// Speed: a block only reruns the pass it can change (two-distance or BFS distance), BFS distance is only
// computed for blocks that can still make the top 5, and the lookahead skips work that can't change the
// chosen block.

namespace {
  constexpr int kInf = INT_MAX;
  constexpr int kBlocked = INT_MAX - 1;  // a blocked cell's value in a distance array
  constexpr int kSearchBlocks = 5;       // best first blocks looked ahead on
  constexpr int kFollowUpBlocks = 16;    // follow-up blocks tried after each cat reply
  constexpr int kSearchDepth = 2;        // catcher moves looked ahead after the first block
  constexpr int kDeepFollowUps = 2;      // follow-up blocks given the deeper look at each level
  constexpr int kModelRadius = 2;        // model playouts try the open cells this many steps from the cat
  constexpr int kModelMoves = 40;        // cat moves a model playout runs at most
  constexpr int kModelEscape = 1 << 20;  // a model playout's length when the cat gets out
  constexpr int kSafetyChecks = 3;       // early model picks checked against the root's worst case

  // cells within kModelRadius steps of the cat
  constexpr int kModelCells = 1 + 3 * kModelRadius * (kModelRadius + 1);

  // the two-distance part of the score: best two-distance among the cat's open neighbors, how many share
  // it, and how many neighbors are open
  struct Escape {
    int bestScore = kInf;
    int ties = 0;
    int open = 0;
  };

  struct Buffers {
    int side = 0;
    // Every per-cell array has one extra entry at index side * side: a sentinel cell that is always
    // blocked and never a border. Off-board neighbors point to it.
    std::vector<std::array<int, 6>> neigh;  // neighbor linear indices
    std::vector<int> borders;
    std::vector<uint8_t> isBorder, open, count, cand;
    std::vector<uint8_t> inScore, inDist;    // cells the cat's neighbors' two-distance / BFS distance depend on
    std::vector<int> score, dist;            // the current board
    std::vector<int> trialScore, trialDist;  // after a hypothetical block
    std::vector<int> nodeScore, nodeDist;    // inside the lookahead
    std::vector<int> queue, list;
    std::vector<Escape> rootEscape;
    std::vector<int> rootBestScore;
    std::vector<std::pair<int64_t, int>> rootValues;
    std::vector<int> fresh;  // a distance pass's start: 0 on open border cells, kInf on other open
                             // cells, kBlocked on blocked cells and the sentinel
    std::vector<int> seeds;  // the open border cells, in index order (a pass's first queue entries)
    bool seedsStale = true;  // a border cell changed since `seeds` was built
  };

  Buffers& buffersFor(int side) {
    static Buffers b;
    if (b.side == side) {
      return b;
    }
    b.side = side;
    int h = side / 2, total = side * side;
    b.neigh.resize(total + 1);
    b.borders.clear();
    for (auto* v : {&b.open, &b.count, &b.cand, &b.inScore, &b.inDist}) {
      v->assign(total + 1, 0);
    }
    for (auto* v : {&b.score, &b.dist, &b.trialScore, &b.trialDist, &b.nodeScore, &b.nodeDist, &b.queue, &b.rootBestScore}) {
      v->resize(total + 1);
    }
    b.rootEscape.resize(total);
    b.isBorder.assign(total + 1, 0);
    b.fresh.assign(total + 1, kBlocked);
    b.seedsStale = true;
    b.neigh[total].fill(total);
    for (int i = 0; i < total; i++) {
      Point2D p = {i % side - h, i / side - h};
      if (std::abs(p.x) == h || std::abs(p.y) == h) {
        b.borders.push_back(i);
        b.isBorder[i] = 1;
      }
      auto ns = CatWorld::neighbors(p);
      for (int k = 0; k < 6; k++) {
        bool inside = std::abs(ns[k].x) <= h && std::abs(ns[k].y) <= h;
        b.neigh[i][k] = inside ? (ns[k].y + h) * side + ns[k].x + h : total;
      }
    }
    return b;
  }

  // opens or blocks a cell
  void setOpen(Buffers& b, int i, bool isOpen) {
    b.open[i] = isOpen;
    if (!isOpen) {
      b.fresh[i] = kBlocked;
    } else if (b.isBorder[i]) {
      b.fresh[i] = 0;
    } else {
      b.fresh[i] = kInf;
    }
    if (b.isBorder[i]) {
      b.seedsStale = true;
    }
  }

  // Distance passes
  // Raw pointers instead of vector calls, and the 6 neighbors written out instead of looped. A pass starts from the template, so one
  // compare (== kInf) means "open and not reached yet", and off-board neighbors point at the sentinel.

  // starts a pass: `dist` from the template, the open border cells queued; returns how many were queued
  int startPass(Buffers& b, std::vector<int>& dist) {
    if (b.seedsStale) {
      b.seeds.clear();
      for (int i : b.borders) {
        if (b.open[i]) {
          b.seeds.push_back(i);
        }
      }
      b.seedsStale = false;
    }
    memcpy(dist.data(), b.fresh.data(), sizeof(int) * b.fresh.size());
    memcpy(b.queue.data(), b.seeds.data(), sizeof(int) * b.seeds.size());
    return static_cast<int>(b.seeds.size());
  }

  // cells leave the queue in nondecreasing score, so the second neighbor to leave is the second best one
  void twoDistance(Buffers& b, std::vector<int>& scoreVec) {
    int tail = startPass(b, scoreVec);
    int* score = scoreVec.data();
    uint8_t* count = b.count.data();
    int* queue = b.queue.data();
    const int* neigh = b.neigh[0].data();
    memset(count, 0, b.count.size());
    for (int head = 0; head < tail; head++) {
      const int c = queue[head];
      const int next = score[c] + 1;
      const int* nb = neigh + c * 6;
      const int m0 = nb[0];
      if (score[m0] == kInf) {
        count[m0]++;
        if (count[m0] == 2) {
          score[m0] = next;
          queue[tail] = m0;
          tail++;
        }
      }
      const int m1 = nb[1];
      if (score[m1] == kInf) {
        count[m1]++;
        if (count[m1] == 2) {
          score[m1] = next;
          queue[tail] = m1;
          tail++;
        }
      }
      const int m2 = nb[2];
      if (score[m2] == kInf) {
        count[m2]++;
        if (count[m2] == 2) {
          score[m2] = next;
          queue[tail] = m2;
          tail++;
        }
      }
      const int m3 = nb[3];
      if (score[m3] == kInf) {
        count[m3]++;
        if (count[m3] == 2) {
          score[m3] = next;
          queue[tail] = m3;
          tail++;
        }
      }
      const int m4 = nb[4];
      if (score[m4] == kInf) {
        count[m4]++;
        if (count[m4] == 2) {
          score[m4] = next;
          queue[tail] = m4;
          tail++;
        }
      }
      const int m5 = nb[5];
      if (score[m5] == kInf) {
        count[m5]++;
        if (count[m5] == 2) {
          score[m5] = next;
          queue[tail] = m5;
          tail++;
        }
      }
    }
  }

  void bfsDistance(Buffers& b, std::vector<int>& distVec) {
    int tail = startPass(b, distVec);
    int* dist = distVec.data();
    int* queue = b.queue.data();
    const int* neigh = b.neigh[0].data();
    for (int head = 0; head < tail; head++) {
      const int c = queue[head];
      const int next = dist[c] + 1;
      const int* nb = neigh + c * 6;
      const int m0 = nb[0];
      if (dist[m0] == kInf) {
        dist[m0] = next;
        queue[tail] = m0;
        tail++;
      }
      const int m1 = nb[1];
      if (dist[m1] == kInf) {
        dist[m1] = next;
        queue[tail] = m1;
        tail++;
      }
      const int m2 = nb[2];
      if (dist[m2] == kInf) {
        dist[m2] = next;
        queue[tail] = m2;
        tail++;
      }
      const int m3 = nb[3];
      if (dist[m3] == kInf) {
        dist[m3] = next;
        queue[tail] = m3;
        tail++;
      }
      const int m4 = nb[4];
      if (dist[m4] == kInf) {
        dist[m4] = next;
        queue[tail] = m4;
        tail++;
      }
      const int m5 = nb[5];
      if (dist[m5] == kInf) {
        dist[m5] = next;
        queue[tail] = m5;
        tail++;
      }
    }
  }

  Escape escapeOf(const Buffers& b, int cat, const std::vector<int>& score) {
    Escape e;
    for (int n : b.neigh[cat]) {
      if (n < 0 || !b.open[n]) {
        continue;
      }
      e.open++;
      if (score[n] < e.bestScore) {
        e.bestScore = score[n];
        e.ties = 1;
      } else if (score[n] == e.bestScore) {
        e.ties++;
      }
    }
    return e;
  }

  int bestDistOf(const Buffers& b, int cat, const std::vector<int>& dist) {
    int bestDist = kInf;
    for (int n : b.neigh[cat]) {
      if (n >= 0 && b.open[n]) {
        bestDist = std::min(bestDist, dist[n]);
      }
    }
    return bestDist;
  }

  // how good the position is for the catcher, cat about to move; bigger is better. The two-distance term
  // dominates: the BFS term adds under 1000000 whenever the two-distance is finite
  int64_t valueOf(const Escape& e, int bestDist) {
    if (e.open == 0) {
      return INT64_MAX;  // trapped
    }
    int64_t v = int64_t(e.bestScore == kInf ? 1000 : e.bestScore) * 1000000 + int64_t(bestDist == kInf ? 1000 : bestDist) * 1000 - e.open;
    return v * 8 - e.ties;
  }

  int64_t value(const Buffers& b, int cat, const std::vector<int>& score, const std::vector<int>& dist) {
    return valueOf(escapeOf(b, cat, score), bestDistOf(b, cat, dist));
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
    for (int i = 0; i < b.side * b.side; i++) {
      if (b.count[i] && i != cat) {
        region.push_back(i);
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

  // the cells one kind of value (`v`) of the cat's neighbors depends on: reachable from them by strictly
  // decreasing `v`. Blocking any other cell leaves that kind of value unchanged at the neighbors.
  void markSupport(Buffers& b, int cat, const std::vector<int>& v, std::vector<uint8_t>& mark) {
    std::fill(mark.begin(), mark.end(), 0);
    int tail = 0;
    for (int n : b.neigh[cat]) {
      if (n >= 0 && b.open[n]) {
        mark[n] = 1;
        b.queue[tail++] = n;
      }
    }
    for (int head = 0; head < tail; head++) {
      int c = b.queue[head];
      for (int m : b.neigh[c]) {
        if (m >= 0 && b.open[m] && !mark[m] && m != cat && v[m] < v[c]) {
          mark[m] = 1;
          b.queue[tail++] = m;
        }
      }
    }
  }

  // the catcher's best value with the cat on `cat` (cat to move after the block), over the cat's
  // neighbors and the nearest cells its scores depend on (at most `limit` blocks tried). Stops once it
  // reaches `cap`: the caller only needs to know the value isn't below it.
  int64_t bestFollowUp(Buffers& b, int cat, int limit, int64_t cap) {
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
    const Escape nodeEscape = escapeOf(b, cat, b.nodeScore);
    const int nodeBestDist = bestDistOf(b, cat, b.nodeDist);
    int64_t best = valueOf(nodeEscape, nodeBestDist);
    if (best >= cap) {
      return best;
    }
    std::vector<int> list = b.list;
    if (static_cast<int>(list.size()) > limit) {
      list.resize(limit);
    }
    markSupport(b, cat, b.nodeScore, b.inScore);
    markSupport(b, cat, b.nodeDist, b.inDist);
    for (int y : list) {
      setOpen(b, y, false);
      Escape e = nodeEscape;
      if (b.inScore[y]) {
        twoDistance(b, b.trialScore);
        e = escapeOf(b, cat, b.trialScore);
      }
      int bestDist = nodeBestDist;
      if (b.inDist[y] && e.open > 0) {
        bfsDistance(b, b.trialDist);
        bestDist = bestDistOf(b, cat, b.trialDist);
      }
      setOpen(b, y, true);
      best = std::max(best, valueOf(e, bestDist));
      if (best >= cap) {
        return best;
      }
    }
    return best;
  }

  // every follow-up block's value (the same blocks bestFollowUp tries), best first
  std::vector<std::pair<int64_t, int>> followUpValues(Buffers& b, int cat, int limit, int64_t& nodeValue) {
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
    const Escape nodeEscape = escapeOf(b, cat, b.nodeScore);
    const int nodeBestDist = bestDistOf(b, cat, b.nodeDist);
    nodeValue = valueOf(nodeEscape, nodeBestDist);
    std::vector<int> list = b.list;
    if (static_cast<int>(list.size()) > limit) {
      list.resize(limit);
    }
    markSupport(b, cat, b.nodeScore, b.inScore);
    markSupport(b, cat, b.nodeDist, b.inDist);
    std::vector<std::pair<int64_t, int>> values;
    for (int y : list) {
      setOpen(b, y, false);
      Escape e = nodeEscape;
      if (b.inScore[y]) {
        twoDistance(b, b.trialScore);
        e = escapeOf(b, cat, b.trialScore);
      }
      int bestDist = nodeBestDist;
      if (b.inDist[y] && e.open > 0) {
        bfsDistance(b, b.trialDist);
        bestDist = bestDistOf(b, cat, b.trialDist);
      }
      setOpen(b, y, true);
      values.push_back({valueOf(e, bestDist), y});
    }
    std::stable_sort(values.begin(), values.end(), [](const auto& a, const auto& c) { return a.first > c.first; });
    return values;
  }

  // the cat's replies from `cat` (after the block just made), most dangerous first; false if it can
  // step onto the border
  bool catReplies(Buffers& b, int cat, std::array<std::pair<int, int>, 6>& replies, int& count) {
    twoDistance(b, b.trialScore);
    count = 0;
    bool escapes = false;
    for (int m : b.neigh[cat]) {
      if (m < 0 || !b.open[m]) {
        continue;
      }
      if (b.isBorder[m]) {
        escapes = true;
      }
      replies[count++] = {b.trialScore[m], m};
    }
    std::stable_sort(replies.begin(), replies.begin() + count, [](const auto& a, const auto& c) { return a.first < c.first; });
    return !escapes;
  }

  // the catcher to move, cat on `cat`. depth 1 = the best follow-up block by its immediate value; deeper =
  // the best of the kDeepFollowUps best follow-ups, each judged by the cat's best reply and the next level.
  // Stops once it reaches `cap` (the caller only needs to know it isn't below it).
  int64_t catcherValue(Buffers& b, int cat, int depth, int64_t cap) {
    if (depth <= 1) {
      return bestFollowUp(b, cat, kFollowUpBlocks, cap);
    }
    int64_t nodeValue = 0;
    auto values = followUpValues(b, cat, kFollowUpBlocks, nodeValue);
    if (values.empty()) {
      return nodeValue;
    }
    if (values[0].first == INT64_MAX) {
      return INT64_MAX;  // a follow-up traps the cat
    }
    int64_t best = INT64_MIN;
    for (int k = 0; k < static_cast<int>(values.size()) && k < kDeepFollowUps; k++) {
      int y = values[k].second;
      setOpen(b, y, false);
      std::array<std::pair<int, int>, 6> replies;
      int count = 0;
      int64_t worst = INT64_MIN;
      if (catReplies(b, cat, replies, count)) {
        worst = INT64_MAX;
        for (int r = 0; r < count && worst > best; r++) {
          worst = std::min(worst, catcherValue(b, replies[r].second, depth - 1, worst));
        }
      }
      setOpen(b, y, true);
      best = std::max(best, worst);
      if (best >= cap) {
        return best;
      }
    }
    return best;
  }

  // Model Playouts: once the cat can't force an escape, catch a plain two-distance cat fastest

  // the cat's best next cell on this board: lowest two-distance, then BFS distance, then most open neighbors; -1 when it has no open neighbor
  int bestNextCell(const Buffers& b, int cat) {
    int best = -1;
    int64_t bestKey = INT64_MAX;
    for (int n : b.neigh[cat]) {
      if (!b.open[n]) {
        continue;
      }
      const int64_t score = b.nodeScore[n] >= kBlocked ? 1000 : b.nodeScore[n];
      const int64_t dist = b.nodeDist[n] >= kBlocked ? 1000 : b.nodeDist[n];
      const int64_t key = (score * 1024 + dist) * 8 + (6 - openAround(b, n, -1));
      if (key < bestKey) {
        bestKey = key;
        best = n;
      }
    }
    return best;
  }

  // the model cat's move: a border cell if it can reach one, else its best next cell; -1 when it is trapped
  int modelCatMove(Buffers& b, int cat) {
    twoDistance(b, b.nodeScore);
    bfsDistance(b, b.nodeDist);
    for (int n : b.neigh[cat]) {
      if (b.open[n] && b.isBorder[n]) {
        return n;
      }
    }
    return bestNextCell(b, cat);
  }

  // Runs until the model cat is trapped (cat to move on `cat`); kModelEscape if it gets out. Stops at
  // `cutoff` once the playout can no longer end sooner than that, since the caller only keeps a shorter one.
  int playout(Buffers& b, int cat, int cutoff) {
    int placed[kModelMoves];
    int count = 0;
    int plies = 0;
    int result = -1;
    for (int t = 0; t < kModelMoves && result < 0; t++) {
      if (plies + 2 >= cutoff) {
        result = cutoff;
        break;
      }
      const int m = modelCatMove(b, cat);
      if (m < 0) {
        result = plies;
        break;
      }
      cat = m;
      plies++;
      if (b.isBorder[m]) {
        result = kModelEscape;
        break;
      }
      // The playout catcher blocks the cat's best next cell, from the distances modelCatMove just computed.
      // About 10 times cheaper per playout move than a real catcher, and its playouts ranked the blocks better.
      const int x = bestNextCell(b, cat);
      if (x < 0 || !b.open[x] || x == cat) {
        result = kModelEscape;
        break;
      }
      setOpen(b, x, false);
      placed[count++] = x;
      plies++;
      if (openAround(b, cat, -1) == 0) {
        result = plies;
      }
    }
    if (result < 0) {
      result = plies + 1000;  // not caught within the playout: long
    }
    for (int k = count - 1; k >= 0; k--) {
      setOpen(b, placed[k], true);
    }
    return result;
  }

  // after blocking x, does every open neighbor of the cat have an infinite two-distance (no forced escape)?
  bool keepsCatIn(Buffers& b, int cat, int x) {
    setOpen(b, x, false);
    twoDistance(b, b.trialScore);
    const Escape e = escapeOf(b, cat, b.trialScore);
    setOpen(b, x, true);
    return e.open == 0 || e.bestScore == kInf;
  }

  // The cells within kModelRadius steps of the cat, nearest first, the cat's own cell at 0 (19 for a radius
  // of 2); returns how many
  int cellsNear(const Buffers& b, int cat, int* cells) {
    int steps[kModelCells];
    int count = 0;
    cells[count] = cat;
    steps[count] = 0;
    count++;
    const int total = b.side * b.side;
    for (int head = 0; head < count; head++) {
      if (steps[head] == kModelRadius) {
        continue;
      }
      for (int m : b.neigh[cells[head]]) {
        if (m >= total) {
          continue;
        }
        bool seen = false;
        for (int k = 0; k < count; k++) {
          if (cells[k] == m) {
            seen = true;
          }
        }
        if (!seen) {
          cells[count] = m;
          steps[count] = steps[head] + 1;
          count++;
        }
      }
    }
    return count;
  }

  // the worst case after blocking x, by the same search as the root lookahead, is at least `need`
  bool worstAtLeast(Buffers& b, int cat, int x, int64_t need) {
    setOpen(b, x, false);
    std::array<std::pair<int, int>, 6> replies;
    int count = 0;
    bool ok = false;
    if (catReplies(b, cat, replies, count)) {
      ok = true;
      for (int r = 0; r < count && ok; r++) {
        if (catcherValue(b, replies[r].second, kSearchDepth, need) < need) {
          ok = false;
        }
      }
    }
    setOpen(b, x, true);
    return ok;
  }

  // before the escape is shut: the blocks within kModelRadius steps whose playouts trap the model cat sooner
  // than `best`'s, shortest first, and the first of up to kSafetyChecks whose worst case is as good as
  // `best`'s (rootWorst).
  int earlyBlock(Buffers& b, int cat, int best, int64_t rootWorst) {
    if (rootWorst == INT64_MIN) {
      return best;
    }
    int cells[kModelCells];
    const int count = cellsNear(b, cat, cells);
    setOpen(b, best, false);
    const int bestLen = openAround(b, cat, -1) == 0 ? 0 : playout(b, cat, INT_MAX);
    setOpen(b, best, true);
    std::array<std::pair<int, int>, kModelCells> better;
    int betterCount = 0;
    for (int k = 1; k < count; k++) {
      const int x = cells[k];
      if (x == best || !b.open[x]) {
        continue;
      }
      setOpen(b, x, false);
      const int len = openAround(b, cat, -1) == 0 ? 0 : playout(b, cat, bestLen);
      setOpen(b, x, true);
      if (len < bestLen) {
        better[betterCount++] = {len, x};
      }
    }
    std::stable_sort(better.begin(), better.begin() + betterCount);
    for (int k = 0; k < betterCount && k < kSafetyChecks; k++) {
      if (worstAtLeast(b, cat, better[k].second, rootWorst)) {
        return better[k].second;
      }
    }
    return best;
  }

  // the block to play instead of `best`: once the escape is shut, among blocks within kModelRadius steps of
  // the cat that keep it shut, the one whose playout traps the model cat soonest (`best` wins ties); before
  // that, earlyBlock
  int modelBlock(Buffers& b, int cat, int best, int64_t rootWorst) {
    // blocking only raises two-distances, so if every escape is already shut any block keeps it shut, and a
    // block outside the cells the cat's neighbors' two-distances are built from can't shut an open one; the
    // pass only runs for the rest. b.score is still this board's two-distance.
    bool allShut = true;
    for (int n : b.neigh[cat]) {
      if (b.open[n] && b.score[n] != kInf) {
        allShut = false;
      }
    }
    if (!allShut) {
      markSupport(b, cat, b.score, b.inScore);
      if (!b.inScore[best] || !keepsCatIn(b, cat, best)) {
        return earlyBlock(b, cat, best, rootWorst);
      }
    }
    int cells[kModelCells];
    const int count = cellsNear(b, cat, cells);
    // decided before the playouts, which reuse inScore
    bool keeps[kModelCells];
    for (int k = 1; k < count; k++) {
      const int x = cells[k];
      keeps[k] = x != best && b.open[x] && (allShut || (b.inScore[x] && keepsCatIn(b, cat, x)));
    }
    setOpen(b, best, false);
    int pickLen = openAround(b, cat, -1) == 0 ? 0 : playout(b, cat, INT_MAX);
    setOpen(b, best, true);
    int pick = best;
    for (int k = 1; k < count; k++) {
      const int x = cells[k];
      if (!keeps[k]) {
        continue;
      }
      setOpen(b, x, false);
      const int len = openAround(b, cat, -1) == 0 ? 0 : playout(b, cat, pickLen);
      setOpen(b, x, true);
      if (len < pickLen) {
        pickLen = len;
        pick = x;
      }
    }
    return pick;
  }

}  // namespace

Point2D Catcher::Move(CatWorld* world) {
  const int side = world->getWorldSideSize(), h = side / 2, total = side * side;
  Buffers& b = buffersFor(side);
  const auto& state = world->worldState();
  for (int i = 0; i < total; i++) {
    setOpen(b, i, !state[i]);
  }
  const Point2D catPos = world->getCat();
  const int cat = (catPos.y + h) * side + catPos.x + h;
  auto toPoint = [&](int i) { return Point2D{i % side - h, i / side - h}; };

  twoDistance(b, b.score);
  bfsDistance(b, b.dist);
  const Escape baseEscape = escapeOf(b, cat, b.score);
  const int baseBestDist = bestDistOf(b, cat, b.dist);
  const int64_t base = valueOf(baseEscape, baseBestDist);

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

  // first pass: the two-distance part for every candidate (rerun only where a block can change it)
  markCandidates(b, cat);
  markSupport(b, cat, b.score, b.inScore);
  markSupport(b, cat, b.dist, b.inDist);
  std::vector<int> candidateScores;
  for (int i = 0; i < total; i++) {
    if (!b.open[i] || i == cat || !b.cand[i]) {
      continue;
    }
    Escape e = baseEscape;
    if (b.inScore[i]) {
      setOpen(b, i, false);
      twoDistance(b, b.trialScore);
      e = escapeOf(b, cat, b.trialScore);
      setOpen(b, i, true);
    }
    b.rootEscape[i] = e;
    b.rootBestScore[i] = e.open == 0 ? INT_MAX : e.bestScore;  // a trapping block outranks everything
    candidateScores.push_back(b.rootBestScore[i]);
  }
  // candidates with a lower two-distance than the 5th best can't be the best block or reach the lookahead
  int scoreCut = INT_MIN;
  if (static_cast<int>(candidateScores.size()) > kSearchBlocks) {
    std::nth_element(candidateScores.begin(), candidateScores.begin() + (kSearchBlocks - 1), candidateScores.end(), std::greater<int>());
    scoreCut = candidateScores[kSearchBlocks - 1];
  }

  // second pass: full values; cells that can't change the score keep the base value
  int best = -1;
  int64_t bestValue = INT64_MIN;
  b.rootValues.clear();
  for (int i = 0; i < total; i++) {
    if (!b.open[i] || i == cat) {
      continue;
    }
    if (b.cand[i] && b.rootBestScore[i] < scoreCut) {
      continue;
    }
    int64_t v = base;
    if (b.cand[i]) {
      const Escape& e = b.rootEscape[i];
      int bestDist = baseBestDist;
      if (b.inDist[i] && e.open > 0) {
        setOpen(b, i, false);
        bfsDistance(b, b.trialDist);
        bestDist = bestDistOf(b, cat, b.trialDist);
        setOpen(b, i, true);
      }
      v = valueOf(e, bestDist);
      b.rootValues.push_back({v, i});
    }
    if (v > bestValue) {
      bestValue = v;
      best = i;
    }
  }

  // lookahead on the best few blocks, unless one of them already traps the cat
  int64_t rootWorst = INT64_MIN;  // the chosen block's worst case, for modelBlock
  if (bestValue != INT64_MAX) {
    std::stable_sort(b.rootValues.begin(), b.rootValues.end(), [](const auto& a, const auto& c) { return a.first > c.first; });
    const auto roots = b.rootValues;
    int64_t bestWorst = INT64_MIN;
    std::array<int64_t, 16> worstOf;
    const int rootCount = std::min<int>(static_cast<int>(roots.size()), kSearchBlocks);
    for (int k = 0; k < rootCount; k++) {
      int x = roots[k].second;
      setOpen(b, x, false);
      std::array<std::pair<int, int>, 6> replies;
      int replyCount = 0;
      int64_t worst = INT64_MIN;  // the cat escapes unless catReplies says otherwise
      if (catReplies(b, cat, replies, replyCount)) {
        worst = INT64_MAX;
        // stop once this block can no longer match the best one found so far (a tie still counts: the
        // tie-break below needs its exact worst case)
        for (int r = 0; r < replyCount && worst >= bestWorst; r++) {
          worst = std::min(worst, catcherValue(b, replies[r].second, kSearchDepth, worst == bestWorst ? INT64_MAX : worst));
        }
      }
      setOpen(b, x, true);
      worstOf[k] = worst;
      if (worst > bestWorst) {
        bestWorst = worst;
        best = x;
      }
    }

    rootWorst = bestWorst;

    // cat model tie-break: among blocks with the same worst case, the one that does best against a cat
    // that follows its shortest path (the generatePath cat)
    if (bestWorst != INT64_MIN) {
      int ties = 0;
      for (int k = 0; k < rootCount; k++) {
        ties += worstOf[k] == bestWorst;
      }
      if (ties > 1) {
        int64_t bestModel = INT64_MIN;
        for (int k = 0; k < rootCount; k++) {
          if (worstOf[k] != bestWorst) {
            continue;
          }
          int x = roots[k].second;
          setOpen(b, x, false);
          bfsDistance(b, b.trialDist);
          int step = -1;
          for (int m : b.neigh[cat]) {
            if (m >= 0 && b.open[m] && (step < 0 || b.trialDist[m] < b.trialDist[step])) {
              step = m;
            }
          }
          int64_t modelValue = step < 0 ? INT64_MAX : catcherValue(b, step, 1, INT64_MAX);
          setOpen(b, x, true);
          if (modelValue > bestModel) {
            bestModel = modelValue;
            best = x;
          }
        }
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
  best = modelBlock(b, cat, best, rootWorst);
  return toPoint(best);
}
