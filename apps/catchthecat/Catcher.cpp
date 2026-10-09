#include "Catcher.h"
#include "Grid.h"
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
//     block within 3 steps of the cat, among those that also leave it none, that traps a model cat soonest
//     in a playout. The playout catcher is steps 1, 2 and 4 without the lookahead, trying only blocks
//     within 2 steps of the cat.
//  6. Before that, take the block within 3 steps whose playout traps that cat sooner, if the lookahead of
//     step 3 rates its worst case at most one two-distance step below the chosen block's (up to 3 tried,
//     shortest playout first).
//  7. The model cat is a plain two-distance cat, unless the cat's last 5 steps, rebuilt from the board, don't
//     fit a two-distance cat (ranked like AaronArchambault's: two-distance, BFS distance, more shortest
//     paths, more room) but do fit one of these: a cat that walks its shortest path to the nearest edge (the
//     generatePath cat), which a catcher that knows its route traps far sooner; else a lookahead cat, which
//     weighs each step against the catcher's worst wall next. Steps that fit a two-distance cat and a copy of
//     AaronArchambault's cat, but not the lookahead cat, make the model his cat.
// Blocking next to the cat (instead of walling the edge) caught last year's cats about 3 times faster.
//
// Speed: a block only reruns the pass it can change (two-distance or BFS distance), BFS distance is only
// computed for blocks that can still make the top 5, and the lookahead skips work that can't change the
// chosen block. The board and the distance passes are in Grid.cpp (shared with the cat).

namespace {
  using grid::bfsDistance;
  using grid::kBlocked;
  using grid::kInf;
  using grid::setOpen;
  using grid::twoDistance;

  constexpr int kSearchBlocks = 5;         // best first blocks looked ahead on
  constexpr int kFollowUpBlocks = 16;      // follow-up blocks tried after each cat reply
  constexpr int kSearchDepth = 2;          // catcher moves looked ahead after the first block
  constexpr int kDeepFollowUps = 2;        // follow-up blocks given the deeper look at each level
  constexpr int kModelRadius = 3;          // model playouts try the open cells this many steps from the cat
  constexpr int kModelMoves = 40;          // cat moves a model playout runs at most
  constexpr int kModelEscape = 1 << 20;    // a model playout's length when the cat gets out
  constexpr int kSafetyChecks = 3;         // early model picks checked against the root's worst case
  constexpr int64_t kScoreStep = 8000000;  // one two-distance step in a value (see valueOf)
  constexpr int kHistorySteps = 5;         // cat steps rebuilt to tell which model cat fits
  constexpr int kHistoryBudget = 400;      // positions each model may try while rebuilding them
  constexpr int kAaronBudget = 100;        // ... AaronArchambault's cat, which fails when it runs out
  constexpr int kHistoryBlocks = 18;       // cells near an earlier cat cell where its next block may have been

  // cells within kModelRadius steps of the cat
  constexpr int kModelCells = 1 + 3 * kModelRadius * (kModelRadius + 1);

  // the two-distance part of the score: best two-distance among the cat's open neighbors, how many share
  // it, and how many neighbors are open
  struct Escape {
    int bestScore = kInf;
    int ties = 0;
    int open = 0;
  };

  // the shared board and the catcher's own arrays,
  // each with the sentinel entry at index side * side
  struct Buffers : grid::Board {
    std::vector<uint8_t> cand;
    std::vector<uint8_t> inScore, inDist;    // cells the cat's neighbors' two-distance / BFS distance depend on
    std::vector<int> score, dist;            // the current board
    std::vector<int> trialScore, trialDist;  // after a hypothetical block
    std::vector<int> nodeScore, nodeDist;    // inside the lookahead
    std::vector<int> list;
    std::vector<Escape> rootEscape;
    std::vector<int> rootBestScore;
    std::vector<std::pair<int64_t, int>> rootValues;
    std::vector<int> parent;                             // the path cat's search
    std::vector<double> paths;                           // shortest paths to the border from each cell (nodeDist's board)
    std::vector<int> lookScore, lookDist;                // the lookahead cat: the board's passes before any wall
    std::vector<uint8_t> lookScoreFrom, lookDistFrom;    // and the cells a step's best values are built from
    std::vector<int> aaronScore, aaronDist, aaronFence;  // AaronArchambault's cat: the board's passes
    std::vector<double> aaronPaths;
  };

  Buffers& buffersFor(int side) {
    static Buffers b;
    if (b.side == side) {
      return b;
    }
    grid::setSide(b, side);
    const int total = b.total;
    for (auto* v : {&b.cand, &b.inScore, &b.inDist, &b.lookScoreFrom, &b.lookDistFrom}) {
      v->assign(total + 1, 0);
    }
    for (auto* v : {&b.score, &b.dist, &b.trialScore, &b.trialDist, &b.nodeScore, &b.nodeDist, &b.rootBestScore, &b.parent, &b.lookScore, &b.lookDist,
                    &b.aaronScore, &b.aaronDist, &b.aaronFence}) {
      v->resize(total + 1);
    }
    for (auto* v : {&b.paths, &b.aaronPaths}) {
      v->resize(total + 1);
    }
    b.rootEscape.resize(total);
    return b;
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

  // Search

  // steps 1 to 4 on the current board: the block to play before the model step (-1 if none), the chosen
  // block's worst case in rootWorst (INT64_MIN when the lookahead didn't run), and trapping set when the cat
  // is sealed in and the block shrinks its region. 'playoutCatcher' skips the lookahead and only tries blocks within 2 steps of the cat.
  int searchBlock(Buffers& b, int cat, bool playoutCatcher, int64_t& rootWorst, bool& trapping) {
    const int total = b.side * b.side;
    rootWorst = INT64_MIN;
    trapping = false;
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
        trapping = true;
        return t;
      }
    }

    // first pass: the two-distance part for every candidate (rerun only where a block can change it)
    markCandidates(b, cat);
    if (playoutCatcher) {
      // only the cells within 2 steps of the cat (inScore is scratch until markSupport fills it)
      std::fill(b.inScore.begin(), b.inScore.end(), 0);
      for (int n : b.neigh[cat]) {
        b.inScore[n] = 1;
        for (int m : b.neigh[n]) {
          b.inScore[m] = 1;
        }
      }
      for (int i = 0; i < total; i++) {
        b.cand[i] = b.cand[i] && b.inScore[i];
      }
    }
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
    if (bestValue != INT64_MAX && !playoutCatcher) {
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

  enum Model { kTwoDistanceCat, kPathCat, kLookaheadCat, kAaronCat };

  // the two-distance cat's move: a border cell if it can reach one, else its best next cell; -1 when it is
  // trapped
  int twoDistanceCatMove(Buffers& b, int cat) {
    twoDistance(b, b.nodeScore);
    bfsDistance(b, b.nodeDist);
    for (int n : b.neigh[cat]) {
      if (b.open[n] && b.isBorder[n]) {
        return n;
      }
    }
    return bestNextCell(b, cat);
  }

  // the path cat's move (the generatePath cat): BFS from the cat in neighbor order, stopping at the first
  // border cell found, and the first step of that path. With no border reachable, its first open neighbor;
  // -1 when it is trapped.
  int pathCatMove(Buffers& b, int cat) {
    const int total = b.side * b.side;
    std::fill(b.parent.begin(), b.parent.end(), -1);
    b.parent[total] = total;  // the sentinel counts as seen
    b.parent[cat] = cat;
    int tail = 0;
    b.queue[tail++] = cat;
    int exit = -1;
    for (int head = 0; head < tail && exit < 0; head++) {
      const int c = b.queue[head];
      for (int m : b.neigh[c]) {
        if (b.parent[m] >= 0 || !b.open[m]) {
          continue;
        }
        b.parent[m] = c;
        if (b.isBorder[m]) {
          exit = m;
          break;
        }
        b.queue[tail++] = m;
      }
    }
    if (exit < 0) {
      for (int n : b.neigh[cat]) {
        if (b.open[n]) {
          return n;
        }
      }
      return -1;
    }
    while (b.parent[exit] != cat) {
      exit = b.parent[exit];
    }
    return exit;
  }

  // open cells within 2 steps of p (counted once per route through a neighbor), not counting `from`
  int roomAround(const Buffers& b, int p, int from) {
    int count = 0;
    for (int a : b.neigh[p]) {
      if (!b.open[a] || a == from) {
        continue;
      }
      count++;
      for (int c : b.neigh[a]) {
        if (b.open[c] && c != from && c != p) {
          count++;
        }
      }
    }
    return count;
  }

  // shortest paths to the border from each cell, for `dist` (this board's BFS distance); 0 where it is unreachable
  void countPaths(Buffers& b, const std::vector<int>& dist, std::vector<double>& paths) {
    std::fill(paths.begin(), paths.end(), 0.0);
    int tail = 0;
    for (int i : b.borders) {
      if (b.open[i]) {
        paths[i] = 1.0;
        b.queue[tail++] = i;
      }
    }
    for (int head = 0; head < tail; head++) {
      const int c = b.queue[head];
      for (int n : b.neigh[c]) {
        if (b.open[n] && dist[n] == dist[c] + 1) {
          if (paths[n] == 0.0) {
            b.queue[tail++] = n;
          }
          paths[n] += paths[c];
        }
      }
    }
  }

  // AaronArchambault's cat (AaronArchambault/mobagen 0b09856, Cat.cpp), rewritten to make the same moves faster.
  // His steps are ranked by two-distance, BFS distance, more shortest paths, farther from the blocked border cells
  // (ignoring other blocks), then more open cells within 2 steps, and he takes the first. When one block could cut
  // that step off from the border he also looks one block ahead, but that never changed his step (none of about
  // 29,000 such positions), so the copy leaves it out: it cost more than the rest of the catcher's playouts.

  // his fence: steps from the nearest blocked border cell, ignoring other blocks (kInf when there is none)
  void aaronFence(Buffers& b) {
    const int total = b.side * b.side;
    int* fence = b.aaronFence.data();
    int tail = 0;
    for (int i = 0; i < total; i++) {
      fence[i] = kInf;
    }
    fence[total] = 0;  // the sentinel is never stepped into
    for (int i : b.borders) {
      if (!b.open[i]) {
        fence[i] = 0;
        b.queue[tail++] = i;
      }
    }
    for (int head = 0; head < tail; head++) {
      const int c = b.queue[head];
      for (int m : b.neigh[c]) {
        if (fence[m] == kInf) {
          fence[m] = fence[c] + 1;
          b.queue[tail++] = m;
        }
      }
    }
  }

  // the board's passes his cat ranks its steps by (the same wherever the cat stands)
  void aaronPasses(Buffers& b) {
    twoDistance(b, b.aaronScore);
    bfsDistance(b, b.aaronDist);
    countPaths(b, b.aaronDist, b.aaronPaths);
  }

  // his cat's step from `cat`, aaronPasses holding this board's passes; -1 when it has no open neighbor
  int aaronStep(Buffers& b, int cat) {
    const int* score = b.aaronScore.data();
    const int* dist = b.aaronDist.data();
    const double* paths = b.aaronPaths.data();
    bool fenceReady = false;
    int best = -1;
    for (int n : b.neigh[cat]) {
      if (!b.open[n]) {
        continue;
      }
      if (best < 0) {
        best = n;
        continue;
      }
      // ties keep the earlier step in neighbor order, as his stable sort does
      bool better = false;
      if (score[n] != score[best]) {
        better = score[n] < score[best];
      } else if (dist[n] != dist[best]) {
        better = dist[n] < dist[best];
      } else if (paths[n] != paths[best]) {
        better = paths[n] > paths[best];
      } else {
        if (!fenceReady) {
          aaronFence(b);
          fenceReady = true;
        }
        if (b.aaronFence[n] != b.aaronFence[best]) {
          better = b.aaronFence[n] > b.aaronFence[best];
        } else {
          better = roomAround(b, n, cat) > roomAround(b, best, cat);
        }
      }
      if (better) {
        best = n;
      }
    }
    return best;
  }

  // his cat's step from `cat`; -1 when it has no open neighbor
  int aaronCatMove(Buffers& b, int cat) {
    aaronPasses(b);
    return aaronStep(b, cat);
  }

  // The lookahead cat: a cat that looks one move ahead (its step, then the catcher's worst wall). An edge
  // step at once; else each step s in neighbor order, against the catcher's worst wall within 2 steps of s
  // (through open cells; sealed in: next to s), scored at s's open neighbors as
  // -100 * min(two-distance, 50) - BFS distance (sealed in: -100000 + room). The first best step wins.
  // Modeled on LogiBear's cat (Logi-Bear/mobagen 08f3acb) at the search depth it finishes on most moves on
  // the competition machine.
  constexpr int kLookaheadUnreachable = 1000000;

  // a step's score with the cat on s, from these passes (a wall, if any, already in place)
  int lookaheadValue(Buffers& b, int s, const std::vector<int>& score, const std::vector<int>& dist) {
    int bestScore = kLookaheadUnreachable, bestDist = kLookaheadUnreachable;
    bool canMove = false;
    for (int n : b.neigh[s]) {
      if (!b.open[n]) {
        continue;
      }
      canMove = true;
      bestScore = std::min(bestScore, score[n] >= kBlocked ? kLookaheadUnreachable : score[n]);
      bestDist = std::min(bestDist, dist[n] >= kBlocked ? kLookaheadUnreachable : dist[n]);
    }
    if (!canMove) {
      return -1000000 + 2;  // trapped
    }
    if (bestDist == kLookaheadUnreachable) {
      return -100000 + regionSize(b, s, -1);
    }
    return -100 * std::min(bestScore, 50) - bestDist;
  }

  // the cells s's open neighbors with value `target` are built from (reachable by strictly decreasing `v`)
  void markBuiltFrom(Buffers& b, int s, const std::vector<int>& v, int target, std::vector<uint8_t>& mark) {
    std::fill(mark.begin(), mark.end(), 0);
    int tail = 0;
    for (int n : b.neigh[s]) {
      if (b.open[n] && v[n] == target && !mark[n]) {
        mark[n] = 1;
        b.queue[tail++] = n;
      }
    }
    for (int head = 0; head < tail; head++) {
      const int c = b.queue[head];
      for (int m : b.neigh[c]) {
        if (b.open[m] && !mark[m] && v[m] < v[c]) {
          mark[m] = 1;
          b.queue[tail++] = m;
        }
      }
    }
  }

  // Walls only lower a step's score, so a step whose score without one can't beat the best so far is
  // skipped, and a wall outside the cells its best neighbors' values are built from leaves it unchanged. A
  // wall only reruns the pass it can change.
  int lookaheadCatMove(Buffers& b, int cat) {
    for (int n : b.neigh[cat]) {
      if (b.open[n] && b.isBorder[n]) {
        return n;
      }
    }
    twoDistance(b, b.lookScore);
    bfsDistance(b, b.lookDist);
    int best = -1, bestValue = INT_MIN;
    for (int s : b.neigh[cat]) {
      if (!b.open[s]) {
        continue;
      }
      const int base = lookaheadValue(b, s, b.lookScore, b.lookDist);
      if (base <= bestValue) {
        continue;
      }
      const bool sealed = b.lookDist[s] >= kBlocked;
      int walls[18];
      int wallCount = 0;
      for (int n : b.neigh[s]) {
        if (b.open[n]) {
          walls[wallCount++] = n;
        }
      }
      if (!sealed) {
        const int ring = wallCount;
        for (int i = 0; i < ring; i++) {
          for (int m : b.neigh[walls[i]]) {
            if (!b.open[m] || m == s) {
              continue;
            }
            bool seen = false;
            for (int j = 0; j < wallCount; j++) {
              seen = seen || walls[j] == m;
            }
            if (!seen) {
              walls[wallCount++] = m;
            }
          }
        }
      }
      // the two-distance only counts below 50
      int bestScore = kLookaheadUnreachable, bestDist = kLookaheadUnreachable;
      for (int n : b.neigh[s]) {
        if (b.open[n]) {
          bestScore = std::min(bestScore, b.lookScore[n] >= kBlocked ? kLookaheadUnreachable : b.lookScore[n]);
          bestDist = std::min(bestDist, b.lookDist[n] >= kBlocked ? kLookaheadUnreachable : b.lookDist[n]);
        }
      }
      const bool scoreCounts = bestScore < 50 && !sealed;
      if (scoreCounts) {
        markBuiltFrom(b, s, b.lookScore, bestScore, b.lookScoreFrom);
      }
      if (!sealed) {
        markBuiltFrom(b, s, b.lookDist, bestDist, b.lookDistFrom);
      }
      int worst = base;
      for (int i = 0; i < wallCount && worst > bestValue; i++) {
        const int w = walls[i];
        const bool changesScore = scoreCounts && b.lookScoreFrom[w];
        const bool changesDist = !sealed && b.lookDistFrom[w];
        if (!sealed && !changesScore && !changesDist) {
          continue;
        }
        setOpen(b, w, false);
        if (changesScore) {
          twoDistance(b, b.nodeScore);
        }
        if (changesDist) {
          bfsDistance(b, b.nodeDist);
        }
        worst = std::min(worst, lookaheadValue(b, s, changesScore ? b.nodeScore : b.lookScore, changesDist ? b.nodeDist : b.lookDist));
        setOpen(b, w, true);
      }
      if (worst > bestValue) {
        bestValue = worst;
        best = s;
      }
    }
    return best;
  }

  int modelCatMove(Buffers& b, int cat, Model model) {
    if (model == kAaronCat) {
      return aaronCatMove(b, cat);
    }
    if (model == kPathCat) {
      return pathCatMove(b, cat);
    }
    return model == kLookaheadCat ? lookaheadCatMove(b, cat) : twoDistanceCatMove(b, cat);
  }

  // Runs until the model cat is trapped (cat to move on `cat`); kModelEscape if it gets out. Stops at
  // `cutoff` once the playout can no longer end sooner than that, since the caller only keeps a shorter one.
  int playout(Buffers& b, int cat, int cutoff, Model model) {
    int placed[kModelMoves];
    int count = 0;
    int plies = 0;
    int result = -1;
    for (int t = 0; t < kModelMoves && result < 0; t++) {
      if (plies + 2 >= cutoff) {
        result = cutoff;
        break;
      }
      const int m = modelCatMove(b, cat, model);
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
      // the playout catcher: this catcher without the lookahead, blocking within 2 steps of the cat. Its playouts rank blocks far better than ones
      // where the catcher just blocks the cat's best next cell, at about a fortieth of the cost of playin out with the full search, which caught
      // them barely sooner.
      int64_t worst = INT64_MIN;
      bool trapping = false;
      const int x = searchBlock(b, cat, true, worst, trapping);
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

  // The cells within kModelRadius steps of the cat, nearest first, the cat's own cell at 0 (37 for a radius
  // of 3); returns how many
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

  // Which model cat? (step 7)

  // nodeScore, nodeDist and the shortest path counts for the board as it is
  void historyPasses(Buffers& b) {
    twoDistance(b, b.nodeScore);
    bfsDistance(b, b.nodeDist);
    countPaths(b, b.nodeDist, b.paths);
  }

  // does the model cat on `from` step to `to`? (historyPasses must hold this board's passes for the
  // two-distance cat, ranked here like AaronArchambault's, and aaronPasses for his cat)
  bool stepsTo(Buffers& b, Model model, int from, int to) {
    if (model == kAaronCat) {
      return aaronStep(b, from) == to;
    }
    if (model == kPathCat) {
      return pathCatMove(b, from) == to;
    }
    if (model == kLookaheadCat) {
      return lookaheadCatMove(b, from) == to;
    }
    int first = -1;
    for (int n : b.neigh[from]) {
      if (!b.open[n]) {
        continue;
      }
      if (first < 0) {
        first = n;
        continue;
      }
      bool better = false;
      if (b.nodeScore[n] != b.nodeScore[first]) {
        better = b.nodeScore[n] < b.nodeScore[first];
      } else if (b.nodeDist[n] != b.nodeDist[first]) {
        better = b.nodeDist[n] < b.nodeDist[first];
      } else if (b.paths[n] != b.paths[first]) {
        better = b.paths[n] > b.paths[first];
      } else {
        better = roomAround(b, n, from) > roomAround(b, first, from);
      }
      if (better) {
        first = n;
      }
    }
    return first == to;
  }

  // Can the model cat explain the cat's last `steps` steps, ending on `cur`? Each earlier step is checked
  // on this board, as it is or with one blocked cell near the earlier cell open again (the block made while
  // the cat stood there). Reaching the center (where the cat starts) ends the history; running out of
  // `budget` counts as explained.
  bool explains(Buffers& b, Model model, int cur, int steps, int& budget) {
    if (steps == 0 || --budget < 0) {
      return true;
    }
    if (model == kTwoDistanceCat) {
      historyPasses(b);
    } else if (model == kAaronCat) {
      aaronPasses(b);
    }
    const int total = b.side * b.side;
    int prev[6];
    int prevCount = 0;
    for (int p : b.neigh[cur]) {
      if (b.open[p] && !b.isBorder[p] && stepsTo(b, model, p, cur)) {
        prev[prevCount++] = p;
      }
    }
    for (int k = 0; k < prevCount; k++) {
      const int p = prev[k];
      if (p == total / 2 || steps == 1 || explains(b, model, p, steps - 1, budget)) {
        return true;
      }
      int cells[kModelCells];
      const int count = cellsNear(b, p, cells);
      for (int j = 1; j < count && j <= kHistoryBlocks; j++) {
        const int x = cells[j];
        if (b.open[x]) {
          continue;
        }
        setOpen(b, x, true);
        const bool fits = explains(b, model, p, steps - 1, budget);
        setOpen(b, x, false);
        if (fits) {
          return true;
        }
      }
    }
    return false;
  }

  // the first model of the two-distance cat, the path cat and the lookahead cat that explains the cat's last
  // kHistorySteps steps, or the two-distance cat if none does. A cat the two-distance cat explains is
  // AaronArchambault's when his cat explains those steps too and the lookahead cat doesn't: his ranking is
  // nearly the two-distance cat's, and modeling LogiBear's deeper cats as his let them escape.
  Model modelFor(Buffers& b, int cat) {
    for (Model model : {kTwoDistanceCat, kPathCat, kLookaheadCat}) {
      int budget = kHistoryBudget;
      if (explains(b, model, cat, kHistorySteps, budget)) {
        if (model == kTwoDistanceCat) {
          int aaronBudget = kAaronBudget, lookBudget = kHistoryBudget;
          if (explains(b, kAaronCat, cat, kHistorySteps, aaronBudget) && aaronBudget >= 0
              && !explains(b, kLookaheadCat, cat, kHistorySteps, lookBudget)) {
            return kAaronCat;
          }
        }
        return model;
      }
    }
    return kTwoDistanceCat;
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
  // than `best`'s, shortest first, and the first of up to kSafetyChecks whose worst case is at most one
  // two-distance step below `best`'s (rootWorst). Demanding a worst case as good as `best`'s turned away
  // nearly every faster block once the playouts used a real catcher.
  int earlyBlock(Buffers& b, int cat, int best, int64_t rootWorst) {
    if (rootWorst == INT64_MIN) {
      return best;
    }
    const int64_t need = (rootWorst / kScoreStep - 1) * kScoreStep;  // one two-distance step below
    const Model model = modelFor(b, cat);
    int cells[kModelCells];
    const int count = cellsNear(b, cat, cells);
    setOpen(b, best, false);
    const int bestLen = openAround(b, cat, -1) == 0 ? 0 : playout(b, cat, INT_MAX, model);
    setOpen(b, best, true);
    std::array<std::pair<int, int>, kModelCells> better;
    int betterCount = 0;
    for (int k = 1; k < count; k++) {
      const int x = cells[k];
      if (x == best || !b.open[x]) {
        continue;
      }
      setOpen(b, x, false);
      const int len = openAround(b, cat, -1) == 0 ? 0 : playout(b, cat, bestLen, model);
      setOpen(b, x, true);
      if (len < bestLen) {
        better[betterCount++] = {len, x};
      }
    }
    std::stable_sort(better.begin(), better.begin() + betterCount);
    for (int k = 0; k < betterCount && k < kSafetyChecks; k++) {
      if (worstAtLeast(b, cat, better[k].second, need)) {
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
    const Model model = modelFor(b, cat);
    setOpen(b, best, false);
    int pickLen = openAround(b, cat, -1) == 0 ? 0 : playout(b, cat, INT_MAX, model);
    setOpen(b, best, true);
    int pick = best;
    for (int k = 1; k < count; k++) {
      const int x = cells[k];
      if (!keeps[k]) {
        continue;
      }
      setOpen(b, x, false);
      const int len = openAround(b, cat, -1) == 0 ? 0 : playout(b, cat, pickLen, model);
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

  int64_t rootWorst = INT64_MIN;  // the chosen block's worst case, for modelBlock
  bool trapping = false;
  int best = searchBlock(b, cat, false, rootWorst, trapping);
  if (trapping) {
    return toPoint(best);
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
