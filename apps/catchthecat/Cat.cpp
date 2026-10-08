#include "Cat.h"
#include "CatcherModels.h"
#include "Grid.h"
#include "World.h"
#include <algorithm>
#include <array>
#include <climits>
#include <chrono>
#include <cstdint>
#include <vector>

// Cat strategy (decided from the current board, plus in the arena a memory of the game):
//  1. Two-distance: an open border cell scores 0; any other open cell scores 1 + the SECOND smallest
//     score among its neighbors, because the catcher will block the best one. Rank moves by it. Moves
//     tied on every ranking key go to the one with the most room: open cells within 2 steps that are at
//     least as close to escaping (over 100,000 games this beat board order on every seed tried).
//  2. Model search: catchers that defend the edge (instead of blocking next to the cat) beat plain
//     two-distance by plugging each exit as the cat laps the board. When the board shows such a style,
//     search up to 6 cat moves for a line that beats cheap copies of those catchers, and take it. On the
//     leaderboard, when an exact copy of JordanCoolbeth's catcher explains the last three blocks, plan against
//     it alone.
//  3. Veto: try the catcher replies that can matter to the top move (cells its score depends on). Only if
//     one of them cuts off every escape, switch to the move whose worst reply leaves the cat best off.
//     Always trusting the worst case made the cat too timid against last year's real catchers.
//  4. If every score is infinite but a border is still reachable, follow the plain BFS distance and hope
//     the catcher is weak.
//  5. If no border is reachable, survive: move to the neighbor with the most open neighbors.
//  6. Survival: once the cat's best two-distance is 14 or more (it can hardly force its way out), play each
//     move out against an exact copy of AaronArchambault's catcher and take the one that lasts longest.
//     Leaderboard only: it scores a lost game by its length, but in the arena a cat only wins by escaping
//     (a game at the move cap goes to the catcher), so there the cat keeps running. The leaderboard starts a
//     new process for every move and the arena keeps the bot loaded for the whole match, so a second Move
//     call in one process means the arena.
//  7. Memory (arena only, where the bot stays loaded): the cat keeps the board it moved on, so each call reads
//     off the catcher's block and checks which models of step 2 predicted it. Models that predicted each of
//     the last 3 blocks are the only ones the search plans against. More models only memory can confirm: exact
//     ports of last year's AylwinMorgan and Cosmey catchers (on the last 6 blocks), each planned against alone
//     8 moves deep; and a catcher that blocks one of the border cells nearest the cat, tie broken its own way.
//     Most of last year's other edge catchers play like this; the search then goes for the move with the best
//     chance of escaping within 8 moves, with each nearest exit taken as equally likely.
//  8. Against AaronArchambault's or LogiBear's catcher (arena only): the cat recognizes the catcher when an exact
//     copy made its blocks (AaronArchambault's, step 6's copy: every block this game; LogiBear's at depth 3 or 1: 2
//     of his last 3 blocks and 60% of them all, as his clock sometimes goes deeper). The copy's replies are certain,
//     so the cat searches for a line that escapes them (planning against LogiBear at the depth his last block came
//     from; up to 700 ms from the start of the move, the search tree kept from move to move), plays it out while
//     the replies match, and until it has one steps toward the most promising position.
//  9. On the leaderboard (no memory), against LogiBear's catcher: the board shows it when his catcher at depth 1 (the
//     depth he mostly reaches on the runner's machine) would have made the last 2 blocks, asked only once the block
//     count (the board starts with at most a tenth of its cells blocked) or the cat's distance from the center shows
//     he has made 2; not when step 2's copy of JordanCoolbeth's catcher explains his last three, nor when
//     AaronArchambault's would have made the last one too. Then a fresh search like step 8's (up to 600 ms from the
//     start of the move) plays the first step of a line that wins against it.
// All buffers are flat arrays reused across calls, so nothing is allocated after the first move.

namespace {
  using grid::kInf;
  using grid::setOpen;

  constexpr int kNoEscape = 1023;  // evaluate() caps scores here, so it also means "no finite escape"

  // replies tried per move, nearest first. With the lookahead only used as a veto, 8 scored best on
  // 21x21 once the runner's time penalty was counted (12 and 22 won no extra games)
  constexpr int kMaxReplies = 8;

  constexpr int kSurvivalTwo = 14;  // best two-distance from which the survival playouts take over (step 6)
  constexpr int kEscapeMs = 700;
  constexpr int kLogiWindow = 3;       // step 8: LogiBear's catcher must explain all but one of his last this many blocks
  constexpr int kLogiShareTenths = 6;  // and this many tenths of all his blocks
  constexpr int kLeaderEscapeMs = 600;

  // the shared board (Grid.h: neighbor table, open cells, distance passes) and the cat's own arrays, each
  // with the sentinel entry at index side * side
  struct Buffers : grid::Board {
    std::vector<int> score;  // two-distance
    std::vector<int> trial;  // two-distance after a hypothetical catcher reply
    std::vector<int> dist;   // plain BFS distance to an open border cell
    std::vector<int> mark;   // stamp per cell for the reply search
    int stamp = 0;
    std::vector<int> replies;
  };

  Buffers& buffersFor(int side) {
    static Buffers b;
    if (b.side == side) {
      return b;
    }
    grid::setSide(b, side);
    for (auto* v : {&b.score, &b.trial, &b.dist}) {
      v->resize(b.total + 1);
    }
    b.mark.assign(b.total + 1, 0);
    b.stamp = 0;
    b.replies.reserve(b.total);
    return b;
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
      setOpen(b, x, false);
      grid::twoDistance(b, b.trial);
      int64_t v = evaluate(b, n, b.trial);
      setOpen(b, x, true);
      worst = std::max(worst, v);
      if (worst >= alpha) {
        break;
      }
    }
    return worst;
  }

  // room around a move to `n`: the open cells within 2 steps of it (n included) whose two-distance is no
  // worse than n's. More room keeps more escape routes alive when moves are otherwise tied.
  int roomAround(const Buffers& b, int n) {
    int cells[19];
    int count = 0;
    cells[count++] = n;
    // n, then its neighbors: their neighbors make the second ring
    for (int head = 0; head < count && head < 7; head++) {
      for (int m : b.neigh[cells[head]]) {
        if (m == b.total) {
          continue;  // off the board
        }
        bool seen = false;
        for (int k = 0; k < count; k++) {
          if (cells[k] == m) {
            seen = true;
          }
        }
        if (!seen) {
          cells[count++] = m;
        }
      }
    }
    int room = 0;
    for (int k = 0; k < count; k++) {
      if (b.open[cells[k]] && b.score[cells[k]] <= b.score[n]) {
        room++;
      }
    }
    return room;
  }

  // the arena cat's memory of the game (step 7)
  struct Memory {
    std::vector<uint8_t> open;  // the board's open cells when the cat last moved
    int cat = -1;               // the cell it moved to
    std::vector<int> hits;      // models::predictedBy for each catcher block this game, oldest first
  };
  Memory memory;

  // reads the catcher's last block off the board (it is the remembered board plus one block, with the cat where
  // it moved; anything else is a new game) and returns the models the memory confirms (models::confirmed)
  int playedLike(Buffers& b, int cat) {
    int changed = 0, added = -1;
    // arena only (step 6's check): the first call has nothing to read yet
    if (Cat::movesThisProcess > 1 && memory.cat == cat && static_cast<int>(memory.open.size()) == b.total) {
      for (int i = 0; i < b.total; i++) {
        if (memory.open[i] != b.open[i]) {
          changed++;
          added = memory.open[i] ? i : -1;
        }
      }
    }
    if (changed != 1 || added < 0) {
      memory.hits.clear();
      return 0;
    }
    memory.hits.push_back(models::predictedBy(b, cat, added));
    return models::confirmed(memory.hits);
  }

  // step 8: the cell the escape search picks against AaronArchambault's or LogiBear's catcher, or -1. Only while
  // every block this game is the copy's (the arena's memory has at least one; on the leaderboard it has none)
  int escapeLine(const Buffers& b, int cat, std::chrono::steady_clock::time_point start) {
    if (memory.hits.empty()) {
      return -1;
    }
    bool aaron = true;
    for (int h : memory.hits) {
      aaron = aaron && (h & models::aaronBit());
    }
    const int n = static_cast<int>(memory.hits.size()), recent = std::min(n, kLogiWindow);
    int matched = 0, matchedRecent = 0;
    for (int k = 0; k < n; k++) {
      const bool hit = (memory.hits[k] & (models::logi3Bit() | models::logi1Bit())) != 0;
      matched += hit;
      matchedRecent += hit && k >= n - recent;
    }
    const bool logi = matchedRecent >= std::min(recent, kLogiWindow - 1) && matched * 10 >= n * kLogiShareTenths;
    if (!aaron && !logi) {
      return -1;
    }
    // his clock decides his depth: plan against the one his last block came from (depth 3 if both or neither)
    const int last = memory.hits.back();
    const int from = aaron                                                         ? models::kEscapeAaron
                     : (last & models::logi1Bit()) && !(last & models::logi3Bit()) ? models::kEscapeLogi1
                                                                                   : models::kEscapeLogi3;
    return models::escapeMove(b, cat, from, start + std::chrono::milliseconds(kEscapeMs), true);
  }

  // step 9 (leaderboard): against LogiBear's catcher, the first step of a line that wins against his catcher at depth
  // 1, or -1. Not once the cat is held: step 6's playouts against AaronArchambault's catcher take over there
  int leaderboardEscape(Buffers& b, int cat, std::chrono::steady_clock::time_point start) {
    int bestTwo = kInf;
    for (int n : b.neigh[cat]) {
      if (n >= 0 && b.open[n]) {
        bestTwo = std::min(bestTwo, b.score[n]);
      }
    }
    if (bestTwo >= kSurvivalTwo) {
      return -1;
    }
    const auto deadline = start + std::chrono::milliseconds(kLeaderEscapeMs);
    if (!models::logiOnBoard(b, cat, deadline)) {
      return -1;
    }
    return models::escapeMove(b, cat, models::kEscapeLogi1, deadline, false);
  }

  // keeps the board the cat's move was made on (usualMove left it loaded) for the next call's playedLike
  Point2D remember(int side, Point2D move) {
    const auto& b = buffersFor(side);
    memory.open.assign(b.open.begin(), b.open.begin() + b.total);
    memory.cat = (move.y + side / 2) * side + move.x + side / 2;
    return move;
  }

  // the cat's move before the survival playouts (strategy steps 1 to 5)
  Point2D usualMove(CatWorld* world) {
    const auto start = std::chrono::steady_clock::now();
    const int side = world->getWorldSideSize();
    const int half = side / 2;
    const auto& state = world->worldState();
    auto& b = buffersFor(side);

    grid::load(b, state);
    grid::twoDistance(b, b.score);
    grid::bfsDistance(b, b.dist);

    // stage 1 ranking of the open neighbors; smaller keys are better
    const Point2D cat = world->getCat();
    const int catIdx = (cat.y + half) * side + cat.x + half;
    const int seen = playedLike(b, catIdx);
    const int line = escapeLine(b, catIdx, start);
    if (line >= 0) {
      return {line % side - half, line / side - half};
    }
    if (Cat::movesThisProcess == 1) {
      const int lb = leaderboardEscape(b, catIdx, start);
      if (lb >= 0) {
        return {lb % side - half, lb / side - half};
      }
    }
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

    // moves tied with the top one on every key: put the one with the most room first (board order breaks what is left)
    int tied = 1;
    while (tied < moveCount && moves[tied].first == moves[0].first) {
      tied++;
    }
    if (tied > 1) {
      int bestK = 0;
      int bestRoom = -1;
      for (int k = 0; k < tied; k++) {
        const int room = roomAround(b, moves[k].second);
        if (room > bestRoom) {
          bestRoom = room;
          bestK = k;
        }
      }
      std::rotate(moves.begin(), moves.begin() + bestK, moves.begin() + bestK + 1);
    }

    // model search, unless the win is already forced
    if (!(moves[0].first[0] == 0 && moves[0].first[1] <= 1)) {
      int cells[6];
      int count = 0;
      while (count < moveCount && moves[count].first[0] < 2) {
        cells[count] = moves[count].second;
        count++;
      }
      int mv = models::modelChoice(b, catIdx, cells, count, Cat::movesThisProcess == 1, seen);
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
}  // namespace

int Cat::movesThisProcess = 0;

Point2D Cat::Move(CatWorld* world) {
  movesThisProcess++;
  const Point2D usual = usualMove(world);
  const int side = world->getWorldSideSize(), half = side / 2;
  const Point2D cat = world->getCat();
  bool anyOpen = false;
  for (const Point2D& n : CatWorld::neighbors(cat)) {
    anyOpen = anyOpen || (world->isValidPosition(n) && !world->getContent(n));
  }
  if (!anyOpen) {
    return remember(side, usual);  // trapped: the catcher already won
  }
  // the survival playouts only once the cat's best two-distance reaches kSurvivalTwo (usualMove left this
  // board's two-distance in b.score): before that, catchers that let the cat out punish anything but running
  auto& b = buffersFor(side);
  const int catIdx = (cat.y + half) * side + cat.x + half;
  int bestTwo = kInf;
  for (int n : b.neigh[catIdx]) {
    if (n >= 0 && b.open[n]) {
      bestTwo = std::min(bestTwo, b.score[n]);
    }
  }
  if (bestTwo < kSurvivalTwo || movesThisProcess > 1) {
    return remember(side, usual);  // not held yet, or the arena (step 6)
  }
  const int pick = models::survivalMove(world->worldState(), side, (cat.y + half) * side + cat.x + half, (usual.y + half) * side + usual.x + half);
  return remember(side, {pick % side - half, pick / side - half});
}
