#ifndef CATCHER_MODELS_H
#define CATCHER_MODELS_H

#include "Grid.h"
#include <chrono>
#include <vector>

// Models of other students' catchers, for the cat to plan against.

namespace models {
  // Cheap copies of catchers that defend the edge (three general styles, close ports of last year's
  // AndrewGenualdo and TOAG21, exact ports of this year's JordanCoolbeth and last year's AylwinMorgan and
  // Cosmey): the first of `cells` (the cat's candidate moves from `cat`, best first) whose line beats the most of
  // the models that fit, or -1 when none beats any. With `playedLike` (a mask from confirmed), exactly those
  // models fit (an exact port alone when it is one); when it includes the nearest-exit model (a catcher that
  // blocks one of the border cells nearest the cat), the move with the best chance against that model alone.
  // Otherwise each model only counts when the board shows its catcher's style; with `tryJordan`, when the JordanCoolbeth
  // port explains the last blocks, it is the only one. Leaves the board as it found it.
  int modelChoice(grid::Board& b, int cat, const int* cells, int count, bool tryJordan, int playedLike);

  // The models (a bit per model, as modelChoice takes them) whose block with the cat on `cat` would have been
  // `block`, which the catcher just blocked on `b`. Leaves the board as it found it.
  int predictedBy(grid::Board& b, int cat, int block);

  // The models the arena cat's memory confirms, from predictedBy for each block this game (oldest first): those
  // that predicted each of the last 3, or for an exact port each of the last 6. 0 before that.
  int confirmed(const std::vector<int>& hits);

  // An exact, faster copy of AaronArchambault's catcher (his 2026-10-07 version, 2e882a2): the cat's move on `state`
  // (cat on cell `cat`, true is blocked) whose playout against it lasts longest; `usual` (the cat's own move) breaks ties.
  int survivalMove(const std::vector<bool>& state, int side, int cat, int usual);

  // predictedBy's bits for AaronArchambault's catcher (the copy above) and LogiBear's at depth 3 and at depth 1
  int aaronBit();
  int logi3Bit();
  int logi1Bit();

  // The arena cat's escape search against an exact copy of a catcher (kEscapeAaron: AaronArchambault's; kEscapeLogi3
  // and kEscapeLogi1: LogiBear's at depth 3 or 1) from `b` with the cat on `cat` to move, until `deadline`: the cell
  // to step to (the next step of a line that wins against the copy, or toward the most promising position found), or
  // -1. Keeps its search tree between calls while the catcher's replies are the copy's.
  constexpr int kEscapeAaron = 0;
  constexpr int kEscapeLogi3 = 1;
  constexpr int kEscapeLogi1 = 2;
  // With `keepTree` false (the leaderboard: a new process every move) the search starts afresh and only returns the
  // first step of a winning line.
  int escapeMove(const grid::Board& b, int cat, int from, std::chrono::steady_clock::time_point deadline, bool keepTree);

  // On the leaderboard (no memory): could LogiBear's catcher at depth 1 have made the last 2 blocks? Only asked once
  // the block count and the cat's distance from the center show he has made at least 2; not when the JordanCoolbeth
  // port explains the last 3, nor when AaronArchambault's catcher would have made the last block too; false if
  // `deadline` comes first. Leaves the board as it found it.
  bool logiOnBoard(grid::Board& b, int cat, std::chrono::steady_clock::time_point deadline);
}  // namespace models

#endif  // CATCHER_MODELS_H
