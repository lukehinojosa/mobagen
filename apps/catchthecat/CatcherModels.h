#ifndef CATCHER_MODELS_H
#define CATCHER_MODELS_H

#include "Grid.h"
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
}  // namespace models

#endif  // CATCHER_MODELS_H
