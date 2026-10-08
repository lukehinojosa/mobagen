#ifndef CATCHER_MODELS_H
#define CATCHER_MODELS_H

#include "Grid.h"
#include <vector>

// Models of other students' catchers, for the cat to plan against.

namespace models {
  // Cheap copies of catchers that defend the edge (three general styles, close ports of last year's
  // AndrewGenualdo and TOAG21, and an exact port of this year's JordanCoolbeth): the first of `cells` (the
  // cat's candidate moves from `cat`, best first) whose line beats the most of the models that fit the board,
  // or -1 when none beats any. Each model only counts when the board shows its catcher's style; with
  // `tryJordan`, when the JordanCoolbeth port explains the last blocks, it is the only one. Leaves the board as
  // it found it.
  int modelChoice(grid::Board& b, int cat, const int* cells, int count, bool tryJordan);

  // An exact, faster copy of AaronArchambault's catcher (his 2026-10-05 version, a5248df: the cat's move on `state`
  // (cat on cell `cat`, true is blocked) whose playout against it lasts longest; `usual` (the cat's own move) breaks ties.
  int survivalMove(const std::vector<bool>& state, int side, int cat, int usual);
}  // namespace models

#endif  // CATCHER_MODELS_H
