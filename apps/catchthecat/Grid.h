#ifndef GRID_H
#define GRID_H

#include <array>
#include <climits>
#include <cstdint>
#include <vector>

// The board the cat and the catcher search on: side * side cells in linear index order, (x, y) at
// (y + side / 2) * side + x + side / 2. Every per-cell array has one extra entry at index side * side: a
// sentinel cell that is always blocked and never a border. Off-board neighbors point to it, so neighbor loops
// need no bounds check.
namespace grid {
  constexpr int kInf = INT_MAX;          // an open cell a distance pass can't reach
  constexpr int kBlocked = INT_MAX - 1;  // a blocked cell's (and the sentinel's) value in a distance array

  struct Board {
    int side = 0;
    int total = 0;                          // side * side, also the sentinel's index
    std::vector<std::array<int, 6>> neigh;  // neighbor indices, in CatWorld::neighbors order
    std::vector<int> borders;               // the border cells, in index order
    std::vector<uint8_t> isBorder, open, count;
    std::vector<int> queue;
    std::vector<int> fresh;  // a distance pass's start: 0 on open border cells, kInf on other open cells,
                             // kBlocked on blocked cells and the sentinel
    std::vector<int> seeds;  // the open border cells, in index order (a pass's first queue entries)
    bool seedsStale = true;  // a border cell changed since `seeds` was built
  };

  // sizes the board for `side` and builds its neighbor table (nothing to do when it already has that side)
  void setSide(Board& b, int side);

  // takes the world's cells (true is blocked)
  void load(Board& b, const std::vector<bool>& state);

  // opens or blocks a cell. Always change `open` through here: it keeps the passes' template in step
  void setOpen(Board& b, int i, bool isOpen);

  // two-distance from the open border cells: 0 there, else 1 + the second smallest of the neighbors'
  void twoDistance(Board& b, std::vector<int>& score);

  // BFS distance from the open border cells
  void bfsDistance(Board& b, std::vector<int>& dist);

  // the same, leaving cells farther than `limit` at kInf
  void bfsDistanceWithin(Board& b, std::vector<int>& dist, int limit);
}  // namespace grid

#endif  // GRID_H
