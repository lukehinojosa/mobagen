#ifndef LIFE_POINTHASH_H
#define LIFE_POINTHASH_H

#include "math/Point2D.h"
#include <cstddef>

// hash for Point2D so it can key to value(s)
struct PointHash {
  std::size_t operator()(const Point2D& p) const {
    // mix the two coordinates into one hash
    std::size_t h = static_cast<std::size_t>(static_cast<unsigned int>(p.x)) * 73856093u;
    h ^= static_cast<std::size_t>(static_cast<unsigned int>(p.y)) * 19349663u;
    return h;
  }
};

#endif  // LIFE_POINTHASH_H
