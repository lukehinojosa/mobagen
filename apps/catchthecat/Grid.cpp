#include "Grid.h"
#include "World.h"
#include <cstdlib>
#include <cstring>

namespace grid {
  void setSide(Board& b, int side) {
    if (b.side == side) {
      return;
    }
    b.side = side;
    const int h = side / 2;
    const int total = side * side;
    b.total = total;
    b.neigh.resize(total + 1);
    b.borders.clear();
    b.isBorder.assign(total + 1, 0);
    b.open.assign(total + 1, 0);
    b.count.assign(total + 1, 0);
    b.queue.resize(total + 1);
    b.fresh.assign(total + 1, kBlocked);
    b.seeds.clear();
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
  }

  void load(Board& b, const std::vector<bool>& state) {
    for (int i = 0; i < b.total; i++) {
      setOpen(b, i, !state[i]);
    }
  }

  void setOpen(Board& b, int i, bool isOpen) {
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
  // Raw pointers instead of vector calls, and the 6 neighbors written out instead of looped (both much faster
  // without compiler optimization). A pass starts from the template, so one compare (== kInf) means "open and
  // not reached yet", and off-board neighbors point at the sentinel.

  // starts a pass: `dist` from the template, the open border cells queued; returns how many were queued
  namespace {
    int startPass(Board& b, std::vector<int>& dist) {
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

    // BFS from the open border cells, stopping before expanding a cell at `limit` or beyond
    void bfsPass(Board& b, std::vector<int>& distVec, int limit) {
      int tail = startPass(b, distVec);
      int* dist = distVec.data();
      int* queue = b.queue.data();
      const int* neigh = b.neigh[0].data();
      for (int head = 0; head < tail; head++) {
        const int c = queue[head];
        if (dist[c] >= limit) {
          break;
        }
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
  }  // namespace

  // cells leave the queue in nondecreasing score, so the second neighbor to leave is the second best one
  void twoDistance(Board& b, std::vector<int>& scoreVec) {
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

  void bfsDistance(Board& b, std::vector<int>& dist) { bfsPass(b, dist, kInf); }

  void bfsDistanceWithin(Board& b, std::vector<int>& dist, int limit) { bfsPass(b, dist, limit); }
}  // namespace grid
