#ifndef CAT_H
#define CAT_H

#include "Agent.h"

class Cat : public Agent {
public:
  explicit Cat() : Agent(){};
  Point2D Move(CatWorld*) override;

  // Move calls made in this process: 1 on the leaderboard (a new process for every move), more in the arena
  // (the bot stays loaded for the whole match)
  static int movesThisProcess;
};

#endif  // CAT_H
