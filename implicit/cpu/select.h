// Copyright 2021 Ben Frederickson
#ifndef IMPLICIT_CPU_SELECT_H_
#define IMPLICIT_CPU_SELECT_H_
#include <algorithm>
#include <functional>
#include <iostream>
#include <utility>
#include <vector>

namespace implicit {

template <typename T>
inline void select(const T *batch, int rows, int cols, int k, int *ids,
                   T *distances) {
  std::vector<std::pair<T, int>> results;
  std::greater<std::pair<T, int>> heap_order;

  for (int row = 0; row < rows; ++row) {
    results.clear();
    for (int col = 0; col < cols; ++col) {
      T score = batch[row * cols + col];

      if ((results.size() < k) || (score > results[0].first)) {
        if (results.size() >= k) {
          std::pop_heap(results.begin(), results.end(), heap_order);
          results.pop_back();
        }
        results.push_back(std::make_pair(score, col));
        std::push_heap(results.begin(), results.end(), heap_order);
      }
    }

    std::sort_heap(results.begin(), results.end(), heap_order);

    for (size_t i = 0; i < results.size(); ++i) {
      ids[row * k + i] = results[i].second;
      distances[row * k + i] = results[i].first;
    }
  }
}
} // namespace implicit
#endif // IMPLICIT_CPU_SELECT_H_
