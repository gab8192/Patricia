#pragma once
#include "defs.h"
#include "simd.h"
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <span>
#include <vector>
#ifdef _MSC_VER
#define W_MSVC
#pragma push_macro("_MSC_VER")
#undef _MSC_VER
#endif

#define INCBIN_PREFIX g_
#include "incbin.h"

#ifdef W_MSVC
#pragma pop_macro("_MSC_VER")
#undef W_MSVC
#endif

constexpr size_t INPUT_SIZE = 768;
constexpr size_t LAYER1_SIZE = 768;

constexpr int SCRELU_MIN = 0;
constexpr int SCRELU_MAX = 255;

constexpr int SCALE = 400;

constexpr int QA = 255;
constexpr int QB = 64;

const auto SCRELU_MIN_VEC = get_int16_vec(SCRELU_MIN);
const auto QA_VEC = get_int16_vec(QA);

constexpr int QAB = QA * QB;

struct alignas(64) NNUE_Params {
  std::array<int16_t, INPUT_SIZE * LAYER1_SIZE> feature_v;
  std::array<int16_t, LAYER1_SIZE> feature_bias;
  std::array<int16_t, LAYER1_SIZE * 2> output_v;
  int16_t output_bias;
};

INCBIN(nnue, "src/abby.nnue");
const NNUE_Params &g_nnue = *reinterpret_cast<const NNUE_Params *>(g_nnueData);

template <size_t HiddenSize> struct alignas(64) Accumulator {
  std::array<std::array<int16_t, HiddenSize>, 2> colors;

  inline void init(std::span<const int16_t, HiddenSize> bias) {
    std::memcpy(colors[1].data(), bias.data(), bias.size_bytes());
    std::memcpy(colors[0].data(), bias.data(), bias.size_bytes());
  }
};

constexpr int32_t screlu(int16_t x) {
  const auto clipped =
      std::clamp(static_cast<int32_t>(x), SCRELU_MIN, SCRELU_MAX);
  return clipped * clipped;
}

template <size_t size, size_t v>
inline void add_to_all(std::array<int16_t, size> &output,
                       std::array<int16_t, size> &input,
                       const std::array<int16_t, v> &delta, size_t offset) {
  for (size_t i = 0; i < size; ++i) {
    output[i] = input[i] + delta[offset + i];
  }
}

template <size_t size, size_t v>
inline void subtract_from_all(std::array<int16_t, size> &output,
                              std::array<int16_t, size> &input,
                              const std::array<int16_t, v> &delta,
                              size_t offset) {

  for (size_t i = 0; i < size; ++i) {
    output[i] = input[i] - delta[offset + i];
  }
}

size_t feature_index(int view, int kingSq, int piece, int sq) {
  constexpr size_t color_stride = 64 * 6;
  constexpr size_t piece_stride = 64;

  // flip piece square if king square is on file E <-> H
  if (kingSq & 4)
    sq ^= 7;

  const auto base = static_cast<int>(piece / 2 - 1);
  const size_t color = piece & 1;

  return (view != color) * color_stride +
         base * piece_stride +
         static_cast<size_t>(sq ^ (56 * view));
}

inline int32_t
screlu_flatten(const std::array<int16_t, LAYER1_SIZE> &us,
               const std::array<int16_t, LAYER1_SIZE> &them,
               const std::array<int16_t, LAYER1_SIZE * 2> &weights) {

#if defined(__AVX512F__) || defined(__AVX2__)

  auto sum = vec_int32_zero();

  for (size_t i = 0; i < LAYER1_SIZE; i += REGISTER_SIZE) {

    auto v_us = int16_load(&us[i]);
    auto w_us = int16_load(&weights[i]);

    v_us = vec_int16_clamp(v_us, SCRELU_MIN_VEC, QA_VEC);

    auto our_product = vec_int16_multiply(v_us, w_us);

    auto our_result = vec_int16_madd_int32(our_product, v_us);

    sum = vec_int32_add(sum, our_result);

    auto v_them = int16_load(&them[i]);
    auto w_them = int16_load(&weights[LAYER1_SIZE + i]);

    v_them = vec_int16_clamp(v_them, SCRELU_MIN_VEC, QA_VEC);

    auto their_product = vec_int16_multiply(v_them, w_them);

    auto their_result = vec_int16_madd_int32(their_product, v_them);

    sum = vec_int32_add(sum, their_result);
  }

  return vec_int32_hadd(sum) / QA;

#else

  int32_t sum = 0;

  for (size_t i = 0; i < LAYER1_SIZE; ++i) {
    sum += screlu(us[i]) * weights[i];
    sum += screlu(them[i]) * weights[LAYER1_SIZE + i];
  }

  return sum / QA;

#endif
}

class NNUE_State {
public:
  Accumulator<LAYER1_SIZE> m_accumulator_stack[MaxSearchDepth];
  Accumulator<LAYER1_SIZE> *m_curr;
  int kings_pos[2];

  void add_sub(int from_piece, int from, int to_piece, int to);
  void add_sub_sub(int from_piece, int from, int to_piece, int to, int captured, int captured_pos);
  void add_add_sub_sub(int piece1, int from1, int to1, int piece2, int from2, int to2);
  void pop();
  int evaluate(int color);
  void reset_nnue(const Position& position);

  inline void add_feature(int piece, int square);

  NNUE_State() {}
};


void NNUE_State::add_sub(int from_piece, int from, int to_piece, int to) {

  for (int view = Colors::White; view <= Colors::Black; ++view) {
    const auto from_ft = feature_index(view, kings_pos[view], from_piece, from);
    const auto to_ft = feature_index(view, kings_pos[view], to_piece, to);
    for (size_t i = 0; i < LAYER1_SIZE; ++i) {
      m_curr[1].colors[view][i] = m_curr->colors[view][i] +
                         g_nnue.feature_v[to_ft * LAYER1_SIZE + i] -
                         g_nnue.feature_v[from_ft * LAYER1_SIZE + i];
    }
  }

  m_curr++;
}

void NNUE_State::add_sub_sub(int from_piece, int from, int to_piece, int to, int captured,
                             int captured_sq) {
  for (int view = Colors::White; view <= Colors::Black; ++view) {
    const auto from_ft = feature_index(view, kings_pos[view], from_piece, from);
    const auto to_ft = feature_index(view, kings_pos[view], to_piece, to);
    const auto cap_ft = feature_index(view, kings_pos[view], captured, captured_sq);
    for (size_t i = 0; i < LAYER1_SIZE; ++i) {
      m_curr[1].colors[view][i] = m_curr->colors[view][i] +
                         g_nnue.feature_v[to_ft * LAYER1_SIZE + i] -
                         g_nnue.feature_v[from_ft * LAYER1_SIZE + i] -
                         g_nnue.feature_v[cap_ft * LAYER1_SIZE + i];
    }
  }

  m_curr++;
}

void NNUE_State::add_add_sub_sub(int piece1, int from1, int to1, int piece2, int from2, int to2){
 for (int view = Colors::White; view <= Colors::Black; ++view) {
    const auto from1_ft = feature_index(view, kings_pos[view], piece1, from1);
    const auto to1_ft = feature_index(view, kings_pos[view], piece1, to1);
    const auto from2_ft = feature_index(view, kings_pos[view], piece2, from2);
    const auto to2_ft = feature_index(view, kings_pos[view], piece2, to2);
    for (size_t i = 0; i < LAYER1_SIZE; ++i) {
      m_curr[1].colors[view][i] = m_curr->colors[view][i] +
                         g_nnue.feature_v[to1_ft * LAYER1_SIZE + i] -
                         g_nnue.feature_v[from1_ft * LAYER1_SIZE + i] +
                         g_nnue.feature_v[to2_ft * LAYER1_SIZE + i] -
                         g_nnue.feature_v[from2_ft * LAYER1_SIZE + i];
    }
  }

  m_curr++;
}

void NNUE_State::pop() { m_curr--; }

int NNUE_State::evaluate(int color) {
  const auto output = screlu_flatten(m_curr->colors[color], m_curr->colors[color^1], g_nnue.output_v);
  return (output + g_nnue.output_bias) * SCALE / QAB;
}

inline void NNUE_State::add_feature(int piece, int square) {
  for (int view = Colors::White; view <= Colors::Black; ++view) {
    add_to_all(m_curr->colors[view], m_curr->colors[view], g_nnue.feature_v,
         feature_index(view, kings_pos[view], piece, square) * LAYER1_SIZE);
  }
}

int get_king_pos(const Position &position, int color);

void NNUE_State::reset_nnue(const Position& position) {
  m_curr = &m_accumulator_stack[0];
  m_curr->init(g_nnue.feature_bias);
  
  for (int view = Colors::White; view <= Colors::Black; ++view)
    kings_pos[view] = get_king_pos(position, view);

  for (int square = a1; square < SqNone; square++) {
    if (position.board[square] != Pieces::Blank) {
      add_feature(position.board[square], square);
    }
  }
}
