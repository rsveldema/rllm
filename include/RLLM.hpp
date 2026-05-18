#pragma once

namespace rllm {

class RLLM {
public:
  RLLM();
  ~RLLM() = default;
  RLLM(const RLLM&) = delete;
  RLLM& operator=(const RLLM&) = delete;

  void train_mode();
  void prompt_mode();
};

} // namespace rllm