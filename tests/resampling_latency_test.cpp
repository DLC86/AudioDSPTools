#include "../dsp/ResamplingContainer/ResamplingContainer.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

namespace
{
using Container = dsp::ResamplingContainer<double, 1, 32>;

const char* ModeName(dsp::EAntiAliasFilterPhase mode)
{
  switch (mode)
  {
    case dsp::EAntiAliasFilterPhase::MinimumPhaseCascadedFIR:
      return "minimum";
    case dsp::EAntiAliasFilterPhase::LinearCascadedFIRShort:
      return "linear-short";
    case dsp::EAntiAliasFilterPhase::LinearCascadedFIRLong:
      return "linear-long";
  }

  return "unknown";
}

bool CheckMode(int factor, dsp::EAntiAliasFilterPhase mode)
{
  constexpr int blockSize = 64;
  constexpr int blockCount = 32;
  constexpr double hostSampleRate = 48000.0;

  Container container(hostSampleRate * factor, mode, hostSampleRate);
  container.Reset(hostSampleRate, blockSize);

  std::vector<double> input(blockSize, 0.0);
  std::vector<double> output(blockSize, 0.0);
  std::vector<double> response(blockSize * blockCount, 0.0);
  double* inputs[] = {input.data()};
  double* outputs[] = {output.data()};

  input[0] = 1.0;

  for (int block = 0; block < blockCount; block++)
  {
    container.ProcessBlock(
      inputs,
      outputs,
      blockSize,
      [](double** in, double** out, int frames)
      {
        std::copy(in[0], in[0] + frames, out[0]);
      });

    std::copy(output.begin(), output.end(), response.begin() + block * blockSize);
    std::fill(input.begin(), input.end(), 0.0);
    std::fill(output.begin(), output.end(), 0.0);
  }

  const auto peakIt = std::max_element(
    response.begin(),
    response.end(),
    [](double a, double b) { return std::abs(a) < std::abs(b); });
  const int peakIndex = static_cast<int>(std::distance(response.begin(), peakIt));
  const int declaredLatency = container.GetLatency();

  int firstNonzero = -1;
  for (size_t i = 0; i < response.size(); i++)
  {
    if (std::abs(response[i]) > 1.0e-12)
    {
      firstNonzero = static_cast<int>(i);
      break;
    }
  }

  const bool valid =
    mode == dsp::EAntiAliasFilterPhase::MinimumPhaseCascadedFIR
      ? declaredLatency == 0
      : std::abs(peakIndex - declaredLatency) <= 1;

  std::cout << ModeName(mode)
            << " " << factor << "x"
            << " declared=" << declaredLatency
            << " peak=" << peakIndex
            << " first=" << firstNonzero
            << " sample0=" << response[0]
            << " peak_value=" << *peakIt
            << (valid ? " OK" : " FAIL")
            << '\n';

  return valid;
}
}

int main()
{
  bool valid = true;

  for (const int factor : {2, 4, 8, 16, 32})
  {
    valid = CheckMode(factor, dsp::EAntiAliasFilterPhase::MinimumPhaseCascadedFIR) && valid;
    valid = CheckMode(factor, dsp::EAntiAliasFilterPhase::LinearCascadedFIRShort) && valid;
    valid = CheckMode(factor, dsp::EAntiAliasFilterPhase::LinearCascadedFIRLong) && valid;
  }

  return valid ? 0 : 1;
}
