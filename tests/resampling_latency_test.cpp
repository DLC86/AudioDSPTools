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

bool CheckMode(
  int factor,
  dsp::EAntiAliasFilterPhase mode,
  double hostSampleRate = 48000.0,
  double bandwidthSampleRate = 48000.0,
  const char* scenario = "standard")
{
  constexpr int blockSize = 64;
  constexpr int blockCount = 64;

  Container container(hostSampleRate * factor, mode, bandwidthSampleRate);
  container.Reset(hostSampleRate, blockSize);

  std::vector<double> input(blockSize, 0.0);
  std::vector<double> output(blockSize, 0.0);
  std::vector<double> response(blockSize * blockCount, 0.0);
  double* inputs[] = {input.data()};
  double* outputs[] = {output.data()};

  const auto renderImpulse = [&]()
  {
    std::vector<double> rendered(blockSize * blockCount, 0.0);
    std::fill(input.begin(), input.end(), 0.0);
    std::fill(output.begin(), output.end(), 0.0);
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

      std::copy(output.begin(), output.end(), rendered.begin() + block * blockSize);
      std::fill(input.begin(), input.end(), 0.0);
      std::fill(output.begin(), output.end(), 0.0);
    }
    return rendered;
  };

  response = renderImpulse();
  container.Reset(hostSampleRate, blockSize);
  const std::vector<double> responseAfterReset = renderImpulse();
  double resetDifference = 0.0;
  for (size_t i = 0; i < response.size(); i++)
    resetDifference = std::max(resetDifference, std::abs(response[i] - responseAfterReset[i]));

  const auto peakIt = std::max_element(
    response.begin(),
    response.end(),
    [](double a, double b) { return std::abs(a) < std::abs(b); });
  const int peakIndex = static_cast<int>(std::distance(response.begin(), peakIt));
  const int declaredLatency = container.GetLatency();
  double responseSum = 0.0;
  double firstMoment = 0.0;
  for (size_t i = 0; i < response.size(); i++)
  {
    responseSum += response[i];
    firstMoment += static_cast<double>(i) * response[i];
  }
  const double centroid =
    std::abs(responseSum) > 1.0e-15 ? firstMoment / responseSum : 0.0;

  int firstNonzero = -1;
  for (size_t i = 0; i < response.size(); i++)
  {
    if (std::abs(response[i]) > 1.0e-12)
    {
      firstNonzero = static_cast<int>(i);
      break;
    }
  }

  const bool causalAtSampleZero = response[0] != 0.0;
  const bool valid =
    mode == dsp::EAntiAliasFilterPhase::MinimumPhaseCascadedFIR
      ? declaredLatency == 0 && causalAtSampleZero && resetDifference <= 1.0e-12
      : std::abs(centroid - static_cast<double>(declaredLatency)) <= 1.0e-6
          && resetDifference <= 1.0e-12;

  std::cout << scenario << " " << ModeName(mode)
            << " " << factor << "x"
            << " declared=" << declaredLatency
            << " peak=" << peakIndex
            << " centroid=" << centroid
            << " first=" << firstNonzero
            << " sample0=" << response[0]
            << " peak_value=" << *peakIt
            << " reset_diff=" << resetDifference
            << (valid ? " OK" : " FAIL")
            << '\n';

  return valid;
}

bool CheckBlockPartitionInvariance(int factor, dsp::EAntiAliasFilterPhase mode)
{
  constexpr double sampleRate = 48000.0;
  constexpr int maxBlockSize = 1024;
  constexpr int totalFrames = 32768;
  std::vector<double> input(totalFrames);
  for (int i = 0; i < totalFrames; i++)
  {
    const double t = static_cast<double>(i) / sampleRate;
    input[i] = 0.2 * std::sin(2.0 * 3.14159265358979323846 * (70.0 + 8000.0 * t) * t);
  }

  const auto process = [&](const std::vector<int>& partitions)
  {
    Container container(sampleRate * factor, mode, sampleRate);
    container.Reset(sampleRate, maxBlockSize);
    std::vector<double> output(totalFrames, 0.0);
    int position = 0;
    size_t partition = 0;
    while (position < totalFrames)
    {
      const int frames = std::min(partitions[partition++ % partitions.size()], totalFrames - position);
      double* inputs[] = {input.data() + position};
      double* outputs[] = {output.data() + position};
      container.ProcessBlock(
        inputs,
        outputs,
        frames,
        [](double** in, double** out, int count) { std::copy(in[0], in[0] + count, out[0]); });
      position += frames;
    }
    return output;
  };

  const auto fixed = process({256});
  const auto variable = process({17, 31, 64, 127, 255, 3, 511, 1024, 9});
  double maxDifference = 0.0;
  double differenceEnergy = 0.0;
  double referenceEnergy = 0.0;
  for (int i = 0; i < totalFrames; i++)
  {
    const double difference = fixed[i] - variable[i];
    maxDifference = std::max(maxDifference, std::abs(difference));
    differenceEnergy += difference * difference;
    referenceEnergy += fixed[i] * fixed[i];
  }
  const double relativeDb =
    10.0 * std::log10((differenceEnergy + 1.0e-300) / (referenceEnergy + 1.0e-300));
  const bool valid = maxDifference <= 1.0e-9;
  std::cout << "partition " << ModeName(mode) << " " << factor << "x"
            << " max_diff=" << maxDifference
            << " relative_db=" << relativeDb
            << (valid ? " OK" : " FAIL") << '\n';
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

  // Host rate above the model bandwidth enables both Minimum Phase strict
  // guards: one before interpolation and one after final decimation.
  for (const int factor : {2, 4, 8, 16, 32})
  {
    valid = CheckMode(
              factor,
              dsp::EAntiAliasFilterPhase::MinimumPhaseCascadedFIR,
              96000.0,
              48000.0,
              "input+output-guards")
            && valid;
  }

  valid = CheckBlockPartitionInvariance(32, dsp::EAntiAliasFilterPhase::MinimumPhaseCascadedFIR) && valid;
  valid = CheckBlockPartitionInvariance(32, dsp::EAntiAliasFilterPhase::LinearCascadedFIRShort) && valid;
  valid = CheckBlockPartitionInvariance(32, dsp::EAntiAliasFilterPhase::LinearCascadedFIRLong) && valid;

  return valid ? 0 : 1;
}
