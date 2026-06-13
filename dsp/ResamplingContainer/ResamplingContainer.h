// File: ResamplingContainer.h
// Created Date: Saturday December 16th 2023
// Author: Steven Atkinson (steven@atkinson.mn)

// A container for real-time resampling using a Lanczos anti-aliasing filter

// This file originally came from the iPlug2 library and has been subsequently modified;
// the following license is copied as required from
// https://github.com/iPlug2/iPlug2/blob/40ebb560eba68f096221e99ef0ae826611fc2bda/LICENSE.txt
// -------------------------------------------------------------------------------------

/*
iPlug 2 C++ Plug-in Framework.

Copyright (C) the iPlug 2 Developers. Portions copyright other contributors, see each source file for more information.

Based on WDL-OL/iPlug by Oli Larkin (2011-2018), and the original iPlug v1 (2008) by John Schwartz / Cockos

LICENSE:

This software is provided 'as-is', without any express or implied warranty.  In no event will the authors be held liable
for any damages arising from the use of this software.

Permission is granted to anyone to use this software for any purpose, including commercial applications, and to alter it
and redistribute it freely, subject to the following restrictions:

1. The origin of this software must not be misrepresented; you must not claim that you wrote the original software. If
you use this software in a product, an acknowledgment in the product documentation would be appreciated but is not
required.
1. Altered source versions must be plainly marked as such, and must not be misrepresented as being the original
software.
1. This notice may not be removed or altered from any source distribution.

iPlug 2 includes the following 3rd party libraries (see each license info):

* Cockos WDL https://www.cockos.com/wdl
* NanoVG https://github.com/memononen/nanovg
* NanoSVG https://github.com/memononen/nanosvg
* MetalNanoVG https://github.com/ollix/MetalNanoVG
* RTAudio https://www.music.mcgill.ca/~gary/rtaudio
* RTMidi https://www.music.mcgill.ca/~gary/rtmidi
*/
// -------------------------------------------------------------------------------------

#pragma once

#include <iostream>
#include <functional>
#include <cmath>
#include <algorithm>
#include <complex>
#include <vector>

// #include "IPlugPlatform.h"

// #include "heapbuf.h"
#include "Dependencies/WDL/ptrlist.h"

#include "Dependencies/LanczosResampler.h"

namespace dsp
{

enum class EAntiAliasFilterPhase
{
  MinimumPhaseIIR = 0,
  MinimumPhaseFIR,
  PolyphaseFIR,
  LinearPhaseFIR
};

/** A multi-channel real-time resampling container that can be used to resample
 * audio processing to a specified sample rate for the situation where you have
 * some arbitary DSP code that requires a specific sample rate, then back to
 * the original external sample rate, encapsulating the arbitrary DSP code.

 * Three modes are supported:
 * - Linear interpolation: simple linear interpolation between samples
 * - Cubic interpolation: cubic interpolation between samples
 * - Lanczos: Lanczos resampling uses an approximation of the sinc function to
 *   interpolate between samples. This is the highest quality resampling mode.
 *
 * The Lanczos resampler has a configurable filter size (A) that affects the
 * latency of the resampler. It can also optionally use SIMD instructions to
 * when T==float.
 *
 *
 * @tparam T the sampletype
 * @tparam NCHANS the number of channels
 * @tparam A The Lanczos filter size for the LanczosResampler resampler mode
   A higher value makes the filter closer to an
   ideal stop-band that rejects high-frequency content (anti-aliasing),
   but at the expense of higher latency
 */
template <typename T = double, int NCHANS = 2, size_t A = 12>
class ResamplingContainer
{
public:
  using BlockProcessFunc = std::function<void(T**, T**, int)>;
  using LanczosResampler = LanczosResampler<T, NCHANS, A>;

  // :param renderingSampleRate: The sample rate required by the code to be encapsulated.
  // :param bandwidthSampleRate: The maximum useful sample rate of the encapsulated DSP. This limits the reconstruction
  // bandwidth when the external context runs at a higher sample rate than the model.
  ResamplingContainer(double renderingSampleRate,
                      EAntiAliasFilterPhase filterPhase = EAntiAliasFilterPhase::LinearPhaseFIR,
                      double bandwidthSampleRate = 0.0)
  : mRenderingSampleRate(renderingSampleRate)
  , mBandwidthSampleRate(bandwidthSampleRate > 0.0 ? bandwidthSampleRate : renderingSampleRate)
  , mFilterPhase(filterPhase)
  {
  }

  ResamplingContainer(const ResamplingContainer&) = delete;
  ResamplingContainer& operator=(const ResamplingContainer&) = delete;

  void SetAntiAliasFilterPhase(EAntiAliasFilterPhase filterPhase)
  {
    mFilterPhase = filterPhase;
  }

  // :param inputSampleRate: The external sample rate interacting with this object.
  // :param blockSize: The largest block size that will be given to this class to process until Reset()  is called
  //     again.
  void Reset(double inputSampleRate, int blockSize = DEFAULT_BLOCK_SIZE)
  {
    if (mInputSampleRate == inputSampleRate && mMaxBlockSize == blockSize && mDesignedFilterPhase == mFilterPhase)
    {
      ClearBuffers();
      return;
    }

    mInputSampleRate = inputSampleRate;
    mRatio1 = mInputSampleRate / mRenderingSampleRate;
    mRatio2 = mRenderingSampleRate / mInputSampleRate;
    const double exactDownsampleFactor = mRenderingSampleRate / mInputSampleRate;
    mIntegerDownsampleFactor = static_cast<int>(std::round(exactDownsampleFactor));
    mUseIntegerDownsampler =
      mIntegerDownsampleFactor > 1 && std::abs(exactDownsampleFactor - mIntegerDownsampleFactor) < 1.0e-9;
    mDecimationPhase = 0;
    // The buffers for the encapsulated code need to be long enough to hold the correesponding number of samples
    mMaxBlockSize = blockSize;
    mMaxEncapsulatedBlockSize = MaxEncapsulatedBlockSize(blockSize);

    mScratchExternalInputData.Resize(mMaxBlockSize * NCHANS); // This may contain junk right now.
    mEncapsulatedInputData.Resize(mMaxEncapsulatedBlockSize * NCHANS); // This may contain junk right now.
    mEncapsulatedOutputData.Resize(mMaxEncapsulatedBlockSize * NCHANS); // This may contain junk right now.
    mAntiAliasOutputData.Resize(mMaxEncapsulatedBlockSize * NCHANS); // This may contain junk right now.
    mScratchExternalInputPointers.Empty();
    mEncapsulatedInputPointers.Empty();
    mEncapsulatedOutputPointers.Empty();
    mAntiAliasOutputPointers.Empty();

    for (auto chan = 0; chan < NCHANS; chan++)
    {
      mScratchExternalInputPointers.Add(mScratchExternalInputData.Get() + (chan * mMaxBlockSize));
      mEncapsulatedInputPointers.Add(mEncapsulatedInputData.Get() + (chan * mMaxEncapsulatedBlockSize));
      mEncapsulatedOutputPointers.Add(mEncapsulatedOutputData.Get() + (chan * mMaxEncapsulatedBlockSize));
      mAntiAliasOutputPointers.Add(mAntiAliasOutputData.Get() + (chan * mMaxEncapsulatedBlockSize));
    }

    DesignAntiAliasFilter();

    mResampler1 = std::make_unique<LanczosResampler>(mInputSampleRate, mRenderingSampleRate);
    mResampler2 = mUseIntegerDownsampler ? nullptr
                                         : std::make_unique<LanczosResampler>(mRenderingSampleRate, mInputSampleRate);

    // Zeroes the scratch pointers so that we warm up with silence.
    ClearBuffers();

    // Warm up the resampling container with enough silence that the first real buffer can yield the required number
    // of output samples.
    const auto midSamples =
      mUseIntegerDownsampler ? static_cast<size_t>(mIntegerDownsampleFactor) : mResampler2->GetNumSamplesRequiredFor(1);
    mLatency = int(mResampler1->GetNumSamplesRequiredFor(midSamples));
    if (mAntiAliasEnabled && mUseIntegerDownsampler
        && (mFilterPhase == EAntiAliasFilterPhase::LinearPhaseFIR
            || mFilterPhase == EAntiAliasFilterPhase::PolyphaseFIR)
        && !mDecimationFirCoefficients.empty())
      mLatency += 2 * GetDecimationFirLatency();
    else if (mAntiAliasEnabled && mFilterPhase == EAntiAliasFilterPhase::LinearPhaseFIR)
      mLatency += 64;
    // 1. Push some silence through the first resampler.
    mResampler1->PushBlock(mScratchExternalInputPointers.GetList(), mLatency);
    const size_t populated = mResampler1->PopBlock(mEncapsulatedInputPointers.GetList(), midSamples);
    if (populated < midSamples)
    {
      throw std::runtime_error("Didn't get enough samples required for pre-population!");
    }
    // 2. "process" the warm-up in the encapsulated DSP.
    // Since this is an audio effect, we can assume that (1) it's causal and (2) that it's silent until
    // a non-silent input is given to it.
    // Therefore, we don't *acutally* need to use `func()`--we can assume that it would output silence!
    // func(mEncapsulatedInputPointers.GetList(), mEncapsulatedOutputPointers.GetList(), (int)populated);
    FallbackFunc(mEncapsulatedInputPointers.GetList(), mEncapsulatedOutputPointers.GetList(), (int)populated);
    T** resamplerInput = PrepareDownsamplerInput(populated);
    if (mUseIntegerDownsampler)
      DecimateBlock(resamplerInput, populated, nullptr, 0);
    else
      mResampler2->PushBlock(resamplerInput, populated);
    // Now we're ready for the first "real" buffer.
  }

  /** Resample an input block with a per-block function (up sample input -> process with function -> down sample)
   * @param inputs Two-dimensional array containing the non-interleaved input buffers of audio samples for all channels
   * @param outputs Two-dimensional array for audio output (non-interleaved).
   * @param nFrames The block size for this block: number of samples per channel.
   * @param func The function that processes the audio sample at the higher sampling rate. NOTE: std::function can call
   * malloc if you pass in captures */
  void ProcessBlock(T** inputs, T** outputs, int nFrames, BlockProcessFunc func)
  {
    mResampler1->PushBlock(inputs, nFrames);
    mOutputWritePos = 0;
    // This is the most samples the encapsualted context might get. Sometimes it'll get fewer.
    const auto maxEncapsulatedLen = MaxEncapsulatedBlockSize(nFrames);

    // Process as much audio as you can with the encapsulated DSP, and push it into the second resampler.
    // This will give the second reasmpler enough for it to pop the required buffer size to complete this function
    // correctly.
    while (mResampler1->GetNumSamplesRequiredFor(1) == 0) // i.e. there's more to process
    {
      // Get a block no larger than the encapsulated DSP is expecting.
      const size_t populated1 = mResampler1->PopBlock(mEncapsulatedInputPointers.GetList(), maxEncapsulatedLen);
      if (populated1 > maxEncapsulatedLen)
      {
        throw std::runtime_error("Got more encapsulated samples than the encapsulated DSP is prepared to handle!");
      }
      func(mEncapsulatedInputPointers.GetList(), mEncapsulatedOutputPointers.GetList(), (int)populated1);
      // And push the results into the second resampler so that it has what the external context requires.
      T** resamplerInput = PrepareDownsamplerInput(populated1);
      if (mUseIntegerDownsampler)
        DecimateBlock(resamplerInput, populated1, outputs, nFrames);
      else
        mResampler2->PushBlock(resamplerInput, populated1);
    }

    // Pop the required output from the second resampler for the external context.
    const auto populated2 = mUseIntegerDownsampler ? mOutputWritePos : mResampler2->PopBlock(outputs, nFrames);
    if (populated2 < nFrames)
    {
      std::cerr << "Did not yield enough samples (" << populated2 << ") to provide the required output buffer (expected"
                << nFrames << ")! Filling with last sample..." << std::endl;
      for (int c = 0; c < NCHANS; c++)
      {
        const T lastSample = populated2 > 0 ? outputs[c][populated2 - 1] : 0.0;
        for (int i = populated2; i < nFrames; i++)
        {
          outputs[c][i] = lastSample;
        }
      }
    }
    // Get ready for the next block:
    mResampler1->RenormalizePhases();
    if (mResampler2 != nullptr)
      mResampler2->RenormalizePhases();
  }

  int GetLatency() const { return mLatency; }

private:
  static inline int LinearInterpolate(T** inputs, T** outputs, int inputLen, double ratio, int maxOutputLen)
  {
    // FIXME check through this!
    const auto outputLen = std::min(static_cast<int>(std::ceil(static_cast<double>(inputLen) / ratio)), maxOutputLen);

    for (auto writePos = 0; writePos < outputLen; writePos++)
    {
      const auto readPos = ratio * static_cast<double>(writePos);
      const auto readPostionTrunc = std::floor(readPos);
      const auto readPosInt = static_cast<int>(readPostionTrunc);

      if (readPosInt < inputLen)
      {
        const auto y = readPos - readPostionTrunc;

        for (auto chan = 0; chan < NCHANS; chan++)
        {
          const auto x0 = inputs[chan][readPosInt];
          const auto x1 = ((readPosInt + 1) < inputLen) ? inputs[chan][readPosInt + 1] : inputs[chan][readPosInt - 1];
          outputs[chan][writePos] = (1.0 - y) * x0 + y * x1;
        }
      }
    }

    return outputLen;
  }

  static inline int CubicInterpolate(T** inputs, T** outputs, int inputLen, double ratio, int maxOutputLen)
  {
    // FIXME check through this!
    const auto outputLen = std::min(static_cast<int>(std::ceil(static_cast<double>(inputLen) / ratio)), maxOutputLen);

    for (auto writePos = 0; writePos < outputLen; writePos++)
    {
      const auto readPos = ratio * static_cast<double>(writePos);
      const auto readPostionTrunc = std::floor(readPos);
      const auto readPosInt = static_cast<int>(readPostionTrunc);

      if (readPosInt < inputLen)
      {
        const auto y = readPos - readPostionTrunc;

        for (auto chan = 0; chan < NCHANS; chan++)
        {
          const auto xm1 = ((readPosInt - 1) > 0) ? inputs[chan][readPosInt - 1] : 0.0f;
          const auto x0 = ((readPosInt) < inputLen) ? inputs[chan][readPosInt] : inputs[chan][readPosInt - 1];
          const auto x1 = ((readPosInt + 1) < inputLen) ? inputs[chan][readPosInt + 1] : inputs[chan][readPosInt - 1];
          const auto x2 = ((readPosInt + 2) < inputLen) ? inputs[chan][readPosInt + 2] : inputs[chan][readPosInt - 1];

          const auto c = (x1 - xm1) * 0.5;
          const auto v = x0 - x1;
          const auto w = c + v;
          const auto a = w + v + (x2 - x0) * 0.5;
          const auto b = w + a;

          outputs[chan][writePos] = ((((a * y) - b) * y + c) * y + x0);
        }
      }
    }

    return outputLen;
  }

  void ClearBuffers()
  {
    memset(mScratchExternalInputData.Get(), 0.0f, DataSize(mMaxBlockSize));
    const auto encapsulatedDataSize = DataSize(mMaxEncapsulatedBlockSize);
    memset(mEncapsulatedInputData.Get(), 0.0f, encapsulatedDataSize);
    memset(mEncapsulatedOutputData.Get(), 0.0f, encapsulatedDataSize);
    memset(mAntiAliasOutputData.Get(), 0.0f, encapsulatedDataSize);
    std::fill(mAntiAliasHistory.begin(), mAntiAliasHistory.end(), T(0.0));
    std::fill(mDecimationFirHistory.begin(), mDecimationFirHistory.end(), T(0.0));

    if (mResampler1 != nullptr)
    {
      mResampler1->ClearBuffer();
    }
    if (mResampler2 != nullptr)
    {
      mResampler2->ClearBuffer();
    }
    mDecimationPhase = 0;
    mOutputWritePos = 0;
  }

  // How big could the corresponding encapsulated buffer be for a buffer at the external sample rate of a given size?
  int MaxEncapsulatedBlockSize(const int externalBlockSize) const
  {
    return static_cast<int>(std::ceil(static_cast<double>(externalBlockSize) / mRatio1));
  }

  // Size of the multi-channel data for a given block size
  size_t DataSize(const int blockSize) const { return blockSize * NCHANS * sizeof(T); };

  void FallbackFunc(T** inputs, T** outputs, int n)
  {
    for (int i = 0; i < NCHANS; i++)
    {
      memcpy(outputs[i], inputs[i], n * sizeof(T));
    }
  }

  void DesignAntiAliasFilter()
  {
    mAntiAliasEnabled = mRenderingSampleRate > (mInputSampleRate * 1.000001)
                        || mInputSampleRate > (mBandwidthSampleRate * 1.000001);
    mAntiAliasCoefficients.clear();
    mAntiAliasHistory.clear();
    mMinimumPhaseSections.clear();
    mMinimumPhaseState.clear();
    mDecimationFirCoefficients.clear();
    mDecimationFirHistory.clear();

    if (!mAntiAliasEnabled)
    {
      mDesignedFilterPhase = mFilterPhase;
      return;
    }

    if (mUseIntegerDownsampler)
    {
      if (mFilterPhase == EAntiAliasFilterPhase::MinimumPhaseIIR)
        DesignMinimumPhaseAntiAliasFilter();
      else
        DesignDecimationFIRAntiAliasFilter(mFilterPhase == EAntiAliasFilterPhase::MinimumPhaseFIR);
      mDesignedFilterPhase = mFilterPhase;
      return;
    }

    if (mFilterPhase == EAntiAliasFilterPhase::MinimumPhaseIIR)
    {
      DesignMinimumPhaseAntiAliasFilter();
      mDesignedFilterPhase = mFilterPhase;
      return;
    }

    if (mFilterPhase == EAntiAliasFilterPhase::MinimumPhaseFIR)
    {
      DesignMinimumPhaseFIRAntiAliasFilter();
      mDesignedFilterPhase = mFilterPhase;
      return;
    }

    DesignLinearPhaseAntiAliasFilter();
    mDesignedFilterPhase = mFilterPhase;
  }

  void DesignLinearPhaseAntiAliasFilter()
  {
    constexpr double pi = 3.14159265358979323846264338327950288;
    constexpr int numTaps = 255;
    const double effectiveInputSampleRate = std::min(mInputSampleRate, mBandwidthSampleRate);
    const double cutoff = std::min(0.49, 0.475 * effectiveInputSampleRate / mRenderingSampleRate);
    double sum = 0.0;

    mAntiAliasCoefficients.resize(numTaps);
    for (int n = 0; n < numTaps; n++)
    {
      const double x = n - 0.5 * (numTaps - 1);
      const double sinc = std::abs(x) < 1.0e-12 ? 2.0 * cutoff : std::sin(2.0 * pi * cutoff * x) / (pi * x);
      const double window = 0.42 - 0.5 * std::cos(2.0 * pi * n / (numTaps - 1))
                            + 0.08 * std::cos(4.0 * pi * n / (numTaps - 1));
      mAntiAliasCoefficients[n] = static_cast<T>(sinc * window);
      sum += mAntiAliasCoefficients[n];
    }

    if (std::abs(sum) > 1.0e-20)
      for (auto& c : mAntiAliasCoefficients)
        c = static_cast<T>(c / sum);

    mAntiAliasHistory.assign(NCHANS * (numTaps - 1), T(0.0));
  }

  void DesignMinimumPhaseFIRAntiAliasFilter()
  {
    DesignLinearPhaseAntiAliasFilter();
    ConvertFIRToMinimumPhase(mAntiAliasCoefficients);
    std::fill(mAntiAliasHistory.begin(), mAntiAliasHistory.end(), T(0.0));
  }

  std::vector<T> DesignKaiserLowpassFIR(int numTaps, double cutoff)
  {
    constexpr double pi = 3.14159265358979323846264338327950288;
    constexpr double beta = 10.5;
    const double denom = BesselI0(beta);
    double sum = 0.0;
    std::vector<T> coefficients(numTaps);

    for (int n = 0; n < numTaps; n++)
    {
      const double x = n - 0.5 * (numTaps - 1);
      const double sinc = std::abs(x) < 1.0e-12 ? 2.0 * cutoff : std::sin(2.0 * pi * cutoff * x) / (pi * x);
      const double r = (2.0 * n) / (numTaps - 1) - 1.0;
      const double window = BesselI0(beta * std::sqrt(std::max(0.0, 1.0 - r * r))) / denom;
      coefficients[n] = static_cast<T>(sinc * window);
      sum += coefficients[n];
    }

    if (std::abs(sum) > 1.0e-20)
      for (auto& c : coefficients)
        c = static_cast<T>(c / sum);

    return coefficients;
  }

  void DesignDecimationFIRAntiAliasFilter(bool minimumPhase)
  {
    constexpr int tapsPerFactor = 128;
    constexpr double cutoffScale = 0.445;
    const double effectiveInputSampleRate = std::min(mInputSampleRate, mBandwidthSampleRate);
    const double effectiveFactor = std::max(1.0, mRenderingSampleRate / effectiveInputSampleRate);
    const int factor = std::max(1, static_cast<int>(std::ceil(effectiveFactor)));
    const int numTaps = tapsPerFactor * factor + 1;
    const double cutoff = std::min(0.49, cutoffScale / effectiveFactor);

    mDecimationFirCoefficients = DesignKaiserLowpassFIR(numTaps, cutoff);
    if (minimumPhase)
      ConvertFIRToMinimumPhase(mDecimationFirCoefficients);

    mDecimationFirHistory.assign(NCHANS * (numTaps - 1), T(0.0));
  }

  void ApplyAntiAliasFilter(size_t nFrames)
  {
    if (mFilterPhase == EAntiAliasFilterPhase::MinimumPhaseIIR)
      ApplyMinimumPhaseAntiAliasFilter(nFrames);
    else
      ApplyLinearPhaseAntiAliasFilter(nFrames);
  }

  T** PrepareDownsamplerInput(size_t nFrames)
  {
    if (!mAntiAliasEnabled)
      return mEncapsulatedOutputPointers.GetList();

    if (mUseIntegerDownsampler && (mFilterPhase == EAntiAliasFilterPhase::PolyphaseFIR
                                   || mFilterPhase == EAntiAliasFilterPhase::LinearPhaseFIR
                                   || mFilterPhase == EAntiAliasFilterPhase::MinimumPhaseFIR))
      return mEncapsulatedOutputPointers.GetList();

    ApplyAntiAliasFilter(nFrames);
    return mAntiAliasOutputPointers.GetList();
  }

  void ApplyLinearPhaseAntiAliasFilter(size_t nFrames)
  {
    const size_t numTaps = mAntiAliasCoefficients.size();
    const size_t historyLen = numTaps - 1;

    for (int chan = 0; chan < NCHANS; chan++)
    {
      T* input = mEncapsulatedOutputPointers.Get(chan);
      T* output = mAntiAliasOutputPointers.Get(chan);
      T* history = mAntiAliasHistory.data() + chan * historyLen;

      for (size_t s = 0; s < nFrames; s++)
      {
        T y = T(0.0);
        for (size_t k = 0; k < numTaps; k++)
        {
          const long inputIndex = static_cast<long>(s) - static_cast<long>(k);
          const T x = inputIndex >= 0 ? input[inputIndex] : history[historyLen + inputIndex];
          y += mAntiAliasCoefficients[k] * x;
        }
        output[s] = y;
      }

      if (nFrames >= historyLen)
      {
        std::copy(input + nFrames - historyLen, input + nFrames, history);
      }
      else
      {
        std::move(history + nFrames, history + historyLen, history);
        std::copy(input, input + nFrames, history + historyLen - nFrames);
      }
    }
  }

  struct BiquadCoefficients
  {
    T b0 = T(1.0);
    T b1 = T(0.0);
    T b2 = T(0.0);
    T a1 = T(0.0);
    T a2 = T(0.0);
  };

  struct BiquadState
  {
    T z1 = T(0.0);
    T z2 = T(0.0);
  };

  static double BesselI0(double x)
  {
    double sum = 1.0;
    double term = 1.0;
    const double y = 0.25 * x * x;
    for (int k = 1; k < 32; k++)
    {
      term *= y / static_cast<double>(k * k);
      sum += term;
      if (term < 1.0e-12 * sum)
        break;
    }
    return sum;
  }

  static void FFT(std::vector<std::complex<double>>& data, bool inverse)
  {
    const size_t n = data.size();
    for (size_t i = 1, j = 0; i < n; i++)
    {
      size_t bit = n >> 1;
      for (; j & bit; bit >>= 1)
        j ^= bit;
      j ^= bit;
      if (i < j)
        std::swap(data[i], data[j]);
    }

    constexpr double pi = 3.14159265358979323846264338327950288;
    for (size_t len = 2; len <= n; len <<= 1)
    {
      const double angle = (inverse ? 2.0 : -2.0) * pi / static_cast<double>(len);
      const std::complex<double> wLen(std::cos(angle), std::sin(angle));
      for (size_t i = 0; i < n; i += len)
      {
        std::complex<double> w(1.0, 0.0);
        for (size_t j = 0; j < len / 2; j++)
        {
          const auto u = data[i + j];
          const auto v = data[i + j + len / 2] * w;
          data[i + j] = u + v;
          data[i + j + len / 2] = u - v;
          w *= wLen;
        }
      }
    }

    if (inverse)
      for (auto& x : data)
        x /= static_cast<double>(n);
  }

  void ConvertFIRToMinimumPhase(std::vector<T>& coefficients)
  {
    const size_t numTaps = coefficients.size();
    size_t fftSize = 1;
    while (fftSize < 16 * numTaps)
      fftSize <<= 1;

    std::vector<std::complex<double>> spectrum(fftSize, std::complex<double>{0.0, 0.0});
    for (size_t i = 0; i < numTaps; i++)
      spectrum[i] = static_cast<double>(coefficients[i]);

    FFT(spectrum, false);
    for (auto& x : spectrum)
      x = std::log(std::max(1.0e-14, std::abs(x)));

    FFT(spectrum, true);
    std::vector<std::complex<double>> minimumCepstrum(fftSize, std::complex<double>{0.0, 0.0});
    minimumCepstrum[0] = spectrum[0].real();
    for (size_t i = 1; i < fftSize / 2; i++)
      minimumCepstrum[i] = 2.0 * spectrum[i].real();
    if ((fftSize & 1U) == 0U)
      minimumCepstrum[fftSize / 2] = spectrum[fftSize / 2].real();

    FFT(minimumCepstrum, false);
    for (auto& x : minimumCepstrum)
      x = std::exp(x);

    FFT(minimumCepstrum, true);
    double sum = 0.0;
    for (size_t i = 0; i < numTaps; i++)
    {
      coefficients[i] = static_cast<T>(minimumCepstrum[i].real());
      sum += coefficients[i];
    }

    if (std::abs(sum) > 1.0e-20)
      for (auto& c : coefficients)
        c = static_cast<T>(c / sum);
  }

  std::vector<BiquadCoefficients> DesignButterworthLowpass(double cutoff, int order)
  {
    constexpr double pi = 3.14159265358979323846264338327950288;
    cutoff = std::min(0.49, std::max(0.001, cutoff));
    const int numSections = order / 2;
    const double omega = 2.0 * pi * cutoff;
    const double sinOmega = std::sin(omega);
    const double cosOmega = std::cos(omega);

    std::vector<BiquadCoefficients> sections(numSections);
    for (int section = 0; section < numSections; section++)
    {
      const double q = 1.0 / (2.0 * std::cos((2.0 * section + 1.0) * pi / (2.0 * order)));
      const double alpha = sinOmega / (2.0 * q);
      const double a0 = 1.0 + alpha;

      BiquadCoefficients coeffs;
      coeffs.b0 = static_cast<T>((1.0 - cosOmega) * 0.5 / a0);
      coeffs.b1 = static_cast<T>((1.0 - cosOmega) / a0);
      coeffs.b2 = static_cast<T>((1.0 - cosOmega) * 0.5 / a0);
      coeffs.a1 = static_cast<T>((-2.0 * cosOmega) / a0);
      coeffs.a2 = static_cast<T>((1.0 - alpha) / a0);
      sections[section] = coeffs;
    }
    return sections;
  }

  void DesignMinimumPhaseAntiAliasFilter()
  {
    constexpr double pi = 3.14159265358979323846264338327950288;
    constexpr int order = 32;
    constexpr int numSections = order / 2;
    constexpr double rippleDb = 0.1;
    constexpr double cutoffScale = 0.445;
    const double effectiveInputSampleRate = std::min(mInputSampleRate, mBandwidthSampleRate);
    const double factor = mRenderingSampleRate / effectiveInputSampleRate;
    const double cutoff = std::min(0.49, cutoffScale / std::max(1.0, factor));
    const double epsilon = std::sqrt(std::pow(10.0, rippleDb / 10.0) - 1.0);
    const double mu = std::asinh(1.0 / epsilon) / static_cast<double>(order);
    const double warpedCutoff = 2.0 * std::tan(pi * cutoff);

    mMinimumPhaseSections.resize(numSections);
    for (int section = 0; section < numSections; section++)
    {
      const double theta = pi * (2.0 * section + 1.0) / (2.0 * order);
      const double sigma = -std::sinh(mu) * std::sin(theta) * warpedCutoff;
      const double omega = std::cosh(mu) * std::cos(theta) * warpedCutoff;
      const double b = -2.0 * sigma;
      const double c = sigma * sigma + omega * omega;
      const double a0 = 4.0 + 2.0 * b + c;

      BiquadCoefficients coeffs;
      coeffs.b0 = static_cast<T>(c / a0);
      coeffs.b1 = static_cast<T>((2.0 * c) / a0);
      coeffs.b2 = static_cast<T>(c / a0);
      coeffs.a1 = static_cast<T>((-8.0 + 2.0 * c) / a0);
      coeffs.a2 = static_cast<T>((4.0 - 2.0 * b + c) / a0);
      mMinimumPhaseSections[section] = coeffs;
    }

    mMinimumPhaseState.assign(NCHANS * mMinimumPhaseSections.size(), BiquadState {});
  }

  void ApplyMinimumPhaseAntiAliasFilter(size_t nFrames)
  {
    ApplyMinimumPhaseFilter(mEncapsulatedOutputPointers.GetList(), mAntiAliasOutputPointers.GetList(), nFrames,
                            mMinimumPhaseState);
  }

  void ApplyMinimumPhaseFilter(T** input, T** output, size_t nFrames, std::vector<BiquadState>& state)
  {
    for (int chan = 0; chan < NCHANS; chan++)
    {
      for (size_t s = 0; s < nFrames; s++)
      {
        T y = input[chan][s];
        for (size_t section = 0; section < mMinimumPhaseSections.size(); section++)
        {
          const auto& coeffs = mMinimumPhaseSections[section];
          auto& sectionState = state[chan * mMinimumPhaseSections.size() + section];
          const T out = coeffs.b0 * y + sectionState.z1;
          sectionState.z1 = coeffs.b1 * y - coeffs.a1 * out + sectionState.z2;
          sectionState.z2 = coeffs.b2 * y - coeffs.a2 * out;
          y = out;

          if (!std::isfinite(static_cast<double>(y)) || std::abs(static_cast<double>(y)) > 1.0e12)
          {
            for (size_t resetSection = 0; resetSection < mMinimumPhaseSections.size(); resetSection++)
              state[chan * mMinimumPhaseSections.size() + resetSection] = {};
            y = T(0.0);
            break;
          }
        }
        output[chan][s] = y;
      }
    }
  }

  void ApplyFIR(T** input, T** output, size_t nFrames, const std::vector<T>& coefficients, std::vector<T>& historyBuffer)
  {
    const size_t numTaps = coefficients.size();
    const size_t historyLen = numTaps - 1;

    for (int chan = 0; chan < NCHANS; chan++)
    {
      T* history = historyBuffer.data() + chan * historyLen;

      for (size_t s = 0; s < nFrames; s++)
      {
        T y = T(0.0);
        for (size_t k = 0; k < numTaps; k++)
        {
          const long inputIndex = static_cast<long>(s) - static_cast<long>(k);
          const T x = inputIndex >= 0 ? input[chan][inputIndex] : history[historyLen + inputIndex];
          y += coefficients[k] * x;
        }
        output[chan][s] = y;
      }

      if (nFrames >= historyLen)
      {
        std::copy(input[chan] + nFrames - historyLen, input[chan] + nFrames, history);
      }
      else
      {
        std::move(history + nFrames, history + historyLen, history);
        std::copy(input[chan], input[chan] + nFrames, history + historyLen - nFrames);
      }
    }
  }

  int GetDecimationFirLatency() const
  {
    if (mDecimationFirCoefficients.empty() || mIntegerDownsampleFactor <= 1)
      return 0;
    return static_cast<int>((mDecimationFirCoefficients.size() - 1) / (2 * mIntegerDownsampleFactor));
  }

  void DecimateBlock(T** input, size_t nFrames, T** outputs, int maxOutputFrames)
  {
    if (!mDecimationFirCoefficients.empty())
    {
      if (mFilterPhase == EAntiAliasFilterPhase::PolyphaseFIR)
        DirectPolyphaseFIRDecimateBlock(input, nFrames, outputs, maxOutputFrames);
      else
        DirectFIRDecimateBlock(input, nFrames, outputs, maxOutputFrames);
    }
    else
      DirectDecimateBlock(input, nFrames, outputs, maxOutputFrames);
  }

  void DirectFIRDecimateBlock(T** input, size_t nFrames, T** outputs, int maxOutputFrames)
  {
    const size_t numTaps = mDecimationFirCoefficients.size();
    const size_t historyLen = numTaps - 1;

    for (size_t s = 0; s < nFrames; s++)
    {
      if (mDecimationPhase == 0)
      {
        if (outputs != nullptr && mOutputWritePos < static_cast<size_t>(maxOutputFrames))
        {
          for (int chan = 0; chan < NCHANS; chan++)
          {
            T y = T(0.0);
            T* history = mDecimationFirHistory.data() + chan * historyLen;
            for (size_t k = 0; k < numTaps; k++)
            {
              const long inputIndex = static_cast<long>(s) - static_cast<long>(k);
              const T x = inputIndex >= 0 ? input[chan][inputIndex] : history[historyLen + inputIndex];
              y += mDecimationFirCoefficients[k] * x;
            }
            outputs[chan][mOutputWritePos] = y;
          }
          mOutputWritePos++;
        }
      }
      mDecimationPhase++;
      if (mDecimationPhase >= mIntegerDownsampleFactor)
        mDecimationPhase = 0;
    }

    for (int chan = 0; chan < NCHANS; chan++)
    {
      T* history = mDecimationFirHistory.data() + chan * historyLen;
      if (nFrames >= historyLen)
      {
        std::copy(input[chan] + nFrames - historyLen, input[chan] + nFrames, history);
      }
      else
      {
        std::move(history + nFrames, history + historyLen, history);
        std::copy(input[chan], input[chan] + nFrames, history + historyLen - nFrames);
      }
    }
  }

  void DirectPolyphaseFIRDecimateBlock(T** input, size_t nFrames, T** outputs, int maxOutputFrames)
  {
    const int factor = std::max(1, mIntegerDownsampleFactor);
    const size_t numTaps = mDecimationFirCoefficients.size();
    const size_t historyLen = numTaps - 1;

    for (size_t s = 0; s < nFrames; s++)
    {
      if (mDecimationPhase == 0)
      {
        if (outputs != nullptr && mOutputWritePos < static_cast<size_t>(maxOutputFrames))
        {
          for (int chan = 0; chan < NCHANS; chan++)
          {
            T y = T(0.0);
            T* history = mDecimationFirHistory.data() + chan * historyLen;
            for (int phase = 0; phase < factor; phase++)
            {
              for (size_t k = static_cast<size_t>(phase); k < numTaps; k += static_cast<size_t>(factor))
              {
                const long inputIndex = static_cast<long>(s) - static_cast<long>(k);
                const T x = inputIndex >= 0 ? input[chan][inputIndex] : history[historyLen + inputIndex];
                y += mDecimationFirCoefficients[k] * x;
              }
            }
            outputs[chan][mOutputWritePos] = y;
          }
          mOutputWritePos++;
        }
      }
      mDecimationPhase++;
      if (mDecimationPhase >= mIntegerDownsampleFactor)
        mDecimationPhase = 0;
    }

    for (int chan = 0; chan < NCHANS; chan++)
    {
      T* history = mDecimationFirHistory.data() + chan * historyLen;
      if (nFrames >= historyLen)
      {
        std::copy(input[chan] + nFrames - historyLen, input[chan] + nFrames, history);
      }
      else
      {
        std::move(history + nFrames, history + historyLen, history);
        std::copy(input[chan], input[chan] + nFrames, history + historyLen - nFrames);
      }
    }
  }

  void DirectDecimateBlock(T** input, size_t nFrames, T** outputs, int maxOutputFrames)
  {
    for (size_t s = 0; s < nFrames; s++)
    {
      if (mDecimationPhase == 0)
      {
        if (outputs != nullptr && mOutputWritePos < static_cast<size_t>(maxOutputFrames))
        {
          for (int chan = 0; chan < NCHANS; chan++)
            outputs[chan][mOutputWritePos] = input[chan][s];
          mOutputWritePos++;
        }
      }
      mDecimationPhase++;
      if (mDecimationPhase >= mIntegerDownsampleFactor)
        mDecimationPhase = 0;
    }
  }

  // Buffers for scratch input data for Reset() to use
  WDL_TypedBuf<T> mScratchExternalInputData;
  WDL_PtrList<T> mScratchExternalInputPointers;
  // Buffers for the input & output to the encapsulated DSP
  WDL_TypedBuf<T> mEncapsulatedInputData;
  WDL_PtrList<T> mEncapsulatedInputPointers;
  WDL_TypedBuf<T> mEncapsulatedOutputData;
  WDL_PtrList<T> mEncapsulatedOutputPointers;
  WDL_TypedBuf<T> mAntiAliasOutputData;
  WDL_PtrList<T> mAntiAliasOutputPointers;
  std::vector<T> mAntiAliasCoefficients;
  std::vector<T> mAntiAliasHistory;
  std::vector<BiquadCoefficients> mMinimumPhaseSections;
  std::vector<BiquadState> mMinimumPhaseState;
  std::vector<T> mDecimationFirCoefficients;
  std::vector<T> mDecimationFirHistory;
  bool mAntiAliasEnabled = false;
  EAntiAliasFilterPhase mFilterPhase = EAntiAliasFilterPhase::LinearPhaseFIR;
  EAntiAliasFilterPhase mDesignedFilterPhase = EAntiAliasFilterPhase::LinearPhaseFIR;
  bool mUseIntegerDownsampler = false;
  int mIntegerDownsampleFactor = 1;
  int mDecimationPhase = 0;
  size_t mOutputWritePos = 0;
  // Sample rate ratio from external to encapsulated, from encapsulated to external.
  double mRatio1 = 0.0, mRatio2 = 0.0;
  // Sample rate of the external context.
  double mInputSampleRate = 0.0;
  // The size of the largest block the external context may provide. (It might provide something smaller.)
  int mMaxBlockSize = 0;
  // The size of the largest possible encapsulated block
  int mMaxEncapsulatedBlockSize = 0;
  // How much latency this object adds due to both of its resamplers. This does _not_ include the latency due to the
  // encapsulated `func()`.
  int mLatency = 0;
  // The sample rate required by the DSP that this object encapsulates
  const double mRenderingSampleRate;
  // The highest sample rate whose bandwidth should be preserved by reconstruction filters.
  const double mBandwidthSampleRate;
  // Pair of resamplers for (1) external -> encapsulated, (2) encapsulated -> external
  std::unique_ptr<LanczosResampler> mResampler1, mResampler2;
};

}; // namespace dsp
