#pragma once

#include "quant.h"

#include <cstdint>
#include <span>

namespace checkpoint::host_dequant
{

bool HasStorageHeader(std::span<char const> bytes);

uint16_t FP8E4M3ToBFloat16(uint8_t value, float scaleInv);
uint16_t MXFP8E4M3ToBFloat16(uint8_t value, uint8_t biasedScale);
uint16_t NVFP4E2M1ToBFloat16(uint8_t value, uint8_t e4m3Scale, float amax, quant::NVFP4E4M3Mode e4m3Mode);

void FP8E4M3ToBFloat16Range(std::span<char const> const& storage, uint64_t elementOffset, std::span<uint16_t> output);
void MXFP8E4M3ToBFloat16(std::span<char const> const& storage, std::span<uint16_t> output);
void MXFP8E4M3ToBFloat16Range(std::span<char const> const& storage, uint64_t elementOffset, std::span<uint16_t> output);
void NVFP4E2M1ToBFloat16Range(std::span<char const> const& storage, uint64_t elementOffset, std::span<uint16_t> output);
void DequantizeToBFloat16Range(std::span<char const> const& storage, uint64_t elementOffset, std::span<uint16_t> output);

}    // namespace checkpoint::host_dequant
