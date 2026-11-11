#pragma once

#include <string_view>

inline constexpr std::string_view ServiceReferenceChildArg = "--service-reference-child";

// Runs the child-process role the service reference tests spawn ckpttest into.
int RunServiceReferenceChild(int argc, char* argv[]);
