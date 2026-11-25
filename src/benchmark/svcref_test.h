#pragma once

#include <string_view>

inline constexpr std::string_view SvcRefChildArg = "--svcref-child";

// Runs the child-process role the service reference tests spawn ckpttest into.
int RunSvcRefChild(int argc, char* argv[]);
