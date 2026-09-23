# Optional C++/pybind11 grid backend `foveamap._fovea_cpp` (README 6.5.7, tier P2).
# Sources are added in Phase 7, task T7.5, only if triggered. The pipeline falls back to NumPy otherwise.
cmake_minimum_required(VERSION 3.20)
project(fovea_cpp LANGUAGES CXX)
message(FATAL_ERROR "The C++ backend is not implemented yet (Phase 7, T7.5). The NumPy backend is used.")
