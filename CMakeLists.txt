cmake_minimum_required(VERSION 3.30)
project(ShipExperiments)

set(CMAKE_CXX_STANDARD 20)

add_executable(main.cpp
        7100.cpp
        dsp.cpp
        7102.cpp
        7203.cpp
        7204.cpp
        7206_arm.cpp
        7109.cpp
        7210_debug.cpp
        7210_debug.patch)
