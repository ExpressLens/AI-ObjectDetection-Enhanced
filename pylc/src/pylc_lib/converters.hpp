#ifndef CONVERTERS_H
#define CONVERTERS_H

#include <memory>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <stdexcept>
#include <opencv2/core/core.hpp>
#include <Eigen/Dense>
#include <eigen3/Eigen/src/Core/Matrix.h>
#include <pybind11/eigen.h>
#include <opencv2/opencv.hpp>
#include <opencv/cxeigen.hpp>

namespace py = pybind11;

// Numpy - cv::Mat interop
namespace pybi