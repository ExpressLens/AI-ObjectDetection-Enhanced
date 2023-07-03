/*
    tests/test_numpy_array.cpp -- test core array functionality

    Copyright (c) 2016 Ivan Smirnov <i.s.smirnov@gmail.com>

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE file.
*/

#include "pybind11_tests.h"

#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <cstdint>

// Size / dtype checks.
struct DtypeCheck {
    py::dtype numpy{};
    py::dtype pybind11{};
};

template <typename T>
DtypeCheck get_dtype_check(const char* name) {
    py::module np = py::module::import("numpy");
    DtypeCheck check{};
    check.numpy = np.attr("dtype")(np.attr(name));
    check.pybind11 = py::dtype::of<T>();
    return check;
}

std::vector<DtypeCheck> get_concrete_dtype_checks() {
    return {
        // Normalization
        get_dtype_check<std::int8_t>("int8"),
        get_dtype_check<std::uint8_t>("uint8"),
        get_dtype_check<std::int16_t>("int16"),
        get_dtype_check<std::uint16_t>("uint16"),
        get_dtype_check<std::int32_t>("int32"),
        get_dtype_check<std::uint32_t>("uint32"),
        get_dtype_check<std::int64_t>("int64"),
        get_dtype_check<std::uint64_t>("uint64")
    };
}

struct DtypeSizeCheck {
    std::string name{};
    int size_cpp{};
    int size_numpy{};
    // For debugging.
    py::dtype dtype{};
};

template <typename T>
DtypeSizeCheck get_dtype_size_check() {
    DtypeSizeCheck check{};
    check.name = py::type_id<T>();
    check.size_cpp = sizeof(T);
    check.dtype = py::dtype::of<T>();
    check.size_numpy = check.dtype.attr("itemsize").template cast<int>();
    return check;
}

std::vector<DtypeSizeCheck> get_platform_dtype_size_checks() {
    return {
        get_dtype_size_check<short>(),
        get_dtype_size_check<unsigned short>(),
        get_dtype_size_check<int>(),
        get_dtype_size_check<unsigned int>(),
        get_dtype_size_check<long>(),
        get_dtype_size_check<unsigned long>(),
        get_dtype_size_check<long long>(),
        get_dtype_size_check<unsigned long long>(),
    };
}

// Arrays.
using arr = py::array;
using arr_t = py::array_t<uint16_t, 0>;
static_assert(std::is_same<arr_t::value_type, uint16_t>::value, "");

template<typename... Ix> arr data(const arr& a, Ix... index) {
    return arr(a.nbytes() - a.offset_at(index...), (const uint8_t *) a.data(index...));
}

template<typename... Ix> arr data_t(const arr_t& a, Ix... index) {
    return arr(a.size() - a.index_at(index...), a.data(index...));
}

template<typename... Ix> arr& mutate_data(arr& a, Ix... index) {
    auto ptr = (uint8_t *) a.mutable_data(index...);
    for (ssize_t i = 0; i < a.nbytes() - a.offset_at(index...); i++)
        ptr[i] = (uint8_t) (ptr[i] * 2);
    return a;
}

template<typename... Ix> arr_t& mutate_data_t(arr_t& a, Ix... index) {
    auto ptr = a.mutable_data(index...);
    for (ssize_t i = 0; i < a.size() - a.index_at(index...); i++)
        ptr[i]++;
    return a;
}

template<typename... Ix> ssize_t index_at(const arr& a, Ix... idx) { return a.index_at(idx...); }
template<typename... Ix> ssize_t index_at_t(const arr_t& a, Ix... idx) { return a.index_at(idx...); }
template<typename... Ix> ssize_t offset_at(const arr& a, Ix... idx) { return a.offset_at(idx...); }
template<typename... Ix> ssize_t offset_at_t(const arr_t& a, Ix... idx) { return a.offset_at(idx...); }
template<typename... Ix> ssize_t at_t(const arr_t& a, Ix... idx) { return a.at(idx...); }
template<typename... Ix> arr_t& mutate_at_t(arr_t& a, Ix... idx) { a.mutable_at(idx...)++; return a; }

#define def_index_fn(name, type) \
    sm.def(#name, [](type a) { return name(a); }); \
    sm.def(#name, [](type a, int i) { return name(a, i); }); \
    sm.def(#name, [](type a, int i, int j) { return name(a, i, j); }); \
    sm.def(#name, [](type a, int i, int j, int k) { return name(a, i, j, k); });

template <typename T, typename T2> py::handle auxiliaries(T &&r, T2 &&r2) {
    if (r.ndim() != 2) throw std::domain_error("error: ndim != 2");
    py::list l;
    l.append(*r.data(0, 0));
    l.append(*r2.mutable_data(0, 0));
    l.append(r.data(0, 1) == r2.mutable_data(0, 1));
    l.append(r.ndim());
    l.append(r.itemsize());
    l.append(r.shape(0));
    l.append(r.shape(1));
    l.append(r.size());
    l.append(r.nbytes());
    return l.release();
}

// note: declaration at local scope would create a dangling reference!
static int data_i = 42;

TEST_SUBMODULE(numpy_array, sm) {
    try { py::module::import("numpy"); }
    catch (...) { return; }

    // test_dtypes
    py::class_<DtypeCheck>(sm, "DtypeCheck")
        .def_readonly("numpy", &DtypeCheck::numpy)
        .def_readonly("pybind11", &DtypeCheck::pybind11)
        .def("__repr__", [](const DtypeCheck& self) {
            return py::str("<DtypeCheck numpy={} pybind11={}>").format(
                self.numpy, self.pybind11);
        });
    sm.def("get_concrete_dtype_checks", &get_concrete_dtype_checks);

    py::class_<DtypeSizeCheck>(sm, "DtypeSizeCheck")
        .def_readonly("name", &DtypeSizeCheck::name)
        .def_readonly("size_cpp", &DtypeSizeCheck::size_cpp)
        .def_readonly("size_numpy", &DtypeSizeCheck::size_numpy)
        .def("__repr__", [](const DtypeSizeCheck& self) {
            return py::str("<DtypeSizeCheck name='{}' size_cpp={} size_numpy={} dtype={}>").format(
                self.name, self.size_cpp, self.size_numpy, self.dtype);
        });
    sm.def("get_platform_dtype_size_checks", &get_platform_dtype_size_checks);

    // test_array_attributes
    sm.def("ndim", [](const arr& a) { return a.ndim(); });
    sm.def("shape", [](const arr& a) { return arr(a.ndim(), a.shape()); });
    sm.def("shape", [](const arr& a, ssize_t dim) { return a.shape(dim); });
    sm.def("strides", [](const arr& a) { return arr(a.ndim(), a.strides()); });
    sm.def("strides", [](const arr& a, ssize_t dim) { return a.strides(dim); });
    sm.def("writeable", [](const arr& a) { return a.writeable(); });
    sm.def("size", [](const arr& a) { return a.size(); });
    sm.def("itemsize", [](const arr& a) { return a.itemsize(); });
    sm.def("nbytes", [](const arr& a) { return a.nbytes(); });
    sm.def("owndata", [](const arr& a) { return a.owndata(); });

    // test_index_offset
    def_index_fn(index_at, const arr&);
    def_index_fn(index_at_t, const arr_t&);
    def_index_fn(offset_at, const arr&);
    def_index_fn(offset_at_t, const arr_t&);
    // test_data
    def_index_fn(data, const arr&);
    def_index_fn(data_t, const arr_t&);
    // test_mutate_data, test_mutate_readonly
    def_index_fn(mutate_data, arr&);
    def_index_fn(mutate_data_t, arr_t&);
    def_index_fn(at_t, const arr_t&);
    def_index_fn(mutate_at_t, arr_t&);

    // test_make_c_f_array
    sm.def("make_f_array", [] { return py::array_t<float>({ 2, 2 }, { 4, 8 }); });
    sm.def("make_c_array", [] { return py::array_t<float>({ 2, 2 }, { 8, 4 }); });

    // test_empty_shaped_array
    sm.def("make_empty_shaped_array", [] { return py::array(py::dtype("f"), {}, {}); });
    // test numpy scalars (empty shape, ndim==0)
    sm.def("scalar_int", []() { return py::array(py::dtype("i"), {}, {}, &data_i); });

    // test_wrap
    sm.def("wrap", [](py::array a) {
        return py::array(
            a.dtype(),
            {a.shape(), a.shape() + a.ndim()},
            {a.strides(), a.strides() + a.ndim()},
            a.data(),
            a
        );
    });

    // test_numpy_view
    struct ArrayClass {
        int data[2] = { 1, 2 };
        ArrayClass() { py::print("ArrayClass()"); }
        ~ArrayClass() { py::print("~ArrayClass()"); }
    };
    py::class_<ArrayClass>(sm, "ArrayClass")
        .def(py::init<>())
        .def("numpy_view", [](py::object &obj) {
            py::print("ArrayClass::numpy_view()");
            ArrayClass &a = obj.cast<ArrayClass&>();
            return py::array_t<int>({2}, {4}, a.data, obj);
        }
    );

    // test_cast_numpy_int64_to_uint64
    sm.def("function_taking_uint64", [](uint64_t) { });

    // test_isinstance
    sm.def("isinstance_untyped", [](py::object yes, py::object no) {
        return py::isinstance<py::array>(yes) && !py::isinstance<py::array>(no);
    });
    sm.def("isinstance_typed", [](py::object o) {
        return py::isinstance<py::array_t<double>>(o) && !py::isinstance<py::array_t<int>>(o);
    });

    // test_constructors
    sm.def("default_constructors", []() {
        return py::dict(
            "array"_a=py::array(),
            "array_t<int32>"_a=py::array_t<std::int32_t>(),
            "array_t<double>"_a=py::array_t<double>()
        );
    });
    sm.def("converting_constructors", [](py::object o) {
        return py::dict(
            "array"_a=py::array(o),
            "array_t<int32>"_a=py::array_t<std::int32_t>(o),
            "array_t<double>"_a=py::array_t<double>(o)
        );
    });

    // test_overload_resolution
    sm.def("overloaded", [](py::array_t<double>) { return "double"; });
    sm.def("overloaded", [](py::array_t<float>) { return "float"; });
    sm.def("overloaded", [](py::array_t<int>) { return "int"; });
    sm.def("overloaded", [](py::array_t<unsigned short>) { return "unsigned short"; });
    sm.def("overloaded", [](py::array_t<long long>) { return "long long"; });
    sm.def("overloaded", [](py::array_t<std::complex<double>