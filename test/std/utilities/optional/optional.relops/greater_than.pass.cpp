//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++98, c++03, c++11, c++14
// <optional>

// template <class T, class U> constexpr bool operator> (const optional<T>& x, const optional<U>& y);

#include "oneapi_std_test_config.h"
#include "test_macros.h"
#include <CL/sycl.hpp>
#include <iostream>

#ifdef USE_ONEAPI_STD
#    include _ONEAPI_STD_TEST_HEADER(optional)
#    include _ONEAPI_STD_TEST_HEADER(type_traits)
namespace s = oneapi_cpp_ns;
#else
#    include <optional>
#    include <type_traits>
namespace s = std;
#endif

using s::optional;

struct X
{
    int i_;

    constexpr X(int i) : i_(i) {}
};

constexpr bool
operator>(const X& lhs, const X& rhs)
{
    return lhs.i_ > rhs.i_;
}

void
kernel_test()
{
    cl::sycl::queue q;
    q.submit([&](cl::sycl::handler& cgh) {
        cgh.single_task<class KernelTest>([=]() {
            {
                typedef optional<X> O;

                constexpr O o1;    // disengaged
                constexpr O o2;    // disengaged
                constexpr O o3{1}; // engaged
                constexpr O o4{2}; // engaged
                constexpr O o5{1}; // engaged

                static_assert(!(o1 > o1), "");
                static_assert(!(o1 > o2), "");
                static_assert(!(o1 > o3), "");
                static_assert(!(o1 > o4), "");
                static_assert(!(o1 > o5), "");

                static_assert(!(o2 > o1), "");
                static_assert(!(o2 > o2), "");
                static_assert(!(o2 > o3), "");
                static_assert(!(o2 > o4), "");
                static_assert(!(o2 > o5), "");

                static_assert((o3 > o1), "");
                static_assert((o3 > o2), "");
                static_assert(!(o3 > o3), "");
                static_assert(!(o3 > o4), "");
                static_assert(!(o3 > o5), "");

                static_assert((o4 > o1), "");
                static_assert((o4 > o2), "");
                static_assert((o4 > o3), "");
                static_assert(!(o4 > o4), "");
                static_assert((o4 > o5), "");

                static_assert((o5 > o1), "");
                static_assert((o5 > o2), "");
                static_assert(!(o5 > o3), "");
                static_assert(!(o5 > o4), "");
                static_assert(!(o5 > o5), "");
            }
            {
                using O1 = optional<int>;
                using O2 = optional<long>;
                constexpr O1 o1(42);
                static_assert(o1 > O2(1), "");
                static_assert(!(O2(42) > o1), "");
            }
            {
                using O1 = optional<int>;
                using O2 = optional<const int>;
                constexpr O1 o1(42);
                static_assert(o1 > O2(1), "");
                static_assert(!(O2(42) > o1), "");
            }
        });
    });
}

int
main(int, char**)
{
    kernel_test();
    std::cout << "Pass" << std::endl;
    return 0;
}
