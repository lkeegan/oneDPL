//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// type_traits

// is_nothrow_copy_assignable

#include "oneapi_std_test_config.h"
#include "test_macros.h"

#include <iostream>

#ifdef USE_ONEAPI_STD
#    include _ONEAPI_STD_TEST_HEADER(type_traits)
namespace s = oneapi_cpp_ns;
#else
#    include <type_traits>
namespace s = std;
#endif

#if TEST_DPCPP_BACKEND_PRESENT
constexpr sycl::access::mode sycl_read = sycl::access::mode::read;
constexpr sycl::access::mode sycl_write = sycl::access::mode::write;

template <class T>
void
test_has_nothrow_assign(sycl::queue& deviceQueue)
{
    deviceQueue.submit([&](sycl::handler& cgh) {
        cgh.single_task<T>([=]() {
            static_assert(s::is_nothrow_copy_assignable<T>::value, "");
#if TEST_STD_VER > 14
            static_assert(s::is_nothrow_copy_assignable_v<T>, "");
#endif
        });
    });
}

template <class T>
void
test_has_not_nothrow_assign(sycl::queue& deviceQueue)
{
    deviceQueue.submit([&](sycl::handler& cgh) {
        cgh.single_task<T>([=]() {
            static_assert(!s::is_nothrow_copy_assignable<T>::value, "");
#if TEST_STD_VER > 14
            static_assert(!s::is_nothrow_copy_assignable_v<T>, "");
#endif
        });
    });
}

class Empty
{
};

union Union {
};

struct bit_zero
{
    int : 0;
};

struct A
{
    A&
    operator=(const A&);
};

void
kernel_test()
{
    sycl::queue deviceQueue = TestUtils::get_test_queue();
    test_has_nothrow_assign<int&>(deviceQueue);
    test_has_nothrow_assign<Union>(deviceQueue);
    test_has_nothrow_assign<Empty>(deviceQueue);
    test_has_nothrow_assign<int>(deviceQueue);
    test_has_nothrow_assign<int*>(deviceQueue);
    test_has_nothrow_assign<const int*>(deviceQueue);
    test_has_nothrow_assign<bit_zero>(deviceQueue);
    if (TestUtils::has_type_support<double>(deviceQueue.get_device()))
    {
        test_has_nothrow_assign<double>(deviceQueue);
    }

    test_has_not_nothrow_assign<const int>(deviceQueue);
    test_has_not_nothrow_assign<void>(deviceQueue);
    test_has_not_nothrow_assign<A>(deviceQueue);
}
#endif // TEST_DPCPP_BACKEND_PRESENT

int
main(int, char**)
{
#if TEST_DPCPP_BACKEND_PRESENT
    kernel_test();
#endif // TEST_DPCPP_BACKEND_PRESENT

    return TestUtils::done(TEST_DPCPP_BACKEND_PRESENT);
}
