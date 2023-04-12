//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// type_traits

// is_assignable

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

struct A
{
};

struct B
{
    void operator=(A);
};

template <class KernelTest, class T, class U>
void
test_is_assignable(sycl::queue& deviceQueue)
{
    deviceQueue.submit([&](sycl::handler& cgh) {
        cgh.single_task<KernelTest>([=]() {
            static_assert((s::is_assignable<T, U>::value), "");
#if TEST_STD_VER > 14
            static_assert(s::is_assignable_v<T, U>, "");
#endif
        });
    });
}

template <class KernelTest, class T, class U>
void
test_is_not_assignable(sycl::queue& deviceQueue)
{
    deviceQueue.submit([&](sycl::handler& cgh) {
        cgh.single_task<KernelTest>([=]() {
            static_assert((!s::is_assignable<T, U>::value), "");
#if TEST_STD_VER > 14
            static_assert(!s::is_assignable_v<T, U>, "");
#endif
        });
    });
}

struct D;

struct C
{
    template <class U>
    D operator,(U&&);
};

struct E
{
    C
    operator=(int);
};

template <typename T>
struct X
{
    T t;
};

class KernelTest1;
class KernelTest2;
class KernelTest3;
class KernelTest4;
class KernelTest5;
class KernelTest6;
class KernelTest7;
class KernelTest8;
class KernelTest9;
class KernelTest10;
class KernelTest11;
class KernelTest12;
class KernelTest13;

void
kernel_test()
{
    sycl::queue deviceQueue;
    test_is_assignable<KernelTest1, int&, int&>(deviceQueue);
    test_is_assignable<KernelTest2, int&, int>(deviceQueue);
    test_is_assignable<KernelTest3, B, A>(deviceQueue);
    test_is_assignable<KernelTest4, void*&, void*>(deviceQueue);

    test_is_assignable<KernelTest5, E, int>(deviceQueue);
    if (TestUtils::has_type_support<double>(deviceQueue.get_device()))
    {
        test_is_assignable<KernelTest6, int&, double>(deviceQueue);
    }

    test_is_not_assignable<KernelTest7, int, int&>(deviceQueue);
    test_is_not_assignable<KernelTest8, int, int>(deviceQueue);
    test_is_not_assignable<KernelTest9, A, B>(deviceQueue);
    test_is_not_assignable<KernelTest10, void, const void>(deviceQueue);
    test_is_not_assignable<KernelTest11, const void, const void>(deviceQueue);
    test_is_not_assignable<KernelTest12, int(), int>(deviceQueue);

    //  pointer to incomplete template type
    test_is_assignable<KernelTest13, X<D>*&, X<D>*>(deviceQueue);
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
