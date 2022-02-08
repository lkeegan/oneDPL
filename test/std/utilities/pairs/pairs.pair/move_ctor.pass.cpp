// <utility>

// template <class T1, class T2> struct pair

// pair(pair&&) = default;

#include "oneapi_std_test_config.h"
#include "test_macros.h"
#include <CL/sycl.hpp>
#include <iostream>

#ifdef USE_ONEAPI_STD
#    include _ONEAPI_STD_TEST_HEADER(utility)
#    include _ONEAPI_STD_TEST_HEADER(type_traits)
namespace s = oneapi_cpp_ns;
#else
#    include <utility>
namespace s = std;
#endif

constexpr cl::sycl::access::mode sycl_read = cl::sycl::access::mode::read;
constexpr cl::sycl::access::mode sycl_write = cl::sycl::access::mode::write;

class KernelPairTest;

struct Dummy
{
    Dummy(Dummy const&) = delete;
    Dummy(Dummy&&) = default;
};

void
kernel_test()
{
    cl::sycl::queue deviceQueue;
    cl::sycl::cl_bool ret = false;
    cl::sycl::range<1> numOfItems{1};
    cl::sycl::buffer<cl::sycl::cl_bool, 1> buffer1(&ret, numOfItems);
    deviceQueue.submit([&](cl::sycl::handler& cgh) {
        auto ret_access = buffer1.get_access<sycl_write>(cgh);
        cgh.single_task<class KernelPairTest>([=]() {
            {
                typedef s::pair<int, short> P1;
                static_assert(s::is_move_constructible<P1>::value, "");
                ret_access[0] = s::is_move_constructible<P1>::value;
                P1 p1(3, static_cast<short>(4));
                P1 p2 = s::move(p1);
                ret_access[0] &= (p2.first == 3);
                ret_access[0] &= (p2.second == 4);
            }

            {
                using P = s::pair<Dummy, int>;
                static_assert(!s::is_copy_constructible<P>::value, "");
                static_assert(s::is_move_constructible<P>::value, "");
                ret_access[0] &= !(s::is_copy_constructible<P>::value);
                ret_access[0] &= (s::is_move_constructible<P>::value);
            }
        });
    });

    auto ret_access_host = buffer1.get_access<sycl_read>();
    if (ret_access_host[0])
    {
        std::cout << "Pass" << std::endl;
    }
    else
    {
        std::cout << "Fail" << std::endl;
    }
}

int
main()
{
    kernel_test();
    return 0;
}
