// Tuple

#include "oneapi_std_test_config.h"
#include "test_macros.h"

#include <iostream>

#ifdef USE_ONEAPI_STD
#    include _ONEAPI_STD_TEST_HEADER(tuple)
#    include _ONEAPI_STD_TEST_HEADER(utility)
namespace s = oneapi_cpp_ns;
#else
#    include <tuple>
#    include <utility>
namespace s = std;
#endif

#if TEST_DPCPP_BACKEND_PRESENT
constexpr sycl::access::mode sycl_read = sycl::access::mode::read;
constexpr sycl::access::mode sycl_write = sycl::access::mode::write;

template <typename X>
bool
test(const X& x)
{
    return (x == x && !(x != x) && x <= x && !(x < x));
}

sycl::cl_bool
kernel_test()
{
    sycl::queue deviceQueue = TestUtils::get_test_queue();
    sycl::cl_bool ret = true;
    sycl::range<1> numOfItem{1};

    {
        sycl::buffer<sycl::cl_bool, 1> buffer1(&ret, numOfItem);
        deviceQueue.submit([&](sycl::handler& cgh) {
            auto ret_acc = buffer1.get_access<sycl_write>(cgh);
            cgh.single_task<class KernelTest>([=]() {
                int i = 0;
                int j = 0;
                int k = 2;
                s::tuple<int, int, int> a(0, 0, 0);
                s::tuple<int, int, int> b(0, 0, 1);
                s::tuple<int&, int&, int&> c(i, j, k);
                s::tuple<const int&, const int&, const int&> d(c);
                test(a);
                test(b);
                test(c);
                test(d);
                ret_acc[0] &= (!(a > a) && !(b > b));
                ret_acc[0] &= (a >= a && b >= b);
                ret_acc[0] &= (a < b && !(b < a) && a <= b && !(b <= a));
                ret_acc[0] &= (b > a && !(a > b) && b >= a && !(a >= b));
            });
        });
    }
    return ret;
}
#endif // TEST_DPCPP_BACKEND_PRESENT

int
main()
{
#if TEST_DPCPP_BACKEND_PRESENT
    auto ret = kernel_test();
    TestUtils::exitOnError(ret);
#endif // TEST_DPCPP_BACKEND_PRESENT

    return TestUtils::done(TEST_DPCPP_BACKEND_PRESENT);
}
