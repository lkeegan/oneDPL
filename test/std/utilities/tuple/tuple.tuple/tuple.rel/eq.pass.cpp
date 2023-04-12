// <tuple>

// template <class... Types> class tuple;

// template<class... TTypes, class... UTypes>
//   bool
//   operator==(const tuple<TTypes...>& t, const tuple<UTypes...>& u);

#include "oneapi_std_test_config.h"
#include "test_macros.h"

#include <iostream>

#ifdef USE_ONEAPI_STD
#    include _ONEAPI_STD_TEST_HEADER(tuple)
namespace s = oneapi_cpp_ns;
#else
#    include <tuple>
namespace s = std;
#endif

#if TEST_DPCPP_BACKEND_PRESENT
constexpr sycl::access::mode sycl_read = sycl::access::mode::read;
constexpr sycl::access::mode sycl_write = sycl::access::mode::write;

class KernelTupleEQTest1;
class KernelTupleEQTest2;

void
kernel_test1(sycl::queue& deviceQueue)
{
    sycl::cl_bool ret = true;
    sycl::range<1> numOfItems{1};
    sycl::buffer<sycl::cl_bool, 1> buffer1(&ret, numOfItems);
    deviceQueue.submit([&](sycl::handler& cgh) {
        auto ret_access = buffer1.get_access<sycl_write>(cgh);
        cgh.single_task<class KernelTupleEQTest1>([=]() {
            {
                typedef s::tuple<> T1;
                typedef s::tuple<> T2;
                const T1 t1;
                const T2 t2;
                ret_access[0] &= (t1 == t2);
                ret_access[0] &= (!(t1 != t2));
            }
            {
                typedef s::tuple<int> T1;
                typedef s::tuple<float> T2;
                const T1 t1(1);
                const T2 t2(1.1f);
                ret_access[0] &= (!(t1 == t2));
                ret_access[0] &= (t1 != t2);
            }
            {
                typedef s::tuple<int> T1;
                typedef s::tuple<float> T2;
                const T1 t1(1);
                const T2 t2(1.f);
                ret_access[0] &= (t1 == t2);
                ret_access[0] &= (!(t1 != t2));
            }
            {
                typedef s::tuple<int, float> T1;
                typedef s::tuple<float, long> T2;
                const T1 t1(1, 2.f);
                const T2 t2(1.f, 2);
                ret_access[0] &= (t1 == t2);
                ret_access[0] &= (!(t1 != t2));
            }
            {
                typedef s::tuple<int, float> T1;
                typedef s::tuple<float, long> T2;
                const T1 t1(1, 2.f);
                const T2 t2(1.f, 3);
                ret_access[0] &= (!(t1 == t2));
                ret_access[0] &= (t1 != t2);
            }
            {
                typedef s::tuple<int, float> T1;
                typedef s::tuple<float, long> T2;
                const T1 t1(1, 2.f);
                const T2 t2(1.1f, 2);
                ret_access[0] &= (!(t1 == t2));
                ret_access[0] &= (t1 != t2);
            }
            {
                typedef s::tuple<int, float> T1;
                typedef s::tuple<float, long> T2;
                const T1 t1(1, 2.f);
                const T2 t2(1.1f, 3);
                ret_access[0] &= (!(t1 == t2));
                ret_access[0] &= (t1 != t2);
            }
            {
                typedef s::tuple<long, int, float> T1;
                typedef s::tuple<float, long, int> T2;
                const T1 t1(1, 2, 3.f);
                const T2 t2(1.f, 2, 3);
                ret_access[0] &= (t1 == t2);
                ret_access[0] &= (!(t1 != t2));
            }
            {
                typedef s::tuple<long, int, float> T1;
                typedef s::tuple<float, long, int> T2;
                const T1 t1(1, 2, 3.f);
                const T2 t2(1.1f, 2, 3);
                ret_access[0] &= (!(t1 == t2));
                ret_access[0] &= (t1 != t2);
            }
            {
                typedef s::tuple<long, int, float> T1;
                typedef s::tuple<float, long, int> T2;
                const T1 t1(1, 2, 3.f);
                const T2 t2(1.f, 3, 3);
                ret_access[0] &= (!(t1 == t2));
                ret_access[0] &= (t1 != t2);
            }
            {
                typedef s::tuple<long, int, float> T1;
                typedef s::tuple<float, long, int> T2;
                const T1 t1(1, 2, 3.f);
                const T2 t2(1.f, 2, 4);
                ret_access[0] &= (!(t1 == t2));
                ret_access[0] &= (t1 != t2);
            }
            {
                typedef s::tuple<long, int, float> T1;
                typedef s::tuple<float, long, int> T2;
                const T1 t1(1, 2, 3.f);
                const T2 t2(1.f, 3, 2);
                ret_access[0] &= (!(t1 == t2));
                ret_access[0] &= (t1 != t2);
            }
            {
                typedef s::tuple<long, int, float> T1;
                typedef s::tuple<float, long, int> T2;
                const T1 t1(1, 2, 3.f);
                const T2 t2(1.1f, 2, 2);
                ret_access[0] &= (!(t1 == t2));
                ret_access[0] &= (t1 != t2);
            }
            {
                typedef s::tuple<long, int, float> T1;
                typedef s::tuple<float, long, int> T2;
                const T1 t1(1, 2, 3.f);
                const T2 t2(1.1f, 3, 3);
                ret_access[0] &= (!(t1 == t2));
                ret_access[0] &= (t1 != t2);
            }
            {
                typedef s::tuple<long, int, float> T1;
                typedef s::tuple<float, long, int> T2;
                const T1 t1(1, 2, 3.f);
                const T2 t2(1.1f, 3, 2);
                ret_access[0] &= (!(t1 == t2));
                ret_access[0] &= (t1 != t2);
            }
        });
    });

    auto ret_access_host = buffer1.get_access<sycl_read>();
    TestUtils::exitOnError(ret_access_host[0]);
}

void
kernel_test2(sycl::queue& deviceQueue)
{
    sycl::cl_bool ret = true;
    sycl::range<1> numOfItems{1};
    sycl::buffer<sycl::cl_bool, 1> buffer1(&ret, numOfItems);
    deviceQueue.submit([&](sycl::handler& cgh) {
        auto ret_access = buffer1.get_access<sycl_write>(cgh);
        cgh.single_task<class KernelTupleEQTest>([=]() {
            {
                typedef s::tuple<> T1;
                typedef s::tuple<> T2;
                const T1 t1;
                const T2 t2;
                ret_access[0] = (t1 == t2);
                ret_access[0] &= (!(t1 != t2));
            }
            {
                typedef s::tuple<int> T1;
                typedef s::tuple<double> T2;
                const T1 t1(1);
                const T2 t2(1.1);
                ret_access[0] &= (!(t1 == t2));
                ret_access[0] &= (t1 != t2);
            }
            {
                typedef s::tuple<int> T1;
                typedef s::tuple<double> T2;
                const T1 t1(1);
                const T2 t2(1);
                ret_access[0] &= (t1 == t2);
                ret_access[0] &= (!(t1 != t2));
            }
            {
                typedef s::tuple<int, double> T1;
                typedef s::tuple<double, long> T2;
                const T1 t1(1, 2);
                const T2 t2(1, 2);
                ret_access[0] &= (t1 == t2);
                ret_access[0] &= (!(t1 != t2));
            }
            {
                typedef s::tuple<int, double> T1;
                typedef s::tuple<double, long> T2;
                const T1 t1(1, 2);
                const T2 t2(1, 3);
                ret_access[0] &= (!(t1 == t2));
                ret_access[0] &= (t1 != t2);
            }
            {
                typedef s::tuple<int, double> T1;
                typedef s::tuple<double, long> T2;
                const T1 t1(1, 2);
                const T2 t2(1.1, 2);
                ret_access[0] &= (!(t1 == t2));
                ret_access[0] &= (t1 != t2);
            }
            {
                typedef s::tuple<int, double> T1;
                typedef s::tuple<double, long> T2;
                const T1 t1(1, 2);
                const T2 t2(1.1, 3);
                ret_access[0] &= (!(t1 == t2));
                ret_access[0] &= (t1 != t2);
            }
            {
                typedef s::tuple<long, int, double> T1;
                typedef s::tuple<double, long, int> T2;
                const T1 t1(1, 2, 3);
                const T2 t2(1, 2, 3);
                ret_access[0] &= (t1 == t2);
                ret_access[0] &= (!(t1 != t2));
            }
            {
                typedef s::tuple<long, int, double> T1;
                typedef s::tuple<double, long, int> T2;
                const T1 t1(1, 2, 3);
                const T2 t2(1.1, 2, 3);
                ret_access[0] &= (!(t1 == t2));
                ret_access[0] &= (t1 != t2);
            }
            {
                typedef s::tuple<long, int, double> T1;
                typedef s::tuple<double, long, int> T2;
                const T1 t1(1, 2, 3);
                const T2 t2(1, 3, 3);
                ret_access[0] &= (!(t1 == t2));
                ret_access[0] &= (t1 != t2);
            }
            {
                typedef s::tuple<long, int, double> T1;
                typedef s::tuple<double, long, int> T2;
                const T1 t1(1, 2, 3);
                const T2 t2(1, 2, 4);
                ret_access[0] &= (!(t1 == t2));
                ret_access[0] &= (t1 != t2);
            }
            {
                typedef s::tuple<long, int, double> T1;
                typedef s::tuple<double, long, int> T2;
                const T1 t1(1, 2, 3);
                const T2 t2(1, 3, 2);
                ret_access[0] &= (!(t1 == t2));
                ret_access[0] &= (t1 != t2);
            }
            {
                typedef s::tuple<long, int, double> T1;
                typedef s::tuple<double, long, int> T2;
                const T1 t1(1, 2, 3);
                const T2 t2(1.1, 2, 2);
                ret_access[0] &= (!(t1 == t2));
                ret_access[0] &= (t1 != t2);
            }
            {
                typedef s::tuple<long, int, double> T1;
                typedef s::tuple<double, long, int> T2;
                const T1 t1(1, 2, 3);
                const T2 t2(1.1, 3, 3);
                ret_access[0] &= (!(t1 == t2));
                ret_access[0] &= (t1 != t2);
            }
            {
                typedef s::tuple<long, int, double> T1;
                typedef s::tuple<double, long, int> T2;
                const T1 t1(1, 2, 3);
                const T2 t2(1.1, 3, 2);
                ret_access[0] &= (!(t1 == t2));
                ret_access[0] &= (t1 != t2);
            }
        });
    });

    auto ret_access_host = buffer1.get_access<sycl_read>();
    TestUtils::exitOnError(ret_access_host[0]);
}
#endif // TEST_DPCPP_BACKEND_PRESENT

int
main()
{
#if TEST_DPCPP_BACKEND_PRESENT
    sycl::queue deviceQueue;
    kernel_test1(deviceQueue);
    if (deviceQueue.get_device().has_extension("cl_khr_fp64"))
    {
        kernel_test2(deviceQueue);
    }
#endif // TEST_DPCPP_BACKEND_PRESENT

    return TestUtils::done(TEST_DPCPP_BACKEND_PRESENT);
}
