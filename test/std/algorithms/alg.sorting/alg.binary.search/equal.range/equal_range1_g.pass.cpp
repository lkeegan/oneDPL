#include "oneapi_std_test_config.h"
#include "testsuite_iterators.h"
#include "checkData.h"

#include <iostream>

#ifdef USE_ONEAPI_STD
#    include _ONEAPI_STD_TEST_HEADER(algorithm)
#    include _ONEAPI_STD_TEST_HEADER(utility)
namespace s = oneapi_cpp_ns;
#else
#    include <algorithm>
#    include <utility>
namespace s = std;
#endif

#if TEST_DPCPP_BACKEND_PRESENT
constexpr sycl::access::mode sycl_read = sycl::access::mode::read;
constexpr sycl::access::mode sycl_write = sycl::access::mode::write;

using s::equal_range;

typedef test_container<int, forward_iterator_wrapper> Container;

sycl::cl_bool
kernel_test1()
{
    sycl::queue deviceQueue;
    int array[] = {0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2};
    const int N = sizeof(array) / sizeof(array[0]);
    auto tmp = array;
    sycl::cl_bool ret = false;
    sycl::cl_bool check = false;
    sycl::range<1> item1{1};
    sycl::range<1> itemN{N};
    {
        sycl::buffer<sycl::cl_bool, 1> buffer1(&ret, item1);
        sycl::buffer<sycl::cl_bool, 1> buffer2(&check, item1);
        sycl::buffer<int, 1> buffer3(array, itemN);
        deviceQueue.submit([&](sycl::handler& cgh) {
            auto ret_access = buffer1.get_access<sycl_write>(cgh);
            auto check_access = buffer2.get_access<sycl_write>(cgh);
            auto access = buffer3.get_access<sycl_write>(cgh);
            cgh.single_task<class KernelTest1>([=]() {
                auto ret = true;
                int arr[] = {0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2};
                // check if there is change after data transfer
                check_access[0] = checkData(&access[0], arr, N);
                if (check_access[0])
                {
                    for (int i = 0; i < 6; ++i)
                    {
                        for (int j = 6; j < 12; ++j)
                        {
                            Container con(&access[0] + i, &access[0] + j);
                            ret &= (equal_range(con.begin(), con.end(), 1).first.ptr == &access[0] + std::max(i, 4));
                            ret &= (equal_range(con.begin(), con.end(), 1).second.ptr == &access[0] + std::min(j, 8));
                        }
                    }
                    ret_access[0] = ret;
                }
            });
        });
    }
    // check if there is change after executing kernel function
    check &= checkData(tmp, array, N);
    if (!check)
        return false;
    return ret;
}

sycl::cl_bool
kernel_test2()
{
    sycl::queue deviceQueue;
    int array[] = {0, 0, 2, 2, 2};
    const int N = sizeof(array) / sizeof(array[0]);
    auto tmp = array;
    sycl::cl_bool ret = false;
    sycl::cl_bool check = false;
    sycl::range<1> item1{1};
    sycl::range<1> itemN{N};
    {
        sycl::buffer<sycl::cl_bool, 1> buffer1(&ret, item1);
        sycl::buffer<sycl::cl_bool, 1> buffer2(&check, item1);
        sycl::buffer<int, 1> buffer3(array, itemN);
        deviceQueue.submit([&](sycl::handler& cgh) {
            auto ret_access = buffer1.get_access<sycl_write>(cgh);
            auto check_access = buffer2.get_access<sycl_write>(cgh);
            auto access = buffer3.get_access<sycl_write>(cgh);
            cgh.single_task<class KernelTest2>([=]() {
                int arr[] = {0, 0, 2, 2, 2};
                // check if there is change after data transfer
                check_access[0] = checkData(&access[0], arr, N);
                if (check_access[0])
                {
                    Container con(&access[0], &access[0] + 5);
                    ret_access[0] = (equal_range(con.begin(), con.end(), 1).first.ptr == &access[0] + 2);
                    ret_access[0] &= (equal_range(con.begin(), con.end(), 1).second.ptr == &access[0] + 2);
                }
            });
        });
    }
    // check if there is change after executing kernel function
    check &= checkData(tmp, array, N);
    if (!check)
        return false;
    return ret;
}

#endif // TEST_DPCPP_BACKEND_PRESENT

int
main()
{
#if TEST_DPCPP_BACKEND_PRESENT
    auto ret = kernel_test1();
    ret &= kernel_test2();
    TestUtils::exitOnError(ret);
#endif // TEST_DPCPP_BACKEND_PRESENT

    return TestUtils::done(TEST_DPCPP_BACKEND_PRESENT);
}
