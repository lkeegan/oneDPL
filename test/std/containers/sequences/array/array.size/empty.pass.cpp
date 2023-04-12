// <array>

// class array

// bool empty() const noexcept;

#include "oneapi_std_test_config.h"

#include <iostream>

#ifdef USE_ONEAPI_STD
#    include _ONEAPI_STD_TEST_HEADER(array)
namespace s = oneapi_cpp_ns;
#else
#    include <array>
namespace s = std;
#endif

int
main(int, char**)
{
#if TEST_DPCPP_BACKEND_PRESENT
    {
        auto ret = true;
        {
            sycl::queue deviceQueue;
            sycl::buffer<bool, 1> buf(&ret, sycl::range<1>(1));

            deviceQueue.submit([&](sycl::handler& cgh) {
                auto ret_acc = buf.get_access<sycl::access::mode::read_write>(cgh);

                cgh.single_task<class KernelEmptyTest1>([=]() {
                    {
                        typedef s::array<int, 2> C;
                        C c;
                        ret_acc[0] &= (!c.empty());
                    }
                    {
                        typedef s::array<int, 0> C;
                        C c;
                        ret_acc[0] &= (c.empty());
                    }
                });
            });
        }

        TestUtils::exitOnError(ret);
    }
#endif // TEST_DPCPP_BACKEND_PRESENT

    return TestUtils::done(TEST_DPCPP_BACKEND_PRESENT);
}
