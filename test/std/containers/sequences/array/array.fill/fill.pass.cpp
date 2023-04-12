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
    bool ret = true;

    {
        sycl::buffer<bool, 1> buf(&ret, sycl::range<1>{1});
        sycl::queue q;
        q.submit([&](sycl::handler& cgh) {
            auto ret_acc = buf.get_access<sycl::access::mode::write>(cgh);
            cgh.single_task<class KernelTest1>([=]() {
                {
                    typedef int T;
                    typedef s::array<T, 3> C;
                    C c = {1, 2, 35};
                    c.fill(55);
                    ret_acc[0] &= (c.size() == 3);
                    ret_acc[0] &= (c[0] == 55);
                    ret_acc[0] &= (c[1] == 55);
                    ret_acc[0] &= (c[2] == 55);
                }
            });
        });
    }

    TestUtils::exitOnError(ret);
#endif // TEST_DPCPP_BACKEND_PRESENT

    return TestUtils::done(TEST_DPCPP_BACKEND_PRESENT);
}
