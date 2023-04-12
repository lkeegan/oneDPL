//===----------------------------------------------------------------------===//
//
// NOTE: nullptr_t emulation cannot handle a reinterpret_cast to an
// integral type
// XFAIL: c++98, c++03
//
// typedef decltype(nullptr) nullptr_t;
//
//===----------------------------------------------------------------------===//

#include "oneapi_std_test_config.h"

#include "test_macros.h"
#include <iostream>

#ifdef USE_ONEAPI_STD
#    include _ONEAPI_STD_TEST_HEADER(cstddef)
namespace s = oneapi_cpp_ns;
#else
#    include <cstddef>
namespace s = std;
#endif

int
main(int, char**)
{
#if TEST_DPCPP_BACKEND_PRESENT
    const s::size_t N = 1;
    bool ret = true;
    {
        sycl::buffer<bool, 1> buf(&ret, sycl::range<1>{N});
        sycl::queue q;
        q.submit([&](sycl::handler& cgh) {
            auto acc = buf.get_access<sycl::access::mode::write>(cgh);
            cgh.single_task<class KernelTest1>([=]() {
                s::ptrdiff_t i = reinterpret_cast<s::ptrdiff_t>(nullptr);
                acc[0] &= (i == 0);
            });
        });
    }

    TestUtils::exitOnError(ret);
#endif // TEST_DPCPP_BACKEND_PRESENT

    return TestUtils::done(TEST_DPCPP_BACKEND_PRESENT);
}
