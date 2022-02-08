#include "oneapi_std_test_config.h"
#include <CL/sycl.hpp>
#include <iostream>

#ifdef USE_ONEAPI_STD
#    include _ONEAPI_STD_TEST_HEADER(functional)
#    include _ONEAPI_STD_TEST_HEADER(type_traits)
namespace s = oneapi_cpp_ns;
#else
#    include <functional>
#    include <type_traits>
namespace s = std;
#endif

constexpr cl::sycl::access::mode sycl_read = cl::sycl::access::mode::read;
constexpr cl::sycl::access::mode sycl_write = cl::sycl::access::mode::write;

class KernelBitXorTest;

void
kernel_test()
{
    cl::sycl::queue deviceQueue;
    cl::sycl::cl_bool ret = false;
    cl::sycl::range<1> numOfItems{1};
    cl::sycl::buffer<cl::sycl::cl_bool, 1> buffer1(&ret, numOfItems);
    deviceQueue.submit([&](cl::sycl::handler& cgh) {
        auto ret_access = buffer1.get_access<sycl_write>(cgh);
        cgh.single_task<class KernelBitXorTest>([=]() {
            {
                typedef s::bit_xor<int> F;
                const F f = F();
                static_assert((s::is_same<int, F::first_argument_type>::value), "");
                static_assert((s::is_same<int, F::second_argument_type>::value), "");
                static_assert((s::is_same<int, F::result_type>::value), "");
                ret_access[0] = (f(0xEA95, 0xEA95) == 0);
                ret_access[0] &= (f(0xEA95, 0x58D3) == 0xB246);
                ret_access[0] &= (f(0x58D3, 0xEA95) == 0xB246);
                ret_access[0] &= (f(0x58D3, 0) == 0x58D3);
                ret_access[0] &= (f(0xFFFF, 0x58D3) == 0xA72C);
            }

            {
                typedef s::bit_xor<long> F2;
                const F2 f = F2();
                ret_access[0] &= (f(0xEA95L, 0xEA95) == 0);
                ret_access[0] &= (f(0xEA95, 0xEA95L) == 0);

                ret_access[0] &= (f(0xEA95L, 0x58D3) == 0xB246);
                ret_access[0] &= (f(0xEA95, 0x58D3L) == 0xB246);

                ret_access[0] &= (f(0x58D3L, 0xEA95) == 0xB246);
                ret_access[0] &= (f(0x58D3, 0xEA95L) == 0xB246);

                ret_access[0] &= (f(0x58D3L, 0) == 0x58D3);
                ret_access[0] &= (f(0x58D3, 0L) == 0x58D3);

                ret_access[0] &= (f(0xFFFF, 0x58D3) == 0xA72C);
                ret_access[0] &= (f(0xFFFFL, 0x58D3) == 0xA72C);
                ret_access[0] &= (f(0xFFFF, 0x58D3L) == 0xA72C);
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
