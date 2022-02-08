// <array>

// reference operator[] (size_type)
// const_reference operator[] (size_type); // constexpr in C++14
// reference at (size_type)
// const_reference at (size_type); // constexpr in C++14
// Libc++ marks these as noexcept

#include "oneapi_std_test_config.h"
#include <CL/sycl.hpp>
#include <iostream>
#ifdef USE_ONEAPI_STD
#    include _ONEAPI_STD_TEST_HEADER(array)
#    include _ONEAPI_STD_TEST_HEADER(type_traits)
namespace s = oneapi_cpp_ns;
#else
#    include <array>
#    include <type_traits>
namespace s = std;
#endif

bool
kernel_test()
{
    auto ret = true;
    {
        cl::sycl::queue deviceQueue;
        cl::sycl::buffer<bool, 1> buf1(&ret, cl::sycl::range<1>(1));

        deviceQueue.submit([&](cl::sycl::handler& cgh) {
            auto ret_acc = buf1.get_access<cl::sycl::access::mode::read_write>(cgh);
            cgh.single_task<class KernelIndexTest1>([=]() {
                {
                    typedef float T;
                    typedef s::array<T, 3> C;
                    C c = {1.f, 2.f, 3.5f};
                    ret_acc[0] &= (s::is_same<C::reference, decltype(c[0])>::value == true);
                    C::reference r1 = c[0];
                    ret_acc[0] &= (r1 == 1.f);
                    r1 = 5.5f;
                    ret_acc[0] &= (c.front() == 5.5f);
                    C::reference r2 = c[2];
                    ret_acc[0] &= (r2 == 3.5f);
                    r2 = 7.5f;
                    ret_acc[0] &= (c.back() == 7.5f);
                }
                {
                    typedef float T;
                    typedef s::array<T, 3> C;
                    const C c = {1.f, 2.f, 3.5f};
                    ret_acc[0] &= (s::is_same<C::const_reference, decltype(c[0])>::value == true);
                    C::const_reference r1 = c[0];
                    ret_acc[0] &= (r1 == 1.f);
                    C::const_reference r2 = c[2];
                    ret_acc[0] &= (r2 == 3.5f);
                }
                {
                    typedef float T;
                    typedef s::array<T, 0> C;
                    C c = {};
                    C const& cc = c;
                    (void)noexcept(c[0]);
                    (void)noexcept(cc[0]);
                    ret_acc[0] &= (s::is_same<C::reference, decltype(c[0])>::value == true);
                    ret_acc[0] &= (s::is_same<C::const_reference, decltype(cc[0])>::value == true);
                }
                {
                    typedef float T;
                    typedef s::array<const T, 0> C;
                    C c = {{}};
                    C const& cc = c;
                    (void)noexcept(c[0]);
                    (void)noexcept(cc[0]);
                    ret_acc[0] &= (s::is_same<C::reference, decltype(c[0])>::value == true);
                    ret_acc[0] &= (s::is_same<C::const_reference, decltype(cc[0])>::value == true);
                }
                {
                    typedef float T;
                    typedef s::array<T, 3> C;
                    constexpr C c = {1.f, 2.f, 3.5f};
                    (void)noexcept(c[0]);
                    ret_acc[0] &= (s::is_same<C::const_reference, decltype(c[0])>::value == true);

                    constexpr T t1 = c[0];
                    ret_acc[0] &= (t1 == 1.f);
                    constexpr T t2 = c[2];
                    ret_acc[0] &= (t2 == 3.5f);
                }
            });
        });
    }
    return ret;
}

int
main(int, char**)
{
    auto ret = kernel_test();
    if (ret)
    {
        std::cout << "Pass" << std::endl;
    }
    else
    {
        std::cout << "Fail" << std::endl;
    }
    return 0;
}
