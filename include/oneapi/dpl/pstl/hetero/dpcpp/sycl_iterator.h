// -*- C++ -*-
//===-- sycl_iterator.h ---------------------------------------------------===//
//
// Copyright (C) Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// This file incorporates work covered by the following copyright and permission
// notice:
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
//
//===----------------------------------------------------------------------===//

#ifndef _ONEDPL_SYCL_ITERATOR_H
#define _ONEDPL_SYCL_ITERATOR_H

#include <iterator>
#include "../../onedpl_config.h"
#include "sycl_defs.h"

namespace oneapi
{
namespace dpl
{

using access_mode = sycl::access::mode;

namespace __internal
{
template <typename T, typename Allocator = __dpl_sycl::__buffer_allocator<T>>
class sycl_const_iterator
{
  public:

    using Size = ::std::size_t;

    using iterator_category = ::std::random_access_iterator_tag;
    using is_hetero = ::std::true_type;
    using is_hetero_const = ::std::true_type;
    using difference_type = ::std::make_signed<Size>::type;

    using value_type = const T;
    using pointer = value_type*;
    using reference = value_type&;

    using Buffer = sycl::buffer<value_type, 1, Allocator>;
    using NonConstBuffer = sycl::buffer<T, 1, Allocator>;

  private:

    Buffer buffer;
    Size idx = 0;

    static Buffer get_buf_from_non_const(NonConstBuffer srcBuf)
    {
        return *reinterpret_cast<Buffer*>(&srcBuf);
    }

  public:

    // required for make_sycl_iterator
    //TODO: sycl::buffer doesn't have a default constructor (SYCL API issue), so we have to create a trivial size buffer
    sycl_const_iterator(Buffer vec = Buffer(0), Size index = 0)
        : buffer(vec), idx(index)
    {
    }
    sycl_const_iterator(NonConstBuffer vec, Size index = 0)
        : buffer(get_buf_from_non_const(vec)), idx(index)
    {
    }
    // required for iter_mode
    sycl_const_iterator(const sycl_const_iterator& in) : buffer(in.get_buffer())
    {
        auto old_iter = sycl_const_iterator{in.get_buffer(), 0};
        idx = in - old_iter;
    }

    sycl_const_iterator& operator=(const sycl_const_iterator& in) = default;

    sycl_const_iterator
    operator+(difference_type forward) const
    {
        return {buffer, idx + forward};
    }
    sycl_const_iterator
    operator-(difference_type backward) const
    {
        return {buffer, idx - backward};
    }
    friend sycl_const_iterator
    operator+(difference_type forward, const sycl_const_iterator& it)
    {
        return it + forward;
    }
    friend sycl_const_iterator
    operator-(difference_type forward, const sycl_const_iterator& it)
    {
        return it - forward;
    }
    difference_type
    operator-(const sycl_const_iterator& it) const
    {
        return idx - it.idx;
    }
    bool
    operator==(const sycl_const_iterator& it) const
    {
        return idx == it.get_index();
    }
    bool
    operator!=(const sycl_const_iterator& it) const
    {
        return !(*this == it);
    }
    bool
    operator<(const sycl_const_iterator& it) const
    {
        return *this - it < 0;
    }

    Buffer
    get_buffer() const
    {
        return buffer;
    }

    Size
    get_index() const
    {
        return idx;
    }
};

template <access_mode Mode, typename T, typename Allocator = __dpl_sycl::__buffer_allocator<T>>
class sycl_iterator : public sycl_const_iterator<T, Allocator>
{
  public:

    using Base = sycl_const_iterator<T, Allocator>;

    using Size = typename Base::Size;

    using iterator_category = typename Base::iterator_category;
    using is_hetero = typename Base::is_hetero;
    using is_hetero_const = ::std::false_type;
    using difference_type = typename Base::difference_type;

    using value_type = T;
    using pointer = value_type*;
    using reference = value_type&;

    using ConstBuffer = typename Base::Buffer;
    using Buffer = sycl::buffer<value_type, 1, Allocator>;

  private:

    Buffer buffer;
    Size idx = 0;

  public:

    // required for make_sycl_iterator
    //TODO: sycl::buffer doesn't have a default constructor (SYCL API issue), so we have to create a trivial size buffer
    sycl_iterator(Buffer vec = Buffer(0), Size index = 0)
        : buffer(vec), idx(index)
    {
    }
    // required for iter_mode
    sycl_iterator(const sycl_iterator& in) : buffer(in.get_buffer())
    {
        auto old_iter = sycl_iterator{in.get_buffer(), 0};
        idx = in - old_iter;
    }

    sycl_iterator&
    operator=(const sycl_iterator& in) = default;

    sycl_iterator
    operator+(difference_type forward) const
    {
        return {buffer, idx + forward};
    }
    sycl_iterator
    operator-(difference_type backward) const
    {
        return {buffer, idx - backward};
    }
    friend sycl_iterator
    operator+(difference_type forward, const sycl_iterator& it)
    {
        return it + forward;
    }
    friend sycl_iterator
    operator-(difference_type forward, const sycl_iterator& it)
    {
        return it - forward;
    }
    difference_type
    operator-(const sycl_iterator& it) const
    {
        return idx - it.idx;
    }
    bool
    operator==(const sycl_iterator& it) const
    {
        return idx == it.get_index();
    }
    bool
    operator==(const sycl_const_iterator<T, Allocator>& it) const
    {
        return idx == it.get_index();
    }
    bool
    operator!=(const sycl_iterator& it) const
    {
        return !(*this == it);
    }
    bool
    operator!=(const sycl_const_iterator<T, Allocator>& it) const
    {
        return !(*this == it);
    }
    bool
    operator<(const sycl_iterator& it) const
    {
        return *this - it < 0;
    }

    Buffer
    get_buffer() const
    {
        return buffer;
    }

    Size
    get_index() const
    {
        return idx;
    }
};

// mode converter when property::noinit present
template <access_mode __mode>
struct _ModeConverter
{
    static constexpr access_mode __value = __mode;
};

template <>
struct _ModeConverter<access_mode::read_write>
{
    static constexpr access_mode __value = access_mode::discard_read_write;
};

template <>
struct _ModeConverter<access_mode::write>
{
    static constexpr access_mode __value = access_mode::discard_write;
};

} // namespace __internal

template <typename T, typename Allocator>
__internal::sycl_iterator<access_mode::read_write, T, Allocator> begin(sycl::buffer<T, /*dim=*/1, Allocator> buf)
{
    return __internal::sycl_iterator<access_mode::read_write, T, Allocator>{buf, 0};
}

template <typename T, typename Allocator>
__internal::sycl_iterator<access_mode::read_write, T, Allocator> end(sycl::buffer<T, /*dim=*/1, Allocator> buf)
{
    return __internal::sycl_iterator<access_mode::read_write, T, Allocator>{buf, __dpl_sycl::__get_buffer_size(buf)};
}

// begin
template <typename T, typename Allocator, access_mode Mode>
__internal::sycl_iterator<Mode, T, Allocator> begin(sycl::buffer<T, /*dim=*/1, Allocator> buf, sycl::mode_tag_t<Mode>)
{
    return __internal::sycl_iterator<Mode, T, Allocator>{buf, 0};
}

template <typename T, typename Allocator, access_mode Mode>
__internal::sycl_iterator<__internal::_ModeConverter<Mode>::__value, T, Allocator>
    begin(sycl::buffer<T, /*dim=*/1, Allocator> buf, sycl::mode_tag_t<Mode>, __dpl_sycl::__no_init)
{
    return __internal::sycl_iterator<__internal::_ModeConverter<Mode>::__value, T, Allocator>{buf, 0};
}

template <typename T, typename Allocator>
__internal::sycl_iterator<access_mode::discard_read_write, T, Allocator>
    begin(sycl::buffer<T, /*dim=*/1, Allocator> buf, __dpl_sycl::__no_init)
{
    return __internal::sycl_iterator<access_mode::discard_read_write, T, Allocator>{buf, 0};
}

// end
template <typename T, typename Allocator, access_mode Mode>
__internal::sycl_iterator<Mode, T, Allocator> end(sycl::buffer<T, /*dim=*/1, Allocator> buf, sycl::mode_tag_t<Mode>)
{
    return __internal::sycl_iterator<Mode, T, Allocator>{buf, __dpl_sycl::__get_buffer_size(buf)};
}

template <typename T, typename Allocator, access_mode Mode>
__internal::sycl_iterator<__internal::_ModeConverter<Mode>::__value, T, Allocator>
    end(sycl::buffer<T, /*dim=*/1, Allocator> buf, sycl::mode_tag_t<Mode>, __dpl_sycl::__no_init)
{
    return __internal::sycl_iterator<__internal::_ModeConverter<Mode>::__value, T, Allocator>{
        buf, __dpl_sycl::__get_buffer_size(buf)};
}

template <typename T, typename Allocator>
__internal::sycl_iterator<access_mode::discard_read_write, T, Allocator> end(sycl::buffer<T, /*dim=*/1, Allocator> buf,
                                                                             __dpl_sycl::__no_init)
{
    return __internal::sycl_iterator<access_mode::discard_read_write, T, Allocator>{buf,
                                                                                    __dpl_sycl::__get_buffer_size(buf)};
}

// cbegin
template <typename T, typename Allocator>
__internal::sycl_const_iterator<T, Allocator>
cbegin(sycl::buffer<T, /*dim=*/1, Allocator> buf)
{
    return __internal::sycl_const_iterator<T, Allocator>{buf, 0};
}

// cend
template <typename T, typename Allocator>
__internal::sycl_const_iterator<T, Allocator>
cend(sycl::buffer<T, /*dim=*/1, Allocator> buf)
{
    return __internal::sycl_const_iterator<T, Allocator>{buf, __dpl_sycl::__get_buffer_size(buf)};
}

} // namespace dpl
} // namespace oneapi

#endif // _ONEDPL_SYCL_ITERATOR_H
