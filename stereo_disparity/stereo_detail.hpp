#pragma once

#include "common/utils.hpp"

#include <opencv2/core.hpp>

#include <vector>

namespace detail {
struct Border {
    int row_border = 0;
    int col_border = 0;
};
bool operator==(Border a, Border b) {
    return a.row_border == b.row_border && a.col_border == b.col_border;
}

template<typename Signature> struct Kernel;
template<typename R, typename... Args> struct Kernel<R(Args...)> {
    using Invokable = R (*)(Args...);

    Kernel(Invokable f, std::vector<Border>&& borders) : m_opaque_function(f), m_borders(borders) {}
    Kernel(const Kernel&) = default;
    Kernel(Kernel&&) = default;
    Kernel& operator=(const Kernel&) = default;
    Kernel& operator=(Kernel&&) = default;
    ~Kernel() = default;

    R operator()(Args... args) { return m_opaque_function(args...); }
    Invokable& get() { return m_opaque_function; }

    Border border(int i) const { return m_borders.at(i); }

private:
    Invokable m_opaque_function;         // opaque function pointer
    std::vector<Border> m_borders = {};  // border sizes for each input buffer for opaque function
};

struct OpaqueKernel {
    template<typename Type>
    OpaqueKernel(Type&& var) : m_holder(std::make_unique<Holder<Type>>(std::forward<Type>(var))) {}

    // TODO: fake copy semantics through move
    OpaqueKernel(const OpaqueKernel& other) : m_holder(other.m_holder->clone()) {}
    OpaqueKernel(OpaqueKernel&& other) : m_holder(std::move(other.m_holder)) {}

    struct Base {
        using Ptr = std::unique_ptr<Base>;
        virtual Border border(int i) const = 0;
        virtual Ptr clone() const = 0;
        virtual ~Base() = default;
    };

    template<typename Type> struct Holder : Base {
        Type m_var;
        Holder(Type var) : m_var(var) {}
        Border border(int i) const override { return m_var.border(i); }
        Base::Ptr clone() const override { return std::make_unique<Holder<Type>>(*this); }
    };

    Border border(int i) const { return m_holder->border(i); }

private:
    typename Base::Ptr m_holder;
};

struct DataView {
    DataView(const cv::Mat& data, Border border) : m_border(border), m_data(data) {}
    const uchar* line(int index) const {
        const int real_index = m_curr_index + index + m_border.row_border;
        REQUIRE(real_index >= 0 && real_index < m_data.rows);
        return m_data.data + real_index * m_data.cols + m_border.col_border;
    }
    uchar* line(int index) {
        const int real_index = m_curr_index + index + m_border.row_border;
        REQUIRE(real_index >= 0 && real_index < m_data.rows);
        return m_data.data + real_index * m_data.cols + m_border.col_border;
    }
    void adjust(int index) { m_curr_index = index; }

private:
    Border m_border = {};
    int m_curr_index = 0;
    cv::Mat m_data;
};
}  // namespace detail
