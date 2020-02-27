#pragma once

#include "common/utils.hpp"

#include <opencv2/core.hpp>

#include <vector>

namespace detail {
struct Border {
    int row_border = 0;
    int col_border = 0;
};
// usage model: [y, y + height)
struct HorSlice {
    int y = 0;
    int height = 0;
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
    const std::vector<Border>& borders() const { return m_borders; }

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
        virtual const std::vector<Border>& borders() const = 0;
        virtual Ptr clone() const = 0;
        virtual ~Base() = default;
    };

    template<typename Type> struct Holder : Base {
        Type m_var;
        Holder(Type var) : m_var(var) {}
        Border border(int i) const override { return m_var.border(i); }
        const std::vector<Border>& borders() const override { return m_var.borders(); }
        Base::Ptr clone() const override { return std::make_unique<Holder<Type>>(*this); }
    };

    Border border(int i) const { return m_holder->border(i); }
    const std::vector<Border>& borders() const { return m_holder->borders(); }

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
    Border border() const { return m_border; }

    cv::Mat& data() { return m_data; }

private:
    Border m_border = {};
    int m_curr_index = 0;
    cv::Mat m_data;
};

// TODO: this class is getting very gross with many different ownership models - better design?
struct KernelData {
    // create kernel data with empty inputs (real inputs are not owned by kernel data)
    KernelData(const std::vector<Border>& borders, const std::vector<cv::Size>& out_sizes) {
        const auto size = borders.size();

        // allocate inputs
        m_inputs.reserve(size);
        for (size_t i = 0; i < size; ++i) {
            m_inputs.emplace_back(cv::Mat(), borders[i]);
        }

        // allocate outputs
        const auto out_size = out_sizes.size();
        m_outputs.reserve(out_size);
        for (size_t i = 0; i < out_size; ++i) {
            m_outputs.emplace_back(cv::Mat::zeros(out_sizes[i], CV_8UC1));
        }
    }

    // create default kernel data with ownership over inputs and outputs
    KernelData(const std::vector<cv::Size>& in_sizes, const std::vector<Border>& borders,
               const std::vector<cv::Size>& out_sizes)
        : KernelData(borders, out_sizes) {
        const auto size = in_sizes.size();
        REQUIRE(size == borders.size());

        // create real inputs on top of pre-allocated views
        for (size_t i = 0; i < size; ++i) {
            m_inputs[i].data() = cv::Mat::zeros(in_sizes[i], CV_8UC1);
        }
    }

    // create kernel data with no ownership over outputs
    KernelData(const std::vector<cv::Size>& in_sizes, const std::vector<Border>& borders,
               const std::vector<cv::Mat>& outs)
        : KernelData(in_sizes, borders, std::vector<cv::Size>{}) {
        m_outputs = outs;
    }

    // create kernel data with no ownership over inputs and outputs (mere "view" over kernel data)
    KernelData(const std::vector<Border>& borders, const std::vector<cv::Mat>& outs)
        : KernelData(borders, std::vector<cv::Size>{}) {
        m_outputs = outs;
    }

    DataView& in_view(int index) { return m_inputs[index]; }
    uchar* out_data(int index) { return m_outputs[index].data; }
    cv::Mat& out_mat(int index) { return m_outputs[index]; }

    void adjust(int index) {
        // adjust all views by the same index
        for (auto& view : m_inputs) {
            view.adjust(index);
        }
    }

    template<typename Updater> void update_src(int index, Updater f) {
        f(m_inputs[index].data(), m_inputs[index].border());
    }
    template<int Index, typename Updater> void update_src(Updater f) { update_src(Index, f); }
    template<int Index, int... Indices, typename Updater, typename... Updaters>
    void update_src(Updater f, Updaters... fs) {
        update_src<Index>(f);
        update_src<Indices...>(fs...);
    }

    template<typename Updater> void update_view(int index, Updater f) { f(m_inputs[index]); }
    template<int Index, typename Updater> void update_view(Updater f) { update_view(Index, f); }
    template<int Index, int... Indices, typename Updater, typename... Updaters>
    void update_view(Updater f, Updaters... fs) {
        update_view<Index>(f);
        update_view<Indices...>(fs...);
    }

private:
    std::vector<DataView> m_inputs = {};  // views for inputs
    std::vector<cv::Mat> m_outputs = {};
};

template<typename Kernel> std::vector<cv::Size> in_sizes(cv::Size input_size, const Kernel& k) {
    OpaqueKernel opaque(k);
    std::vector<cv::Size> sizes;
    for (const auto& border : opaque.borders()) {
        sizes.emplace_back(input_size.width + border.col_border * 2,
                           input_size.height + border.row_border * 2);
    }
    return sizes;
}
}  // namespace detail
