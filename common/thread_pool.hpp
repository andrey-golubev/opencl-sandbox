#pragma once

#include <cstdint>

#include <atomic>
#include <functional>
#include <future>
#include <list>
#include <mutex>
#include <thread>
#include <vector>

#include "require.hpp"

namespace thr {
namespace detail {
struct ThreadPool {
    ThreadPool(std::size_t threads) : m_alive(true) {
        REQUIRE(threads > 0);
        m_threads.reserve(threads);
        for (std::size_t _ = 0; _ < threads; ++_) {
            m_threads.emplace_back([this]() {
                while (m_alive) {
                    std::packaged_task<void()> task;
                    {
                        std::unique_lock<std::mutex> lock(m_queue_mutex);
                        m_queue_cv.wait(lock,
                                        [this]() { return !m_queue.empty() || !m_alive.load(); });
                        if (m_queue.empty()) {
                            continue;
                        }
                        task = std::move(m_queue.front());
                        m_queue.pop_front();
                    }

                    task();
                }
            });
        }
    }

    template<typename Callable, typename... Args>
    using task_return_t = typename std::result_of<Callable(Args...)>::type;

    template<typename Callable, typename... Args>
    std::future<task_return_t<Callable, Args...>> add_task(Callable f, Args&&... args) {

        using return_t = task_return_t<Callable, Args...>;
        auto this_task =
            std::make_unique<std::packaged_task<return_t()>>([=]() { return f(args...); });

        std::future<return_t> future = this_task->get_future();
        {
            std::unique_lock<std::mutex> lock(m_queue_mutex);
            m_queue.emplace_back([task = std::move(this_task)]() { return (*task)(); });
        }
        m_queue_cv.notify_one();
        return future;
    }

    ~ThreadPool() {
        m_alive = false;
        m_queue_cv.notify_all();
        for (auto& t : m_threads) {
            REQUIRE(t.joinable());
            t.join();
        }
    }

private:
    std::vector<std::thread> m_threads;
    std::list<std::packaged_task<void()>> m_queue;
    std::mutex m_queue_mutex;
    std::condition_variable m_queue_cv;
    std::atomic<bool> m_alive;
};
}  // namespace detail
}  // namespace thr
