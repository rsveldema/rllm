#pragma once

#include <array>
#include <atomic>
#include <optional>

namespace fastfork
{
    // Dmitry Vyukov's MPMC bounded queue (lock-free).
    // N must be a power of two.
    // try_push returns false when full  (t is left unchanged).
    // try_pop  returns false when empty.
    template<typename T, int N>
    class CircularQueue
    {
        static_assert((N & (N - 1)) == 0, "N must be a power of two");
        static constexpr unsigned MASK = static_cast<unsigned>(N) - 1u;

        struct Slot
        {
            std::atomic<unsigned> sequence{0};
            T item{};
        };

        std::array<Slot, N>               m_slots{};
        alignas(64) std::atomic<unsigned> m_enqueue_pos{0};
        alignas(64) std::atomic<unsigned> m_dequeue_pos{0};

    public:
        CircularQueue() noexcept
        {
            for (unsigned i = 0; i < static_cast<unsigned>(N); ++i)
                m_slots[i].sequence.store(i, std::memory_order_relaxed);
        }

        // Returns false if full; t is NOT moved from in that case.
        bool try_push(T&& t)
        {
            unsigned pos = m_enqueue_pos.load(std::memory_order_relaxed);
            for (;;)
            {
                Slot& slot = m_slots[pos & MASK];
                const unsigned seq  = slot.sequence.load(std::memory_order_acquire);
                const auto     diff = static_cast<int>(seq - pos);
                if (diff == 0)
                {
                    if (m_enqueue_pos.compare_exchange_weak(
                            pos, pos + 1, std::memory_order_relaxed))
                        break;
                    // pos refreshed by CAS on failure
                }
                else if (diff < 0)
                    return false; // full
                else
                    pos = m_enqueue_pos.load(std::memory_order_relaxed);
            }
            m_slots[pos & MASK].item = std::move(t);
            m_slots[pos & MASK].sequence.store(pos + 1, std::memory_order_release);
            return true;
        }

        // Returns false if empty.
        bool try_pop(T& t)
        {
            unsigned pos = m_dequeue_pos.load(std::memory_order_relaxed);
            for (;;)
            {
                Slot& slot = m_slots[pos & MASK];
                const unsigned seq  = slot.sequence.load(std::memory_order_acquire);
                const auto     diff = static_cast<int>(seq - (pos + 1));
                if (diff == 0)
                {
                    if (m_dequeue_pos.compare_exchange_weak(
                            pos, pos + 1, std::memory_order_relaxed))
                        break;
                }
                else if (diff < 0)
                    return false; // empty
                else
                    pos = m_dequeue_pos.load(std::memory_order_relaxed);
            }
            t = std::move(m_slots[pos & MASK].item);
            m_slots[pos & MASK].sequence.store(
                pos + static_cast<unsigned>(N), std::memory_order_release);
            return true;
        }

        // Approximate occupancy (racy; used only as a batch-steal hint).
        int approx_size() const noexcept
        {
            const unsigned enq = m_enqueue_pos.load(std::memory_order_relaxed);
            const unsigned deq = m_dequeue_pos.load(std::memory_order_relaxed);
            return static_cast<int>(enq - deq);
        }
    };

} // namespace fastfork


namespace fastfork
{
    // Chase-Lev work-stealing deque (fixed-size, single-producer / multi-consumer).
    //
    // Interface
    //   push_bottom(T&&) → bool   owner-only LIFO push; false if full (t unchanged)
    //   pop_bottom(T&)   → bool   owner-only LIFO pop;  false if empty
    //   steal_top(T&)    → bool   any-thread FIFO steal; false if empty / lost CAS
    //
    // Correctness: the (t == b) "last element" race between pop_bottom and
    // steal_top is resolved by a seq_cst CAS on m_top; the loser returns false
    // without touching the item.  Items live in std::optional<T> so that moves
    // happen only after ownership has been established.
    template<typename T, unsigned N>
    class alignas(64) WSDeque
    {
        static_assert((N & (N - 1)) == 0, "N must be a power of two");
        static constexpr int64_t MASK = static_cast<int64_t>(N) - 1;

        struct Slot { std::optional<T> val; };

        Slot                             m_slots[N];
        alignas(64) std::atomic<int64_t> m_bottom{0};
        alignas(64) std::atomic<int64_t> m_top{0};

    public:
        WSDeque() = default;
        WSDeque(const WSDeque&)            = delete;
        WSDeque& operator=(const WSDeque&) = delete;

        // Owner only — push to bottom (LIFO end).  Returns false if full; t unchanged.
        bool push_bottom(T&& t) noexcept(std::is_nothrow_move_constructible_v<T>)
        {
            const int64_t b  = m_bottom.load(std::memory_order_relaxed);
            const int64_t tp = m_top.load(std::memory_order_acquire);
            if (b - tp >= static_cast<int64_t>(N)) return false;
            m_slots[b & MASK].val = std::move(t);
            std::atomic_thread_fence(std::memory_order_release);
            m_bottom.store(b + 1, std::memory_order_relaxed);
            return true;
        }

        // Owner only — pop from bottom (LIFO).  Returns false if empty.
        bool pop_bottom(T& item) noexcept(std::is_nothrow_move_assignable_v<T>)
        {
            const int64_t b = m_bottom.load(std::memory_order_relaxed) - 1;
            m_bottom.store(b, std::memory_order_relaxed);
            std::atomic_thread_fence(std::memory_order_seq_cst);
            int64_t t = m_top.load(std::memory_order_relaxed);

            if (t > b)                          // deque is empty
            {
                m_bottom.store(b + 1, std::memory_order_relaxed);
                return false;
            }
            if (t == b)                         // last element — may race with steal_top
            {
                m_bottom.store(b + 1, std::memory_order_relaxed);
                if (!m_top.compare_exchange_strong(t, t + 1,
                        std::memory_order_seq_cst, std::memory_order_relaxed))
                    return false;               // stealer won
            }
            item = std::move(*m_slots[b & MASK].val);
            m_slots[b & MASK].val.reset();
            return true;
        }

        // Any thread — steal from top (FIFO).  Returns false if empty or CAS lost.
        bool steal_top(T& item) noexcept(std::is_nothrow_move_assignable_v<T>)
        {
            int64_t t = m_top.load(std::memory_order_acquire);
            std::atomic_thread_fence(std::memory_order_seq_cst);
            const int64_t b = m_bottom.load(std::memory_order_acquire);
            if (t >= b) return false;
            if (!m_top.compare_exchange_strong(t, t + 1,
                    std::memory_order_seq_cst, std::memory_order_relaxed))
                return false;
            item = std::move(*m_slots[t & MASK].val);
            m_slots[t & MASK].val.reset();
            return true;
        }

        // Approximate occupancy — racy, used only as a batch-steal hint.
        int64_t approx_size() const noexcept
        {
            const int64_t b = m_bottom.load(std::memory_order_relaxed);
            const int64_t t = m_top.load(std::memory_order_relaxed);
            return std::max(int64_t{0}, b - t);
        }
    };

} // namespace fastfork
