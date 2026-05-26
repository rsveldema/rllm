#include <fastfork/fastfork.hpp>
#include <fastfork/circular_queue.hpp>

#include <hwloc.h>

#include <algorithm>
#include <atomic>
#include <latch>
#include <optional>
#include <thread>
#include <unordered_set>
#include <vector>

namespace fastfork
{
    namespace
    {
        static int                      s_num_threads = 0;
        static std::vector<std::thread> s_workers;
        static std::atomic<bool>        s_stop{false};
        static std::atomic<uint64_t>    s_work_epoch{0}; // bumped + notify_one per enqueue
        static thread_local int         tl_thread_id   = 0;
        static thread_local int         tl_fork_cursor = 0; // per-core, no atomic needed

        // Per-worker stats; cache-line padded to avoid false sharing.
        struct alignas(64) Stats
        {
            uint64_t tasks_executed_own{0};
            uint64_t tasks_stolen{0};
            uint64_t idle_polls{0};
            uint64_t tasks_enqueued{0};
        };
        static std::vector<Stats> s_stats;

        hwloc_topology_t s_topology = nullptr;

        static constexpr int QUEUE_CAPACITY = 4096 * 128;

        // ── TaskQueue tree ─────────────────────────────────────────────────
        // One node per hwloc object.  Only leaf nodes (thread_id ≥ 0) carry
        // a queue.  steal_order is pre-computed at init time: own queue first,
        // then sibling-group leaves ordered by increasing topology distance.
        struct TaskQueueNode
        {
            hwloc_obj_type_t             type;
            int                          thread_id = -1;   // ≥ 0 at leaves
            TaskQueueNode*               parent    = nullptr;
            std::vector<TaskQueueNode*>  children;          // non-owning
            WSDeque<task_t, QUEUE_CAPACITY>       queue;     // leaves only
            std::vector<TaskQueueNode*>  steal_order;       // leaves only
            hwloc_cpuset_t               cpuset    = nullptr; // leaves only; bind target
            ~TaskQueueNode() { if (cpuset) hwloc_bitmap_free(cpuset); }
        };

        // s_nodes owns every TaskQueueNode; s_leaves[i] is thread i's leaf.
        static std::vector<std::unique_ptr<TaskQueueNode>> s_nodes;
        static std::vector<TaskQueueNode*>                 s_leaves;

        TaskQueueNode* alloc_node(hwloc_obj_type_t type, TaskQueueNode* parent)
        {
            auto uptr = std::make_unique<TaskQueueNode>();
            auto* node = uptr.get();
            node->type   = type;
            node->parent = parent;
            s_nodes.push_back(std::move(uptr));
            return node;
        }

        // Recurse the hwloc tree; stop at PU (hardware-thread) level and
        // assign sequential thread IDs.  Prunes once s_num_threads leaves found.
        void build_tree(hwloc_obj_t obj, TaskQueueNode* parent, int& leaf_counter)
        {
            if (leaf_counter >= s_num_threads) return;

            auto* node = alloc_node(obj->type, parent);
            if (parent) parent->children.push_back(node);

            if (obj->type == HWLOC_OBJ_PU || obj->arity == 0)
            {
                node->thread_id = leaf_counter++;
                node->cpuset    = hwloc_bitmap_dup(obj->cpuset);
                s_leaves.push_back(node);
            }
            else
            {
                for (unsigned i = 0; i < obj->arity && leaf_counter < s_num_threads; ++i)
                    build_tree(obj->children[i], node, leaf_counter);
            }
        }

        // Collect all leaf descendants of `node` (skipping `skip`) that have
        // not already been visited.
        void collect_leaves(TaskQueueNode* node, const TaskQueueNode* skip,
                            std::vector<TaskQueueNode*>& out,
                            std::unordered_set<TaskQueueNode*>& visited)
        {
            if (node == skip) return;
            if (node->thread_id >= 0)
            {
                if (!visited.count(node)) { visited.insert(node); out.push_back(node); }
                return;
            }
            for (auto* child : node->children)
                collect_leaves(child, skip, out, visited);
        }

        // For each leaf, build steal_order = [self, topology-nearest … farthest].
        void compute_steal_orders()
        {
            for (auto* leaf : s_leaves)
            {
                leaf->steal_order.push_back(leaf);
                std::unordered_set<TaskQueueNode*> visited{leaf};

                // Walk up the tree; at each level collect unclaimed leaves
                // from sibling subtrees before continuing to grandparent.
                for (auto* anc = leaf->parent; anc; anc = anc->parent)
                    for (auto* sibling : anc->children)
                        collect_leaves(sibling, leaf, leaf->steal_order, visited);
            }
        }

        static constexpr int MAX_STEAL       = 32;
        static constexpr int IDLE_SPIN_COUNT = 2000; // iters before sleeping via futex

        // Cilk-style poll:
        //   own queue  → pop_bottom  (LIFO: hottest cache lines first)
        //   other queues → steal_top (FIFO: coarsest/oldest tasks first)
        // Both paths steal up to min(MAX_STEAL, max(1, approx/n_threads)) tasks
        // into a local batch to amortise per-op overhead.
        bool poll_and_execute_one(int thread_id)
        {
            auto* own = s_leaves[thread_id];

            // ── own deque: LIFO pop ────────────────────────────────────────
            {
                const auto approx = own->queue.approx_size();
                if (approx > 0)
                {
                    std::array<std::optional<task_t>, MAX_STEAL> batch;
                    int count = 0;
                    const int want = static_cast<int>(
                        std::min((int64_t)MAX_STEAL,
                                 std::max((int64_t)1, approx / s_num_threads)));
                    for (; count < want; ++count)
                    {
                        task_t tmp;
                        if (!own->queue.pop_bottom(tmp)) break;
                        batch[count] = std::move(tmp);
                    }
                    if (count > 0)
                    {
                        s_stats[thread_id].tasks_executed_own += count;
                        for (int i = 0; i < count; ++i) (*batch[i])();
                        return true;
                    }
                }
            }

            // ── other deques: FIFO steal (topology order) ─────────────────
            for (auto* qnode : own->steal_order)
            {
                if (qnode == own) continue;
                const auto approx = qnode->queue.approx_size();
                if (approx <= 0) continue;
                std::array<std::optional<task_t>, MAX_STEAL> batch;
                int count = 0;
                {
                    const int want = static_cast<int>(
                        std::min((int64_t)MAX_STEAL,
                                 std::max((int64_t)1, approx / s_num_threads)));
                    for (; count < want; ++count)
                    {
                        task_t tmp;
                        if (!qnode->queue.steal_top(tmp)) break;
                        batch[count] = std::move(tmp);
                    }
                }
                if (count > 0)
                {
                    s_stats[thread_id].tasks_stolen += count;
                    for (int i = 0; i < count; ++i) (*batch[i])();
                    return true;
                }
            }

            ++s_stats[thread_id].idle_polls;
            return false;
        }

        void shutdown()
        {
            s_stop.store(true, std::memory_order_release);
            // Bump epoch with release so the s_stop write above is visible to any
            // thread that acquire-loads the epoch just before entering wait().
            s_work_epoch.fetch_add(1, std::memory_order_release);
            s_work_epoch.notify_all();
            for (auto& w : s_workers)
                if (w.joinable()) w.join();
            s_workers.clear();
        }

        struct ShutdownGuard { ~ShutdownGuard() { shutdown(); } };

        // Build the topology tree and spawn workers.  May be called again after
        // shutdown() to restart with a new s_num_threads value.
        // Blocks until every worker thread is polling (ready to accept tasks).
        void spawn_workers()
        {
            s_nodes.clear();
            s_leaves.clear();
            s_stats.assign(s_num_threads, Stats{});
            int leaf_counter = 0;
            build_tree(hwloc_get_root_obj(s_topology), nullptr, leaf_counter);
            compute_steal_orders();

            s_stop = false;
            const int n_workers = s_num_threads - 1;
            s_workers.reserve(n_workers);
            if (n_workers > 0)
            {
                std::latch ready{n_workers};
                for (int i = 1; i < s_num_threads; ++i)
                    s_workers.emplace_back([i, &ready]() {
                        tl_thread_id   = i;
                        tl_fork_cursor = i;
                        ready.count_down(); // signal: this thread is running
                        while (!s_stop.load(std::memory_order_relaxed))
                        {
                            // Spin briefly so we respond quickly to new work.
                            bool found = false;
                            for (int k = 0; k < IDLE_SPIN_COUNT; ++k)
                            {
                                if (poll_and_execute_one(i)) { found = true; break; }
#ifdef __x86_64__
                                __builtin_ia32_pause();
#endif
                            }
                            if (!found)
                            {
                                // Load epoch first, then check s_stop to avoid a
                                // lost-wakeup race with shutdown(): if shutdown()
                                // increments epoch between our load and wait(), then
                                // wait(old_epoch) returns immediately because the
                                // value already changed.
                                const uint64_t epoch =
                                    s_work_epoch.load(std::memory_order_acquire);
                                if (s_stop.load(std::memory_order_relaxed)) break;
                                if (!poll_and_execute_one(i))
                                    s_work_epoch.wait(epoch, std::memory_order_relaxed);
                            }
                        }
                    });
                ready.wait(); // returns only after all workers have checked in
            }
        }
    } // namespace

    void init()
    {
        hwloc_topology_init(&s_topology);
        hwloc_topology_load(s_topology);

        if (s_num_threads == 0)
        {
            s_num_threads = hwloc_get_nbobjs_by_type(s_topology, HWLOC_OBJ_PU);
            if (s_num_threads <= 0)
                s_num_threads = static_cast<int>(std::thread::hardware_concurrency());
        }

        spawn_workers();

        static ShutdownGuard guard;
    }

    int get_max_threads()
    {
        return s_num_threads > 0 ? s_num_threads
                                 : static_cast<int>(std::thread::hardware_concurrency());
    }

    void set_num_threads(int n)
    {
        shutdown();
        s_num_threads = n;
        if (s_topology)        // only respawn if init() has already been called
            spawn_workers();
    }
    int  get_thread_num()       { return tl_thread_id; }

    void fork_task(task_t t)
    {
        // Cilk-style: always push to the bottom of the calling thread's own
        // deque.  Other workers will steal from our top when they run out of
        // work.  tl_thread_id is 0 for the main thread (which acts as worker 0)
        // and 1..n-1 for spawned workers, so s_leaves[tl_thread_id] is always
        // valid after init().
        if (s_leaves[tl_thread_id]->queue.push_bottom(std::move(t)))
        {
            ++s_stats[tl_thread_id].tasks_enqueued;
            // Wake one parked worker (if any).
            s_work_epoch.fetch_add(1, std::memory_order_release);
            s_work_epoch.notify_one();
            return;
        }
        t(); // own deque full — execute inline
    }

    void fork_task(Context& ctx, task_t t)
    {
        ++ctx;
        fork_task([&ctx, t = std::move(t)]() {
            t();
            --ctx;
        });
    }

    void wait_local(Context& ctx)
    {
        while (!ctx.empty())
            poll_and_execute_one(tl_thread_id);
    }

    Context::~Context() { wait_local(*this); }

    std::vector<WorkerStats> get_worker_stats()
    {
        std::vector<WorkerStats> out(s_stats.size());
        for (size_t i = 0; i < s_stats.size(); ++i)
        {
            out[i].tasks_executed_own = s_stats[i].tasks_executed_own;
            out[i].tasks_stolen       = s_stats[i].tasks_stolen;
            out[i].idle_polls         = s_stats[i].idle_polls;
            out[i].tasks_enqueued     = s_stats[i].tasks_enqueued;
        }
        return out;
    }

    void reset_worker_stats()
    {
        for (auto& s : s_stats) s = Stats{};
    }

} // namespace fastfork