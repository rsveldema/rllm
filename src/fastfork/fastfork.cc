#include <fastfork/fastfork.hpp>

#include <hwloc.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <latch>
#include <mutex>
#include <queue>
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
        static thread_local int         tl_thread_id   = 0;
        static thread_local int         tl_fork_cursor = 0; // per-core, no atomic needed

        hwloc_topology_t s_topology = nullptr;

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
            std::queue<task_t>           queue;             // leaves only
            std::mutex                   mutex;             // leaves only
            std::vector<TaskQueueNode*>  steal_order;       // leaves only
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

        static constexpr int MAX_STEAL = 32;

        // Try own queue first, then steal in topology order.
        // Steals min(MAX_STEAL, max(1, queue_size / num_threads)) tasks per lock
        // acquisition into a stack-local array, amortising mutex cost without
        // heap allocation.
        bool poll_and_execute_one(int thread_id)
        {
            for (auto* qnode : s_leaves[thread_id]->steal_order)
            {
                std::array<task_t, MAX_STEAL> batch;
                int count = 0;
                {
                    std::lock_guard<std::mutex> lock(qnode->mutex);
                    const int n = static_cast<int>(qnode->queue.size());
                    if (n > 0)
                    {
                        count = std::min(MAX_STEAL, std::max(1, n / s_num_threads));
                        for (int i = 0; i < count; ++i)
                        {
                            batch[i] = std::move(qnode->queue.front());
                            qnode->queue.pop();
                        }
                    }
                }
                if (count > 0)
                {
                    for (int i = 0; i < count; ++i)
                        batch[i]();
                    return true;
                }
            }
            return false;
        }

        void shutdown()
        {
            s_stop.store(true, std::memory_order_release);
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
                            poll_and_execute_one(i);
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
        const int target = tl_fork_cursor % s_num_threads;
        tl_fork_cursor   = target + 1; // pre-wrap to avoid a second % on the next call
        if (tl_fork_cursor >= s_num_threads) tl_fork_cursor = 0;
        std::lock_guard<std::mutex> lock(s_leaves[target]->mutex);
        s_leaves[target]->queue.push(std::move(t));
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

} // namespace fastfork