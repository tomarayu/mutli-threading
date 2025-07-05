#ifndef __PROGTEST__
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <climits>
#include <cfloat>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <numeric>
#include <vector>
#include <set>
#include <list>
#include <map>
#include <unordered_set>
#include <unordered_map>
#include <queue>
#include <stack>
#include <deque>
#include <memory>
#include <functional>
#include <thread>
#include <mutex>
#include <semaphore>
#include <atomic>
#include <condition_variable>
#include "progtest_solver.h"
#include "sample_tester.h"
#endif /* __PROGTEST__ */

// ------------------------------------------------------------------
// Helper comparator for merging price list items (CProd).
// Allows quick detection of duplicates (W/H swapped).
// ------------------------------------------------------------------
struct CompareSteelSizes
{
    bool operator()(const CProd &lhs, const CProd &rhs) const
    {
        // Always normalize (width <= height) for the comparison
        auto a = std::make_pair(std::min(lhs.m_W, lhs.m_H), std::max(lhs.m_W, lhs.m_H));
        auto b = std::make_pair(std::min(rhs.m_W, rhs.m_H), std::max(rhs.m_W, rhs.m_H));
        return a < b;
    }
};

// ------------------------------------------------------------------
// Our main class implementing the custom solution
// ------------------------------------------------------------------
class CWeldingCompany
{
public:
    // We do NOT rely on the built‐in (Progtest) solver, but use our own DP solver
    static bool usingProgtestSolver()
    {
        return false;
    }

    // ------------------------------------------------------------------
    // Our custom solver (a classic DP approach to the 2D "welding" problem)
    // ------------------------------------------------------------------
    static void seqSolve(APriceList mergedList, COrder &targetOrder)
    {
        // Dimensions to solve
        unsigned widthNeeded  = targetOrder.m_W;
        unsigned heightNeeded = targetOrder.m_H;
        double weldPrice      = targetOrder.m_WeldingStrength;

        // Table to store minimal cost for each dimension
        // The table is (widthNeeded+1) x (heightNeeded+1).
        std::vector<std::vector<double>> costTable(
            widthNeeded + 1, 
            std::vector<double>(heightNeeded + 1, std::numeric_limits<double>::max())
        );

        // 1) Initialize the direct purchase costs from mergedList
        for (const CProd &item : mergedList->m_List)
        {
            unsigned w = item.m_W;
            unsigned h = item.m_H;
            double price = item.m_Cost;

            if (w <= widthNeeded && h <= heightNeeded)
            {
                costTable[w][h] = std::min(costTable[w][h], price);
            }
            // Because we can rotate
            if (h <= widthNeeded && w <= heightNeeded)
            {
                costTable[h][w] = std::min(costTable[h][w], price);
            }
        }

        // 2) Try all possible ways of splitting
        for (unsigned w = 1; w <= widthNeeded; w++)
        {
            for (unsigned h = 1; h <= heightNeeded; h++)
            {
                // Try vertical splits
                for (unsigned cut = 1; cut < w; ++cut)
                {
                    double weldCost   = h * weldPrice; // the "seam" length is 'h'
                    double combined   = costTable[cut][h] + costTable[w - cut][h] + weldCost;
                    if (combined < costTable[w][h])
                        costTable[w][h] = combined;
                }
                // Try horizontal splits
                for (unsigned cut = 1; cut < h; ++cut)
                {
                    double weldCost   = w * weldPrice; // the "seam" length is 'w'
                    double combined   = costTable[w][cut] + costTable[w][h - cut] + weldCost;
                    if (combined < costTable[w][h])
                        costTable[w][h] = combined;
                }
            }
        }

        // The final cost is stored at (widthNeeded, heightNeeded)
        targetOrder.m_Cost = costTable[widthNeeded][heightNeeded];
    }

    // Public methods
    void addProducer(AProducer prod);
    void addCustomer(ACustomer cust);
    void addPriceList(AProducer prod, APriceList priceList);
    void start(unsigned thrCount);
    void stop();

private:
    // We will define a few private methods for clarity
    void customerThread(ACustomer c);
    void workerThread();

    // Bounded queue
    static constexpr size_t MAX_QUEUE_LEN = 50; // tweak as needed

    unsigned m_WorkerCount{0};
    unsigned m_RemainingCustomers{0};

    // references to producers / customers
    std::vector<AProducer> m_Producers;
    std::vector<ACustomer> m_Customers;

    // aggregated price lists
    std::map<unsigned, APriceList> m_CombinedPriceLists;
    // how many producers have delivered for a given material
    std::map<unsigned, size_t> m_PriceListCounter;
    // set of materials for which we've called sendPriceList()
    std::set<unsigned> m_MaterialRequested;

    // the queue of (orders, customer)
    std::deque<std::pair<AOrderList, ACustomer>> m_OrderQueue;

    // threads
    std::vector<std::thread> m_CustomerThreads;
    std::vector<std::thread> m_WorkerThreads;

    // synchronization
    std::mutex m_MutexOrders;
    std::condition_variable m_OrdersNotEmpty;
    std::condition_variable m_OrdersHaveSpace;

    std::mutex m_MutexPrice;
    std::condition_variable m_PriceListReady;

    // to handle "final sentinel" logic
    std::mutex m_MutexLastCustomer;

    // separate from above: to avoid collisions
    std::mutex m_MutexRequested;

    // Helpers
    void mergePriceList(APriceList &base, const APriceList &incoming);
};

// ------------------------------------------------------------------
// Add a new customer
// ------------------------------------------------------------------
void CWeldingCompany::addCustomer(ACustomer cust)
{
    m_Customers.push_back(cust);
    m_RemainingCustomers++;
}

// ------------------------------------------------------------------
// Add a new producer
// ------------------------------------------------------------------
void CWeldingCompany::addProducer(AProducer prod)
{
    m_Producers.push_back(prod);
}

// ------------------------------------------------------------------
// addPriceList: called by the producer. Merge it into our stored data.
// ------------------------------------------------------------------
void CWeldingCompany::addPriceList(AProducer /*unused*/, APriceList newList)
{
    std::unique_lock<std::mutex> lock(m_MutexPrice);

    unsigned matID = newList->m_MaterialID;

    // if we have never stored a list for matID, just store & track
    if (!m_CombinedPriceLists.count(matID))
    {
        m_CombinedPriceLists[matID] = newList;
        m_PriceListCounter[matID] = 1;

        if (m_PriceListCounter[matID] == m_Producers.size())
        {
            m_PriceListReady.notify_all();
        }
        return;
    }

    // otherwise, we merge the new one
    APriceList &existing = m_CombinedPriceLists[matID];
    std::set<CProd, CompareSteelSizes> mergedSet(existing->m_List.begin(), existing->m_List.end());

    for (const CProd &item : newList->m_List)
    {
        auto it = mergedSet.find(item);
        if (it == mergedSet.end())
        {
            // not present => insert
            mergedSet.insert(item);
        }
        else
        {
            // present => maybe update cost if cheaper
            if (it->m_Cost > item.m_Cost)
            {
                mergedSet.erase(it);
                mergedSet.insert(item);
            }
        }
    }
    // assign it back
    existing->m_List.assign(mergedSet.begin(), mergedSet.end());

    // increment the # of producers that have delivered
    m_PriceListCounter[matID] += 1;
    if (m_PriceListCounter[matID] == m_Producers.size())
    {
        m_PriceListReady.notify_all();
    }
}

// ------------------------------------------------------------------
// Launch threads
// ------------------------------------------------------------------
void CWeldingCompany::start(unsigned thrCount)
{
    m_WorkerCount = thrCount;

    // spawn 1 thread per customer
    for (ACustomer &c : m_Customers)
    {
        m_CustomerThreads.emplace_back(&CWeldingCompany::customerThread, this, c);
    }

    // spawn thrCount worker threads
    for (size_t i = 0; i < thrCount; i++)
    {
        m_WorkerThreads.emplace_back(&CWeldingCompany::workerThread, this);
    }
}

// ------------------------------------------------------------------
// stop: join threads, then reset the data
// ------------------------------------------------------------------
void CWeldingCompany::stop()
{
    // 1) join customer threads
    for (auto &ct : m_CustomerThreads)
    {
        if (ct.joinable())
            ct.join();
    }
    m_CustomerThreads.clear();

    // 2) join worker threads
    for (auto &wt : m_WorkerThreads)
    {
        if (wt.joinable())
            wt.join();
    }
    m_WorkerThreads.clear();

    // 3) reset
    m_Customers.clear();
    m_Producers.clear();
    m_OrderQueue.clear();
    m_CombinedPriceLists.clear();
    m_PriceListCounter.clear();
    m_MaterialRequested.clear();
    m_RemainingCustomers = 0;
}

// ------------------------------------------------------------------
// A single customer thread: repeatedly produce new orders
// ------------------------------------------------------------------
void CWeldingCompany::customerThread(ACustomer c)
{
    while (true)
    {
        {
            // Wait until there's space in the queue *before* we ask for a new order
            std::unique_lock<std::mutex> lock(m_MutexOrders);
            while (m_OrderQueue.size() >= MAX_QUEUE_LEN)
            {
                m_OrdersHaveSpace.wait(lock);
            }
        }

        // Now read next demand from the customer
        AOrderList nextDemand = c->waitForDemand();
        if (!nextDemand)
        {
            // no more demands => break
            break;
        }

        // Possibly call sendPriceList if first time for this matID
        {
            std::lock_guard<std::mutex> g(m_MutexRequested);
            if (m_MaterialRequested.count(nextDemand->m_MaterialID) == 0)
            {
                // we haven't requested for matID yet => do it
                m_MaterialRequested.insert(nextDemand->m_MaterialID);
                // call each producer
                for (auto &prod : m_Producers)
                {
                    prod->sendPriceList(nextDemand->m_MaterialID);
                }
            }
        }

        // Now enqueue
        {
            std::unique_lock<std::mutex> lock2(m_MutexOrders);
            m_OrderQueue.push_back({ nextDemand, c });
            m_OrdersNotEmpty.notify_one();
        }
    }

    // If this is the last customer thread => push sentinel orders for each worker
    {
        std::lock_guard<std::mutex> guard(m_MutexLastCustomer);
        if (m_RemainingCustomers > 1)
        {
            // not the last => just decrement
            m_RemainingCustomers--;
            return;
        }
        // otherwise, we are the last => push sentinel
    }

    // push sentinel
    for (size_t i = 0; i < m_WorkerCount; i++)
    {
        {
            std::unique_lock<std::mutex> lockQ(m_MutexOrders);
            // also respect bounding here if you like
            while (m_OrderQueue.size() >= MAX_QUEUE_LEN)
            {
                m_OrdersHaveSpace.wait(lockQ);
            }

            // sentinel: materialID = max => worker sees it => stops
            auto sentinelList = std::make_shared<COrderList>(std::numeric_limits<unsigned>::max());
            m_OrderQueue.push_back({ sentinelList, c });
            m_OrdersNotEmpty.notify_all();
        }
    }
}

// ------------------------------------------------------------------
// Worker thread: pop from queue, wait for the price list to be ready,
// run seqSolve on each sub‐order, call completed(...). Repeat forever.
// ------------------------------------------------------------------
void CWeldingCompany::workerThread()
{
    while (true)
    {
        AOrderList incoming;
        ACustomer clientRef;
        {
            std::unique_lock<std::mutex> lock(m_MutexOrders);
            // wait until queue is nonempty
            m_OrdersNotEmpty.wait(lock, [this] { return !m_OrderQueue.empty(); });

            // pop front
            incoming  = m_OrderQueue.front().first;
            clientRef = m_OrderQueue.front().second;
            m_OrderQueue.pop_front();

            // we freed up a slot
            m_OrdersHaveSpace.notify_one();
        }

        // check sentinel
        unsigned matID = incoming->m_MaterialID;
        if (matID == std::numeric_limits<unsigned>::max())
        {
            // sentinel => time to finish
            return;
        }

        // wait until all producers have delivered the full price list
        {
            std::unique_lock<std::mutex> lockPrice(m_MutexPrice);
            m_PriceListReady.wait(lockPrice, [this, matID]() {
                auto it = m_PriceListCounter.find(matID);
                if (it == m_PriceListCounter.end())
                    return false;
                return (it->second == m_Producers.size());
            });
        }

        // Now we have the completed price list => solve each sub‐order
        APriceList completeList;
        {
            // no complicated merges here; we already have m_CombinedPriceLists
            std::lock_guard<std::mutex> guard(m_MutexPrice);
            completeList = m_CombinedPriceLists[matID];
        }

        for (COrder &oneOrder : incoming->m_List)
        {
            seqSolve(completeList, oneOrder);
        }

        // done => notify the customer
        clientRef->completed(incoming);
    }
}

// ------------------------------------------------------------------
// A small main just for local usage
// ------------------------------------------------------------------
#ifndef __PROGTEST__
int main()
{
    using namespace std::placeholders;
    CWeldingCompany testSystem;

    AProducer p1 = std::make_shared<CProducerSync>(std::bind(&CWeldingCompany::addPriceList, &testSystem, _1, _2));
    AProducerAsync p2 = std::make_shared<CProducerAsync>(std::bind(&CWeldingCompany::addPriceList, &testSystem, _1, _2));

    testSystem.addProducer(p1);
    testSystem.addProducer(p2);

    // A single customer that will produce multiple demands
    testSystem.addCustomer(std::make_shared<CCustomerTest>(10000));

    p2->start();

    // Just one worker in this example
    testSystem.start(1);

    // stop everything
    testSystem.stop();

    p2->stop();

    return 0;
}
#endif /* __PROGTEST__ */
