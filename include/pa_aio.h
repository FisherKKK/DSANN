//
// Created by ECNU on 2024/9/27.
//

#ifndef DISKANN_PAG_AIO_H
#define DISKANN_PAG_AIO_H

#include <libaio.h>
#include <liburing.h>
#include <memory>
#include <utility>
#include <vector>
#include <mutex>
#include <condition_variable>
#include <algorithm>
#include <thread>
#include <iostream>
#include <queue>
#include "pa_aio.h"
#include "distance.h"

namespace diskann {

//class ThreadPool {
//  public:
//    ThreadPool(size_t sz);
//
//  private:
//    std::vector<std::thread> worker_;
//    std::vector<std::function<void()>> tasks;
//    std::mutex queue_mutex;
//
//};

using Element = std::pair<float, uint32_t>;
/** Target for vector calculation
 */
template<typename dist_t>
class AsyncIOManager
{
  public:
    virtual ~AsyncIOManager() = default;
    virtual void read(int fd, uint32_t ivf_id) = 0;
    virtual void process(uint32_t k, std::vector<Element> &pq) = 0;
    virtual void set_query(dist_t *query) = 0;

    // io times / io size
    virtual std::pair<uint64_t, uint64_t> get_info() { return {}; }
};


// class AsyncIOManagerAIO : public AsyncIOManager
// {
//   public:
//     AsyncIOManagerAIO();
//     ~AsyncIOManagerAIO();
//     void read(int fd, off_t offset, size_t size) override;
//     void process(std::vector<Element> &pq) override;
//     void set_query(float *query, size_t dim) override;

//   private:
//     io_context_t ctx_ = 0;
//     static constexpr int MAX_EVENT = 1024;
//     static constexpr int MIN_EVENT_RETURN = 4;
// };

/** Here we have two type of implementation, 
 *  with cblas for large batch calculation and iterate for small batch calculation
 */
template<typename dist_t>
class AsyncIOManagerUring : public AsyncIOManager<dist_t>
{
  public:
    AsyncIOManagerUring(size_t dim, size_t nx, std::vector<std::vector<uint32_t>> &ivf, std::vector<size_t> &ivf_off,
                        std::vector<uint32_t> &ivf_sz,
                        std::unique_ptr<Distance<dist_t>> &fn,
                        unsigned depth = DEFAULT_DEPTH);
    ~AsyncIOManagerUring();

    void read(int fd, uint32_t ivf_id) override;

    void process(uint32_t k, std::vector<Element> &pq) override;

    void set_query(dist_t *query) override {
      query_ = query;
      memset(visited_.data(), 0, sizeof(uint8_t) * nx_);
      issue_ = 0;
      io_sz_ = 0;
    }

    std::pair<uint64_t, uint64_t> get_info() override {
        return {issue_, io_sz_};
    }

    struct Request {
      uint32_t ivf_id;
      size_t size;
      char* buffer = nullptr;

      ~Request() {
          free(buffer);
          buffer = nullptr;
      }
    };

  private:
    struct io_uring ring_;
    static constexpr int DEFAULT_DEPTH = 128;
    std::unique_ptr<Distance<dist_t>> &distance_fn_;
    std::vector<std::vector<uint32_t>> &ivf_;
    std::vector<size_t> &ivf_off_;
    std::vector<uint32_t> &ivf_sz_;
    std::vector<uint8_t> visited_;
    dist_t *query_;
    size_t dim_;
    size_t nx_;
    size_t issue_;
    size_t off_;
    size_t io_sz_;
};


class AsyncIOManagerUringLimitQueue {
  public:
    AsyncIOManagerUringLimitQueue(char *filename, size_t capacity, size_t dim, int depth = DEFAULT_DEPTH);
    ~AsyncIOManagerUringLimitQueue() {
        io_uring_unregister_files(&ring_);
        io_uring_unregister_buffers(&ring_);
        for (size_t i = 0; i < depth_; i++) {
            delete[] (char*)(iovs_[i].iov_base);
        }
        delete[] iovs_;
        close(fd_[0]);
    }

    void read_wait();

    void process_single() {

    }

    void process_all() {

    }





  private:
    int fd_[1];
    struct iovec *iovs_;
    struct io_uring ring_;
    int depth_;
    int issue_;
    static constexpr int DEFAULT_DEPTH = 256;
};

template<typename dist_t>
class IOManagerPool {
  public:
    IOManagerPool(size_t size, size_t dim, size_t nx, std::vector<std::vector<uint32_t>> &ivf, std::vector<size_t> &ivf_off,
                  std::vector<uint32_t> &ivf_sz,
                  std::unique_ptr<Distance<dist_t>> &fn,
                  unsigned depth = DEFAULT_DEPTH) : size_(size) {
        for (size_t i = 0; i < size; i++) {
            pool_.emplace_back(std::make_unique<AsyncIOManagerUring<dist_t>>(dim, nx, ivf, ivf_off, ivf_sz, fn, depth));
            ids_.push(i);
        }
    }


    ~IOManagerPool() {

    }

     std::unique_ptr<AsyncIOManager<dist_t>>& Get(size_t &id) {
        std::unique_lock<std::mutex> lk(latch_);
        cv_.wait(lk, [&]{return !ids_.empty();});
        id = ids_.front();
        ids_.pop();
        return pool_[id];
    }

    void Put(size_t id) {
        std::scoped_lock slk(latch_);
        ids_.push(id);
        cv_.notify_one();
    }





  private:
    static constexpr int DEFAULT_DEPTH = 128;
    size_t size_;
    std::queue<size_t> ids_;
    std::vector<std::unique_ptr<AsyncIOManager<dist_t>>> pool_;
    std::mutex latch_;
    std::condition_variable cv_;
};


}



#endif // DISKANN_PAG_AIO_H
