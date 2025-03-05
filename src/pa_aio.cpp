//
// Created by ECNU on 2024/9/27.
//
#include "pa_aio.h"


namespace diskann {

// AsyncIOManagerAIO::AsyncIOManagerAIO()
// {
//     if (io_setup(MAX_EVENT, &ctx_) < 0) {
//         exit(1);
//     }
// }

// AsyncIOManagerAIO::~AsyncIOManagerAIO()
// {
//     io_destroy(ctx_);
// }

// void AsyncIOManagerAIO::read(int fd, off_t offset, size_t size)
// {
//     char* buffer = new char[size];
//     struct iocb *cb = new iocb;
//     io_prep_pread(cb, fd, buffer, size, offset);
//     struct iocb* cbs[1] = {cb};
//     if (io_submit(ctx_, 1, cbs) < 0) {
//         exit(1);
//     }
// }

// void AsyncIOManagerAIO::process(std::vector<Element> &pq)
// {
//     struct io_event events[MIN_EVENT_RETURN];
//     int ret;
//     while ((ret = io_getevents(ctx_, 1, MIN_EVENT_RETURN, events, NULL)) > 0) {
//         for (int i = 0; i < ret; i++) {
//             struct io_event *event = events + i;
//             struct iocb *cb = reinterpret_cast<struct iocb*>(event->obj);
//             char *buffer = reinterpret_cast<char*>(cb->u.c.buf);
//             size_t bytes_read = event->res;
//             std::function<void(char*, size_t)>* callback = reinterpret_cast<std::function<void(char*, size_t)>*>(cb->data);
//             (*callback)(buffer, bytes_read);
//             delete[] buffer;
//             delete callback;
//             delete cb;
//         }
//     }   
// }

template<typename dist_t>
AsyncIOManagerUring<dist_t>::AsyncIOManagerUring(size_t dim, size_t nx, std::vector<std::vector<uint32_t>> &ivf, std::vector<size_t> &ivf_off,
                                         std::vector<uint32_t> &ivf_sz, std::unique_ptr<Distance<dist_t>> &fn,
                                         unsigned depth):
      dim_(dim), nx_(nx), ivf_(ivf), ivf_off_(ivf_off), ivf_sz_(ivf_sz), distance_fn_(fn)
{
    if (io_uring_queue_init(depth, &ring_, 0) < 0) {
        std::cerr << "io_uring_queue_init" << std::endl;
        exit(1);
    }
    visited_.resize(nx, 0);
    off_ = sizeof(dist_t) * dim_ + sizeof(uint32_t) * 8; // for aligned
}

template<typename dist_t>
AsyncIOManagerUring<dist_t>::~AsyncIOManagerUring() {
    io_uring_queue_exit(&ring_);
}

template<typename dist_t>
void AsyncIOManagerUring<dist_t>::read(int fd, uint32_t ivf_id) {
    auto *req = new Request;
    auto size = ivf_sz_[ivf_id] * off_;
    req->size = size;
    req->ivf_id = ivf_id;
    req->buffer = (char*)aligned_alloc(32, size);
    
    struct io_uring_sqe *sqe;
    if (!(sqe = io_uring_get_sqe(&ring_))) {
        std::cerr << "io_uring_get_sqe" << std::endl;
        exit(1);
    }
    io_uring_prep_read(sqe, fd, req->buffer, size, ivf_off_[ivf_id]);
    io_uring_sqe_set_data(sqe, (void*)req);

    if (io_uring_submit(&ring_) < 0) {
        std::cerr << "io_uring_submit" << std::endl;
        exit(1);
    }
    issue_ += 1;
    io_sz_ += size;
}

template<typename dist_t>
void AsyncIOManagerUring<dist_t>::process(uint32_t k, std::vector<Element> &pq) {
    struct io_uring_cqe *cqe;
    int ret;
    for (size_t j = 0; j < issue_; j++) {
        ret = io_uring_wait_cqe(&ring_, &cqe);
        if (ret < 0) {
            std::cerr << "io_uring_wait_cqe" << std::endl;
        }
        Request *req = reinterpret_cast<Request*>(io_uring_cqe_get_data(cqe));
        if (cqe->res >= 0) {
//            std::cout << cqe->res << std::endl;
            char *id_base = req->buffer;
            char *vec_base = req->buffer + sizeof(uint32_t) * 8;
            uint32_t sz = ivf_sz_[req->ivf_id];

            for (int i = 0; i < sz; i++) {
                uint32_t id = *(uint32_t*)(id_base + off_ * i); // ivf_[ivf_id][i];
                if (visited_[id])
                    continue;
                visited_[id] = 1;

                const dist_t *vec = (const dist_t* __restrict)(vec_base + off_ * i);
//                for (size_t l = 0; l < dim_; l++)
//                    std::cout << vec[l] << " ";
//                std::cout << std::endl;
                float dist = distance_fn_->compare(vec, query_, dim_);

                if (pq.size() < k) {
                    pq.emplace_back(dist, id);
                    std::push_heap(pq.begin(), pq.end(), std::less<Element>());
                } else if (dist < pq[0].first) {
                    pq.emplace_back(dist, id);
                    std::push_heap(pq.begin(), pq.end(), std::less<Element>());
                    std::pop_heap(pq.begin(), pq.end(), std::less<Element>());
                    pq.pop_back();
                }
            }

        }

//        std::cout << req->ivf_id << ", " << req->size << std::endl;
        io_uring_cqe_seen(&ring_, cqe);
        delete req;
    }
}

//AsyncIOManagerUringLimitQueue::AsyncIOManagerUringLimitQueue(char *filename, size_t capacity, size_t dim, int depth)
//    : depth_(depth)
//{
//    if (io_uring_queue_init(depth, &ring_, 0) < 0) {
//        std::cerr << "io_uring_queue_init" << std::endl;
//        exit(1);
//    }
//    fd_[0] = open(filename, O_RDONLY);
//    io_uring_register_files(&ring_, fd_, 1);
//
//
//    iovs_ = new iovec[depth];
//    size_t sz = capacity * dim * sizeof(float);
//    for (size_t i = 0; i < capacity; i++) {
//        iovs_[i].iov_base = new char[sz];
//        iovs_[i].iov_len = sz;
//    }
//    io_uring_register_buffers(&ring_, iovs_, depth);
//
//
//
//}
//
//void AsyncIOManagerUringLimitQueue::read_wait()
//{
//
//    if (issue_ < depth_) {
//        struct io_uring_sqe *sqe;
//        sqe = io_uring_get_sqe(&ring_);
//
//        // 准备读操作，使用已经注册的文件描述符索引和缓冲区索引
//        io_uring_prep_read_fixed(sqe, 0 /* 文件描述符索引 */, buf, sizeof(buf), 0 /* 偏移 */);
//        io_uring_sqe_set_flags(sqe, IOSQE_FIXED_FILE);  // 表示使用已注册的文件描述符
//        sqe->buf_index = 0;  // 使用已注册的缓冲区索引
//
//        // 提交 I/O 请求
//        io_uring_submit(&ring);
//    } else {
//
//    }
//}


template class AsyncIOManagerUring<float>;
template class AsyncIOManagerUring<uint8_t>;

template class IOManagerPool<float>;
template class IOManagerPool<uint8_t>;


}
