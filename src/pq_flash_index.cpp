// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include "common_includes.h"

#include "timer.h"
#include "pq.h"
#include "pq_scratch.h"
#include "pq_flash_index.h"
#include "cosine_similarity.h"

#ifdef _WINDOWS
#include "windows_aligned_file_reader.h"
#else
#include "linux_aligned_file_reader.h"
#endif

#define READ_U64(stream, val) stream.read((char *)&val, sizeof(uint64_t))
#define READ_U32(stream, val) stream.read((char *)&val, sizeof(uint32_t))
#define READ_UNSIGNED(stream, val) stream.read((char *)&val, sizeof(unsigned))

// sector # beyond the end of graph where data for id is present for reordering
#define VECTOR_SECTOR_NO(id) (((uint64_t)(id)) / _nvecs_per_sector + _reorder_data_start_sector)

// sector # beyond the end of graph where data for id is present for reordering
#define VECTOR_SECTOR_OFFSET(id) ((((uint64_t)(id)) % _nvecs_per_sector) * _data_dim * sizeof(float))

namespace diskann
{

// 量化压缩的内存索引
template <typename T, typename LabelT>
PQFlashIndex<T, LabelT>::PQFlashIndex(std::shared_ptr<AlignedFileReader> &fileReader, diskann::Metric m)
    : reader(fileReader), metric(m), _thread_data(nullptr)
{
    if (m == diskann::Metric::COSINE || m == diskann::Metric::INNER_PRODUCT)
    {
        if (std::is_floating_point<T>::value)
        {
            diskann::cout << "Cosine metric chosen for (normalized) float data."
                             "Changing distance to L2 to boost accuracy."
                          << std::endl;
            metric = diskann::Metric::L2;
        }
        else
        {
            diskann::cerr << "WARNING: Cannot normalize integral data types."
                          << " This may result in erroneous results or poor recall."
                          << " Consider using L2 distance with integral data types." << std::endl;
        }
    }

    this->_dist_cmp.reset(diskann::get_distance_function<T>(metric));
    this->_dist_cmp_float.reset(diskann::get_distance_function<float>(metric));
}

template <typename T, typename LabelT> PQFlashIndex<T, LabelT>::~PQFlashIndex()
{
#ifndef EXEC_ENV_OLS
    if (data != nullptr)
    {
        delete[] data;
    }
#endif

    if (_centroid_data != nullptr)
        aligned_free(_centroid_data);
    // delete backing bufs for nhood and coord cache
    if (_nhood_cache_buf != nullptr)
    {
        delete[] _nhood_cache_buf;
        diskann::aligned_free(_coord_cache_buf);
    }

    if (_load_flag)
    {
        diskann::cout << "Clearing scratch" << std::endl;
        ScratchStoreManager<SSDThreadData<T>> manager(this->_thread_data);
        manager.destroy();
        this->reader->deregister_all_threads();
        reader->close();
    }
    if (_pts_to_label_offsets != nullptr)
    {
        delete[] _pts_to_label_offsets;
    }
    if (_pts_to_label_counts != nullptr)
    {
        delete[] _pts_to_label_counts;
    }
    if (_pts_to_labels != nullptr)
    {
        delete[] _pts_to_labels;
    }
}

template <typename T, typename LabelT> inline uint64_t PQFlashIndex<T, LabelT>::get_node_sector(uint64_t node_id)
{
    return 1 + (_nnodes_per_sector > 0 ? node_id / _nnodes_per_sector
                                       : node_id * DIV_ROUND_UP(_max_node_len, defaults::SECTOR_LEN));
}

template <typename T, typename LabelT>
inline char *PQFlashIndex<T, LabelT>::offset_to_node(char *sector_buf, uint64_t node_id)
{
    return sector_buf + (_nnodes_per_sector == 0 ? 0 : (node_id % _nnodes_per_sector) * _max_node_len);
}

template <typename T, typename LabelT> inline uint32_t *PQFlashIndex<T, LabelT>::offset_to_node_nhood(char *node_buf)
{
    return (unsigned *)(node_buf + _disk_bytes_per_point);
}

template <typename T, typename LabelT> inline T *PQFlashIndex<T, LabelT>::offset_to_node_coords(char *node_buf)
{
    return (T *)(node_buf);
}

template <typename T, typename LabelT>
void PQFlashIndex<T, LabelT>::setup_thread_data(uint64_t nthreads, uint64_t visited_reserve)
{
    diskann::cout << "Setting up thread-specific contexts for nthreads: " << nthreads << std::endl;
// omp parallel for to generate unique thread IDs
#pragma omp parallel for num_threads((int)nthreads)
    for (int64_t thread = 0; thread < (int64_t)nthreads; thread++)
    {
#pragma omp critical
        {
            // 每一个线程都是一个scratch, 负责单独处理查询
            SSDThreadData<T> *data = new SSDThreadData<T>(this->_aligned_dim, visited_reserve);
            // 为当前线程注册aio
            this->reader->register_thread();
            // 当前线程数据的context
            data->ctx = this->reader->get_ctx();
            // 将当前data压入到ConcurrentQueue中
            this->_thread_data.push(data);
        }
    }
    _load_flag = true;
}

// 读取节点请求, 以sector为单位进行读取
// 每个需要被访问的节点, 都读取其对应的sector
// 然后将数据copy out, 保存real vector, 及其邻居
template <typename T, typename LabelT>
std::vector<bool> PQFlashIndex<T, LabelT>::read_nodes(const std::vector<uint32_t> &node_ids,
                                                      std::vector<T *> &coord_buffers,
                                                      std::vector<std::pair<uint32_t, uint32_t *>> &nbr_buffers)
{
    std::vector<AlignedRead> read_reqs;
    std::vector<bool> retval(node_ids.size(), true);

    char *buf = nullptr;
    auto num_sectors = _nnodes_per_sector > 0 ? 1 : DIV_ROUND_UP(_max_node_len, defaults::SECTOR_LEN);
    alloc_aligned((void **)&buf, node_ids.size() * num_sectors * defaults::SECTOR_LEN, defaults::SECTOR_LEN);

    // create read requests
    // 这里相当于读取每个node对应的sector
    // num_sectors就是每个node所需要的sector数目
    for (size_t i = 0; i < node_ids.size(); ++i)
    {
        auto node_id = node_ids[i];

        AlignedRead read;
        read.len = num_sectors * defaults::SECTOR_LEN;
        // 当前node读取的buf位置
        read.buf = buf + i * num_sectors * defaults::SECTOR_LEN;
        // 获取该node的offset
        read.offset = get_node_sector(node_id) * defaults::SECTOR_LEN;
        // push到reqs中
        read_reqs.push_back(read);
    }

    // borrow thread data and issue reads
    ScratchStoreManager<SSDThreadData<T>> manager(this->_thread_data);
    auto this_thread_data = manager.scratch_space();
    IOContext &ctx = this_thread_data->ctx;
    // 在指定上下文下进行读请求
    // 这里就是异步执行IO, 然后将结果存到buf中
    reader->read(read_reqs, ctx);

    // copy reads into buffers
    // 对每个read_reqs进行读取
    for (uint32_t i = 0; i < read_reqs.size(); i++)
    {
#if defined(_WINDOWS) && defined(USE_BING_INFRA) // this block is to handle failed reads in
                                                 // production settings
        if ((*ctx.m_pRequestsStatus)[i] != IOContext::READ_SUCCESS)
        {
            retval[i] = false;
            continue;
        }
#endif

        char *node_buf = offset_to_node((char *)read_reqs[i].buf, node_ids[i]);

        // 拷贝节点的vector
        if (coord_buffers[i] != nullptr)
        {
            T *node_coords = offset_to_node_coords(node_buf);
            memcpy(coord_buffers[i], node_coords, _disk_bytes_per_point);
        }

        // copy节点的邻居数目, 以及节点的id
        if (nbr_buffers[i].second != nullptr)
        {
            uint32_t *node_nhood = offset_to_node_nhood(node_buf);
            auto num_nbrs = *node_nhood;
            nbr_buffers[i].first = num_nbrs;
            memcpy(nbr_buffers[i].second, node_nhood + 1, num_nbrs * sizeof(uint32_t));
        }
    }

    aligned_free(buf);

    return retval;
}

/** 载入缓存的节点
 *  分配一个缓存空间读取vector, neighbors
 *  将读取的缓存buff汇总到PQFlashIndex的Map中
 */
template <typename T, typename LabelT> void PQFlashIndex<T, LabelT>::load_cache_list(std::vector<uint32_t> &node_list)
{
    diskann::cout << "Loading the cache list into memory.." << std::flush;
    size_t num_cached_nodes = node_list.size();

    // borrow thread data
    // 异步IO
    ScratchStoreManager<SSDThreadData<T>> manager(this->_thread_data);
    auto this_thread_data = manager.scratch_space();
    IOContext &ctx = this_thread_data->ctx;

    // Allocate space for neighborhood cache
    // 为每个node的邻居分配缓存空间
    _nhood_cache_buf = new uint32_t[num_cached_nodes * (_max_degree + 1)];
    memset(_nhood_cache_buf, 0, num_cached_nodes * (_max_degree + 1));

    // Allocate space for coordinate cache
    // 为original vector分配空间, 并进行重置
    size_t coord_cache_buf_len = num_cached_nodes * _aligned_dim;
    diskann::alloc_aligned((void **)&_coord_cache_buf, coord_cache_buf_len * sizeof(T), 8 * sizeof(T));
    memset(_coord_cache_buf, 0, coord_cache_buf_len * sizeof(T));

    size_t BLOCK_SIZE = 8;
    size_t num_blocks = DIV_ROUND_UP(num_cached_nodes, BLOCK_SIZE);
    // 对每一块进行遍历
    // 两层关系: idx => id | id => vector | idx => vector
    for (size_t block = 0; block < num_blocks; block++)
    {
        // 块的开始id和结束id
        size_t start_idx = block * BLOCK_SIZE;
        size_t end_idx = (std::min)(num_cached_nodes, (block + 1) * BLOCK_SIZE);

        // Copy offset into buffers to read into
        std::vector<uint32_t> nodes_to_read;
        std::vector<T *> coord_buffers;
        std::vector<std::pair<uint32_t, uint32_t *>> nbr_buffers;
        for (size_t node_idx = start_idx; node_idx < end_idx; node_idx++)
        {
            // 需要被读取的节点
            nodes_to_read.push_back(node_list[node_idx]);
            // vector buf container
            coord_buffers.push_back(_coord_cache_buf + node_idx * _aligned_dim);
            // 每个节点对应的 邻居
            nbr_buffers.emplace_back(0, _nhood_cache_buf + node_idx * (_max_degree + 1));
        }

        // issue the reads
        // 进行读请求, 结果将被记录在coord_buffers和nbr_buffers中
        auto read_status = read_nodes(nodes_to_read, coord_buffers, nbr_buffers);

        // check for success and insert into the cache.
        for (size_t i = 0; i < read_status.size(); i++)
        {
            // 将结果汇总到map中
            // _coord_cache: id => vector
            // _nhood_cache: id => neighbors(num, T[])
            if (read_status[i] == true)
            {
                _coord_cache.insert(std::make_pair(nodes_to_read[i], coord_buffers[i]));
                _nhood_cache.insert(std::make_pair(nodes_to_read[i], nbr_buffers[i]));
            }
        }
    }
    diskann::cout << "..done." << std::endl;
}

#ifdef EXEC_ENV_OLS
template <typename T, typename LabelT>
void PQFlashIndex<T, LabelT>::generate_cache_list_from_sample_queries(MemoryMappedFiles &files, std::string sample_bin,
                                                                      uint64_t l_search, uint64_t beamwidth,
                                                                      uint64_t num_nodes_to_cache, uint32_t nthreads,
                                                                      std::vector<uint32_t> &node_list)
{
#else
template <typename T, typename LabelT>
void PQFlashIndex<T, LabelT>::generate_cache_list_from_sample_queries(std::string sample_bin, uint64_t l_search,
                                                                      uint64_t beamwidth, uint64_t num_nodes_to_cache,
                                                                      uint32_t nthreads,
                                                                      std::vector<uint32_t> &node_list)
{
#endif
    if (num_nodes_to_cache >= this->_num_points)
    {
        // for small num_points and big num_nodes_to_cache, use below way to get the node_list quickly
        node_list.resize(this->_num_points);
        for (uint32_t i = 0; i < this->_num_points; ++i)
        {
            node_list[i] = i;
        }
        return;
    }

    this->_count_visited_nodes = true;
    this->_node_visit_counter.clear();
    this->_node_visit_counter.resize(this->_num_points);
    for (uint32_t i = 0; i < _node_visit_counter.size(); i++)
    {
        this->_node_visit_counter[i].first = i;
        this->_node_visit_counter[i].second = 0;
    }

    uint64_t sample_num, sample_dim, sample_aligned_dim;
    T *samples;

#ifdef EXEC_ENV_OLS
    if (files.fileExists(sample_bin))
    {
        diskann::load_aligned_bin<T>(files, sample_bin, samples, sample_num, sample_dim, sample_aligned_dim);
    }
#else
    if (file_exists(sample_bin))
    {
        diskann::load_aligned_bin<T>(sample_bin, samples, sample_num, sample_dim, sample_aligned_dim);
    }
#endif
    else
    {
        diskann::cerr << "Sample bin file not found. Not generating cache." << std::endl;
        return;
    }

    std::vector<uint64_t> tmp_result_ids_64(sample_num, 0);
    std::vector<float> tmp_result_dists(sample_num, 0);

    bool filtered_search = false;
    std::vector<LabelT> random_query_filters(sample_num);
    if (_filter_to_medoid_ids.size() != 0)
    {
        filtered_search = true;
        generate_random_labels(random_query_filters, (uint32_t)sample_num, nthreads);
    }

#pragma omp parallel for schedule(dynamic, 1) num_threads(nthreads)
    for (int64_t i = 0; i < (int64_t)sample_num; i++)
    {
        auto &label_for_search = random_query_filters[i];
        // run a search on the sample query with a random label (sampled from base label distribution), and it will
        // concurrently update the node_visit_counter to track most visited nodes. The last false is to not use the
        // "use_reorder_data" option which enables a final reranking if the disk index itself contains only PQ data.
        cached_beam_search(samples + (i * sample_aligned_dim), 1, l_search, tmp_result_ids_64.data() + i,
                           tmp_result_dists.data() + i, beamwidth, filtered_search, label_for_search, false);
    }

    std::sort(this->_node_visit_counter.begin(), _node_visit_counter.end(),
              [](std::pair<uint32_t, uint32_t> &left, std::pair<uint32_t, uint32_t> &right) {
                  return left.second > right.second;
              });
    node_list.clear();
    node_list.shrink_to_fit();
    num_nodes_to_cache = std::min(num_nodes_to_cache, this->_node_visit_counter.size());
    node_list.reserve(num_nodes_to_cache);
    for (uint64_t i = 0; i < num_nodes_to_cache; i++)
    {
        node_list.push_back(this->_node_visit_counter[i].first);
    }
    this->_count_visited_nodes = false;

    diskann::aligned_free(samples);
}

// 采用bfs的方式缓存指定数目的点, 围绕着memoid
// 从memoid点出发进行BFS => 对每一个点载入其对应的sector => 载入其邻居 => 直到达到cache size
// 结果保存在node_list中, 即所有被载入的node
template <typename T, typename LabelT>
void PQFlashIndex<T, LabelT>::cache_bfs_levels(uint64_t num_nodes_to_cache, std::vector<uint32_t> &node_list,
                                               const bool shuffle)
{
    std::random_device rng;
    std::mt19937 urng(rng());

    tsl::robin_set<uint32_t> node_set;

    // Do not cache more than 10% of the nodes in the index
    uint64_t tenp_nodes = (uint64_t)(std::round(this->_num_points * 0.1));
    if (num_nodes_to_cache > tenp_nodes)
    {
        diskann::cout << "Reducing nodes to cache from: " << num_nodes_to_cache << " to: " << tenp_nodes
                      << "(10 percent of total nodes:" << this->_num_points << ")" << std::endl;
        num_nodes_to_cache = tenp_nodes == 0 ? 1 : tenp_nodes;
    }
    diskann::cout << "Caching " << num_nodes_to_cache << "..." << std::endl;

    // borrow thread data
    // 这里相当于是线程数据了
    ScratchStoreManager<SSDThreadData<T>> manager(this->_thread_data);
    // 当前thread的数目
    auto this_thread_data = manager.scratch_space();
    IOContext &ctx = this_thread_data->ctx;

    std::unique_ptr<tsl::robin_set<uint32_t>> cur_level, prev_level;
    cur_level = std::make_unique<tsl::robin_set<uint32_t>>();
    prev_level = std::make_unique<tsl::robin_set<uint32_t>>();

    for (uint64_t miter = 0; miter < _num_medoids && cur_level->size() < num_nodes_to_cache; miter++)
    {
        cur_level->insert(_medoids[miter]);
    }

    if ((_filter_to_medoid_ids.size() > 0) && (cur_level->size() < num_nodes_to_cache))
    {
        for (auto &x : _filter_to_medoid_ids)
        {
            for (auto &y : x.second)
            {
                cur_level->insert(y);
                if (cur_level->size() == num_nodes_to_cache)
                    break;
            }
            if (cur_level->size() == num_nodes_to_cache)
                break;
        }
    }

    uint64_t lvl = 1;
    uint64_t prev_node_set_size = 0;
    while ((node_set.size() + cur_level->size() < num_nodes_to_cache) && cur_level->size() != 0)
    {
        // swap prev_level and cur_level
        std::swap(prev_level, cur_level);
        // clear cur_level
        cur_level->clear();

        std::vector<uint32_t> nodes_to_expand;

        // 首先将medoid插入到node_set当中
        for (const uint32_t &id : *prev_level)
        {
            // 将节点插入到node_set
            if (node_set.find(id) != node_set.end())
            {
                continue;
            }
            node_set.insert(id);
            nodes_to_expand.push_back(id);
        }

        if (shuffle)
            std::shuffle(nodes_to_expand.begin(), nodes_to_expand.end(), urng);
        else
            std::sort(nodes_to_expand.begin(), nodes_to_expand.end());

        diskann::cout << "Level: " << lvl << std::flush;
        bool finish_flag = false;

        uint64_t BLOCK_SIZE = 1024;
        uint64_t nblocks = DIV_ROUND_UP(nodes_to_expand.size(), BLOCK_SIZE);
        for (size_t block = 0; block < nblocks && !finish_flag; block++)
        {
            diskann::cout << "." << std::flush;
            size_t start = block * BLOCK_SIZE;
            size_t end = (std::min)((block + 1) * BLOCK_SIZE, nodes_to_expand.size());

            std::vector<uint32_t> nodes_to_read;
            std::vector<T *> coord_buffers(end - start, nullptr);
            std::vector<std::pair<uint32_t, uint32_t *>> nbr_buffers;

            for (size_t cur_pt = start; cur_pt < end; cur_pt++)
            {
                nodes_to_read.push_back(nodes_to_expand[cur_pt]);
                // pair(邻居数目, 邻居id)
                nbr_buffers.emplace_back(0, new uint32_t[_max_degree + 1]);
            }

            // issue read requests
            // 读取对应的节点 => nodes_to_read
            // 这里相当于按sector读取数据, 并将相关数据放入coord_buffer, nbr_buffers中
            //! 但是这里的coord_buffers还都是null
            auto read_status = read_nodes(nodes_to_read, coord_buffers, nbr_buffers);

            // process each nhood buf
            // 对每个neighbor进行处理
            for (uint32_t i = 0; i < read_status.size(); i++)
            {
                if (read_status[i] == false)
                {
                    continue;
                }
                else
                {
                    // first是数目
                    uint32_t nnbrs = nbr_buffers[i].first;
                    // second是邻居id
                    uint32_t *nbrs = nbr_buffers[i].second;

                    // explore next level
                    // 探索下一层
                    for (uint32_t j = 0; j < nnbrs && !finish_flag; j++)
                    {
                        // 插入未见过的邻居id
                        // 但是这里还没有插入到node_set
                        if (node_set.find(nbrs[j]) == node_set.end())
                        {
                            cur_level->insert(nbrs[j]);
                        }
                        if (cur_level->size() + node_set.size() >= num_nodes_to_cache)
                        {
                            finish_flag = true;
                        }
                    }
                }
                delete[] nbr_buffers[i].second;
            }
        }

        diskann::cout << ". #nodes: " << node_set.size() - prev_node_set_size
                      << ", #nodes thus far: " << node_set.size() << std::endl;
        prev_node_set_size = node_set.size();
        lvl++;
    }

    assert(node_set.size() + cur_level->size() == num_nodes_to_cache || cur_level->size() == 0);

    node_list.clear();
    node_list.reserve(node_set.size() + cur_level->size());
    for (auto node : node_set)
        node_list.push_back(node);
    for (auto node : *cur_level)
        node_list.push_back(node);

    diskann::cout << "Level: " << lvl << std::flush;
    diskann::cout << ". #nodes: " << node_list.size() - prev_node_set_size << ", #nodes thus far: " << node_list.size()
                  << std::endl;
    diskann::cout << "done" << std::endl;
}

template <typename T, typename LabelT> void PQFlashIndex<T, LabelT>::use_medoids_data_as_centroids()
{
    if (_centroid_data != nullptr)
        aligned_free(_centroid_data);
    alloc_aligned(((void **)&_centroid_data), _num_medoids * _aligned_dim * sizeof(float), 32);
    std::memset(_centroid_data, 0, _num_medoids * _aligned_dim * sizeof(float));

    // borrow ctx
    ScratchStoreManager<SSDThreadData<T>> manager(this->_thread_data);
    auto data = manager.scratch_space();
    IOContext &ctx = data->ctx;
    diskann::cout << "Loading centroid data from medoids vector data of " << _num_medoids << " medoid(s)" << std::endl;

    std::vector<uint32_t> nodes_to_read;
    std::vector<T *> medoid_bufs;
    std::vector<std::pair<uint32_t, uint32_t *>> nbr_bufs;

    for (uint64_t cur_m = 0; cur_m < _num_medoids; cur_m++)
    {
        nodes_to_read.push_back(_medoids[cur_m]);
        medoid_bufs.push_back(new T[_data_dim]);
        nbr_bufs.emplace_back(0, nullptr);
    }

    auto read_status = read_nodes(nodes_to_read, medoid_bufs, nbr_bufs);

    for (uint64_t cur_m = 0; cur_m < _num_medoids; cur_m++)
    {
        if (read_status[cur_m] == true)
        {
            if (!_use_disk_index_pq)
            {
                for (uint32_t i = 0; i < _data_dim; i++)
                    _centroid_data[cur_m * _aligned_dim + i] = medoid_bufs[cur_m][i];
            }
            else
            {
                _disk_pq_table.inflate_vector((uint8_t *)medoid_bufs[cur_m], (_centroid_data + cur_m * _aligned_dim));
            }
        }
        else
        {
            throw ANNException("Unable to read a medoid", -1, __FUNCSIG__, __FILE__, __LINE__);
        }
        delete[] medoid_bufs[cur_m];
    }
}

template <typename T, typename LabelT>
void PQFlashIndex<T, LabelT>::generate_random_labels(std::vector<LabelT> &labels, const uint32_t num_labels,
                                                     const uint32_t nthreads)
{
    std::random_device rd;
    labels.clear();
    labels.resize(num_labels);

    uint64_t num_total_labels = _pts_to_label_offsets[_num_points - 1] + _pts_to_label_counts[_num_points - 1];
    std::mt19937 gen(rd());
    if (num_total_labels == 0)
    {
        std::stringstream stream;
        stream << "No labels found in data. Not sampling random labels ";
        diskann::cerr << stream.str() << std::endl;
        throw diskann::ANNException(stream.str(), -1, __FUNCSIG__, __FILE__, __LINE__);
    }
    std::uniform_int_distribution<uint64_t> dis(0, num_total_labels - 1);

#pragma omp parallel for schedule(dynamic, 1) num_threads(nthreads)
    for (int64_t i = 0; i < num_labels; i++)
    {
        uint64_t rnd_loc = dis(gen);
        labels[i] = (LabelT)_pts_to_labels[rnd_loc];
    }
}

template <typename T, typename LabelT>
std::unordered_map<std::string, LabelT> PQFlashIndex<T, LabelT>::load_label_map(std::basic_istream<char> &map_reader)
{
    std::unordered_map<std::string, LabelT> string_to_int_mp;
    std::string line, token;
    LabelT token_as_num;
    std::string label_str;
    while (std::getline(map_reader, line))
    {
        std::istringstream iss(line);
        getline(iss, token, '\t');
        label_str = token;
        getline(iss, token, '\t');
        token_as_num = (LabelT)std::stoul(token);
        string_to_int_mp[label_str] = token_as_num;
    }
    return string_to_int_mp;
}

template <typename T, typename LabelT>
LabelT PQFlashIndex<T, LabelT>::get_converted_label(const std::string &filter_label)
{
    if (_label_map.find(filter_label) != _label_map.end())
    {
        return _label_map[filter_label];
    }
    if (_use_universal_label)
    {
        return _universal_filter_label;
    }
    std::stringstream stream;
    stream << "Unable to find label in the Label Map";
    diskann::cerr << stream.str() << std::endl;
    throw diskann::ANNException(stream.str(), -1, __FUNCSIG__, __FILE__, __LINE__);
}

template <typename T, typename LabelT>
void PQFlashIndex<T, LabelT>::reset_stream_for_reading(std::basic_istream<char> &infile)
{
    infile.clear();
    infile.seekg(0);
}

template <typename T, typename LabelT>
void PQFlashIndex<T, LabelT>::get_label_file_metadata(const std::string &fileContent, uint32_t &num_pts,
                                                      uint32_t &num_total_labels)
{
    num_pts = 0;
    num_total_labels = 0;

    size_t file_size = fileContent.length();

    std::string label_str;
    size_t cur_pos = 0;
    size_t next_pos = 0;
    while (cur_pos < file_size && cur_pos != std::string::npos)
    {
        next_pos = fileContent.find('\n', cur_pos);
        if (next_pos == std::string::npos)
        {
            break;
        }

        size_t lbl_pos = cur_pos;
        size_t next_lbl_pos = 0;
        while (lbl_pos < next_pos && lbl_pos != std::string::npos)
        {
            next_lbl_pos = fileContent.find(',', lbl_pos);
            if (next_lbl_pos == std::string::npos) // the last label
            {
                next_lbl_pos = next_pos;
            }

            num_total_labels++;

            lbl_pos = next_lbl_pos + 1;
        }

        cur_pos = next_pos + 1;

        num_pts++;
    }

    diskann::cout << "Labels file metadata: num_points: " << num_pts << ", #total_labels: " << num_total_labels
                  << std::endl;
}

template <typename T, typename LabelT>
inline bool PQFlashIndex<T, LabelT>::point_has_label(uint32_t point_id, LabelT label_id)
{
    uint32_t start_vec = _pts_to_label_offsets[point_id];
    uint32_t num_lbls = _pts_to_label_counts[point_id];
    bool ret_val = false;
    for (uint32_t i = 0; i < num_lbls; i++)
    {
        if (_pts_to_labels[start_vec + i] == label_id)
        {
            ret_val = true;
            break;
        }
    }
    return ret_val;
}

template <typename T, typename LabelT>
void PQFlashIndex<T, LabelT>::parse_label_file(std::basic_istream<char> &infile, size_t &num_points_labels)
{
    infile.seekg(0, std::ios::end);
    size_t file_size = infile.tellg();

    std::string buffer(file_size, ' ');

    infile.seekg(0, std::ios::beg);
    infile.read(&buffer[0], file_size);

    std::string line;
    uint32_t line_cnt = 0;

    uint32_t num_pts_in_label_file;
    uint32_t num_total_labels;
    get_label_file_metadata(buffer, num_pts_in_label_file, num_total_labels);

    _pts_to_label_offsets = new uint32_t[num_pts_in_label_file];
    _pts_to_label_counts = new uint32_t[num_pts_in_label_file];
    _pts_to_labels = new LabelT[num_total_labels];
    uint32_t labels_seen_so_far = 0;

    std::string label_str;
    size_t cur_pos = 0;
    size_t next_pos = 0;
    while (cur_pos < file_size && cur_pos != std::string::npos)
    {
        next_pos = buffer.find('\n', cur_pos);
        if (next_pos == std::string::npos)
        {
            break;
        }

        _pts_to_label_offsets[line_cnt] = labels_seen_so_far;
        uint32_t &num_lbls_in_cur_pt = _pts_to_label_counts[line_cnt];
        num_lbls_in_cur_pt = 0;

        size_t lbl_pos = cur_pos;
        size_t next_lbl_pos = 0;
        while (lbl_pos < next_pos && lbl_pos != std::string::npos)
        {
            next_lbl_pos = buffer.find(',', lbl_pos);
            if (next_lbl_pos == std::string::npos) // the last label in the whole file
            {
                next_lbl_pos = next_pos;
            }

            if (next_lbl_pos > next_pos) // the last label in one line, just read to the end
            {
                next_lbl_pos = next_pos;
            }

            label_str.assign(buffer.c_str() + lbl_pos, next_lbl_pos - lbl_pos);
            if (label_str[label_str.length() - 1] == '\t') // '\t' won't exist in label file?
            {
                label_str.erase(label_str.length() - 1);
            }

            LabelT token_as_num = (LabelT)std::stoul(label_str);
            _pts_to_labels[labels_seen_so_far++] = (LabelT)token_as_num;
            num_lbls_in_cur_pt++;

            // move to next label
            lbl_pos = next_lbl_pos + 1;
        }

        // move to next line
        cur_pos = next_pos + 1;

        if (num_lbls_in_cur_pt == 0)
        {
            diskann::cout << "No label found for point " << line_cnt << std::endl;
            exit(-1);
        }

        line_cnt++;
    }

    num_points_labels = line_cnt;
    reset_stream_for_reading(infile);
}

template <typename T, typename LabelT> void PQFlashIndex<T, LabelT>::set_universal_label(const LabelT &label)
{
    _use_universal_label = true;
    _universal_filter_label = label;
}

#ifdef EXEC_ENV_OLS
template <typename T, typename LabelT>
int PQFlashIndex<T, LabelT>::load(MemoryMappedFiles &files, uint32_t num_threads, const char *index_prefix)
{
#else
// 本质上通过磁盘index和PQ数据构建内存中PQ index
template <typename T, typename LabelT> int PQFlashIndex<T, LabelT>::load(uint32_t num_threads, const char *index_prefix)
{
#endif
    std::string pq_table_bin = std::string(index_prefix) + "_pq_pivots.bin";
    std::string pq_compressed_vectors = std::string(index_prefix) + "_pq_compressed.bin";
    std::string _disk_index_file = std::string(index_prefix) + "_disk.index";
#ifdef EXEC_ENV_OLS
    return load_from_separate_paths(files, num_threads, _disk_index_file.c_str(), pq_table_bin.c_str(),
                                    pq_compressed_vectors.c_str());
#else
    return load_from_separate_paths(num_threads, _disk_index_file.c_str(), pq_table_bin.c_str(),
                                    pq_compressed_vectors.c_str());
#endif
}

#ifdef EXEC_ENV_OLS
template <typename T, typename LabelT>
int PQFlashIndex<T, LabelT>::load_from_separate_paths(diskann::MemoryMappedFiles &files, uint32_t num_threads,
                                                      const char *index_filepath, const char *pivots_filepath,
                                                      const char *compressed_filepath)
{
#else
// 本质上这里就是从build_disk中保存的文件进行读取
template <typename T, typename LabelT>
int PQFlashIndex<T, LabelT>::load_from_separate_paths(uint32_t num_threads, const char *index_filepath,
                                                      const char *pivots_filepath, const char *compressed_filepath)
{
#endif
    std::string pq_table_bin = pivots_filepath;
    std::string pq_compressed_vectors = compressed_filepath;
    std::string _disk_index_file = index_filepath;
    std::string medoids_file = std::string(_disk_index_file) + "_medoids.bin";
    std::string centroids_file = std::string(_disk_index_file) + "_centroids.bin";

    std::string labels_file = std ::string(_disk_index_file) + "_labels.txt";
    std::string labels_to_medoids = std ::string(_disk_index_file) + "_labels_to_medoids.txt";
    std::string dummy_map_file = std ::string(_disk_index_file) + "_dummy_map.txt";
    std::string labels_map_file = std ::string(_disk_index_file) + "_labels_map.txt";
    size_t num_pts_in_label_file = 0;

    size_t pq_file_dim, pq_file_num_centroids;
#ifdef EXEC_ENV_OLS
    get_bin_metadata(files, pq_table_bin, pq_file_num_centroids, pq_file_dim, METADATA_SIZE);
#else
    get_bin_metadata(pq_table_bin, pq_file_num_centroids, pq_file_dim, METADATA_SIZE);
#endif

    this->_disk_index_file = _disk_index_file;

    if (pq_file_num_centroids != 256)
    {
        diskann::cout << "Error. Number of PQ centroids is not 256. Exiting." << std::endl;
        return -1;
    }

    this->_data_dim = pq_file_dim;
    // will change later if we use PQ on disk or if we are using
    // inner product without PQ
    this->_disk_bytes_per_point = this->_data_dim * sizeof(T);
    this->_aligned_dim = ROUND_UP(pq_file_dim, 8);

    size_t npts_u64, nchunks_u64;
#ifdef EXEC_ENV_OLS
    diskann::load_bin<uint8_t>(files, pq_compressed_vectors, this->data, npts_u64, nchunks_u64);
#else
    diskann::load_bin<uint8_t>(pq_compressed_vectors, this->data, npts_u64, nchunks_u64);
#endif

    this->_num_points = npts_u64;
    this->_n_chunks = nchunks_u64;
#ifdef EXEC_ENV_OLS
    if (files.fileExists(labels_file))
    {
        FileContent &content_labels = files.getContent(labels_file);
        std::stringstream infile(std::string((const char *)content_labels._content, content_labels._size));
#else
    if (file_exists(labels_file))
    {
        std::ifstream infile(labels_file, std::ios::binary);
        if (infile.fail())
        {
            throw diskann::ANNException(std::string("Failed to open file ") + labels_file, -1);
        }
#endif
        parse_label_file(infile, num_pts_in_label_file);
        assert(num_pts_in_label_file == this->_num_points);

#ifndef EXEC_ENV_OLS
        infile.close();
#endif

#ifdef EXEC_ENV_OLS
        FileContent &content_labels_map = files.getContent(labels_map_file);
        std::stringstream map_reader(std::string((const char *)content_labels_map._content, content_labels_map._size));
#else
        std::ifstream map_reader(labels_map_file);
#endif
        _label_map = load_label_map(map_reader);

#ifndef EXEC_ENV_OLS
        map_reader.close();
#endif

#ifdef EXEC_ENV_OLS
        if (files.fileExists(labels_to_medoids))
        {
            FileContent &content_labels_to_meoids = files.getContent(labels_to_medoids);
            std::stringstream medoid_stream(
                std::string((const char *)content_labels_to_meoids._content, content_labels_to_meoids._size));
#else
        if (file_exists(labels_to_medoids))
        {
            std::ifstream medoid_stream(labels_to_medoids);
            assert(medoid_stream.is_open());
#endif
            std::string line, token;

            _filter_to_medoid_ids.clear();
            try
            {
                while (std::getline(medoid_stream, line))
                {
                    std::istringstream iss(line);
                    uint32_t cnt = 0;
                    std::vector<uint32_t> medoids;
                    LabelT label;
                    while (std::getline(iss, token, ','))
                    {
                        if (cnt == 0)
                            label = (LabelT)std::stoul(token);
                        else
                            medoids.push_back((uint32_t)stoul(token));
                        cnt++;
                    }
                    _filter_to_medoid_ids[label].swap(medoids);
                }
            }
            catch (std::system_error &e)
            {
                throw FileException(labels_to_medoids, e, __FUNCSIG__, __FILE__, __LINE__);
            }
        }
        std::string univ_label_file = std ::string(_disk_index_file) + "_universal_label.txt";

#ifdef EXEC_ENV_OLS
        if (files.fileExists(univ_label_file))
        {
            FileContent &content_univ_label = files.getContent(univ_label_file);
            std::stringstream universal_label_reader(
                std::string((const char *)content_univ_label._content, content_univ_label._size));
#else
        if (file_exists(univ_label_file))
        {
            std::ifstream universal_label_reader(univ_label_file);
            assert(universal_label_reader.is_open());
#endif
            std::string univ_label;
            universal_label_reader >> univ_label;
#ifndef EXEC_ENV_OLS
            universal_label_reader.close();
#endif
            LabelT label_as_num = (LabelT)std::stoul(univ_label);
            set_universal_label(label_as_num);
        }

#ifdef EXEC_ENV_OLS
        if (files.fileExists(dummy_map_file))
        {
            FileContent &content_dummy_map = files.getContent(dummy_map_file);
            std::stringstream dummy_map_stream(
                std::string((const char *)content_dummy_map._content, content_dummy_map._size));
#else
        if (file_exists(dummy_map_file))
        {
            std::ifstream dummy_map_stream(dummy_map_file);
            assert(dummy_map_stream.is_open());
#endif
            std::string line, token;

            while (std::getline(dummy_map_stream, line))
            {
                std::istringstream iss(line);
                uint32_t cnt = 0;
                uint32_t dummy_id;
                uint32_t real_id;
                while (std::getline(iss, token, ','))
                {
                    if (cnt == 0)
                        dummy_id = (uint32_t)stoul(token);
                    else
                        real_id = (uint32_t)stoul(token);
                    cnt++;
                }
                _dummy_pts.insert(dummy_id);
                _has_dummy_pts.insert(real_id);
                _dummy_to_real_map[dummy_id] = real_id;

                if (_real_to_dummy_map.find(real_id) == _real_to_dummy_map.end())
                    _real_to_dummy_map[real_id] = std::vector<uint32_t>();

                _real_to_dummy_map[real_id].emplace_back(dummy_id);
            }
#ifndef EXEC_ENV_OLS
            dummy_map_stream.close();
#endif
            diskann::cout << "Loaded dummy map" << std::endl;
        }
    }

#ifdef EXEC_ENV_OLS
    _pq_table.load_pq_centroid_bin(files, pq_table_bin.c_str(), nchunks_u64);
#else
    _pq_table.load_pq_centroid_bin(pq_table_bin.c_str(), nchunks_u64);
#endif

    diskann::cout << "Loaded PQ centroids and in-memory compressed vectors. #points: " << _num_points
                  << " #dim: " << _data_dim << " #aligned_dim: " << _aligned_dim << " #chunks: " << _n_chunks
                  << std::endl;

    if (_n_chunks > MAX_PQ_CHUNKS)
    {
        std::stringstream stream;
        stream << "Error loading index. Ensure that max PQ bytes for in-memory "
                  "PQ data does not exceed "
               << MAX_PQ_CHUNKS << std::endl;
        throw diskann::ANNException(stream.str(), -1, __FUNCSIG__, __FILE__, __LINE__);
    }

    std::string disk_pq_pivots_path = this->_disk_index_file + "_pq_pivots.bin";
#ifdef EXEC_ENV_OLS
    if (files.fileExists(disk_pq_pivots_path))
    {
        _use_disk_index_pq = true;
        // giving 0 chunks to make the _pq_table infer from the
        // chunk_offsets file the correct value
        _disk_pq_table.load_pq_centroid_bin(files, disk_pq_pivots_path.c_str(), 0);
#else
    if (file_exists(disk_pq_pivots_path)) // 磁盘中就使用PQ数据的情况
    {
        _use_disk_index_pq = true;
        // giving 0 chunks to make the _pq_table infer from the
        // chunk_offsets file the correct value
        _disk_pq_table.load_pq_centroid_bin(disk_pq_pivots_path.c_str(), 0);
#endif
        _disk_pq_n_chunks = _disk_pq_table.get_num_chunks();
        _disk_bytes_per_point =
            _disk_pq_n_chunks * sizeof(uint8_t); // revising disk_bytes_per_point since DISK PQ is used.
        diskann::cout << "Disk index uses PQ data compressed down to " << _disk_pq_n_chunks << " bytes per point."
                      << std::endl;
    }

// read index metadata
#ifdef EXEC_ENV_OLS
    // This is a bit tricky. We have to read the header from the
    // disk_index_file. But  this is now exclusively a preserve of the
    // DiskPriorityIO class. So, we need to estimate how many
    // bytes are needed to store the header and read in that many using our
    // 'standard' aligned file reader approach.
    reader->open(_disk_index_file);
    this->setup_thread_data(num_threads);
    this->_max_nthreads = num_threads;

    char *bytes = getHeaderBytes();
    ContentBuf buf(bytes, HEADER_SIZE);
    std::basic_istream<char> index_metadata(&buf);
#else
    std::ifstream index_metadata(_disk_index_file, std::ios::binary);
#endif

    uint32_t nr, nc; // metadata itself is stored as bin format (nr is number of
                     // metadata, nc should be 1)
    READ_U32(index_metadata, nr);
    READ_U32(index_metadata, nc);

    uint64_t disk_nnodes;
    uint64_t disk_ndims; // can be disk PQ dim if disk_PQ is set to true
    READ_U64(index_metadata, disk_nnodes);
    READ_U64(index_metadata, disk_ndims);

    if (disk_nnodes != _num_points)
    {
        diskann::cout << "Mismatch in #points for compressed data file and disk "
                         "index file: "
                      << disk_nnodes << " vs " << _num_points << std::endl;
        return -1;
    }

    size_t medoid_id_on_file;
    READ_U64(index_metadata, medoid_id_on_file);
    READ_U64(index_metadata, _max_node_len);
    READ_U64(index_metadata, _nnodes_per_sector);
    _max_degree = ((_max_node_len - _disk_bytes_per_point) / sizeof(uint32_t)) - 1;

    if (_max_degree > defaults::MAX_GRAPH_DEGREE)
    {
        std::stringstream stream;
        stream << "Error loading index. Ensure that max graph degree (R) does "
                  "not exceed "
               << defaults::MAX_GRAPH_DEGREE << std::endl;
        throw diskann::ANNException(stream.str(), -1, __FUNCSIG__, __FILE__, __LINE__);
    }

    // setting up concept of frozen points in disk index for streaming-DiskANN
    READ_U64(index_metadata, this->_num_frozen_points);
    uint64_t file_frozen_id;
    READ_U64(index_metadata, file_frozen_id);
    if (this->_num_frozen_points == 1)
        this->_frozen_location = file_frozen_id;
    if (this->_num_frozen_points == 1)
    {
        diskann::cout << " Detected frozen point in index at location " << this->_frozen_location
                      << ". Will not output it at search time." << std::endl;
    }

    READ_U64(index_metadata, this->_reorder_data_exists);
    if (this->_reorder_data_exists)
    {
        if (this->_use_disk_index_pq == false)
        {
            throw ANNException("Reordering is designed for used with disk PQ "
                               "compression option",
                               -1, __FUNCSIG__, __FILE__, __LINE__);
        }
        READ_U64(index_metadata, this->_reorder_data_start_sector);
        READ_U64(index_metadata, this->_ndims_reorder_vecs);
        READ_U64(index_metadata, this->_nvecs_per_sector);
    }

    diskann::cout << "Disk-Index File Meta-data: ";
    diskann::cout << "# nodes per sector: " << _nnodes_per_sector;
    diskann::cout << ", max node len (bytes): " << _max_node_len;
    diskann::cout << ", max node degree: " << _max_degree << std::endl;

#ifdef EXEC_ENV_OLS
    delete[] bytes;
#else
    index_metadata.close();
#endif

#ifndef EXEC_ENV_OLS
    // open AlignedFileReader handle to index_file
    // 设置线程数, 这里setup了thread_scratch
    std::string index_fname(_disk_index_file);
    reader->open(index_fname);
    this->setup_thread_data(num_threads);
    this->_max_nthreads = num_threads;

#endif

#ifdef EXEC_ENV_OLS
    if (files.fileExists(medoids_file))
    {
        size_t tmp_dim;
        diskann::load_bin<uint32_t>(files, medoids_file, _medoids, _num_medoids, tmp_dim);
#else
    if (file_exists(medoids_file))
    {
        size_t tmp_dim;
        diskann::load_bin<uint32_t>(medoids_file, _medoids, _num_medoids, tmp_dim);
#endif

        if (tmp_dim != 1)
        {
            std::stringstream stream;
            stream << "Error loading medoids file. Expected bin format of m times "
                      "1 vector of uint32_t."
                   << std::endl;
            throw diskann::ANNException(stream.str(), -1, __FUNCSIG__, __FILE__, __LINE__);
        }
#ifdef EXEC_ENV_OLS
        if (!files.fileExists(centroids_file))
        {
#else
        if (!file_exists(centroids_file))
        {
#endif
            diskann::cout << "Centroid data file not found. Using corresponding vectors "
                             "for the medoids "
                          << std::endl;
            use_medoids_data_as_centroids();
        }
        else
        {
            size_t num_centroids, aligned_tmp_dim;
#ifdef EXEC_ENV_OLS
            diskann::load_aligned_bin<float>(files, centroids_file, _centroid_data, num_centroids, tmp_dim,
                                             aligned_tmp_dim);
#else
            diskann::load_aligned_bin<float>(centroids_file, _centroid_data, num_centroids, tmp_dim, aligned_tmp_dim);
#endif
            if (aligned_tmp_dim != _aligned_dim || num_centroids != _num_medoids)
            {
                std::stringstream stream;
                stream << "Error loading centroids data file. Expected bin format "
                          "of "
                          "m times data_dim vector of float, where m is number of "
                          "medoids "
                          "in medoids file.";
                diskann::cerr << stream.str() << std::endl;
                throw diskann::ANNException(stream.str(), -1, __FUNCSIG__, __FILE__, __LINE__);
            }
        }
    }
    else
    {
        _num_medoids = 1;
        _medoids = new uint32_t[1];
        _medoids[0] = (uint32_t)(medoid_id_on_file);
        use_medoids_data_as_centroids();
    }

    std::string norm_file = std::string(_disk_index_file) + "_max_base_norm.bin";

#ifdef EXEC_ENV_OLS
    if (files.fileExists(norm_file) && metric == diskann::Metric::INNER_PRODUCT)
    {
        uint64_t dumr, dumc;
        float *norm_val;
        diskann::load_bin<float>(files, norm_val, dumr, dumc);
#else
    if (file_exists(norm_file) && metric == diskann::Metric::INNER_PRODUCT)
    {
        uint64_t dumr, dumc;
        float *norm_val;
        diskann::load_bin<float>(norm_file, norm_val, dumr, dumc);
#endif
        this->_max_base_norm = norm_val[0];
        diskann::cout << "Setting re-scaling factor of base vectors to " << this->_max_base_norm << std::endl;
        delete[] norm_val;
    }
    diskann::cout << "done.." << std::endl;
    return 0;
}

#ifdef USE_BING_INFRA
bool getNextCompletedRequest(const IOContext &ctx, size_t size, int &completedIndex)
{
    bool waitsRemaining = false;
    long completeCount = ctx.m_completeCount;
    do
    {
        for (int i = 0; i < size; i++)
        {
            auto ithStatus = (*ctx.m_pRequestsStatus)[i];
            if (ithStatus == IOContext::Status::READ_SUCCESS)
            {
                completedIndex = i;
                return true;
            }
            else if (ithStatus == IOContext::Status::READ_WAIT)
            {
                waitsRemaining = true;
            }
        }

        // if we didn't find one in READ_SUCCESS, wait for one to complete.
        if (waitsRemaining)
        {
            WaitOnAddress(&ctx.m_completeCount, &completeCount, sizeof(completeCount), 100);
            // this assumes the knowledge of the reader behavior (implicit
            // contract). need better factoring?
        }
    } while (waitsRemaining);

    completedIndex = -1;
    return false;
}
#endif

template <typename T, typename LabelT>
void PQFlashIndex<T, LabelT>::cached_beam_search(const T *query1, const uint64_t k_search, const uint64_t l_search,
                                                 uint64_t *indices, float *distances, const uint64_t beam_width,
                                                 const bool use_reorder_data, QueryStats *stats)
{
    cached_beam_search(query1, k_search, l_search, indices, distances, beam_width, std::numeric_limits<uint32_t>::max(),
                       use_reorder_data, stats);
}

template <typename T, typename LabelT>
void PQFlashIndex<T, LabelT>::cached_beam_search(const T *query1, const uint64_t k_search, const uint64_t l_search,
                                                 uint64_t *indices, float *distances, const uint64_t beam_width,
                                                 const bool use_filter, const LabelT &filter_label,
                                                 const bool use_reorder_data, QueryStats *stats)
{
    cached_beam_search(query1, k_search, l_search, indices, distances, beam_width, use_filter, filter_label,
                       std::numeric_limits<uint32_t>::max(), use_reorder_data, stats);
}

template <typename T, typename LabelT>
void PQFlashIndex<T, LabelT>::cached_beam_search(const T *query1, const uint64_t k_search, const uint64_t l_search,
                                                 uint64_t *indices, float *distances, const uint64_t beam_width,
                                                 const uint32_t io_limit, const bool use_reorder_data,
                                                 QueryStats *stats)
{
    LabelT dummy_filter = 0;
    cached_beam_search(query1, k_search, l_search, indices, distances, beam_width, false, dummy_filter, io_limit,
                       use_reorder_data, stats);
}

/** 真正的cached beam search都落在了这里, indices就是result id, distance就是对应的distance
 *      1. 对数据进行预处理, 并且获取pq_table
 *      2. 从最近邻的memoid点出发
 *      3. 不断扩展节点, 但是节点分为两部分:
 *          3.1 缓存命中节点 cached, 在fullset中计算精确距离, 邻居计算PQ距离
 *          3.2 缓存未命中节点 frontier(被beamwidth大小限制), 同样如上操作, 但是多了一步磁盘请求
 *
 */
template <typename T, typename LabelT>
void PQFlashIndex<T, LabelT>::cached_beam_search(const T *query1, const uint64_t k_search, const uint64_t l_search,
                                                 uint64_t *indices, float *distances, const uint64_t beam_width,
                                                 const bool use_filter, const LabelT &filter_label,
                                                 const uint32_t io_limit, const bool use_reorder_data,
                                                 QueryStats *stats)
{

    // 每个节点占用的sector数目
    uint64_t num_sector_per_nodes = DIV_ROUND_UP(_max_node_len, defaults::SECTOR_LEN);
    if (beam_width > num_sector_per_nodes * defaults::MAX_N_SECTOR_READS)
        throw ANNException("Beamwidth can not be higher than defaults::MAX_N_SECTOR_READS", -1, __FUNCSIG__, __FILE__,
                           __LINE__);

    // read request
    ScratchStoreManager<SSDThreadData<T>> manager(this->_thread_data);
    auto data = manager.scratch_space();
    IOContext &ctx = data->ctx;
    auto query_scratch = &(data->scratch);
    auto pq_query_scratch = query_scratch->pq_scratch();

    // reset query scratch
    query_scratch->reset();

    // copy query to thread specific aligned and allocated memory (for distance
    // calculations we need aligned data)
    // 这里都相当于时获取query scratch的内存指针
    float query_norm = 0;
    T *aligned_query_T = query_scratch->aligned_query_T();
    float *query_float = pq_query_scratch->aligned_query_float;
    float *query_rotated = pq_query_scratch->rotated_query;

    // if inner product, we laso normalize the query and set the last coordinate
    // to 0 (this is the extra coordindate used to convert MIPS to L2 search)
    if (metric == diskann::Metric::INNER_PRODUCT)
    {
        for (size_t i = 0; i < this->_data_dim - 1; i++)
        {
            aligned_query_T[i] = query1[i];
            query_norm += query1[i] * query1[i];
        }
        aligned_query_T[this->_data_dim - 1] = 0;

        query_norm = std::sqrt(query_norm);

        for (size_t i = 0; i < this->_data_dim - 1; i++)
        {
            aligned_query_T[i] = (T)(aligned_query_T[i] / query_norm);
        }
        pq_query_scratch->initialize(this->_data_dim, aligned_query_T);
    }
    else
    {
        // copy query数据 => aligned_query_T中
        for (size_t i = 0; i < this->_data_dim; i++)
        {
            aligned_query_T[i] = query1[i];
        }
        // scratch初始化这个query
        // 这里初始化之后相当于将数据从aligned_query_T => rotated_query_float中
        pq_query_scratch->initialize(this->_data_dim, aligned_query_T);
    }

    // pointers to buffers for data
    // 数据的buff
    T *data_buf = query_scratch->coord_scratch;
    _mm_prefetch((char *)data_buf, _MM_HINT_T1);

    // sector scratch
    char *sector_scratch = query_scratch->sector_scratch;
    uint64_t &sector_scratch_idx = query_scratch->sector_idx;
    const uint64_t num_sectors_per_node =
        _nnodes_per_sector > 0 ? 1 : DIV_ROUND_UP(_max_node_len, defaults::SECTOR_LEN);

    // query <-> PQ chunk centers distances
    // query_rotated中的数据和query数据一致, 只不过它一直都是float对齐类型
    //! 这里首先对query进行处理, 0均值或者旋转处理
    _pq_table.preprocess_query(query_rotated); // center the query and rotate if
                                               // we have a rotation matrix
    // 将pq_dist指针赋值, 方便后面对之进行操作
    float *pq_dists = pq_query_scratch->aligned_pqtable_dist_scratch;
    //! 这里就是计算pq_table
    _pq_table.populate_chunk_distances(query_rotated, pq_dists);

    // query <-> neighbor list
    float *dist_scratch = pq_query_scratch->aligned_dist_scratch;
    uint8_t *pq_coord_scratch = pq_query_scratch->aligned_pq_coord_scratch;

    // lambda to batch compute query<-> node distances in PQ space
    // 计算pq空间下的距离
    auto compute_dists = [this, pq_coord_scratch, pq_dists](const uint32_t *ids, const uint64_t n_ids,
                                                            float *dists_out) {
        diskann::aggregate_coords(ids, n_ids, this->data, this->_n_chunks, pq_coord_scratch); // 将pq数据拷贝到pq_coord_scratch中
        diskann::pq_dist_lookup(pq_coord_scratch, n_ids, this->_n_chunks, pq_dists, dists_out); // 计算asynic pq  distance with coord
    };
    Timer query_timer, io_timer, cpu_timer;

    tsl::robin_set<uint64_t> &visited = query_scratch->visited;
    NeighborPriorityQueue &retset = query_scratch->retset; // 存储邻居距离, PQ距离, 指导查询方向
    retset.reserve(l_search);
    std::vector<Neighbor> &full_retset = query_scratch->full_retset; // 存储最后全精度结果

    uint32_t best_medoid = 0;
    float best_dist = (std::numeric_limits<float>::max)();
    if (!use_filter)
    {
        // 这里相当于计算最紧邻的一个memoid
        for (uint64_t cur_m = 0; cur_m < _num_medoids; cur_m++)
        {
            float cur_expanded_dist =
                _dist_cmp_float->compare(query_float, _centroid_data + _aligned_dim * cur_m, (uint32_t)_aligned_dim);
            if (cur_expanded_dist < best_dist)
            {
                best_medoid = _medoids[cur_m];
                best_dist = cur_expanded_dist;
            }
        }
    }
    else
    {
        if (_filter_to_medoid_ids.find(filter_label) != _filter_to_medoid_ids.end())
        {
            const auto &medoid_ids = _filter_to_medoid_ids[filter_label];
            for (uint64_t cur_m = 0; cur_m < medoid_ids.size(); cur_m++)
            {
                // for filtered index, we dont store global centroid data as for unfiltered index, so we use PQ distance
                // as approximation to decide closest medoid matching the query filter.
                compute_dists(&medoid_ids[cur_m], 1, dist_scratch);
                float cur_expanded_dist = dist_scratch[0];
                if (cur_expanded_dist < best_dist)
                {
                    best_medoid = medoid_ids[cur_m];
                    best_dist = cur_expanded_dist;
                }
            }
        }
        else
        {
            throw ANNException("Cannot find medoid for specified filter.", -1, __FUNCSIG__, __FILE__, __LINE__);
        }
    }

    // 计算从best_memoid开始的距离
    compute_dists(&best_medoid, 1, dist_scratch);
    // 加入到结果集中
    retset.insert(Neighbor(best_medoid, dist_scratch[0]));
    // 加入到visited集中
    visited.insert(best_medoid);

    uint32_t cmps = 0;
    uint32_t hops = 0;
    uint32_t num_ios = 0;

    // cleared every iteration
    std::vector<uint32_t> frontier;
    frontier.reserve(2 * beam_width);
    std::vector<std::pair<uint32_t, char *>> frontier_nhoods;
    frontier_nhoods.reserve(2 * beam_width);
    std::vector<AlignedRead> frontier_read_reqs;
    frontier_read_reqs.reserve(2 * beam_width);
    std::vector<std::pair<uint32_t, std::pair<uint32_t, uint32_t *>>> cached_nhoods;
    cached_nhoods.reserve(2 * beam_width);

    // 如果存在没有扩展的节点, 并且io没有达到限制
    while (retset.has_unexpanded_node() && num_ios < io_limit)
    {
        // clear iteration state
        frontier.clear();
        frontier_nhoods.clear();
        frontier_read_reqs.clear();
        cached_nhoods.clear();
        sector_scratch_idx = 0;
        // find new beam
        uint32_t num_seen = 0;
        // 也就是说没有缓存命中的点达到一定的数量就会跳出循环
        while (retset.has_unexpanded_node() && frontier.size() < beam_width && num_seen < beam_width)
        {
            // 返回一个没有被expand的point
            auto nbr = retset.closest_unexpanded();
            num_seen++;
            // 在缓存中查找这个节点
            auto iter = _nhood_cache.find(nbr.id);
            // 缓存命中
            if (iter != _nhood_cache.end())
            {
                cached_nhoods.push_back(std::make_pair(nbr.id, iter->second));
                if (stats != nullptr)
                {
                    stats->n_cache_hits++;
                }
            } // 缓存未命中存入frontier中
            else
            {
                frontier.push_back(nbr.id);
            }
            if (this->_count_visited_nodes) // 计数visited_nodes
            {
                reinterpret_cast<std::atomic<uint32_t> &>(this->_node_visit_counter[nbr.id].second).fetch_add(1);
            }
        }

        // read nhoods of frontier ids, 对frontier中未命中的数据进行操作
        if (!frontier.empty())
        {
            if (stats != nullptr)
                stats->n_hops++;
            for (uint64_t i = 0; i < frontier.size(); i++)
            {
                // 对frontier中的id进行读取调用
                auto id = frontier[i];
                std::pair<uint32_t, char *> fnhood;
                fnhood.first = id;
                fnhood.second = sector_scratch + num_sectors_per_node * sector_scratch_idx * defaults::SECTOR_LEN;
                sector_scratch_idx++;
                frontier_nhoods.push_back(fnhood);
                frontier_read_reqs.emplace_back(get_node_sector((size_t)id) * defaults::SECTOR_LEN,
                                                num_sectors_per_node * defaults::SECTOR_LEN, fnhood.second);
                if (stats != nullptr)
                {
                    stats->n_4k++;
                    stats->n_ios++;
                }
                // io次数增加
                num_ios++;
            }
            io_timer.reset();
#ifdef USE_BING_INFRA   // 发起异步读请求
            reader->read(frontier_read_reqs, ctx,
                         true); // asynhronous reader for Bing.
#else
            reader->read(frontier_read_reqs, ctx); // synchronous IO linux
#endif
            if (stats != nullptr)
            {
                stats->io_us += (float)io_timer.elapsed();
            }
        }

        // process cached nhoods
        // 处理刚刚缓存的节点
        for (auto &cached_nhood : cached_nhoods)
        {
            // 这里的节点一定都能找到
            auto global_cache_iter = _coord_cache.find(cached_nhood.first);
            T *node_fp_coords_copy = global_cache_iter->second;
            float cur_expanded_dist;
            if (!_use_disk_index_pq)
            {
                // 计算当前扩展节点的实际距离(非PQ距离)
                cur_expanded_dist = _dist_cmp->compare(aligned_query_T, node_fp_coords_copy, (uint32_t)_aligned_dim);
            }
            else
            {
                if (metric == diskann::Metric::INNER_PRODUCT)
                    cur_expanded_dist = _disk_pq_table.inner_product(query_float, (uint8_t *)node_fp_coords_copy);
                else
                    cur_expanded_dist = _disk_pq_table.l2_distance( // disk_pq does not support OPQ yet
                        query_float, (uint8_t *)node_fp_coords_copy);
            }
            // 将当前节点加入到全部的结果集
            full_retset.push_back(Neighbor((uint32_t)cached_nhood.first, cur_expanded_dist));

            uint64_t nnbrs = cached_nhood.second.first;
            uint32_t *node_nbrs = cached_nhood.second.second;

            // compute node_nbrs <-> query dists in PQ space
            cpu_timer.reset();
            // 计算邻居节点和query的PQ距离
            compute_dists(node_nbrs, nnbrs, dist_scratch);
            if (stats != nullptr)
            {
                stats->n_cmps += (uint32_t)nnbrs;
                stats->cpu_us += (float)cpu_timer.elapsed();
            }

            // process prefetched nhood
            // 处理预取的nhood, 加入到visited中, 以及将dist加入到retset中
            for (uint64_t m = 0; m < nnbrs; ++m)
            {
                uint32_t id = node_nbrs[m];
                if (visited.insert(id).second)
                {
                    if (!use_filter && _dummy_pts.find(id) != _dummy_pts.end())
                        continue;

                    if (use_filter && !(point_has_label(id, filter_label)) &&
                        (!_use_universal_label || !point_has_label(id, _universal_filter_label)))
                        continue;
                    cmps++;
                    float dist = dist_scratch[m];
                    Neighbor nn(id, dist);
                    // 加入到结果集中
                    retset.insert(nn);
                }
            }
        }
#ifdef USE_BING_INFRA
        // process each frontier nhood - compute distances to unvisited nodes
        int completedIndex = -1;
        long requestCount = static_cast<long>(frontier_read_reqs.size());
        // If we issued read requests and if a read is complete or there are
        // reads in wait state, then enter the while loop.
        while (requestCount > 0 && getNextCompletedRequest(ctx, requestCount, completedIndex))
        {
            assert(completedIndex >= 0);
            auto &frontier_nhood = frontier_nhoods[completedIndex];
            (*ctx.m_pRequestsStatus)[completedIndex] = IOContext::PROCESS_COMPLETE;
#else

        // 对未缓存的节点进行操作
        for (auto &frontier_nhood : frontier_nhoods)
        {
#endif
            // 这里相当于定位到了id所对应的sector
            // 那么可以找到它的origin vector, nn, neighbor
            // 所以这里的含义和上述cached_node是一样的
            char *node_disk_buf = offset_to_node(frontier_nhood.second, frontier_nhood.first);
            uint32_t *node_buf = offset_to_node_nhood(node_disk_buf);
            uint64_t nnbrs = (uint64_t)(*node_buf);
            T *node_fp_coords = offset_to_node_coords(node_disk_buf);
            memcpy(data_buf, node_fp_coords, _disk_bytes_per_point);
            float cur_expanded_dist;
            if (!_use_disk_index_pq)
            {
                cur_expanded_dist = _dist_cmp->compare(aligned_query_T, data_buf, (uint32_t)_aligned_dim);
            }
            else
            {
                if (metric == diskann::Metric::INNER_PRODUCT)
                    cur_expanded_dist = _disk_pq_table.inner_product(query_float, (uint8_t *)data_buf);
                else
                    cur_expanded_dist = _disk_pq_table.l2_distance(query_float, (uint8_t *)data_buf);
            }
            full_retset.push_back(Neighbor(frontier_nhood.first, cur_expanded_dist));
            uint32_t *node_nbrs = (node_buf + 1);
            // compute node_nbrs <-> query dist in PQ space
            cpu_timer.reset();
            compute_dists(node_nbrs, nnbrs, dist_scratch);
            if (stats != nullptr)
            {
                stats->n_cmps += (uint32_t)nnbrs;
                stats->cpu_us += (float)cpu_timer.elapsed();
            }

            cpu_timer.reset();
            // process prefetch-ed nhood
            for (uint64_t m = 0; m < nnbrs; ++m)
            {
                uint32_t id = node_nbrs[m];
                if (visited.insert(id).second)
                {
                    if (!use_filter && _dummy_pts.find(id) != _dummy_pts.end())
                        continue;

                    if (use_filter && !(point_has_label(id, filter_label)) &&
                        (!_use_universal_label || !point_has_label(id, _universal_filter_label)))
                        continue;
                    cmps++;
                    float dist = dist_scratch[m];
                    if (stats != nullptr)
                    {
                        stats->n_cmps++;
                    }

                    Neighbor nn(id, dist);
                    retset.insert(nn);
                }
            }

            if (stats != nullptr)
            {
                stats->cpu_us += (float)cpu_timer.elapsed();
            }
        }

        hops++;
    }

    // re-sort by distance
    // 按照距离进行排序上述的距离
    //! full retset放的是全精度距离, retset放的是PQ距离
    std::sort(full_retset.begin(), full_retset.end());

    if (use_reorder_data)
    {
        if (!(this->_reorder_data_exists))
        {
            throw ANNException("Requested use of reordering data which does "
                               "not exist in index "
                               "file",
                               -1, __FUNCSIG__, __FILE__, __LINE__);
        }

        std::vector<AlignedRead> vec_read_reqs;

        if (full_retset.size() > k_search * FULL_PRECISION_REORDER_MULTIPLIER)
            full_retset.erase(full_retset.begin() + k_search * FULL_PRECISION_REORDER_MULTIPLIER, full_retset.end());

        for (size_t i = 0; i < full_retset.size(); ++i)
        {
            // MULTISECTORFIX
            vec_read_reqs.emplace_back(VECTOR_SECTOR_NO(((size_t)full_retset[i].id)) * defaults::SECTOR_LEN,
                                       defaults::SECTOR_LEN, sector_scratch + i * defaults::SECTOR_LEN);

            if (stats != nullptr)
            {
                stats->n_4k++;
                stats->n_ios++;
            }
        }

        io_timer.reset();
#ifdef USE_BING_INFRA
        reader->read(vec_read_reqs, ctx, true); // async reader windows.
#else
        reader->read(vec_read_reqs, ctx); // synchronous IO linux
#endif
        if (stats != nullptr)
        {
            stats->io_us += io_timer.elapsed();
        }

        for (size_t i = 0; i < full_retset.size(); ++i)
        {
            auto id = full_retset[i].id;
            // MULTISECTORFIX
            auto location = (sector_scratch + i * defaults::SECTOR_LEN) + VECTOR_SECTOR_OFFSET(id);
            full_retset[i].distance = _dist_cmp->compare(aligned_query_T, (T *)location, (uint32_t)this->_data_dim);
        }

        std::sort(full_retset.begin(), full_retset.end());
    }

    // copy k_search values
    // 拷贝K个最近邻到indices中
    for (uint64_t i = 0; i < k_search; i++)
    {
        indices[i] = full_retset[i].id;
        auto key = (uint32_t)indices[i];
        if (_dummy_pts.find(key) != _dummy_pts.end())
        {
            indices[i] = _dummy_to_real_map[key];
        }
        // 设置对应点的距离
        if (distances != nullptr)
        {
            distances[i] = full_retset[i].distance;
            if (metric == diskann::Metric::INNER_PRODUCT)
            {
                // flip the sign to convert min to max
                distances[i] = (-distances[i]);
                // rescale to revert back to original norms (cancelling the
                // effect of base and query pre-processing)
                if (_max_base_norm != 0)
                    distances[i] *= (_max_base_norm * query_norm);
            }
        }
    }

#ifdef USE_BING_INFRA
    ctx.m_completeCount = 0;
#endif

    if (stats != nullptr)
    {
        stats->total_us = (float)query_timer.elapsed();
    }
}

// range search returns results of all neighbors within distance of range.
// indices and distances need to be pre-allocated of size l_search and the
// return value is the number of matching hits.
template <typename T, typename LabelT>
uint32_t PQFlashIndex<T, LabelT>::range_search(const T *query1, const double range, const uint64_t min_l_search,
                                               const uint64_t max_l_search, std::vector<uint64_t> &indices,
                                               std::vector<float> &distances, const uint64_t min_beam_width,
                                               QueryStats *stats)
{
    uint32_t res_count = 0;

    bool stop_flag = false;

    uint32_t l_search = (uint32_t)min_l_search; // starting size of the candidate list
    while (!stop_flag)
    {
        indices.resize(l_search);
        distances.resize(l_search);
        uint64_t cur_bw = min_beam_width > (l_search / 5) ? min_beam_width : l_search / 5;
        cur_bw = (cur_bw > 100) ? 100 : cur_bw;
        for (auto &x : distances)
            x = std::numeric_limits<float>::max();
        this->cached_beam_search(query1, l_search, l_search, indices.data(), distances.data(), cur_bw, false, stats);
        for (uint32_t i = 0; i < l_search; i++)
        {
            if (distances[i] > (float)range)
            {
                res_count = i;
                break;
            }
            else if (i == l_search - 1)
                res_count = l_search;
        }
        if (res_count < (uint32_t)(l_search / 2.0))
            stop_flag = true;
        l_search = l_search * 2;
        if (l_search > max_l_search)
            stop_flag = true;
    }
    indices.resize(res_count);
    distances.resize(res_count);
    return res_count;
}

template <typename T, typename LabelT> uint64_t PQFlashIndex<T, LabelT>::get_data_dim()
{
    return _data_dim;
}

template <typename T, typename LabelT> diskann::Metric PQFlashIndex<T, LabelT>::get_metric()
{
    return this->metric;
}

#ifdef EXEC_ENV_OLS
template <typename T, typename LabelT> char *PQFlashIndex<T, LabelT>::getHeaderBytes()
{
    IOContext &ctx = reader->get_ctx();
    AlignedRead readReq;
    readReq.buf = new char[PQFlashIndex<T, LabelT>::HEADER_SIZE];
    readReq.len = PQFlashIndex<T, LabelT>::HEADER_SIZE;
    readReq.offset = 0;

    std::vector<AlignedRead> readReqs;
    readReqs.push_back(readReq);

    reader->read(readReqs, ctx, false);

    return (char *)readReq.buf;
}
#endif

template <typename T, typename LabelT>
std::vector<std::uint8_t> PQFlashIndex<T, LabelT>::get_pq_vector(std::uint64_t vid)
{
    std::uint8_t *pqVec = &this->data[vid * this->_n_chunks];
    return std::vector<std::uint8_t>(pqVec, pqVec + this->_n_chunks);
}

template <typename T, typename LabelT> std::uint64_t PQFlashIndex<T, LabelT>::get_num_points()
{
    return _num_points;
}

// instantiations
template class PQFlashIndex<uint8_t>;
template class PQFlashIndex<int8_t>;
template class PQFlashIndex<float>;
template class PQFlashIndex<uint8_t, uint16_t>;
template class PQFlashIndex<int8_t, uint16_t>;
template class PQFlashIndex<float, uint16_t>;

} // namespace diskann
