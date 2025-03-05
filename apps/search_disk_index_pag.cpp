// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include "common_includes.h"
#include <boost/program_options.hpp>

#include "index.h"
#include "disk_utils.h"
#include "math_utils.h"
#include "memory_mapper.h"
#include "partition.h"
#include "pq_flash_index.h"
#include "timer.h"
#include "percentile_stats.h"
#include "program_options_utils.hpp"

#ifndef _WINDOWS
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include "linux_aligned_file_reader.h"
#include "pa_graph.h"
#else
#ifdef USE_BING_INFRA
#include "bing_aligned_file_reader.h"
#else
#include "windows_aligned_file_reader.h"
#endif
#endif

#define WARMUP false

namespace po = boost::program_options;

void print_stats(std::string category, std::vector<float> percentiles, std::vector<float> results)
{
    diskann::cout << std::setw(20) << category << ": " << std::flush;
    for (uint32_t s = 0; s < percentiles.size(); s++)
    {
        diskann::cout << std::setw(8) << percentiles[s] << "%";
    }
    diskann::cout << std::endl;
    diskann::cout << std::setw(22) << " " << std::flush;
    for (uint32_t s = 0; s < percentiles.size(); s++)
    {
        diskann::cout << std::setw(9) << results[s];
    }
    diskann::cout << std::endl;
}


template <typename T, typename LabelT = uint32_t>
int search_disk_index(diskann::Metric &metric, const std::string &index_path_prefix,
                      const std::string &result_output_prefix, const std::string &query_file, std::string &gt_file,
                      const uint32_t num_threads, const uint32_t recall_at, const uint32_t beamwidth,
                      const uint32_t num_nodes_to_cache, const uint32_t search_io_limit,
                      const std::vector<uint32_t> &Lvec, const float fail_if_recall_below,
                      const std::vector<std::string> &query_filters, const bool use_reorder_data = false)
{
    diskann::cout << "Search parameters: #threads: " << num_threads << ", ";
    if (beamwidth <= 0)
        diskann::cout << "beamwidth to be optimized for each L value" << std::flush;
    else
        diskann::cout << " beamwidth: " << beamwidth << std::flush;
    if (search_io_limit == std::numeric_limits<uint32_t>::max())
        diskann::cout << "." << std::endl;
    else
        diskann::cout << ", io_limit: " << search_io_limit << "." << std::endl;

    // warm_up的data file
    std::string warmup_query_file = index_path_prefix + "_sample_data.bin";

    // load query bin
    T *query = nullptr;
    uint32_t *gt_ids = nullptr;
    float *gt_dists = nullptr;
    size_t query_num, query_dim, query_aligned_dim, gt_num, gt_dim;
    // 读取query
    diskann::load_aligned_bin<T>(query_file, query, query_num, query_dim, query_aligned_dim);

    bool filtered_search = false;
    if (!query_filters.empty())
    {
        filtered_search = true;
        if (query_filters.size() != 1 && query_filters.size() != query_num)
        {
            std::cout << "Error. Mismatch in number of queries and size of query "
                         "filters file"
                      << std::endl;
            return -1; // To return -1 or some other error handling?
        }
    }

    bool calc_recall_flag = false;
    // 读取ground_truth
    if (gt_file != std::string("null") && gt_file != std::string("NULL") && file_exists(gt_file))
    {
        diskann::load_truthset(gt_file, gt_ids, gt_dists, gt_num, gt_dim);
        if (gt_num != query_num)
        {
            diskann::cout << "Error. Mismatch in number of queries and ground truth data" << std::endl;
        }
        calc_recall_flag = true;
    }

    std::shared_ptr<AlignedFileReader> reader = nullptr;
#ifdef _WINDOWS
#ifndef USE_BING_INFRA
    reader.reset(new WindowsAlignedFileReader());
#else
    reader.reset(new diskann::BingAlignedFileReader());
#endif
#else
    reader.reset(new LinuxAlignedFileReader());
#endif

    std::unique_ptr<diskann::PQFlashIndex<T, LabelT>> _pFlashIndex(
        new diskann::PQFlashIndex<T, LabelT>(reader, metric));

    // 读取disk索引数据
    int res = _pFlashIndex->load(num_threads, index_path_prefix.c_str());

    if (res != 0)
    {
        return res;
    }

    std::vector<uint32_t> node_list;
    diskann::cout << "Caching " << num_nodes_to_cache << " nodes around medoid(s)" << std::endl;
    // 缓存memoid附近的点, 这里只拿到了需要缓存的node id, 保存在node_list
    _pFlashIndex->cache_bfs_levels(num_nodes_to_cache, node_list);
    // if (num_nodes_to_cache > 0)
    //     _pFlashIndex->generate_cache_list_from_sample_queries(warmup_query_file, 15, 6, num_nodes_to_cache,
    //     num_threads, node_list);

    // 载入缓存的节点, 保存在_pFlashIndex的内部属性map中
    _pFlashIndex->load_cache_list(node_list);
    node_list.clear();
    node_list.shrink_to_fit();

    // 设置线程数目
    omp_set_num_threads(num_threads);

    uint64_t warmup_L = 20;
    uint64_t warmup_num = 0, warmup_dim = 0, warmup_aligned_dim = 0;
    T *warmup = nullptr;

    if (WARMUP)
    {
        if (file_exists(warmup_query_file))
        {
            diskann::load_aligned_bin<T>(warmup_query_file, warmup, warmup_num, warmup_dim, warmup_aligned_dim);
        }
        else
        {
            warmup_num = (std::min)((uint32_t)150000, (uint32_t)15000 * num_threads);
            warmup_dim = query_dim;
            warmup_aligned_dim = query_aligned_dim;
            diskann::alloc_aligned(((void **)&warmup), warmup_num * warmup_aligned_dim * sizeof(T), 8 * sizeof(T));
            std::memset(warmup, 0, warmup_num * warmup_aligned_dim * sizeof(T));
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_int_distribution<> dis(-128, 127);
            for (uint32_t i = 0; i < warmup_num; i++)
            {
                for (uint32_t d = 0; d < warmup_dim; d++)
                {
                    warmup[i * warmup_aligned_dim + d] = (T)dis(gen);
                }
            }
        }
        diskann::cout << "Warming up index... " << std::flush;
        std::vector<uint64_t> warmup_result_ids_64(warmup_num, 0);
        std::vector<float> warmup_result_dists(warmup_num, 0);

#pragma omp parallel for schedule(dynamic, 1)
        for (int64_t i = 0; i < (int64_t)warmup_num; i++)
        {
            _pFlashIndex->cached_beam_search(warmup + (i * warmup_aligned_dim), 1, warmup_L,
                                             warmup_result_ids_64.data() + (i * 1),
                                             warmup_result_dists.data() + (i * 1), 4);
        }
        diskann::cout << "..done" << std::endl;
    }

    diskann::cout.setf(std::ios_base::fixed, std::ios_base::floatfield);
    diskann::cout.precision(2);

    std::string recall_string = "Recall@" + std::to_string(recall_at);
    diskann::cout << std::setw(6) << "L" << std::setw(12) << "Beamwidth" << std::setw(16) << "QPS" << std::setw(16)
                  << "Mean Latency" << std::setw(16) << "99.9 Latency" << std::setw(16) << "Mean IOs" << std::setw(16)
                  << "CPU (s)";
    if (calc_recall_flag)
    {
        diskann::cout << std::setw(16) << recall_string << std::endl;
    }
    else
        diskann::cout << std::endl;
    diskann::cout << "==============================================================="
                     "======================================================="
                  << std::endl;

    // L是search_list大小
    // query结果的id => L1, L2, L3
    std::vector<std::vector<uint32_t>> query_result_ids(Lvec.size());
    // query结果的距离 => L1, L2, L3
    std::vector<std::vector<float>> query_result_dists(Lvec.size());

    uint32_t optimized_beamwidth = 2;

    double best_recall = 0.0;

    for (uint32_t test_id = 0; test_id < Lvec.size(); test_id++)
    {
        // 选择的L大小
        uint32_t L = Lvec[test_id];

        // 如果L小于K
        if (L < recall_at)
        {
//            diskann::cout << "Ignoring search with L:" << L << " since it's smaller than K:" << recall_at << std::endl;
//            continue;
        }

        // 没有指定beam时会添加一个优化的beam
        if (beamwidth <= 0)
        {
            diskann::cout << "Tuning beamwidth.." << std::endl;
            optimized_beamwidth =
                optimize_beamwidth(_pFlashIndex, warmup, warmup_num, warmup_aligned_dim, L, optimized_beamwidth);
        }
        else
            optimized_beamwidth = beamwidth;

        // query_num就是query的数目
        query_result_ids[test_id].resize(recall_at * query_num);
        query_result_dists[test_id].resize(recall_at * query_num);

        // 每一个query的状态
        auto stats = new diskann::QueryStats[query_num];

        // 查询结果的id
        std::vector<uint64_t> query_result_ids_64(recall_at * query_num);
        auto s = std::chrono::high_resolution_clock::now();

        // 并发执行query
#pragma omp parallel for schedule(dynamic, 1)
        for (int64_t i = 0; i < (int64_t)query_num; i++)
        {
            if (!filtered_search)
            {   // 没有filter, 使用cache_beam_search
                // 搜索k个最近邻保存在结果集中
                _pFlashIndex->cached_beam_search(query + (i * query_aligned_dim), recall_at, L,
                                                 query_result_ids_64.data() + (i * recall_at),
                                                 query_result_dists[test_id].data() + (i * recall_at),
                                                 optimized_beamwidth, use_reorder_data, stats + i);
            }
            else
            {
                LabelT label_for_search;
                if (query_filters.size() == 1)
                { // one label for all queries
                    label_for_search = _pFlashIndex->get_converted_label(query_filters[0]);
                }
                else
                { // one label for each query
                    label_for_search = _pFlashIndex->get_converted_label(query_filters[i]);
                }
                _pFlashIndex->cached_beam_search(
                    query + (i * query_aligned_dim), recall_at, L, query_result_ids_64.data() + (i * recall_at),
                    query_result_dists[test_id].data() + (i * recall_at), optimized_beamwidth, true, label_for_search,
                    use_reorder_data, stats + i);
            }
        }
        auto e = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff = e - s;
        double qps = (1.0 * query_num) / (1.0 * diff.count());

        // 进行类型转换
        diskann::convert_types<uint64_t, uint32_t>(query_result_ids_64.data(), query_result_ids[test_id].data(),
                                                   query_num, recall_at);

        // 计算延迟
        auto mean_latency = diskann::get_mean_stats<float>(
            stats, query_num, [](const diskann::QueryStats &stats) { return stats.total_us; });

        auto latency_999 = diskann::get_percentile_stats<float>(
            stats, query_num, 0.999, [](const diskann::QueryStats &stats) { return stats.total_us; });

        // 计算io
        auto mean_ios = diskann::get_mean_stats<uint32_t>(stats, query_num,
                                                          [](const diskann::QueryStats &stats) { return stats.n_ios; });

        // 计算cpu占用
        auto mean_cpuus = diskann::get_mean_stats<float>(stats, query_num,
                                                         [](const diskann::QueryStats &stats) { return stats.cpu_us; });

        double recall = 0;
        if (calc_recall_flag)
        {
            // 计算召回率
            recall = diskann::calculate_recall((uint32_t)query_num, gt_ids, gt_dists, (uint32_t)gt_dim,
                                               query_result_ids[test_id].data(), recall_at, recall_at);
            best_recall = std::max(recall, best_recall);
        }

        diskann::cout << std::setw(6) << L << std::setw(12) << optimized_beamwidth << std::setw(16) << qps
                      << std::setw(16) << mean_latency << std::setw(16) << latency_999 << std::setw(16) << mean_ios
                      << std::setw(16) << mean_cpuus;
        if (calc_recall_flag)
        {
            diskann::cout << std::setw(16) << recall << std::endl;
        }
        else
            diskann::cout << std::endl;
        delete[] stats;
    }

    // 保存结果
    diskann::cout << "Done searching. Now saving results " << std::endl;
    uint64_t test_id = 0;
    for (auto L : Lvec)
    {
        if (L < recall_at)
            continue;

        std::string cur_result_path = result_output_prefix + "_" + std::to_string(L) + "_idx_uint32.bin";
        diskann::save_bin<uint32_t>(cur_result_path, query_result_ids[test_id].data(), query_num, recall_at);

        cur_result_path = result_output_prefix + "_" + std::to_string(L) + "_dists_float.bin";
        diskann::save_bin<float>(cur_result_path, query_result_dists[test_id++].data(), query_num, recall_at);
    }

    diskann::aligned_free(query);
    if (warmup != nullptr)
        diskann::aligned_free(warmup);
    return best_recall >= fail_if_recall_below ? 0 : -1;
}

template <typename T>
int search_disk_index(diskann::Metric &metric, const std::string &index_path, const std::string &result_path_prefix,
                        const std::string &query_file, const std::string &truthset_file, const uint32_t num_threads,
                        const uint32_t recall_at, const bool print_all_recalls, const std::vector<uint32_t> &Lvec,
                        const bool dynamic, const bool tags, const bool show_qps_per_thread,
                        const std::vector<std::string> &query_filters, const float fail_if_recall_below)
{
    using TagT = uint32_t;
    // Load the query file
    T *query = nullptr;
    uint32_t *gt_ids = nullptr;
    float *gt_dists = nullptr;
    size_t query_num, query_dim, query_aligned_dim, gt_num, gt_dim;
    diskann::load_aligned_bin<T>(query_file, query, query_num, query_dim, query_aligned_dim);
    query_num = 1000;
    if (metric == diskann::Metric::COSINE) {
        auto dist_fn = std::make_unique<diskann::AVXNormalizedCosineDistanceFloat>();
        dist_fn->preprocess_base_points((float*)query, query_dim, query_num);
    }

    {
        std::ifstream status("/proc/self/status");
        std::string line;
        while (std::getline(status, line)) {
            if (line.substr(0, 6) == "VmRSS:") {
                std::cout << "12@@@@@" << std::stoul(line.substr(7)) << "@@@@@2\n"; // 物理内存占用
            }
        }
    }


    bool calc_recall_flag = false;
    if (truthset_file != std::string("null") && file_exists(truthset_file))
    {
        diskann::load_truthset(truthset_file, gt_ids, gt_dists, gt_num, gt_dim);
        if (gt_num != query_num)
        {
            std::cout << "Error. Mismatch in number of queries and ground truth data" << std::endl;
        }
        calc_recall_flag = true;
    }
    else
    {
        diskann::cout << " Truthset file " << truthset_file << " not found. Not computing recall." << std::endl;
    }

    {
        std::ifstream status("/proc/self/status");
        std::string line;
        while (std::getline(status, line)) {
            if (line.substr(0, 6) == "VmRSS:") {
                std::cout << "2@@@@@" << std::stoul(line.substr(7)) << "@@@@@2\n"; // 物理内存占用
            }
        }
    }

    bool filtered_search = false;
    if (!query_filters.empty())
    {
        filtered_search = true;
        if (query_filters.size() != 1 && query_filters.size() != query_num)
        {
            std::cout << "Error. Mismatch in number of queries and size of query "
                         "filters file"
                      << std::endl;
            return -1; // To return -1 or some other error handling?
        }
    }

    std::ifstream status("/proc/self/status");
    std::string line;
    while (std::getline(status, line)) {
        if (line.substr(0, 6) == "VmRSS:") {
            std::cout << "2@@@@@" << std::stoul(line.substr(7)) << "@@@@@2\n"; // 物理内存占用
        }
    }

    auto index = std::make_unique<diskann::PointAggregationGraph<T>>();
//    index->LoadFromMemoryPAG(index_path.c_str(), *(std::max_element(Lvec.begin(), Lvec.end())));
    index->LoadFromDiskPAG(index_path.c_str(), *(std::max_element(Lvec.begin(), Lvec.end())));

    //    const size_t num_frozen_pts = diskann::get_graph_num_frozen_points(index_path);
    //
    //    auto config = diskann::IndexConfigBuilder()
    //                      .with_metric(metric)
    //                      .with_dimension(query_dim)
    //                      .with_max_points(0)
    //                      .with_data_load_store_strategy(diskann::DataStoreStrategy::MEMORY)
    //                      .with_graph_load_store_strategy(diskann::GraphStoreStrategy::MEMORY)
    //                      .with_data_type(diskann_type_to_name<T>())
    //                      .with_label_type(diskann_type_to_name<LabelT>())
    //                      .with_tag_type(diskann_type_to_name<TagT>())
    //                      .is_dynamic_index(dynamic)
    //                      .is_enable_tags(tags)
    //                      .is_concurrent_consolidate(false)
    //                      .is_pq_dist_build(false)
    //                      .is_use_opq(false)
    //                      .with_num_pq_chunks(0)
    //                      .with_num_frozen_pts(num_frozen_pts)
    //                      .build();
    //
    //    auto index_factory = diskann::IndexFactory(config);
    //    auto index = index_factory.create_instance();
    //    index->load(index_path.c_str(), num_threads, *(std::max_element(Lvec.begin(), Lvec.end())));
    //    std::cout << "Index loaded" << std::endl;

    //    if (metric == diskann::FAST_L2)
    //        index->optimize_index_layout();

    std::cout << "Using " << num_threads << " threads to search" << std::endl;
    std::cout.setf(std::ios_base::fixed, std::ios_base::floatfield);
    std::cout.precision(2);
    const std::string qps_title = show_qps_per_thread ? "QPS/thread" : "QPS";
    uint32_t table_width = 0;
    if (tags)
    {
        std::cout << std::setw(4) << "Ls" << std::setw(12) << qps_title << std::setw(20) << "Mean Latency (mus)"
                  << std::setw(15) << "99.9 Latency";
        table_width += 4 + 12 + 20 + 15;
    }
    else
    {
        std::cout << std::setw(4) << "Ls" << std::setw(12) << qps_title << std::setw(18) << "Avg dist cmps"
                  << std::setw(20) << "Mean Latency (mus)" << std::setw(15) << "99.9 Latency"
                  << std::setw(15) << "I/O Times" << std::setw(15) << "I/O Size (KB)";
        table_width += 4 + 12 + 18 + 20 + 15 + 15 + 15;
    }
    uint32_t recalls_to_print = 0;
    const uint32_t first_recall = print_all_recalls ? 1 : recall_at;
    if (calc_recall_flag)
    {
        for (uint32_t curr_recall = first_recall; curr_recall <= recall_at; curr_recall++)
        {
            std::cout << std::setw(12) << ("Recall@" + std::to_string(curr_recall));
        }
        recalls_to_print = recall_at + 1 - first_recall;
        table_width += recalls_to_print * 12;
    }
    std::cout << std::endl;
    std::cout << std::string(table_width, '=') << std::endl;

    std::vector<std::vector<uint32_t>> query_result_ids(Lvec.size());
    std::vector<std::vector<float>> query_result_dists(Lvec.size());
    std::vector<float> latency_stats(query_num, 0);
    std::vector<uint32_t> cmp_stats;
    if (not tags || filtered_search)
    {
        cmp_stats = std::vector<uint32_t>(query_num, 0);
    }

    std::vector<TagT> query_result_tags;
    if (tags)
    {
        query_result_tags.resize(recall_at * query_num);
    }

    double best_recall = 0.0;

//    std::ifstream status("/proc/self/status");
//    std::string line;
    while (std::getline(status, line)) {
        if (line.substr(0, 6) == "VmRSS:") {
            std::cout << "@@@@@" << std::stoul(line.substr(7)) << "@@@@@\n"; // 物理内存占用
        }
    }

    for (uint32_t test_id = 0; test_id < Lvec.size(); test_id++)
    {
        uint32_t L = Lvec[test_id];
        if (L < recall_at)
        {
//            diskann::cout << "Ignoring search with L:" << L << " since it's smaller than K:" << recall_at << std::endl;
//            continue;
        }

        query_result_ids[test_id].resize(recall_at * query_num);
        query_result_dists[test_id].resize(recall_at * query_num);
        std::vector<float *> res = std::vector<float *>();

        auto s = std::chrono::high_resolution_clock::now();
        omp_set_num_threads(num_threads);
        size_t io_times = 0, io_sz = 0;
//#pragma omp parallel for schedule(dynamic, 1)
        for (int64_t i = 0; i < (int64_t)query_num; i++)
        {
            auto qs = std::chrono::high_resolution_clock::now();
            //            if (filtered_search)
            //            {
            //                std::string raw_filter = query_filters.size() == 1 ? query_filters[0] : query_filters[i];
            //
            //                auto retval = index->search_with_filters(query + i * query_aligned_dim, raw_filter, recall_at, L,
            //                                                         query_result_ids[test_id].data() + i * recall_at,
            //                                                         query_result_dists[test_id].data() + i * recall_at);
            //                cmp_stats[i] = retval.second;
            //            }
            //            else if (metric == diskann::FAST_L2)
            //            {
            //                index->search_with_optimized_layout(query + i * query_aligned_dim, recall_at, L,
            //                                                    query_result_ids[test_id].data() + i * recall_at);
            //            }
            //            else if (tags)
            //            {
            //                index->search_with_tags(query + i * query_aligned_dim, recall_at, L,
            //                                        query_result_tags.data() + i * recall_at, nullptr, res);
            //                for (int64_t r = 0; r < (int64_t)recall_at; r++)
            //                {
            //                    query_result_ids[test_id][recall_at * i + r] = query_result_tags[recall_at * i + r];
            //                }
            //            }
            //            else
            {
                //                cmp_stats[i] = index
                //                                   ->search(query + i * query_aligned_dim, recall_at, L,
                //                                            query_result_ids[test_id].data() + i * recall_at)
                //                                   .second;

                std::pair<size_t, size_t> info;

                if (index->RoutingPathSearch()) {
                    info = index->SearchInDiskByRoute(query + i * query_aligned_dim, recall_at, L,
                                                 query_result_ids[test_id].data() + i * recall_at);
                } else {
                    size_t nprobe = L;
                    // *(std::max_element(Lvec.begin(), Lvec.end()))
                    info = index->SearchInDiskByTopK(query + i * query_aligned_dim, recall_at, L, nprobe,
                                                query_result_ids[test_id].data() + i * recall_at);
                }

                io_times += info.first;
                io_sz += info.second;
            }

            auto qe = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> diff = qe - qs;
            latency_stats[i] = (float)(diff.count() * 1000000);
        }
        std::chrono::duration<double> diff = std::chrono::high_resolution_clock::now() - s;

        double displayed_qps = query_num / diff.count();

        if (show_qps_per_thread)
            displayed_qps /= num_threads;

        std::vector<double> recalls;
        if (calc_recall_flag)
        {
            recalls.reserve(recalls_to_print);
            for (uint32_t curr_recall = first_recall; curr_recall <= recall_at; curr_recall++)
            {
                recalls.push_back(diskann::calculate_recall((uint32_t)query_num, gt_ids, gt_dists, (uint32_t)gt_dim,
                                                            query_result_ids[test_id].data(), recall_at, curr_recall));
            }
        }

        std::sort(latency_stats.begin(), latency_stats.end());
        double mean_latency =
            std::accumulate(latency_stats.begin(), latency_stats.end(), 0.0) / static_cast<float>(query_num);

        float avg_cmps = (float)std::accumulate(cmp_stats.begin(), cmp_stats.end(), 0) / (float)query_num;

        if (tags && !filtered_search)
        {
            std::cout << std::setw(4) << L << std::setw(12) << displayed_qps << std::setw(20) << (float)mean_latency
                      << std::setw(15) << (float)latency_stats[(uint64_t)(0.999 * query_num)]
                      << std::setw(15)  << io_times / query_num << std::setw(15) << io_sz / 1024 / query_num;
        }
        else
        {
            std::cout << std::setw(4) << L << std::setw(12) << displayed_qps << std::setw(18) << avg_cmps
                      << std::setw(20) << (float)mean_latency << std::setw(15)
                      << (float)latency_stats[(uint64_t)(0.999 * query_num)]
                      << std::setw(15) << io_times / query_num <<  std::setw(15) << io_sz / 1024 / query_num;
        }
        for (double recall : recalls)
        {
            std::cout << std::setw(12) << recall;
            best_recall = std::max(recall, best_recall);
        }
        std::cout << std::endl;
    }

    std::cout << "Done searching. Now saving results " << std::endl;
    uint64_t test_id = 0;
    for (auto L : Lvec)
    {
        if (L < recall_at)
        {
//            diskann::cout << "Ignoring search with L:" << L << " since it's smaller than K:" << recall_at << std::endl;
//            continue;
        }
        std::string cur_result_path_prefix = result_path_prefix + "_" + std::to_string(L);

        std::string cur_result_path = cur_result_path_prefix + "_idx_uint32.bin";
        diskann::save_bin<uint32_t>(cur_result_path, query_result_ids[test_id].data(), query_num, recall_at);

        cur_result_path = cur_result_path_prefix + "_dists_float.bin";
        diskann::save_bin<float>(cur_result_path, query_result_dists[test_id].data(), query_num, recall_at);

        test_id++;
    }

    diskann::aligned_free(query);
    return best_recall >= fail_if_recall_below ? 0 : -1;
}
int main(int argc, char **argv)
{
    uint32_t W, num_nodes_to_cache, search_io_limit;
    bool use_reorder_data = false;

    std::string data_type, dist_fn, index_path_prefix, result_path, query_file, gt_file, filter_label, label_type,
        query_filters_file;
    uint32_t num_threads, K;
    std::vector<uint32_t> Lvec;
    bool print_all_recalls, dynamic, tags, show_qps_per_thread;
    float fail_if_recall_below = 0.0f;

    po::options_description desc{
        program_options_utils::make_program_description("search_memory_index", "Searches in-memory DiskANN indexes")};
    try
    {
        desc.add_options()("help,h", "Print this information on arguments");

        // Required parameters
        po::options_description required_configs("Required");
        required_configs.add_options()("data_type", po::value<std::string>(&data_type)->required(),
                                       program_options_utils::DATA_TYPE_DESCRIPTION);
        required_configs.add_options()("dist_fn", po::value<std::string>(&dist_fn)->required(),
                                       program_options_utils::DISTANCE_FUNCTION_DESCRIPTION);
        required_configs.add_options()("index_path_prefix", po::value<std::string>(&index_path_prefix)->required(),
                                       program_options_utils::INDEX_PATH_PREFIX_DESCRIPTION);
        required_configs.add_options()("result_path", po::value<std::string>(&result_path)->required(),
                                       program_options_utils::RESULT_PATH_DESCRIPTION);
        required_configs.add_options()("query_file", po::value<std::string>(&query_file)->required(),
                                       program_options_utils::QUERY_FILE_DESCRIPTION);
        required_configs.add_options()("recall_at,K", po::value<uint32_t>(&K)->required(),
                                       program_options_utils::NUMBER_OF_RESULTS_DESCRIPTION);
        required_configs.add_options()("search_list,L",
                                       po::value<std::vector<uint32_t>>(&Lvec)->multitoken()->required(),
                                       program_options_utils::SEARCH_LIST_DESCRIPTION);
        //        required_configs.add_options()("nprobe",
        //                                       po::value<std::vector<uint32_t>>(&nprobes)->multitoken()->required(),
        //                                       "probe number of partitions");

        // Optional parameters
        po::options_description optional_configs("Optional");
        optional_configs.add_options()("filter_label",
                                       po::value<std::string>(&filter_label)->default_value(std::string("")),
                                       program_options_utils::FILTER_LABEL_DESCRIPTION);
        optional_configs.add_options()("query_filters_file",
                                       po::value<std::string>(&query_filters_file)->default_value(std::string("")),
                                       program_options_utils::FILTERS_FILE_DESCRIPTION);
        optional_configs.add_options()("label_type", po::value<std::string>(&label_type)->default_value("uint"),
                                       program_options_utils::LABEL_TYPE_DESCRIPTION);
        optional_configs.add_options()("gt_file", po::value<std::string>(&gt_file)->default_value(std::string("null")),
                                       program_options_utils::GROUND_TRUTH_FILE_DESCRIPTION);
        optional_configs.add_options()("num_threads,T",
                                       po::value<uint32_t>(&num_threads)->default_value(omp_get_num_procs()),
                                       program_options_utils::NUMBER_THREADS_DESCRIPTION);
        optional_configs.add_options()(
            "dynamic", po::value<bool>(&dynamic)->default_value(false),
            "Whether the index is dynamic. Dynamic indices must have associated tags.  Default false.");
        optional_configs.add_options()("tags", po::value<bool>(&tags)->default_value(false),
                                       "Whether to search with external identifiers (tags). Default false.");
        optional_configs.add_options()("fail_if_recall_below",
                                       po::value<float>(&fail_if_recall_below)->default_value(0.0f),
                                       program_options_utils::FAIL_IF_RECALL_BELOW);

        // Output controls
        po::options_description output_controls("Output controls");
        output_controls.add_options()("print_all_recalls", po::bool_switch(&print_all_recalls),
                                      "Print recalls at all positions, from 1 up to specified "
                                      "recall_at value");
        output_controls.add_options()("print_qps_per_thread", po::bool_switch(&show_qps_per_thread),
                                      "Print overall QPS divided by the number of threads in "
                                      "the output table");



        // Merge required and optional parameters
        desc.add(required_configs).add(optional_configs).add(output_controls);

        po::variables_map vm;
        po::store(po::parse_command_line(argc, argv, desc), vm);
        if (vm.count("help"))
        {
            std::cout << desc;
            return 0;
        }
        po::notify(vm);
    }
    catch (const std::exception &ex)
    {
        std::cerr << ex.what() << '\n';
        return -1;
    }

    diskann::Metric metric;
    if (dist_fn == std::string("mips"))
    {
        metric = diskann::Metric::INNER_PRODUCT;
    }
    else if (dist_fn == std::string("l2"))
    {
        metric = diskann::Metric::L2;
    }
    else if (dist_fn == std::string("cosine"))
    {
        metric = diskann::Metric::COSINE;
    }
    else
    {
        std::cout << "Unsupported distance function. Currently only L2/ Inner "
                     "Product/Cosine are supported."
                  << std::endl;
        return -1;
    }

    if ((data_type != std::string("float")) && (metric == diskann::Metric::INNER_PRODUCT))
    {
        std::cout << "Currently support only floating point data for Inner Product." << std::endl;
        return -1;
    }

    if (use_reorder_data && data_type != std::string("float"))
    {
        std::cout << "Error: Reorder data for reordering currently only "
                     "supported for float data type."
                  << std::endl;
        return -1;
    }

    if (filter_label != "" && query_filters_file != "")
    {
        std::cerr << "Only one of filter_label and query_filters_file should be provided" << std::endl;
        return -1;
    }

    std::vector<std::string> query_filters;
    if (filter_label != "")
    {
        query_filters.push_back(filter_label);
    }
    else if (query_filters_file != "")
    {
        query_filters = read_file_to_vector_of_strings(query_filters_file);
    }

    try
    {

        if (!query_filters.empty() && label_type == "ushort")
        {
            if (data_type == std::string("float"))
            {
                return search_disk_index<float>(metric, index_path_prefix, result_path, query_file, gt_file,
                                         num_threads, K, print_all_recalls, Lvec, dynamic, tags,
                                         show_qps_per_thread, query_filters, fail_if_recall_below);
            } else if (data_type == std::string("uint8")) {
                return search_disk_index<uint8_t>(metric, index_path_prefix, result_path, query_file, gt_file,
                                                num_threads, K, print_all_recalls, Lvec, dynamic, tags,
                                                show_qps_per_thread, query_filters, fail_if_recall_below);
            }
            else
            {
                std::cerr << "Unsupported data type. Use float or int8 or uint8" << std::endl;
                return -1;
            }
        }
        else
        {
            if (data_type == std::string("float"))
            {
                std::ifstream status("/proc/self/status");
                std::string line;
                while (std::getline(status, line)) {
                    if (line.substr(0, 6) == "VmRSS:") {
                        std::cout << "2@@@@@" << std::stoul(line.substr(7)) << "@@@@@2\n"; // 物理内存占用
                    }
                }

                return search_disk_index<float>(metric, index_path_prefix, result_path, query_file, gt_file,
                                           num_threads, K, print_all_recalls, Lvec, dynamic, tags,
                                           show_qps_per_thread, query_filters, fail_if_recall_below);
            } else if (data_type == std::string("uint8")) {
                return search_disk_index<uint8_t>(metric, index_path_prefix, result_path, query_file, gt_file,
                                                  num_threads, K, print_all_recalls, Lvec, dynamic, tags,
                                                  show_qps_per_thread, query_filters, fail_if_recall_below);
            }
            else
            {
                std::cerr << "Unsupported data type. Use float or int8 or uint8" << std::endl;
                return -1;
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cout << std::string(e.what()) << std::endl;
        diskann::cerr << "Index search failed." << std::endl;
        return -1;
    }
}