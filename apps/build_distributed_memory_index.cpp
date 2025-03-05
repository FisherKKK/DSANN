//
// Created by FisherKK on 2024/1/31.
//

#include <omp.h>
#include <boost/program_options.hpp>

#include "utils.h"
#include "disk_utils.h"
#include "math_utils.h"
#include "index.h"
#include "partition.h"
#include "defaults.h"
#include "program_options_utils.hpp"

namespace po = boost::program_options;

/** 并行构建Vamana Index, 然后再进行合并:
 *      1. read data
 *      2. k-means划分数据 => 保存到磁盘
 *      3. parallel build vamana index => 保存到磁盘
 *      4. merge vamana index => 保存到磁盘
 */
int  main(int argc, char *argv[]) {
    /** 参数处理开始 **/
    std::string data_type, dist_fn, data_path, index_path_prefix;
    uint32_t num_threads, R, L; // R是最大度数, L是search list大小
    po::options_description desc{
        program_options_utils::make_program_description("build_distributed_memory_index", "Parallel build a memory-based index.")};
    try
    {
        desc.add_options()("help,h", "Print information on arguments");

        // Required parameters
        po::options_description required_configs("Required");
        required_configs.add_options()("data_type", po::value<std::string>(&data_type)->required(),
                                       program_options_utils::DATA_TYPE_DESCRIPTION);
        required_configs.add_options()("dist_fn", po::value<std::string>(&dist_fn)->required(),
                                       program_options_utils::DISTANCE_FUNCTION_DESCRIPTION);
        required_configs.add_options()("index_path_prefix", po::value<std::string>(&index_path_prefix)->required(),
                                       program_options_utils::INDEX_PATH_PREFIX_DESCRIPTION);
        required_configs.add_options()("data_path", po::value<std::string>(&data_path)->required(),
                                       program_options_utils::INPUT_DATA_PATH);

        // Optional parameters
        po::options_description optional_configs("Optional");
        optional_configs.add_options()("num_threads,T",
                                       po::value<uint32_t>(&num_threads)->default_value(omp_get_num_procs()),
                                       program_options_utils::NUMBER_THREADS_DESCRIPTION);
        optional_configs.add_options()("max_degree,R", po::value<uint32_t>(&R)->default_value(64),
                                       program_options_utils::MAX_BUILD_DEGREE);
        optional_configs.add_options()("Lbuild,L", po::value<uint32_t>(&L)->default_value(100),
                                       program_options_utils::GRAPH_BUILD_COMPLEXITY);

        // Merge required and optional parameters
        desc.add(required_configs).add(optional_configs);

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
    if (dist_fn == std::string("l2"))
        metric = diskann::Metric::L2;
    else if (dist_fn == std::string("mips"))
        metric = diskann::Metric::INNER_PRODUCT;
    else
    {
        std::cout << "Error. Only l2 and mips distance functions are supported" << std::endl;
        return -1;
    }

    std::string params = std::string(std::to_string(R)) + " " + std::string(std::to_string(L)) + " " +
                         std::string(std::to_string(num_threads));

    /** 参数处理结束 **/


    /** 数据划分, 产生如下文件:
     *      1. index_path_prefix + "_centroids.bin" 保存中心点 [num_center * dim]
     *      2. index_path_prefix + "_subshard-" + part + ".bins" 保存part的数据点 [npts] | [dim] | [npts * dim]
     *      3. index_path_prefix + "_subshard-" + part + "_ids_uint32.bin" 保存从index => ids [npts] [npts * 1]
     */
    const float sample_rate = 0.6; // k-means采样率
    const size_t num_centers = 3, // k-means中心数目
                 max_k_means_reps = 1000, // k-means迭代次数
                 k_base = 1; // 每个节点所属的cluster数目
    if (data_type == std::string("float"))
        partition<float>(data_path, sample_rate, num_centers, max_k_means_reps, index_path_prefix, k_base);
    else
    {
        diskann::cerr << "Error. Unsupported data type" << std::endl;
        return -1;
    }
    diskann::cout << "****** Complete Data Partition ******" << std::endl;
    /** 数据划分结束 **/


    /** 并行构建Vamana Index **/
    omp_set_num_threads(num_threads);
#pragma omp parallel for schedule(static, 1)
    for (size_t part = 0; part < num_centers; part++)
    {
        diskann::IndexWriteParameters low_degree_params = diskann::IndexWriteParametersBuilder(L, 2 * R / 3)
                                                              .with_filter_list_size(0)
                                                              .with_saturate_graph(false)
                                                              .with_num_threads(num_threads)
                                                              .build();

        std::string shard_base_file = index_path_prefix + "_subshard-" + std::to_string(part) + ".bin";
        std::string shard_index_file = index_path_prefix + "_subshard-" + std::to_string(part) + "_mem.index";
        size_t shard_base_pts, shard_base_dim;
        diskann::get_bin_metadata(shard_base_file, shard_base_pts, shard_base_dim);
        diskann::Index<float> _index(metric, shard_base_dim, shard_base_pts,
                                 std::make_shared<diskann::IndexWriteParameters>(low_degree_params), nullptr,
                                 diskann::defaults::NUM_FROZEN_POINTS_STATIC, false, false, false, false,
                                 0, false);

        _index.build(shard_base_file.c_str(), shard_base_pts);
        _index.save(shard_index_file.c_str());
        std::remove(shard_base_file.c_str());
    }
    diskann::cout << "****** Complete Parallel Build Vamana ******" << std::endl;

    /** 并行构建Index 结束 **/

    /** 索引合并 **/
    diskann::merge_index(index_path_prefix + "_subshard-", "_mem.index", index_path_prefix + "_subshard-",
                          "_ids_uint32.bin", num_centers, R, index_path_prefix, index_path_prefix + "_centroids.bin");

    size_t base_pts, base_dim;

    /** 保存datastore **/
    diskann::get_bin_metadata(data_path, base_pts, base_dim);
    std::unique_ptr<diskann::Distance<float>> distance;
    distance.reset((diskann::Distance<float> *)diskann::get_distance_function<float>(metric));
    auto data_store = std::make_shared<diskann::InMemDataStore<float>>(base_pts, base_dim,std::move(distance));
    data_store->populate_data(data_path, 0U);
    data_store->save(index_path_prefix + ".data", base_pts);


    diskann::cout << "****** Complete Index Merge ******" << std::endl;
    /** 合并结束 **/
    return 0;
}