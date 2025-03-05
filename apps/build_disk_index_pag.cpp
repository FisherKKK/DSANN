// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include <omp.h>
#include <boost/program_options.hpp>

#include "utils.h"
#include "disk_utils.h"
#include "math_utils.h"
#include "index.h"
#include "partition.h"
#include "program_options_utils.hpp"
#include "pa_graph.h"

namespace po = boost::program_options;

// 构建磁盘index
int main(int argc, char **argv)
{
    std::string data_type, dist_fn, data_path, index_path_prefix;
    uint32_t sample_freq, num_threads, R, L, redundant_num, scale_factor;
    float sample_rate, extra_rate, sigma, delta, early_stop_factor, alpha;
    uint8_t use_routing_path_redundancy;

//    std::string data_type, dist_fn, data_path, index_path_prefix, codebook_prefix, label_file, universal_label,
//        label_type;
//    uint32_t num_threads, R, L, disk_PQ, build_PQ, QD, Lf, filter_threshold;
//    float B, M;
//    bool append_reorder_data = false;
//    bool use_opq = false;

    po::options_description desc{
        program_options_utils::make_program_description("build_disk_index", "Build a disk-based index.")};
    // 这里就是捕捉程序的参数
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

        optional_configs.add_options()("alpha", po::value<float>(&alpha)->default_value(1.2f),
                                       program_options_utils::GRAPH_BUILD_ALPHA);

        optional_configs.add_options()("sample_freq", po::value<uint32_t>(&sample_freq)->default_value(32),
                                       "sample radius frequency");

        optional_configs.add_options()("redundant_num", po::value<uint32_t>(&redundant_num)->default_value(4),
                                       "redundant number of partition");

        optional_configs.add_options()("scale_factor", po::value<uint32_t>(&scale_factor)->default_value(5),
                                       "pre-scale up the ivf size");

        optional_configs.add_options()("sample_rate", po::value<float>(&sample_rate)->default_value(0.04),
                                       "sample rate of aggregation point");

        optional_configs.add_options()("extra_rate", po::value<float>(&extra_rate)->default_value(0.1),
                                       "allow extra overflow of ivf");

        optional_configs.add_options()("sigma", po::value<float>(&sigma)->default_value(0.8),
                                       "relax RNG-base rule");

        optional_configs.add_options()("delta", po::value<float>(&delta)->default_value(0.1),
                                       "select nearly the same partition portion");

        optional_configs.add_options()("early_stop_factor", po::value<float>(&early_stop_factor)->default_value(1.2),
                                       "early stop condition: d  > stop_factor * (minimal_d + 2 * r)");

        optional_configs.add_options()("use_routing_path_redundancy", po::bool_switch()->default_value(false),
                                       "redundancy with path");


//        optional_configs.add_options()("QD", po::value<uint32_t>(&QD)->default_value(0),
//                                       " Quantized Dimension for compression");
//        optional_configs.add_options()("codebook_prefix", po::value<std::string>(&codebook_prefix)->default_value(""),
//                                       "Path prefix for pre-trained codebook");
//        optional_configs.add_options()("PQ_disk_bytes", po::value<uint32_t>(&disk_PQ)->default_value(0),
//                                       "Number of bytes to which vectors should be compressed "
//                                       "on SSD; 0 for no compression");
//        optional_configs.add_options()("append_reorder_data", po::bool_switch()->default_value(false),
//                                       "Include full precision data in the index. Use only in "
//                                       "conjuction with compressed data on SSD.");
//        optional_configs.add_options()("build_PQ_bytes", po::value<uint32_t>(&build_PQ)->default_value(0),
//                                       program_options_utils::BUIlD_GRAPH_PQ_BYTES);
//        optional_configs.add_options()("use_opq", po::bool_switch()->default_value(false),
//                                       program_options_utils::USE_OPQ);
//        optional_configs.add_options()("label_file", po::value<std::string>(&label_file)->default_value(""),
//                                       program_options_utils::LABEL_FILE);
//        optional_configs.add_options()("universal_label", po::value<std::string>(&universal_label)->default_value(""),
//                                       program_options_utils::UNIVERSAL_LABEL);
//        optional_configs.add_options()("FilteredLbuild", po::value<uint32_t>(&Lf)->default_value(0),
//                                       program_options_utils::FILTERED_LBUILD);
//        optional_configs.add_options()("filter_threshold,F", po::value<uint32_t>(&filter_threshold)->default_value(0),
//                                       "Threshold to break up the existing nodes to generate new graph "
//                                       "internally where each node has a maximum F labels.");
//        optional_configs.add_options()("label_type", po::value<std::string>(&label_type)->default_value("uint"),
//                                       program_options_utils::LABEL_TYPE_DESCRIPTION);

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
        use_routing_path_redundancy = vm["use_routing_path_redundancy"].as<bool>() ? 1 : 0;
    }
    catch (const std::exception &ex)
    {
        std::cerr << ex.what() << '\n';
        return -1;
    }

//    bool use_filters = (label_file != "") ? true : false;
    // Metric
    diskann::Metric metric;

    if (dist_fn == std::string("mips"))
    {   // 内积
        metric = diskann::Metric::INNER_PRODUCT;
    }
    else if (dist_fn == std::string("l2"))
    {   // l2范数
        metric = diskann::Metric::L2;
    }
    else if (dist_fn == std::string("cosine"))
    {
        // 余弦
        metric = diskann::Metric::COSINE;
    }
    else
    {   // 报错
        std::cout << "Unsupported distance function. Currently only L2/ Inner "
                     "Product/Cosine are supported."
                  << std::endl;
        return -1;
    }

    // 索引中是否使用全精度向量
//    if (append_reorder_data)
//    {
//        if (disk_PQ == 0)
//        {
//            std::cout << "Error: It is not necessary to append data for reordering "
//                         "when vectors are not compressed on disk."
//                      << std::endl;
//            return -1;
//        }
//        if (data_type != std::string("float"))
//        {
//            std::cout << "Error: Appending data for reordering currently only "
//                         "supported for float data type."
//                      << std::endl;
//            return -1;
//        }
//    }

//    std::string params = std::string(std::to_string(R)) + " " + std::string(std::to_string(L)) + " " +
//                         std::string(std::to_string(B)) + " " + std::string(std::to_string(M)) + " " +
//                         std::string(std::to_string(num_threads)) + " " + std::string(std::to_string(disk_PQ)) + " " +
//                         std::string(std::to_string(append_reorder_data)) + " " +
//                         std::string(std::to_string(build_PQ)) + " " + std::string(std::to_string(QD));

    // 这里就是核心部分, 根据数值的类型选择合适的函数
//    try
//    {
//        if (label_file != "" && label_type == "ushort")
//        {
//            if (data_type == std::string("int8"))
//                return diskann::build_disk_index<int8_t>(data_path.c_str(), index_path_prefix.c_str(), params.c_str(),
//                                                         metric, use_opq, codebook_prefix, use_filters, label_file,
//                                                         universal_label, filter_threshold, Lf);
//            else if (data_type == std::string("uint8"))
//                return diskann::build_disk_index<uint8_t, uint16_t>(
//                    data_path.c_str(), index_path_prefix.c_str(), params.c_str(), metric, use_opq, codebook_prefix,
//                    use_filters, label_file, universal_label, filter_threshold, Lf);
//            else if (data_type == std::string("float"))
//                return diskann::build_disk_index<float, uint16_t>(
//                    data_path.c_str(), index_path_prefix.c_str(), params.c_str(), metric, use_opq, codebook_prefix,
//                    use_filters, label_file, universal_label, filter_threshold, Lf);
//            else
//            {
//                diskann::cerr << "Error. Unsupported data type" << std::endl;
//                return -1;
//            }
//        }
//        else
//        {
//            if (data_type == std::string("int8"))
//                return diskann::build_disk_index<int8_t>(data_path.c_str(), index_path_prefix.c_str(), params.c_str(),
//                                                         metric, use_opq, codebook_prefix, use_filters, label_file,
//                                                         universal_label, filter_threshold, Lf);
//            else if (data_type == std::string("uint8"))
//                return diskann::build_disk_index<uint8_t>(data_path.c_str(), index_path_prefix.c_str(), params.c_str(),
//                                                          metric, use_opq, codebook_prefix, use_filters, label_file,
//                                                          universal_label, filter_threshold, Lf);
//            else if (data_type == std::string("float"))
//                return diskann::build_disk_index<float>(data_path.c_str(), index_path_prefix.c_str(), params.c_str(),
//                                                        metric, use_opq, codebook_prefix, use_filters, label_file,
//                                                        universal_label, filter_threshold, Lf);
//            else
//            {
//                diskann::cerr << "Error. Unsupported data type" << std::endl;
//                return -1;
//            }
//        }
//    }
//    catch (const std::exception &e)
//    {
//        std::cout << std::string(e.what()) << std::endl;
//        diskann::cerr << "Index build failed." << std::endl;
//        return -1;
//    }



    try {
        diskann::cout << "Starting index build" << std::endl;
        if (data_type == std::string("uint8"))
        {

            std::unique_ptr<diskann::Distance<uint8_t >> fn;
            fn.reset(diskann::get_distance_function<uint8_t >(metric));
            auto index = std::make_unique<diskann::PointAggregationGraph<uint8_t>>(
                sample_rate, sample_freq, L, R, num_threads, alpha, use_routing_path_redundancy, extra_rate,
                redundant_num, sigma, delta, scale_factor, early_stop_factor, std::move(fn));
            index->BuildInDisk(data_path.c_str(), index_path_prefix.c_str());
            index->SaveFromDiskPAG();
            index.reset();
        } else if (data_type == std::string("float")) {
            std::unique_ptr<diskann::Distance<float>> fn;
            if (metric == diskann::Metric::COSINE)
                fn.reset(new diskann::AVXNormalizedCosineDistanceFloat());
            else
                fn.reset(diskann::get_distance_function<float>(metric));
            auto index = std::make_unique<diskann::PointAggregationGraph<float>>(
                sample_rate, sample_freq, L, R, num_threads, alpha, use_routing_path_redundancy, extra_rate,
                redundant_num, sigma, delta, scale_factor, early_stop_factor, std::move(fn));
            index->BuildInDisk(data_path.c_str(), index_path_prefix.c_str());
            index->SaveFromDiskPAG();
            index.reset();
        }
        return 0;
    } catch (const std::exception &e) {
        std::cout << std::string(e.what()) << std::endl;
        diskann::cerr << "Index build failed." << std::endl;
        return -1;
    }
}
