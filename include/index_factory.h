#include "index.h"
#include "abstract_graph_store.h"
#include "in_mem_graph_store.h"
#include "pq_data_store.h"

namespace diskann
{
class IndexFactory
{
  public:
    DISKANN_DLLEXPORT explicit IndexFactory(const IndexConfig &config);

    /**
     * open的create_instance函数, 一共有三个整体的顺序是:
     *  1. open -> privte重载
     *  2. private重载 -> 模板函数
     */
    
    DISKANN_DLLEXPORT std::unique_ptr<AbstractIndex> create_instance();

    DISKANN_DLLEXPORT static std::unique_ptr<AbstractGraphStore> construct_graphstore(
        const GraphStoreStrategy stratagy, const size_t size, const size_t reserve_graph_degree);

    template <typename T>
    DISKANN_DLLEXPORT static std::shared_ptr<AbstractDataStore<T>> construct_datastore(DataStoreStrategy stratagy,
                                                                                       size_t num_points,
                                                                                       size_t dimension, Metric m);
    // For now PQDataStore incorporates within itself all variants of quantization that we support. In the
    // future it may be necessary to introduce an AbstractPQDataStore class to spearate various quantization
    // flavours.
    template <typename T>
    DISKANN_DLLEXPORT static std::shared_ptr<PQDataStore<T>> construct_pq_datastore(DataStoreStrategy strategy,
                                                                                    size_t num_points, size_t dimension,
                                                                                    Metric m, size_t num_pq_chunks,
                                                                                    bool use_opq);
    template <typename T> static Distance<T> *construct_inmem_distance_fn(Metric m);

  private:
    void check_config();

    // 模板create_instance函数
    template <typename data_type, typename tag_type, typename label_type>
    std::unique_ptr<AbstractIndex> create_instance();

    // 重载create_instance函数
    std::unique_ptr<AbstractIndex> create_instance(const std::string &data_type, const std::string &tag_type,
                                                   const std::string &label_type);

    template <typename data_type>
    std::unique_ptr<AbstractIndex> create_instance(const std::string &tag_type, const std::string &label_type);

    template <typename data_type, typename tag_type>
    std::unique_ptr<AbstractIndex> create_instance(const std::string &label_type);

    std::unique_ptr<IndexConfig> _config;
};

} // namespace diskann
