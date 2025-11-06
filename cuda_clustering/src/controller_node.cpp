#include "cuda_clustering/controller_node.hpp"

ControllerNode::ControllerNode() : Node("clustering_node")
{
    this->loadParameters();


    this->segmentation = new CudaSegmentation(segP);
    this->filter = new CudaFilter(upFilterLimits, downFilterLimits);

    this->clustering = new CudaClustering(param);

    this->clustering->getInfo();

    /* Define QoS for Best Effort messages transport */
    auto qos = rclcpp::QoS(rclcpp::KeepLast(10), rmw_qos_profile_sensor_data);

    this->cones_array_pub = this->create_publisher<visualization_msgs::msg::Marker>("/perception/newclusters", 100);
    if(this->filterOnZ && this->publishFilteredPc)
        this->filtered_cp_pub = this->create_publisher<sensor_msgs::msg::PointCloud2>("/filtered_pc", 100);
    if(this->segmentFlag && this->publishSegmentedPc)
        this->segmented_cp_pub = this->create_publisher<sensor_msgs::msg::PointCloud2>("/segmented_pc", 100);

    /* Create subscriber */
    this->cloud_sub = this->create_subscription<sensor_msgs::msg::PointCloud2>(this->input_topic, qos,
                                                                               std::bind(&ControllerNode::scanCallback, this, std::placeholders::_1));

    /* Cones topic init */
    cones->header.frame_id = this->frame_id;
    cones->ns = "ListaConiRilevati";
    cones->type = visualization_msgs::msg::Marker::SPHERE_LIST;
    cones->action = visualization_msgs::msg::Marker::ADD;
    cones->scale.x = 0.3; // 0.5
    cones->scale.y = 0.2;
    cones->scale.z = 0.2;
    cones->color.a = 1.0; // 1.0
    cones->color.r = 1.0;
    cones->color.g = 0.0;
    cones->color.b = 1.0;
    cones->pose.orientation.x = 0.0;
    cones->pose.orientation.y = 0.0;
    cones->pose.orientation.z = 0.0;
    cones->pose.orientation.w = 1.0;

    cudaStreamCreate ( &stream );
}

void ControllerNode::loadParameters()
{

    declare_parameter("input_topic", "");
    declare_parameter("frame_id", "");
    declare_parameter("minClusterSize", 0);
    declare_parameter("maxClusterSize", 0);
    declare_parameter("voxelX", 0.0);
    declare_parameter("voxelY", 0.0);
    declare_parameter("voxelZ", 0.0);
    declare_parameter("countThreshold", 0);

    declare_parameter("clusterMaxX", 0.0);
    declare_parameter("clusterMaxY", 0.0);
    declare_parameter("clusterMaxZ", 0.0);
    declare_parameter("clusterMinX", 0.0);
    declare_parameter("clusterMinY", 0.0);
    declare_parameter("clusterMinZ", 0.0);
    declare_parameter("maxHeight", 0.0);

    declare_parameter("downFilterLimits", 0.0);
    declare_parameter("upFilterLimits", 0.0);

    declare_parameter("filterOnZ", false);
    declare_parameter("segment", false);
    declare_parameter("publishFilteredPc", false);
    declare_parameter("publishSegmentedPc", false);
    declare_parameter("distanceThreshold", 0.05);
    declare_parameter("maxIterations", 50);
    declare_parameter("probability", 0.99);
    declare_parameter("optimizeCoefficients", true);


    get_parameter("input_topic", this->input_topic);
    get_parameter("frame_id", this->frame_id);
    get_parameter("minClusterSize", this->param.clustering.minClusterSize);
    get_parameter("maxClusterSize", this->param.clustering.maxClusterSize);
    get_parameter("voxelX", this->param.clustering.voxelX);
    get_parameter("voxelY", this->param.clustering.voxelY);
    get_parameter("voxelZ", this->param.clustering.voxelZ);
    get_parameter("countThreshold", this->param.clustering.countThreshold);

    get_parameter("clusterMaxX", this->param.filtering.clusterMaxX);
    get_parameter("clusterMaxY", this->param.filtering.clusterMaxY);
    get_parameter("clusterMaxZ", this->param.filtering.clusterMaxZ);
    get_parameter("clusterMinX", this->param.filtering.clusterMinX);
    get_parameter("clusterMinY", this->param.filtering.clusterMinY);
    get_parameter("clusterMinZ", this->param.filtering.clusterMinZ);
    get_parameter("maxHeight", this->param.filtering.maxHeight);

    get_parameter("downFilterLimits", this->downFilterLimits);
    get_parameter("upFilterLimits", this->upFilterLimits);
    
    get_parameter("filterOnZ", this->filterOnZ);
    get_parameter("segment", this->segmentFlag);
    get_parameter("publishFilteredPc", this->publishFilteredPc);
    get_parameter("publishSegmentedPc", this->publishSegmentedPc);
    get_parameter("distanceThreshold", this->segP.distanceThreshold);
    get_parameter("maxIterations", this->segP.maxIterations);
    get_parameter("probability", this->segP.probability);
    get_parameter("optimizeCoefficients", this->segP.optimizeCoefficients);
}

void ControllerNode::publishPc(float *points, unsigned int size, rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub)
{
    sensor_msgs::msg::PointCloud2 pc;
    pcl::PointCloud<pcl::PointXYZ>::Ptr pcl_cloud(new pcl::PointCloud<pcl::PointXYZ>);

    pcl_cloud->width = size;
    pcl_cloud->height = 1;
    pcl_cloud->points.resize(size);

    memcpy(pcl_cloud->points.data(), points, size * 4 * sizeof(float));

    pcl::toROSMsg(*pcl_cloud, pc);
    pc.header.frame_id = this->frame_id;
    pub->publish(pc);
}

void ControllerNode::scanCallback(sensor_msgs::msg::PointCloud2::SharedPtr sub_cloud)
{
    std::chrono::steady_clock::time_point tstart = std::chrono::steady_clock::now();
    cones->points = {};
    unsigned int size = 0;
    float* tmp = NULL;

    unsigned int inputSize = sub_cloud->width * sub_cloud->height;
    
    if(memoryAllocated < inputSize){
        cudaFree(inputData);
        cudaMallocManaged(&inputData, sizeof(float) * 4 * inputSize, cudaMemAttachHost);
        cudaStreamAttachMemAsync (stream, inputData);
        cudaFree(partialOutput);
        cudaMallocManaged(&partialOutput, sizeof(float) * 4 * inputSize, cudaMemAttachHost);
        cudaStreamAttachMemAsync (stream, partialOutput);
        memoryAllocated = inputSize;
    }

    auto t1 = std::chrono::steady_clock::now();
    /* ----------------------------------------- */
    
    pointcloud_utils::convertPointCloud2ToFloatArray(sub_cloud, inputData);
    
    /* ----------------------------------------- */

    auto t2 = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(t2 - t1);
    RCLCPP_INFO(rclcpp::get_logger("CudaSegmentation"), "conversione in: %.3f ms", duration.count());

    if (this->filterOnZ)
    {
        this->filter->filterPoints(inputData, inputSize, &partialOutput, &size);
        inputSize = size;

        if (this->publishFilteredPc)
        {
            this->publishPc(partialOutput, size, filtered_cp_pub);
        }

        tmp = partialOutput;
        partialOutput = inputData;
        inputData = tmp;
    }

    if (this->segmentFlag)
    {
        segmentation->segment(inputData, inputSize, &partialOutput, &size);
        inputSize = size;

        if (this->publishSegmentedPc)
        {
            if(size != 0){
                publishPc(partialOutput, size, segmented_cp_pub);
            }
        }

        tmp = partialOutput;
        partialOutput = inputData;
        inputData = tmp;
    }

    this->clustering->extractClusters(inputData, inputSize, partialOutput, cones);
    // RCLCPP_INFO(this->get_logger(), "Marker: %ld data points.", cones->points.size());
    std::chrono::steady_clock::time_point tend = std::chrono::steady_clock::now();
    std::chrono::duration<double, std::ratio<1, 1000>> time_span = std::chrono::duration_cast<std::chrono::duration<double, std::ratio<1, 1000>>>(tend - tstart);
    RCLCPP_INFO(rclcpp::get_logger("clustering_node"), ">>>> TOTAL TIME: %f ms.", time_span.count());
    std::cout << "\n------------------------------------ "<< std::endl;


    cones->header.stamp = this->now();
    if(cones->points.size() != 0)
        cones_array_pub->publish(*cones);
    
}