#pragma once
#include <string.h>

#include "cuda_clustering/clustering/cuda_clustering.hpp"
#include "cuda_clustering/filtering/cuda_filtering.hpp"
#include "cuda_clustering/clustering/iclustering.hpp"
#include "cuda_clustering/filtering/ifiltering.hpp"
#include "cuda_clustering/segmentation/cuda_segmentation.hpp"
#include "cuda_clustering/segmentation/isegmentation.hpp"
#include "cuda_clustering/utils/pointcloud_converter.hpp"

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

#include <pcl_conversions/pcl_conversions.h>

class ControllerNode : public rclcpp::Node
{
private:
        std::shared_ptr<visualization_msgs::msg::Marker> cones{new visualization_msgs::msg::Marker()};
        std::string input_topic, segmented_topic, filtered_topic, cluster_topic, frame_id;
        bool filterOnZ, segmentFlag, publishFilteredPc, publishSegmentedPc;
        float downFilterLimits, upFilterLimits;
        clustering_parameters param;
        segParam_t segP;
        bool autoOptimizeCoefficients;
        int failed_segmentations;
        int maxFailedSegmentations;
        bool skipClustering;

        cudaStream_t stream = NULL;
        unsigned int memoryAllocated = 0;
        float *inputData = NULL;
        float *partialOutput = NULL;


        IFilter *filter;
        IClustering *clustering;
        Isegmentation *segmentation;

        /* Publisher */
        // rclcpp::Publisher<geometry_msgs::msg::PoseArray>::SharedPtr pose_array_pub_;

        /* Subscriber */
        rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr cloud_sub;

        rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr cones_array_pub;
        rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr filtered_cp_pub;
        rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr segmented_cp_pub;

        /* Load parameters function */
        void loadParameters();

        /* PointCloud Callback */
        void scanCallback(const sensor_msgs::msg::PointCloud2::SharedPtr sub_cloud);

        /* Publish PointCloud */
        void publishPc(float *points, unsigned int size, rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub);

public:
        ControllerNode();
};