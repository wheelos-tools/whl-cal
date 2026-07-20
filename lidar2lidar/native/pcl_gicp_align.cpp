// PCL Generalized ICP helper for manual/auto LiDAR alignment.
// Build:
//   g++ -O3 -std=c++17 lidar2lidar/native/pcl_gicp_align.cpp \
//     -o lidar2lidar/bin/pcl_gicp_align \
//     $(pkg-config --cflags --libs pcl_common pcl_io pcl_registration)

#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/registration/gicp.h>

#include <Eigen/Dense>

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

namespace {

bool read_matrix4f(const std::string& path, Eigen::Matrix4f* matrix) {
    std::ifstream input(path);
    if (!input) {
        return false;
    }
    for (int row = 0; row < 4; ++row) {
        for (int col = 0; col < 4; ++col) {
            if (!(input >> (*matrix)(row, col))) {
                return false;
            }
        }
    }
    return true;
}

void write_json_matrix(std::ostream& out, const Eigen::Matrix4f& matrix) {
    out << "[";
    for (int row = 0; row < 4; ++row) {
        if (row > 0) {
            out << ",";
        }
        out << "[";
        for (int col = 0; col < 4; ++col) {
            if (col > 0) {
                out << ",";
            }
            out << matrix(row, col);
        }
        out << "]";
    }
    out << "]";
}

std::string arg_value(int argc, char** argv, const std::string& key) {
    for (int index = 1; index + 1 < argc; ++index) {
        if (std::string(argv[index]) == key) {
            return argv[index + 1];
        }
    }
    return "";
}

}  // namespace

int main(int argc, char** argv) {
    const std::string source_path = arg_value(argc, argv, "--source");
    const std::string target_path = arg_value(argc, argv, "--target");
    const std::string initial_path = arg_value(argc, argv, "--initial");
    const std::string output_path = arg_value(argc, argv, "--output");
    const std::string max_corr_text = arg_value(argc, argv, "--max-correspondence-distance");
    const std::string max_iter_text = arg_value(argc, argv, "--max-iterations");

    if (source_path.empty() || target_path.empty() || output_path.empty()) {
        std::cerr
            << "Usage: pcl_gicp_align --source SRC.pcd --target TGT.pcd "
            << "[--initial INIT.txt] [--max-correspondence-distance 0.2] "
            << "[--max-iterations 120] --output result.json\n";
        return 2;
    }

    const float max_correspondence_distance =
        max_corr_text.empty() ? 0.2f : std::stof(max_corr_text);
    const int max_iterations = max_iter_text.empty() ? 120 : std::stoi(max_iter_text);

    pcl::PointCloud<pcl::PointXYZ>::Ptr source(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::PointCloud<pcl::PointXYZ>::Ptr target(new pcl::PointCloud<pcl::PointXYZ>());
    if (pcl::io::loadPCDFile(source_path, *source) != 0 ||
        pcl::io::loadPCDFile(target_path, *target) != 0) {
        std::cerr << "Failed to load input point clouds.\n";
        return 3;
    }
    if (source->empty() || target->empty()) {
        std::cerr << "Input point cloud is empty.\n";
        return 4;
    }

    Eigen::Matrix4f initial = Eigen::Matrix4f::Identity();
    if (!initial_path.empty() && !read_matrix4f(initial_path, &initial)) {
        std::cerr << "Failed to read initial transform.\n";
        return 5;
    }

    pcl::GeneralizedIterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> gicp;
    gicp.setInputSource(source);
    gicp.setInputTarget(target);
    gicp.setMaxCorrespondenceDistance(max_correspondence_distance);
    gicp.setMaximumIterations(max_iterations);
    gicp.setTransformationEpsilon(1e-6);
    gicp.setEuclideanFitnessEpsilon(1e-6);
    gicp.setRANSACIterations(0);

    pcl::PointCloud<pcl::PointXYZ> aligned;
    gicp.align(aligned, initial);

    const bool converged = gicp.hasConverged();
    const double fitness_score = gicp.getFitnessScore();
    const Eigen::Matrix4f final_transform = gicp.getFinalTransformation();
    const double inlier_rmse = converged ? std::sqrt(std::max(fitness_score, 0.0)) : -1.0;

    std::ofstream output(output_path);
    if (!output) {
        std::cerr << "Failed to open output file.\n";
        return 6;
    }

    output << "{";
    output << "\"success\":" << (converged ? "true" : "false") << ",";
    output << "\"fitness_score\":" << fitness_score << ",";
    output << "\"inlier_rmse\":" << inlier_rmse << ",";
    output << "\"transform\":";
    write_json_matrix(output, final_transform);
    output << "}";
    return converged ? 0 : 1;
}
