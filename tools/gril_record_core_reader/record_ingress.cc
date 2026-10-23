#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_set>
#include <utility>
#include <vector>

#include "cyber/message/protobuf_factory.h"
#include "cyber/record/record_message.h"
#include "cyber/record/record_reader.h"
#include "google/protobuf/descriptor.h"
#include "google/protobuf/message.h"

namespace {

using apollo::cyber::message::ProtobufFactory;
using apollo::cyber::record::RecordMessage;
using apollo::cyber::record::RecordReader;
using google::protobuf::FieldDescriptor;
using google::protobuf::Message;

constexpr char kMagic[] = "GRIL_CANONICAL_NATIVE 1\n";
constexpr char kLidarTopicDefault[] =
    "/apollo/sensor/vanjeelidar/up/PointCloud2";
constexpr char kImuTopicDefault[] = "/apollo/sensor/gnss/imu";

struct Point {
  float x;
  float y;
  float z;
  float intensity;
  uint16_t ring;
  float time_s;
};

struct Scan {
  int64_t timestamp_ns;
  std::vector<Point> points;
};

struct ImuSample {
  int64_t timestamp_ns;
  double angular_velocity[3];
  double linear_acceleration[3];
};

struct Transform {
  std::string parent;
  std::string child;
  std::string source_topic;
  double matrix[16];
};

const FieldDescriptor *Field(const Message &message, const std::string &name,
                             bool required = false) {
  const auto *field = message.GetDescriptor()->FindFieldByName(name);
  if (required && field == nullptr) {
    throw std::runtime_error("Missing protobuf field " +
                             message.GetDescriptor()->full_name() + "." + name);
  }
  return field;
}

double NumericValue(const Message &message, const FieldDescriptor *field) {
  const auto *reflection = message.GetReflection();
  switch (field->cpp_type()) {
  case FieldDescriptor::CPPTYPE_DOUBLE:
    return reflection->GetDouble(message, field);
  case FieldDescriptor::CPPTYPE_FLOAT:
    return reflection->GetFloat(message, field);
  case FieldDescriptor::CPPTYPE_INT32:
    return reflection->GetInt32(message, field);
  case FieldDescriptor::CPPTYPE_INT64:
    return static_cast<double>(reflection->GetInt64(message, field));
  case FieldDescriptor::CPPTYPE_UINT32:
    return reflection->GetUInt32(message, field);
  case FieldDescriptor::CPPTYPE_UINT64:
    return static_cast<double>(reflection->GetUInt64(message, field));
  default:
    throw std::runtime_error("Protobuf field is not numeric: " +
                             field->full_name());
  }
}

double Number(const Message &message, const std::string &name,
              double fallback = 0.0) {
  const auto *field = Field(message, name);
  if (field == nullptr)
    return fallback;
  const auto *reflection = message.GetReflection();
  if (field->is_optional() && !reflection->HasField(message, field)) {
    return fallback;
  }
  return NumericValue(message, field);
}

std::string String(const Message &message, const std::string &name) {
  const auto *field = Field(message, name);
  if (field == nullptr)
    return {};
  return message.GetReflection()->GetString(message, field);
}

const Message *Nested(const Message &message, const std::string &name) {
  const auto *field = Field(message, name);
  if (field == nullptr ||
      field->cpp_type() != FieldDescriptor::CPPTYPE_MESSAGE) {
    return nullptr;
  }
  const auto *reflection = message.GetReflection();
  if (field->is_optional() && !reflection->HasField(message, field)) {
    return nullptr;
  }
  return &reflection->GetMessage(message, field);
}

std::string FrameId(const Message &message, const std::string &fallback) {
  const std::string top_level = String(message, "frame_id");
  if (!top_level.empty())
    return top_level;
  const Message *header = Nested(message, "header");
  if (header != nullptr) {
    const std::string header_frame = String(*header, "frame_id");
    if (!header_frame.empty())
      return header_frame;
  }
  return fallback;
}

int64_t SecondsToNanoseconds(double seconds) {
  if (!std::isfinite(seconds) || seconds <= 0.0)
    return 0;
  const double nanoseconds = seconds * 1.0e9;
  if (nanoseconds > static_cast<double>(std::numeric_limits<int64_t>::max())) {
    throw std::runtime_error("Timestamp exceeds signed nanosecond range");
  }
  return static_cast<int64_t>(std::nearbyint(nanoseconds));
}

int64_t MessageTimestamp(const Message &message, uint64_t fallback) {
  const int64_t measurement =
      SecondsToNanoseconds(Number(message, "measurement_time"));
  if (measurement != 0)
    return measurement;
  const Message *header = Nested(message, "header");
  if (header != nullptr) {
    const int64_t header_time =
        SecondsToNanoseconds(Number(*header, "timestamp_sec"));
    if (header_time != 0)
      return header_time;
  }
  if (fallback > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
    throw std::runtime_error(
        "Record timestamp exceeds signed nanosecond range");
  }
  return static_cast<int64_t>(fallback);
}

bool Includes(int64_t timestamp_ns, int64_t start_ns, int64_t end_ns) {
  return (start_ns == std::numeric_limits<int64_t>::min() ||
          timestamp_ns >= start_ns) &&
         (end_ns == std::numeric_limits<int64_t>::max() ||
          timestamp_ns < end_ns);
}

std::unique_ptr<Message> Decode(RecordReader &reader,
                                const RecordMessage &record,
                                ProtobufFactory &factory,
                                std::unordered_set<std::string> *registered) {
  const std::string type = reader.GetMessageType(record.channel_name);
  if (type.empty()) {
    throw std::runtime_error("Record channel has no protobuf type: " +
                             record.channel_name);
  }
  if (registered->insert(type).second) {
    const std::string descriptor = reader.GetProtoDesc(record.channel_name);
    if (descriptor.empty() || !factory.RegisterMessage(descriptor)) {
      throw std::runtime_error(
          "Unable to register record protobuf schema for " + type);
    }
  }
  std::unique_ptr<Message> message(factory.GenerateMessageByType(type));
  if (message == nullptr || !message->ParseFromString(record.content)) {
    throw std::runtime_error("Unable to decode record payload for " + type);
  }
  return message;
}

void ReadPointCloud(const Message &cloud, int64_t start_ns, int64_t end_ns,
                    uint32_t scan_lines, std::string *frame_id,
                    std::vector<Scan> *scans) {
  const auto *point_field = Field(cloud, "point", true);
  if (!point_field->is_repeated() ||
      point_field->cpp_type() != FieldDescriptor::CPPTYPE_MESSAGE) {
    throw std::runtime_error("PointCloud.point must be repeated messages");
  }
  const auto *reflection = cloud.GetReflection();
  const int point_count = reflection->FieldSize(cloud, point_field);
  if (point_count == 0)
    return;
  std::vector<const Message *> source_points;
  source_points.reserve(point_count);
  uint64_t timestamp_ns = std::numeric_limits<uint64_t>::max();
  for (int i = 0; i < point_count; ++i) {
    const Message &point =
        reflection->GetRepeatedMessage(cloud, point_field, i);
    const auto *timestamp_field = Field(point, "timestamp", true);
    const uint64_t point_timestamp =
        point.GetReflection()->GetUInt64(point, timestamp_field);
    timestamp_ns = std::min(timestamp_ns, point_timestamp);
    source_points.push_back(&point);
  }
  if (timestamp_ns >
      static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
    throw std::runtime_error("Point timestamp exceeds signed nanosecond range");
  }
  const int64_t scan_timestamp = static_cast<int64_t>(timestamp_ns);
  if (!Includes(scan_timestamp, start_ns, end_ns))
    return;

  if (frame_id->empty())
    *frame_id = FrameId(cloud, "lidar");
  Scan scan;
  scan.timestamp_ns = scan_timestamp;
  scan.points.reserve(point_count);
  for (size_t i = 0; i < source_points.size(); ++i) {
    const Message &source = *source_points[i];
    const float x = static_cast<float>(Number(source, "x"));
    const float y = static_cast<float>(Number(source, "y"));
    const float z = static_cast<float>(Number(source, "z"));
    const float intensity = static_cast<float>(Number(source, "intensity"));
    if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z) ||
        !std::isfinite(intensity)) {
      continue;
    }
    const uint64_t point_timestamp = source.GetReflection()->GetUInt64(
        source, Field(source, "timestamp", true));
    if (point_timestamp < timestamp_ns) {
      throw std::runtime_error("Point timestamp precedes scan timestamp");
    }
    const float point_time = static_cast<float>(
        static_cast<double>(point_timestamp - timestamp_ns) * 1.0e-9);
    scan.points.push_back(Point{
        x, y, z, intensity, static_cast<uint16_t>(i % scan_lines), point_time});
  }
  if (!scan.points.empty())
    scans->push_back(std::move(scan));
}

void ReadImu(const Message &source, const std::string &topic,
             uint64_t record_timestamp, int64_t start_ns, int64_t end_ns,
             std::string *frame_id, std::vector<ImuSample> *samples) {
  const int64_t timestamp_ns = MessageTimestamp(source, record_timestamp);
  if (!Includes(timestamp_ns, start_ns, end_ns))
    return;
  if (frame_id->empty())
    *frame_id = FrameId(source, "imu");
  const Message *imu = Nested(source, "imu");
  if (imu == nullptr)
    imu = &source;
  const Message *angular = Nested(*imu, "angular_velocity");
  const Message *acceleration = Nested(*imu, "linear_acceleration");
  if (angular == nullptr || acceleration == nullptr) {
    throw std::runtime_error("IMU payload is missing angular_velocity or "
                             "linear_acceleration on " +
                             topic);
  }
  ImuSample sample{};
  sample.timestamp_ns = timestamp_ns;
  for (int axis = 0; axis < 3; ++axis) {
    const char *names[] = {"x", "y", "z"};
    sample.angular_velocity[axis] = Number(*angular, names[axis]);
    sample.linear_acceleration[axis] = Number(*acceleration, names[axis]);
    if (!std::isfinite(sample.angular_velocity[axis]) ||
        !std::isfinite(sample.linear_acceleration[axis])) {
      throw std::runtime_error("IMU payload contains non-finite values");
    }
  }
  samples->push_back(sample);
}

void QuaternionToMatrix(double x, double y, double z, double w,
                        double matrix[16]) {
  const double norm = std::sqrt(x * x + y * y + z * z + w * w);
  if (!std::isfinite(norm) || norm <= 0.0) {
    throw std::runtime_error("Transform quaternion must have a finite "
                             "non-zero norm");
  }
  x /= norm;
  y /= norm;
  z /= norm;
  w /= norm;
  const double xx = x * x;
  const double yy = y * y;
  const double zz = z * z;
  const double xy = x * y;
  const double xz = x * z;
  const double yz = y * z;
  const double wx = w * x;
  const double wy = w * y;
  const double wz = w * z;
  matrix[0] = 1.0 - 2.0 * (yy + zz);
  matrix[1] = 2.0 * (xy - wz);
  matrix[2] = 2.0 * (xz + wy);
  matrix[3] = 0.0;
  matrix[4] = 2.0 * (xy + wz);
  matrix[5] = 1.0 - 2.0 * (xx + zz);
  matrix[6] = 2.0 * (yz - wx);
  matrix[7] = 0.0;
  matrix[8] = 2.0 * (xz - wy);
  matrix[9] = 2.0 * (yz + wx);
  matrix[10] = 1.0 - 2.0 * (xx + yy);
  matrix[11] = 0.0;
  matrix[12] = matrix[13] = matrix[14] = 0.0;
  matrix[15] = 1.0;
}

void ReadTransforms(
    const Message &message, const std::string &topic,
    std::map<std::pair<std::string, std::string>, Transform> *transforms) {
  const auto *field = Field(message, "transforms");
  if (field == nullptr ||
      field->cpp_type() != FieldDescriptor::CPPTYPE_MESSAGE ||
      !field->is_repeated()) {
    throw std::runtime_error("TF message is missing repeated transforms");
  }
  const auto *reflection = message.GetReflection();
  for (int i = 0; i < reflection->FieldSize(message, field); ++i) {
    const Message &stamped = reflection->GetRepeatedMessage(message, field, i);
    const Message *header = Nested(stamped, "header");
    const Message *transform = Nested(stamped, "transform");
    if (header == nullptr || transform == nullptr)
      continue;
    const std::string parent = String(*header, "frame_id");
    const std::string child = String(stamped, "child_frame_id");
    if (parent.empty() || child.empty())
      continue;
    const Message *translation = Nested(*transform, "translation");
    const Message *rotation = Nested(*transform, "rotation");
    if (translation == nullptr || rotation == nullptr)
      continue;
    Transform edge;
    edge.parent = parent;
    edge.child = child;
    edge.source_topic = topic;
    QuaternionToMatrix(Number(*rotation, "qx"), Number(*rotation, "qy"),
                       Number(*rotation, "qz"), Number(*rotation, "qw"),
                       edge.matrix);
    edge.matrix[3] = Number(*translation, "x");
    edge.matrix[7] = Number(*translation, "y");
    edge.matrix[11] = Number(*translation, "z");
    (*transforms)[{parent, child}] = std::move(edge);
  }
}

template <typename T> void WriteLittleEndian(std::ostream &output, T value) {
  static_assert(std::is_trivially_copyable<T>::value, "trivial type required");
#if __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
  char bytes[sizeof(T)];
  std::memcpy(bytes, &value, sizeof(T));
  std::reverse(bytes, bytes + sizeof(T));
  output.write(bytes, sizeof(T));
#else
  output.write(reinterpret_cast<const char *>(&value), sizeof(T));
#endif
  if (!output)
    throw std::runtime_error("Failed writing canonical data");
}

void WriteString(std::ostream &output, const std::string &value) {
  if (value.size() > std::numeric_limits<uint32_t>::max()) {
    throw std::runtime_error("Canonical string exceeds uint32 length");
  }
  WriteLittleEndian(output, static_cast<uint32_t>(value.size()));
  output.write(value.data(), value.size());
  if (!output)
    throw std::runtime_error("Failed writing canonical string");
}

void WriteCanonical(const std::string &output_path,
                    const std::string &lidar_frame,
                    const std::string &imu_frame,
                    const std::vector<Scan> &scans,
                    const std::vector<ImuSample> &imu_samples,
                    const std::map<std::pair<std::string, std::string>,
                                   Transform> &transforms) {
  std::ofstream output(output_path, std::ios::binary);
  if (!output)
    throw std::runtime_error("Unable to open output: " + output_path);
  output.write(kMagic, sizeof(kMagic) - 1);
  WriteString(output, lidar_frame);
  WriteString(output, imu_frame);
  uint64_t point_count = 0;
  for (const auto &scan : scans)
    point_count += scan.points.size();
  WriteLittleEndian(output, static_cast<uint64_t>(scans.size()));
  WriteLittleEndian(output, point_count);
  WriteLittleEndian(output, static_cast<uint64_t>(imu_samples.size()));
  WriteLittleEndian(output, static_cast<uint64_t>(transforms.size()));
  for (const auto &scan : scans) {
    WriteLittleEndian(output, scan.timestamp_ns);
  }
  uint64_t offset = 0;
  WriteLittleEndian(output, offset);
  for (const auto &scan : scans) {
    offset += scan.points.size();
    WriteLittleEndian(output, offset);
  }
  for (const auto &scan : scans) {
    for (const auto &point : scan.points) {
      WriteLittleEndian(output, point.x);
      WriteLittleEndian(output, point.y);
      WriteLittleEndian(output, point.z);
      WriteLittleEndian(output, point.intensity);
      WriteLittleEndian(output, point.ring);
      WriteLittleEndian(output, point.time_s);
    }
  }
  for (const auto &sample : imu_samples) {
    WriteLittleEndian(output, sample.timestamp_ns);
  }
  for (const auto &sample : imu_samples) {
    for (double value : sample.angular_velocity)
      WriteLittleEndian(output, value);
    for (double value : sample.linear_acceleration) {
      WriteLittleEndian(output, value);
    }
  }
  for (const auto &item : transforms) {
    const auto &edge = item.second;
    WriteString(output, edge.parent);
    WriteString(output, edge.child);
    WriteString(output, edge.source_topic);
    for (double value : edge.matrix)
      WriteLittleEndian(output, value);
  }
}

int64_t ParseInt64(const char *value, const std::string &name) {
  try {
    size_t parsed = 0;
    const long long result = std::stoll(value, &parsed);
    if (parsed != std::strlen(value))
      throw std::invalid_argument("suffix");
    return static_cast<int64_t>(result);
  } catch (const std::exception &) {
    throw std::runtime_error("Invalid " + name + ": " + value);
  }
}

} // namespace

int main(int argc, char **argv) {
  if (argc < 8) {
    std::cerr
        << "usage: record_ingress OUTPUT LIDAR_TOPIC IMU_TOPIC SCAN_LINES "
           "START_NS END_NS RECORD [RECORD ...]\n";
    return 2;
  }
  try {
    const std::string output_path = argv[1];
    const std::string lidar_topic = argv[2];
    const std::string imu_topic = argv[3];
    const int scan_lines = std::stoi(argv[4]);
    const int64_t start_ns = ParseInt64(argv[5], "start_ns");
    const int64_t end_ns = ParseInt64(argv[6], "end_ns");
    if (scan_lines <= 0 || scan_lines > std::numeric_limits<uint16_t>::max()) {
      throw std::runtime_error("scan_lines must fit uint16 and be positive");
    }

    ProtobufFactory *factory = ProtobufFactory::Instance();
    if (factory == nullptr) {
      throw std::runtime_error("Unable to initialize Core protobuf factory");
    }
    std::unordered_set<std::string> registered_types;
    std::vector<Scan> scans;
    std::vector<ImuSample> imu_samples;
    std::map<std::pair<std::string, std::string>, Transform> transforms;
    std::string lidar_frame;
    std::string imu_frame;

    for (int arg = 7; arg < argc; ++arg) {
      RecordReader reader(argv[arg]);
      if (!reader.IsValid()) {
        throw std::runtime_error("Unable to open record: " +
                                 std::string(argv[arg]));
      }
      RecordMessage record;
      while (reader.ReadMessage(&record)) {
        if (record.channel_name != lidar_topic &&
            record.channel_name != imu_topic &&
            record.channel_name != "/tf_static") {
          continue;
        }
        auto message = Decode(reader, record, *factory, &registered_types);
        if (record.channel_name == lidar_topic) {
          ReadPointCloud(*message, start_ns, end_ns,
                         static_cast<uint32_t>(scan_lines), &lidar_frame,
                         &scans);
        } else if (record.channel_name == imu_topic) {
          ReadImu(*message, record.channel_name, record.time, start_ns, end_ns,
                  &imu_frame, &imu_samples);
        } else {
          ReadTransforms(*message, record.channel_name, &transforms);
        }
      }
    }
    if (scans.empty())
      throw std::runtime_error("No LiDAR scans were decoded");
    if (imu_samples.empty())
      throw std::runtime_error("No IMU samples were decoded");
    if (!std::is_sorted(scans.begin(), scans.end(),
                        [](const Scan &lhs, const Scan &rhs) {
                          return lhs.timestamp_ns < rhs.timestamp_ns;
                        })) {
      throw std::runtime_error("LiDAR scans are not time ordered");
    }
    if (!std::is_sorted(imu_samples.begin(), imu_samples.end(),
                        [](const ImuSample &lhs, const ImuSample &rhs) {
                          return lhs.timestamp_ns < rhs.timestamp_ns;
                        })) {
      throw std::runtime_error("IMU samples are not time ordered");
    }
    WriteCanonical(output_path, lidar_frame.empty() ? "lidar" : lidar_frame,
                   imu_frame.empty() ? "imu" : imu_frame, scans, imu_samples,
                   transforms);
    uint64_t points = 0;
    for (const auto &scan : scans)
      points += scan.points.size();
    std::cout << "lidar_scans=" << scans.size() << " lidar_points=" << points
              << " imu_samples=" << imu_samples.size()
              << " static_transforms=" << transforms.size() << '\n';
  } catch (const std::exception &error) {
    std::cerr << "record_ingress: " << error.what() << '\n';
    return 1;
  }
  return 0;
}
